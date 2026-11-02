"""Stage 6. Rigid registration of every Real3D-AD test scan to its category's templates.

Five categories ship pre-registered (median nearest-neighbour distance to a template about
0.05 mm). The other seven sit 1.6 to 2.6 mm away with rotated principal axes, so the
nominal-reference stages skipped them. This stage registers every test scan with a
trimmed point-to-point ICP so that the nominal stages can run on all 597 objects.

Per item: the test scan is subsampled, and for every template a set of rigid
initializations is tried: the identity, and the 24 proper rotations of the cube group
applied in the PCA frame (test principal axes mapped onto template principal axes, which
handles arbitrary orientation of a partial view up to the symmetry of the object). Each
initialization runs a short trimmed ICP (keep the 70 % closest correspondences, Kabsch
update); the best few by median nearest-neighbour distance are refined, and the best
(template, transform) pair is kept. Acceptance: median NN distance of a fresh 20,000-point
subsample of the test scan to the template at or below ALIGN_TOL_MM after registration.
Nothing here reads a label. The transform maps TEST coordinates into the template frame:
p' = R p + t.

Output: registration/<cat>__<stem>.json with R, t, template, fit before and after, accepted.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, time, argparse, itertools
import r3_common as C
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool

ALIGN_TOL_MM = 0.3          # the pre-registered categories' level (0.03 to 0.05 mm) with margin
LOOSE_TOL_MM = 0.6          # accepted with a flag; the residual is carried as a covariate
OUTDIR = os.path.join(C.OUT, "registration")
N_COARSE, N_FINE, N_CHECK = 1500, 8000, 20000
TEMPLATE_SUB = 250000
TRIM = 0.7
EXTRA_DEG = (45.0, 135.0, 225.0)   # in-plane rotations about each principal axis, on top of the cube group
_TPL = {}


def axis_rotation(axis, deg):
    a = np.zeros(3); a[axis] = 1.0
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


EXTRA = [axis_rotation(ax, d) for ax in range(3) for d in EXTRA_DEG]


def cube_rotations():
    """The 24 proper rotations of the cube (signed axis permutations with det +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


CUBE = cube_rotations()


def pca_frame(X):
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return c, R


def kabsch(A, B):
    """Rigid transform (R, t) minimizing |R a + t - b| over paired rows."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def icp(P, tree, T, R, t, iters, trim=TRIM):
    for _ in range(iters):
        Q = P @ R.T + t
        d, idx = tree.query(Q)
        keep = d <= np.percentile(d, 100 * trim)
        R2, t2 = kabsch(Q[keep], T[idx[keep]])
        R, t = R2 @ R, R2 @ t + t2
        if np.abs(R2 - np.eye(3)).max() < 1e-6 and np.linalg.norm(t2) < 1e-5:
            break
    return R, t


def median_nn(P, tree, R, t):
    return float(np.median(tree.query(P @ R.T + t)[0]))


def templates(cat):
    if cat not in _TPL:
        d = {}
        for name in C.enumerate_templates(cat):
            T = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "train", name + ".pcd"))
            rng = np.random.default_rng(0)
            Ts = T[rng.choice(len(T), min(TEMPLATE_SUB, len(T)), replace=False)]
            d[name] = (Ts, cKDTree(Ts), pca_frame(Ts))
        _TPL[cat] = d
    return _TPL[cat]


def register_item(args):
    cat, stem, lab = args
    dst = os.path.join(OUTDIR, "%s__%s.json" % (cat, stem))
    if os.path.exists(dst):
        return dst, "skip"
    t0 = time.time()
    P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "test", stem + ".pcd"))
    rng = np.random.default_rng(0)
    Pc = P[rng.choice(len(P), min(N_COARSE, len(P)), replace=False)]
    Pf = P[rng.choice(len(P), min(N_FINE, len(P)), replace=False)]
    Pk = P[rng.choice(len(P), min(N_CHECK, len(P)), replace=False)]
    cP, RP = pca_frame(Pf)
    cands = []
    before = {}
    # fast path: scans that already sit on a template (the pre-registered categories) need
    # only a polish from the identity; the full initialization sweep runs otherwise
    quick = []
    for name, (Ts, tree, (cT, RT)) in templates(cat).items():
        before[name] = median_nn(Pk, tree, np.eye(3), np.zeros(3))
        R, t = icp(Pf, tree, Ts, np.eye(3), np.zeros(3), 30)
        quick.append((median_nn(Pk, tree, R, t), name, R, t))
    quick.sort(key=lambda x: x[0])
    inits = []
    if quick[0][0] <= 0.1:
        cands = quick
    for name, (Ts, tree, (cT, RT)) in ([] if cands else templates(cat).items()):
        inits = [(np.eye(3), np.zeros(3))]
        for G in CUBE + [G @ E for G in CUBE for E in EXTRA]:
            R0 = RT @ G @ RP.T                  # test PCA axes -> (cube rotation, extra) -> template PCA axes
            inits.append((R0, cT - R0 @ cP))
        for R0, t0_ in inits:
            R, t = icp(Pc, tree, Ts, R0, t0_, 6)
            cands.append((median_nn(Pc, tree, R, t), name, R, t))
    cands.sort(key=lambda x: x[0])
    best = None
    for fit, name, R, t in cands[:8]:
        Ts, tree, _ = templates(cat)[name]
        R, t = icp(Pf, tree, Ts, R, t, 80)
        R, t = icp(Pk, tree, Ts, R, t, 20)      # polish on the check subsample
        fit = median_nn(Pk, tree, R, t)
        if best is None or fit < best[0]:
            best = (fit, name, R, t)
    fit, name, R, t = best
    rec = {"cat": cat, "stem": stem, "label": lab, "template": name,
           "R": R.tolist(), "t": t.tolist(), "n_inits_per_template": len(inits),
           "fit_before_mm": before, "fit_after_mm": fit,
           "accepted": bool(fit <= ALIGN_TOL_MM), "accepted_loose": bool(fit <= LOOSE_TOL_MM),
           "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(OUTDIR, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(rec, f, indent=1)
    return dst, "%.1fs %s before %.2f after %.3f rot %.0f deg %s" % (
        rec["seconds"], name, min(before.values()), fit, rec["rotation_deg"],
        "OK" if rec["accepted"] else ("LOOSE" if rec["accepted_loose"] else "REJECT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=C.CATS)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    items = sorted(i for i in C.enumerate_test_items() if i[0] in a.cats)
    print("items:", len(items), flush=True)
    with Pool(a.procs) as pool:
        for k, (p, msg) in enumerate(pool.imap(register_item, items, chunksize=2), 1):
            if msg != "skip":
                print("[%d/%d] %s %s" % (k, len(items), os.path.basename(p), msg), flush=True)
    recs = [json.load(open(os.path.join(OUTDIR, f))) for f in sorted(os.listdir(OUTDIR)) if f.endswith(".json")]
    summary = {}
    for cat in C.CATS:
        rs = [r for r in recs if r["cat"] == cat]
        if rs:
            summary[cat] = {"n": len(rs), "accepted": sum(r["accepted"] for r in rs),
                            "accepted_loose": sum(r.get("accepted_loose", r["accepted"]) for r in rs),
                            "median_fit_after_mm": float(np.median([r["fit_after_mm"] for r in rs])),
                            "median_fit_before_mm": float(np.median([min(r["fit_before_mm"].values()) for r in rs])),
                            "median_rotation_deg": float(np.median([r["rotation_deg"] for r in rs]))}
    with open(os.path.join(C.OUT, "registration_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for cat, s in summary.items():
        print("%-9s n=%d accepted=%d before %.2f -> after %.3f mm, rotation %.0f deg" % (
            cat, s["n"], s["accepted"], s["median_fit_before_mm"], s["median_fit_after_mm"], s["median_rotation_deg"]), flush=True)
    print("ALL_DONE_R3_REGISTER", flush=True)


if __name__ == "__main__":
    main()
