"""Stage 5. Subspace methods GIVEN the nominal reference, extended
to every category through the registration of stage 6 and with scores
normalized by the nominal set's own spread.

On Real3D-AD nothing local beats chance while a plain point-to-template distance does.
The question here is what subspace methods do when they are given the same nominal
reference. This stage answers it at every seed of the
stage 1 lattice, in the identical tangent frame.

The four good templates of a category are mutually registered to 0.03 to 0.17 mm (checked
2026-09-15), so at every seed a location-specific NOMINAL SET exists: the four template
patches at the seed plus four quarter-radius jitters of each (20 height fields on the
canonical grid). From it:

  nom_rec_k{0,1,3,6}  reconstruction error of the test height field in the PCA subspace of
                      the nominal set (CLAFIC / one-class subspace at the location; K=0 is
                      the distance to the nominal mean)
  nom_rec_k6_norm     the same at K=6 divided by the median leave-one-out reconstruction
                      error of the nominal patches themselves: how far the test patch sits
                      off the nominal sheet in units of how far the templates sit off it.
                      Windows that run off the object have a wide nominal spread and are
                      suppressed by construction
  nom_nn              RMS distance to the nearest nominal patch (amplitude, jitter tolerant)
  diff_rms, diff_absmean, diff_mean
                      difference to the best-fitting template's patch at the seed (the
                      round-1 baseline; diff_mean keeps the sign, positive = outward)
  magd_tmpl           Mag(D) test vs best-template patch
  magd_nom_min        min over the nominal set of Mag(D) test vs nominal patch
  magd_nom_sub        canonical angle of the test plane-residual DIRECTION against the
                      3-D subspace spanned by the nominal plane residuals, 2(1 - sigma):
                      amplitude-free by construction (Proposition 1)
  nn_p95              plain point-to-template distance

Registration: with --reg (default on when ~/accv_results/real3d/registration exists) the
TEST cloud and its seeds are moved into the template frame by the stage-6 transform, so
the cached templates need no per-item transform. Items whose registration was rejected are
skipped and listed. Every other choice (jitter 0.25 radius, K in {0,1,3,6}, same seeds,
frames and grid as stages 1 and 3) was fixed before any label was read.
Output: <items-dir>/<cat>__<stem>__r<r>.npz in the stage-3 format.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, time, argparse
import r3_common as C
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool

ALIGN_TOL_MM = 0.3
JITTER = 0.25
OFFS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
K_LIST = [0, 1, 3, 6]
STATS = (["nom_rec_k%d" % k for k in K_LIST] +
         ["nom_rec_k6_norm", "nom_rec_k6_pl", "nom_rec_k6_pl_norm", "nom_nn", "diff_rms",
          "diff_absmean", "diff_mean", "magd_tmpl", "magd_nom_min", "magd_nom_sub", "nn_p95", "cov_max"])
# nom_rec_k6_pl / _pl_norm: the same reconstruction error and its LOO-normalized form on
# PLANE-DETRENDED height fields (test and nominal patches each projected by grid.P), which
# removes a residual offset along the normal and a residual tilt, i.e. the part of a small
# registration error a local rigid plane can absorb.
REGDIR = os.path.join(C.OUT, "registration")
_TREES = {}
_ARGS = {"items_dir": "items_nominal", "reg": False, "tol": ALIGN_TOL_MM}


TEMPLATE_SUB = 300000   # templates carry 0.2 to 2.4 million points on 20 to 30 mm objects; a
                        # seeded 300k subsample is still several times denser than a test scan
                        # and keeps the ball queries of the dense categories tractable


def template_trees(cat):
    if cat not in _TREES:
        d = {}
        for t in C.enumerate_templates(cat):
            T = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "train", t + ".pcd"))
            if len(T) > TEMPLATE_SUB:
                T = T[np.random.default_rng(0).choice(len(T), TEMPLATE_SUB, replace=False)]
            d[t] = (T, cKDTree(T))
        _TREES[cat] = d
    return _TREES[cat]


def load_registration(cat, stem):
    p = os.path.join(REGDIR, "%s__%s.json" % (cat, stem))
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return np.asarray(r["R"]), np.asarray(r["t"]), bool(r["accepted"]), r["fit_after_mm"]


def loo_error(Nc, K):
    """Median leave-one-out reconstruction RMS of the centred nominal patches at rank K."""
    m = len(Nc)
    errs = []
    for i in range(m):
        rest = np.delete(Nc, i, axis=0)
        mu = rest.mean(0)
        _, _, Vt = np.linalg.svd(rest - mu, full_matrices=False)
        d = Nc[i] - mu
        r = d - Vt[:K].T @ (Vt[:K] @ d) if K > 0 else d
        errs.append(np.sqrt(np.mean(r ** 2)))
    return float(np.median(errs))


def score_item(args):
    cat, stem, lab, radius = args
    src = os.path.join(C.OUT, "items", "%s__%s__r%.1f.npz" % (cat, stem, radius))
    dst = os.path.join(C.OUT, _ARGS["items_dir"], "%s__%s__r%.1f.npz" % (cat, stem, radius))
    if os.path.exists(dst):
        return dst, "skip"
    if not os.path.exists(src):
        return dst, "no-stage1-file"
    t0 = time.time()
    z = np.load(src)
    P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "test", stem + ".pcd"))
    seeds = z["seeds"].copy()
    reg_fit = None
    if _ARGS["reg"]:
        reg = load_registration(cat, stem)
        if reg is None:
            return dst, "no-registration"
        R, t, ok, reg_fit = reg
        if reg_fit > _ARGS["tol"]:
            return dst, "unaligned %.2fmm" % reg_fit
        P = P @ R.T + t
        seeds = seeds @ R.T + t
    trees = template_trees(cat)
    sub = P[:: max(1, len(P) // 4000)]
    fit = {t: float(np.median(tr.query(sub)[0])) for t, (T, tr) in trees.items()}
    tbest = min(fit, key=fit.get)
    if fit[tbest] > _ARGS["tol"]:
        return dst, "unaligned %.2fmm" % fit[tbest]
    nominal = [t for t in trees if fit[t] <= max(_ARGS["tol"], 2.0 * fit[tbest])]

    grid = C.Grid()
    XY = np.c_[grid.X * radius, grid.Y * radius]
    valid = z["valid"].copy()
    tree = cKDTree(P)
    obj_c = P.mean(0)
    m = len(seeds)
    f = {k: np.full(m, np.nan) for k in STATS}
    ball_t = radius * (1.6 + JITTER)
    for j in np.flatnonzero(valid):
        c = seeds[j]
        idx = tree.query_ball_point(c, radius * 1.6)
        if len(idx) < C.MIN_BALL_PTS:
            valid[j] = False
            continue
        nb = P[idx]
        F = C.local_frame(nb, c, obj_c)
        h, cov = C.resample(nb, c, F, grid, radius, return_cov=True)
        N, N0, covs = [], None, [cov]
        for t in nominal:
            T, ttree = trees[t]
            tidx = ttree.query_ball_point(c, ball_t)
            if len(tidx) < C.MIN_BALL_PTS:
                continue
            Tn = T[tidx]
            for ox, oy in OFFS:
                cc = c + F[:, 0] * (ox * JITTER * radius) + F[:, 1] * (oy * JITTER * radius)
                ht, cv = C.resample(Tn, cc, F, grid, radius, return_cov=True)
                N.append(ht)
                if t == tbest and ox == 0 and oy == 0:
                    N0, cov0 = ht, cv
        if N0 is None or len(N) < 5:
            valid[j] = False
            continue
        covs.append(cov0)
        N = np.asarray(N)
        mu = N.mean(0)
        Nc = N - mu
        _, _, Vt = np.linalg.svd(Nc, full_matrices=False)
        d = h - mu
        for K in K_LIST:
            r = d - Vt[:K].T @ (Vt[:K] @ d) if K > 0 else d
            f["nom_rec_k%d" % K][j] = np.sqrt(np.mean(r ** 2))
        f["nom_rec_k6_norm"][j] = f["nom_rec_k6"][j] / max(loo_error(N, 6), 1e-6)
        # plane-detrended variant: offset and tilt removed from test and nominal patches alike
        hp = grid.P @ h
        Np = (grid.P @ N.T).T
        mup = Np.mean(0)
        _, _, Vtp = np.linalg.svd(Np - mup, full_matrices=False)
        dp = hp - mup
        rp = dp - Vtp[:6].T @ (Vtp[:6] @ dp)
        f["nom_rec_k6_pl"][j] = np.sqrt(np.mean(rp ** 2))
        f["nom_rec_k6_pl_norm"][j] = f["nom_rec_k6_pl"][j] / max(loo_error(Np, 6), 1e-6)
        f["nom_nn"][j] = np.sqrt(np.mean((N - h) ** 2, axis=1)).min()
        dd = h - N0
        f["diff_rms"][j] = np.sqrt(np.mean(dd ** 2))
        f["diff_absmean"][j] = abs(dd.mean())
        f["diff_mean"][j] = dd.mean()
        U, _ = C.shape_subspace(np.c_[XY, h])
        U0, _ = C.shape_subspace(np.c_[XY, N0])
        f["magd_tmpl"][j] = C.mag_d(U, U0)
        f["magd_nom_min"][j] = min(C.mag_d(U, C.shape_subspace(np.c_[XY, Nk])[0]) for Nk in N)
        rt = grid.P @ h
        rt = rt / max(np.linalg.norm(rt), 1e-12)
        Rn = (grid.P @ Nc.T).T
        _, _, Vr = np.linalg.svd(Rn, full_matrices=False)
        sig = np.linalg.norm(Vr[:3] @ rt)
        f["magd_nom_sub"][j] = 2.0 * (1.0 - min(sig, 1.0))
        inwin = nb[np.max(np.abs((nb - c) @ F[:, :2]), axis=1) <= radius]
        f["nn_p95"][j] = np.percentile(trees[tbest][1].query(inwin)[0], 95) if len(inwin) else np.nan
        f["cov_max"][j] = max(covs)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, cat=cat, stem=stem, label=lab, radius=radius,
                        seeds=z["seeds"], valid=valid, p1_check=z["p1_check"],
                        template=tbest, template_fit_mm=fit[tbest], n_nominal_templates=len(nominal),
                        registered=bool(_ARGS["reg"]), registration_fit_mm=reg_fit if reg_fit is not None else -1.0,
                        n_points=len(P), n_gt=z["n_gt"], n_defect=z["n_defect"],
                        pt_label=z["pt_label"], pt_seed=z["pt_seed"],
                        seed_defect_dist=z["seed_defect_dist"],
                        ref_med=json.dumps({}), ref_mad=json.dumps({}),
                        h_ptp=z["h_ptp"], **f)
    return dst, "%.1fs %d valid tmpl=%s nominal=%d" % (time.time() - t0, int(valid.sum()), tbest, len(nominal))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=C.CATS)
    ap.add_argument("--radii", type=float, nargs="*", default=C.RADII_MM)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--items-dir", default="items_nominal_reg")
    ap.add_argument("--no-reg", action="store_true", help="use the raw poses (round-2 behaviour)")
    ap.add_argument("--tol", type=float, default=ALIGN_TOL_MM,
                    help="max median NN distance (mm) between test scan and template to score an item")
    a = ap.parse_args()
    _ARGS["items_dir"] = a.items_dir
    _ARGS["reg"] = (not a.no_reg) and os.path.isdir(REGDIR)
    _ARGS["tol"] = a.tol
    items = [i for i in C.enumerate_test_items() if i[0] in a.cats]
    jobs = sorted([(c, s, l, r) for (c, s, l) in items for r in a.radii])
    print("jobs:", len(jobs), "cats:", a.cats, "registration:", _ARGS["reg"], "->", a.items_dir, flush=True)
    skipped = []
    with Pool(a.procs, initializer=_init, initargs=(dict(_ARGS),)) as pool:
        for k, (p, msg) in enumerate(pool.imap(score_item, jobs, chunksize=4), 1):
            if msg.startswith("unaligned") or msg.startswith("no-"):
                skipped.append((os.path.basename(p), msg))
            elif msg != "skip":
                print("[%d/%d] %s %s" % (k, len(jobs), os.path.basename(p), msg), flush=True)
    print("skipped %d:" % len(skipped), skipped[:20], flush=True)
    with open(os.path.join(C.OUT, a.items_dir + "_skipped.json"), "w") as fh:
        json.dump(skipped, fh, indent=1)
    print("ALL_DONE_R3_NOMINAL", flush=True)


def _init(args):
    _ARGS.update(args)


if __name__ == "__main__":
    main()
