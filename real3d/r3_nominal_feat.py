"""Stage 5b. The HYBRID: frozen learned features plus the nominal reference, extended to every category
through the registration of stage 6 and with a score normalized by the nominal spread.

Same seeds, frames and 64x64 ring-detrended residual windows as stage 4 at the pre-declared
scale 6.6 (the protocol where the fuselage encoders carry any signal on these objects). For
every seed the window is built on the test scan and on each of the four mutually registered
templates in the identical frame, and both are pushed through the frozen DentNet encoder
(1024-D) and the frozen PointNet++ encoder (128-D).

Per encoder, reference-based scores at the seed:
  <enc>_nom_nn         distance to the nearest of the four nominal feature vectors
  <enc>_nom_mean       distance to their mean
  <enc>_nom_rec3       reconstruction error in the 3-D PCA subspace of the four (their
                       affine hull): the local nominal subspace
  <enc>_nom_rec3_norm  the same divided by the median leave-one-out distance of each
                       nominal feature to the affine hull of the other three, i.e. the
                       test residual in units of the templates' own disagreement at that
                       location. Windows that run off the object have templates
                       that disagree, so they are suppressed by construction
and one category-level one-class subspace (CLAFIC on the encoder), fitted on the nominal
features of every seed of every item in the category, never on a test window:
  <enc>_cat_rec50      reconstruction error in the K=50 nominal subspace of the category
plus the stage-4 in-domain PBSS margin for reference (dentnet_pbss_k50).

With --reg (default on when the registration directory exists) the test cloud and seeds are
moved into the template frame by the stage-6 transform.
Output: <items-dir>/<cat>__<stem>__r3.6.npz, stage-4 format.
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
import r3_learned as L

SCALE = 6.6
ALIGN_TOL_MM = 0.3
JITTER = 0.25                                  # in-plane jitter of the nominal windows, in units of the half-window
OFFS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
N_NOM = 4 * len(OFFS)                          # 20 nominal windows per seed
REGDIR = os.path.join(C.OUT, "registration")
_TREES = {}
_ARGS = {"reg": False, "tol": ALIGN_TOL_MM, "pert_rot": 0.0, "pert_trans": 0.0, "pert_seed": 0}


def perturb_pose(P, seeds, cat, stem):
    """Registration sensitivity: after the stage-6 registration, rotate
    the test cloud and its seeds by --perturb-rot-deg about a random axis through the cloud
    centroid and translate them by --perturb-trans-mm along a random direction. The axis and
    direction are drawn per item from a generator seeded by (--perturb-seed, category, item
    number), so a run is reproducible and every level of the sweep perturbs the same item in
    the same direction. Off by default; the published runs are untouched."""
    if _ARGS["pert_rot"] <= 0 and _ARGS["pert_trans"] <= 0:
        return P, seeds
    prng = np.random.default_rng([int(_ARGS["pert_seed"]), C.CATS.index(cat), int(stem.split("_")[0])])
    axis = prng.normal(size=3); axis /= np.linalg.norm(axis)
    th = np.radians(_ARGS["pert_rot"])
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rp = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    u = prng.normal(size=3); u /= np.linalg.norm(u)
    tp = u * _ARGS["pert_trans"]
    c0 = P.mean(0)
    return (P - c0) @ Rp.T + c0 + tp, (seeds - c0) @ Rp.T + c0 + tp


TEMPLATE_SUB = 300000   # same seeded template subsample as r3_nominal.py


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


def build_pair(args):
    """Test windows and, per kept seed, the nominal windows of every aligned template."""
    cat, stem, lab, outdir = args
    dst = os.path.join(outdir, "%s__%s__r%.1f.npz" % (cat, stem, L.SEED_RADIUS))
    if os.path.exists(dst):
        return None
    z = np.load(os.path.join(C.OUT, "items", "%s__%s__r%.1f.npz" % (cat, stem, L.SEED_RADIUS)))
    P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "test", stem + ".pcd"))
    seeds = z["seeds"].copy()
    reg_fit = -1.0
    if _ARGS["reg"]:
        reg = load_registration(cat, stem)
        if reg is None:
            return dict(cat=cat, stem=stem, skipped="no-registration")
        R, t, ok, reg_fit = reg
        if reg_fit > _ARGS["tol"]:
            return dict(cat=cat, stem=stem, skipped="unaligned %.2f" % reg_fit)
        P = P @ R.T + t
        seeds = seeds @ R.T + t
    P, seeds = perturb_pose(P, seeds, cat, stem)
    trees = template_trees(cat)
    sub = P[:: max(1, len(P) // 4000)]
    fit = {t: float(np.median(tr.query(sub)[0])) for t, (T, tr) in trees.items()}
    tbest = min(fit, key=fit.get)
    if fit[tbest] > _ARGS["tol"]:
        return dict(cat=cat, stem=stem, skipped="unaligned %.2f" % fit[tbest])
    nominal = [t for t in trees if fit[t] <= max(_ARGS["tol"], 2.0 * fit[tbest])]
    tree = cKDTree(P)
    obj_c = P.mean(0)
    valid = z["valid"].copy()
    pitch = L.PITCH_MM / SCALE
    ax = (np.arange(L.PATCH) - (L.PATCH - 1) / 2.0) * pitch
    Yg, Xg = np.meshgrid(ax, ax, indexing="ij")
    Xf, Yf = Xg.ravel(), Yg.ravel()
    half = ax[-1]
    ring = np.maximum(np.abs(Xf), np.abs(Yf)) > 0.5 * half

    class _G:
        X, Y, g = Xf / half, Yf / half, L.PATCH
    grid = _G()
    keep, Rw, RN, cov, hptp, nnom = [], [], [], [], [], []
    for j in np.flatnonzero(valid):
        c = seeds[j]
        fidx = tree.query_ball_point(c, L.FRAME_BALL)
        widx = tree.query_ball_point(c, half * 1.5)
        if len(fidx) < C.MIN_BALL_PTS or len(widx) < C.MIN_BALL_PTS:
            valid[j] = False
            continue
        F = C.local_frame(P[fidx], c, obj_c)
        h, cv = C.resample(P[widx], c, F, grid, half, return_cov=True)
        noms = []
        for t in nominal:
            T, ttree = trees[t]
            tidx = ttree.query_ball_point(c, half * (1.5 + JITTER))
            if len(tidx) < C.MIN_BALL_PTS:
                continue
            Tn = T[tidx]
            for ox, oy in OFFS:               # the template window at the seed and at four in-plane jitters
                cc = c + F[:, 0] * (ox * JITTER * half) + F[:, 1] * (oy * JITTER * half)
                ht = C.resample(Tn, cc, F, grid, half)
                noms.append(L.ring_quadric_residual(ht, Xf, Yf, ring) * SCALE)
        if len(noms) < len(OFFS):       # at least one template within tolerance (with its jitters)
            valid[j] = False
            continue
        nnom.append(len(noms))
        while len(noms) < N_NOM:            # pad for fixed-shape storage; only the first nnom are used
            noms.append(noms[-1])
        keep.append(j)
        Rw.append((L.ring_quadric_residual(h, Xf, Yf, ring) * SCALE).astype(np.float32))
        RN.append(np.asarray(noms[:N_NOM], np.float32))
        cov.append(cv)
        hptp.append(np.ptp(h) * SCALE)
    src = {k: z[k] for k in ("seeds", "p1_check", "n_points", "n_gt", "n_defect",
                             "pt_label", "pt_seed", "seed_defect_dist")}
    return dict(cat=cat, stem=stem, label=lab, src=src, keep=np.array(keep, int),
                R=np.array(Rw, np.float32).reshape(-1, L.PATCH, L.PATCH),
                RN=np.array(RN, np.float32).reshape(-1, N_NOM, L.PATCH, L.PATCH),
                cov=np.array(cov), hptp=np.array(hptp), valid=valid, nnom=np.array(nnom, int),
                Xf=(Xf * SCALE).astype(np.float32), Yf=(Yf * SCALE).astype(np.float32),
                n_nominal=len(nominal), dst=dst, reg_fit=reg_fit, template_fit=fit[tbest])


class Encoders:
    def __init__(self, device):
        import torch
        from feature_extractor import DentNetFeatureExtractor
        from subspace_builder import SubspaceBuilder
        from pnpp_model import PatchClassifier
        self.torch, self.device = torch, device
        self.extractor = DentNetFeatureExtractor(model_path=L.DENTNET_CKPT, device=device)
        tr = np.load(L.FEATURES_TRAIN)
        X, y = tr["features"], tr["labels"].astype(np.int8)
        self.pbss = SubspaceBuilder(n_components=50)
        self.pbss.build_subspaces(X[y == 1], X[y == 0])
        self.pnpp = PatchClassifier(norm_radius=0.3).to(device)
        self.pnpp.load_state_dict(torch.load(L.PNPP_CKPT, map_location=device))
        self.pnpp.eval()

    def dentnet(self, R):
        if len(R) == 0:
            return np.zeros((0, 1024), np.float32)
        return np.asarray(self.extractor.extract_features_batch([r for r in R], batch_size=256), np.float32)

    def pointnet(self, R, Xf, Yf, rng):
        torch = self.torch
        n = len(R)
        if n == 0:
            return np.zeros((0, 128), np.float32)
        pts = np.empty((n, L.N_POINTS, 3), np.float32)
        for k in range(n):
            sel = rng.choice(L.PATCH * L.PATCH, L.N_POINTS, replace=False)
            pts[k] = L.normalize_patch(np.c_[Xf[sel], Yf[sel], R[k].ravel()[sel]])
        out = []
        with torch.no_grad():
            for i in range(0, n, 256):
                b = torch.from_numpy(pts[i:i + 256]).to(self.device)
                l1_xyz, l1_pts = self.pnpp.sa1(b, None)
                _, l2 = self.pnpp.sa2(l1_xyz, l1_pts)
                out.append(l2.view(l2.shape[0], -1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


K_LOCAL = 3


def subspace_residual(x, G, K):
    """Distance of x to the K-dimensional PCA subspace (through the mean) of the rows of G."""
    mu = G.mean(0)
    _, _, Vt = np.linalg.svd(G - mu, full_matrices=False)
    K = min(K, len(Vt))
    d = x - mu
    return float(np.linalg.norm(d - Vt[:K].T @ (Vt[:K] @ d)))


def local_scores(f, G, nnom):
    """f: (n, d) test features; G: (n, N_NOM, d) nominal features at the same seeds (up to
    four templates at the seed and at four in-plane jitters each); nnom[i] is the number of
    distinct nominal windows at seed i (the rest of G[i] is padding)."""
    nn = np.empty(len(f)); mean = np.empty(len(f)); rec3 = np.empty(len(f)); norm = np.empty(len(f))
    rng = np.random.default_rng(0)
    for i in range(len(f)):
        Gi = G[i][:max(int(nnom[i]), 2)]
        nn[i] = np.linalg.norm(Gi - f[i], axis=1).min()
        mean[i] = np.linalg.norm(f[i] - Gi.mean(0))
        rec3[i] = subspace_residual(f[i], Gi, K_LOCAL)
        # leave-one-out on (a subset of) the nominal windows, for the scale of their own spread
        loo = [subspace_residual(Gi[k], np.delete(Gi, k, axis=0), min(K_LOCAL, len(Gi) - 2))
               for k in rng.choice(len(Gi), min(8, len(Gi)), replace=False)]
        norm[i] = rec3[i] / max(float(np.median(loo)), 1e-6)
    return nn, mean, rec3, norm


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=C.CATS)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--items-dir", default="items_nominal_feat_reg")
    ap.add_argument("--no-reg", action="store_true")
    ap.add_argument("--tol", type=float, default=ALIGN_TOL_MM,
                    help="max median NN distance (mm) between test scan and template to score an item")
    ap.add_argument("--save-feats", action="store_true",
                    help="also write the encoded test and nominal features per item to "
                         "<items-dir>_feats/<cat>__<stem>.npz for r3_nominal_ablate.py")
    ap.add_argument("--perturb-rot-deg", type=float, default=0.0, help="see perturb_pose")
    ap.add_argument("--perturb-trans-mm", type=float, default=0.0, help="see perturb_pose")
    ap.add_argument("--perturb-seed", type=int, default=0)
    a = ap.parse_args()
    _ARGS["reg"] = (not a.no_reg) and os.path.isdir(REGDIR)
    _ARGS["tol"] = a.tol
    _ARGS["pert_rot"] = a.perturb_rot_deg
    _ARGS["pert_trans"] = a.perturb_trans_mm
    _ARGS["pert_seed"] = a.perturb_seed
    outdir = os.path.join(C.OUT, a.items_dir)
    os.makedirs(outdir, exist_ok=True)
    fdir = outdir + "_feats"
    if a.save_feats:
        os.makedirs(fdir, exist_ok=True)
    print("registration:", _ARGS["reg"], "->", outdir, "perturb rot %.2f deg trans %.2f mm seed %d" % (
        a.perturb_rot_deg, a.perturb_trans_mm, a.perturb_seed), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoders(device)
    rng = np.random.default_rng(0)
    skipped = []
    for cat in a.cats:
        items = [(c, s, l, outdir) for (c, s, l) in C.enumerate_test_items() if c == cat]
        t0 = time.time()
        with Pool(a.procs, initializer=_init, initargs=(dict(_ARGS),)) as pool:
            built = [b for b in pool.imap(build_pair, items, chunksize=2) if b is not None]
        skipped += [(b["cat"], b["stem"], b["skipped"]) for b in built if "skipped" in b]
        skipped += [(b["cat"], b["stem"], "no-valid-window") for b in built if "skipped" not in b and len(b["keep"]) == 0]
        built = [b for b in built if "skipped" not in b and len(b["keep"]) > 0]
        print("[%s] %d items built in %.0fs" % (cat, len(built), time.time() - t0), flush=True)
        if not built:
            continue
        feats = []
        for b in built:
            n = len(b["keep"])
            fd = enc.dentnet(b["R"])
            gd = enc.dentnet(b["RN"].reshape(-1, L.PATCH, L.PATCH)).reshape(n, N_NOM, -1)
            fp = enc.pointnet(b["R"], b["Xf"], b["Yf"], rng)
            gp = enc.pointnet(b["RN"].reshape(-1, L.PATCH, L.PATCH), b["Xf"], b["Yf"], rng).reshape(n, N_NOM, -1)
            feats.append((fd, gd, fp, gp))
            if a.save_feats:
                np.savez_compressed(os.path.join(fdir, "%s__%s.npz" % (b["cat"], b["stem"])),
                                    cat=b["cat"], stem=b["stem"], label=b["label"], keep=b["keep"],
                                    nnom=b["nnom"], cov=b["cov"], n_nominal=b["n_nominal"],
                                    reg_fit=b["reg_fit"], template_fit=b["template_fit"],
                                    fd=fd, gd=gd, fp=fp, gp=gp)
        cat_sub = {}
        for name, col in (("dentnet", 1), ("pnpp", 3)):
            Gall = np.concatenate([f[col].reshape(-1, f[col].shape[-1]) for f in feats])
            mu = Gall.mean(0)
            _, _, Vt = np.linalg.svd(Gall - mu, full_matrices=False)
            cat_sub[name] = (mu, Vt[:50])
        print("[%s] category nominal subspaces from %d nominal windows" % (cat, len(Gall)), flush=True)
        for b, (fd, gd, fp, gp) in zip(built, feats):
            z = b["src"]
            m = len(z["seeds"])
            out = {}
            for name, f, G in (("dentnet", fd, gd), ("pnpp", fp, gp)):
                nn, mean, rec3, norm = local_scores(f, G, b["nnom"])
                mu, V = cat_sub[name]
                d = f - mu
                rec50 = np.linalg.norm(d - (d @ V.T) @ V, axis=1)
                for k, v in (("nom_nn", nn), ("nom_mean", mean), ("nom_rec3", rec3),
                             ("nom_rec3_norm", norm), ("cat_rec50", rec50)):
                    arr = np.full(m, np.nan)
                    arr[b["keep"]] = v
                    out["%s_%s" % (name, k)] = arr
            _, conf = enc.pbss.classify_features(fd)
            arr = np.full(m, np.nan); arr[b["keep"]] = np.asarray(conf, float)
            out["dentnet_pbss_k50"] = arr
            cov = np.full(m, np.nan); cov[b["keep"]] = b["cov"]
            hptp = np.full(m, np.nan); hptp[b["keep"]] = b["hptp"]
            np.savez_compressed(b["dst"], cat=b["cat"], stem=b["stem"], label=b["label"], radius=L.SEED_RADIUS,
                                scale=SCALE, seeds=z["seeds"], valid=b["valid"], p1_check=z["p1_check"],
                                n_points=z["n_points"], n_gt=z["n_gt"], n_defect=z["n_defect"],
                                pt_label=z["pt_label"], pt_seed=z["pt_seed"],
                                seed_defect_dist=z["seed_defect_dist"],
                                ref_med=json.dumps({}), ref_mad=json.dumps({}),
                                h_ptp=hptp, cov_max=cov, n_nominal=b["n_nominal"],
                                registered=bool(_ARGS["reg"]), registration_fit_mm=b["reg_fit"],
                                template_fit_mm=b["template_fit"], **out)
        print("[%s] wrote %d items (%.0fs)" % (cat, len(built), time.time() - t0), flush=True)
    print("skipped %d:" % len(skipped), skipped[:20], flush=True)
    with open(os.path.join(C.OUT, a.items_dir + "_skipped.json"), "w") as fh:
        json.dump(skipped, fh, indent=1)
    print("ALL_DONE_R3_NOMINAL_FEAT", flush=True)


def _init(args):
    _ARGS.update(args)


if __name__ == "__main__":
    main()
