"""Stage 4. The learned methods on Real3D-AD, zero-shot from the fuselage checkpoints.

Every valid seed of the stage 1 lattice (radius 3.6 mm files, stride 1.8 mm) gets the
64x64 residual window the learned methods were trained on: the surface is resampled in the
seed's tangent frame onto a 64x64 grid at the benchmark pixel pitch (0.1875 mm), slab
filtered as in r3_common.resample, and detrended by a quadric fitted on the outer ring of
the window (the window-local analogue of the paper's RANSAC quadric, deterministic and
unable to absorb a defect that sits in the centre). All three models read that window:

  dentnet_mean / dentnet_max   softmax dent probability of the end-to-end U-Net, mean and
                               max over the window (the paper's patch score is the mean)
  pbss_k50                     frozen encoder, PCA nearest subspace K=50, e_normal - e_dent
  nearest_mean                 frozen encoder, nearest class mean
  logistic                     frozen encoder, logistic head, logit
  pnpp                         PointNet++ (grouping radius 0.3, the fuselage checkpoint) on
                               1024 (x, y, r) points drawn from the same window, sigmoid

The rules for the frozen encoder are fitted on the cached synthetic training features
(~/accv_results/pbss/features_train.npz), exactly as in p1_pbss_head.py. Nothing is fitted
to Real3D-AD.

Two protocols, both fixed before any score was looked at:
  --scale 1     metric pitch, the paper's real-scan protocol; a 5 mm defect is 27 px wide
                inside the 64 px window, whereas synthetic dents are 34 to 119 mm across
  --scale 6.6   one similarity transform of the coordinates (x, y, h all multiplied) that
                maps the median Real3D-AD defect extent (5.17 mm) onto the 10th percentile of
                the synthetic dent extent (34.1 mm), i.e. the smallest dents the networks saw
                in training. It also brings the 0.03 mm sensor floor to 0.2 mm, close to the
                0.15 mm synthetic noise.

Writes ~/accv_results/real3d/items_learned_s<scale>/<cat>__<stem>__r3.6.npz with the keys
r3_aggregate.py needs, so
  r3_aggregate.py --items-dir items_learned_s1 --radii 3.6 --stats dentnet_mean dentnet_max
                  pbss_k50 nearest_mean logistic pnpp
evaluates it.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")       # main process; workers are capped below
import r3_common as C
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCH = 64
PITCH_MM = 0.1875
SEED_RADIUS = 3.6                 # the stage 1 lattice these windows are placed on
FRAME_BALL = SEED_RADIUS * 1.6    # same frame definition as stage 1
N_POINTS = 1024
FEATURES_TRAIN = os.path.expanduser("~/accv_results/pbss/features_train.npz")
PNPP_CKPT = os.path.expanduser("~/accv_results_real/pointnetpp_fuselage_r0.3/best.pth")
DENTNET_CKPT = os.path.join(ROOT, "best_dentnet_real_scale.pth")


# ------------------------------------------------------------------- windows (CPU workers)

def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def ring_quadric_residual(h, X, Y, ring):
    A = np.c_[np.ones_like(X), X, Y, X * X, X * Y, Y * Y]
    coef, *_ = np.linalg.lstsq(A[ring], h[ring], rcond=None)
    return h - A @ coef


def build_windows(args):
    cat, stem, lab, scale = args
    src = os.path.join(C.OUT, "items", "%s__%s__r%.1f.npz" % (cat, stem, SEED_RADIUS))
    z = np.load(src)
    P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "test", stem + ".pcd"))
    tree = cKDTree(P)
    obj_c = P.mean(0)
    seeds, valid = z["seeds"], z["valid"].copy()
    pitch = PITCH_MM / scale                       # metric pitch in the object's own mm
    ax = (np.arange(PATCH) - (PATCH - 1) / 2.0) * pitch
    Yg, Xg = np.meshgrid(ax, ax, indexing="ij")
    Xf, Yf = Xg.ravel(), Yg.ravel()
    half = ax[-1]
    ring = np.maximum(np.abs(Xf), np.abs(Yf)) > 0.5 * half

    class _G:                                        # duck-typed grid for C.resample
        X, Y, g = Xf / half, Yf / half, PATCH
    grid = _G()
    keep, R, cov, hptp = [], [], [], []
    for j in np.flatnonzero(valid):
        c = seeds[j]
        fidx = tree.query_ball_point(c, FRAME_BALL)
        widx = tree.query_ball_point(c, half * 1.5)
        if len(fidx) < C.MIN_BALL_PTS or len(widx) < C.MIN_BALL_PTS:
            valid[j] = False
            continue
        F = C.local_frame(P[fidx], c, obj_c)
        h, cv = C.resample(P[widx], c, F, grid, half, return_cov=True)
        r = ring_quadric_residual(h, Xf, Yf, ring) * scale
        keep.append(j); R.append(r.astype(np.float32)); cov.append(cv); hptp.append(np.ptp(h) * scale)
    return dict(cat=cat, stem=stem, label=lab, scale=scale, src=src, keep=np.array(keep, int),
                R=np.array(R, np.float32).reshape(-1, PATCH, PATCH), cov=np.array(cov), hptp=np.array(hptp),
                valid=valid, Xf=(Xf * scale).astype(np.float32), Yf=(Yf * scale).astype(np.float32))


# --------------------------------------------------------------------- models (GPU, main)

def normalize_patch(p):
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    s = np.linalg.norm(q, axis=1).mean()
    return (q / (s if s > 1e-8 else 1.0)).astype(np.float32)


class Models:
    def __init__(self, device):
        import torch
        from inference_ransac import load_model
        from feature_extractor import DentNetFeatureExtractor
        from subspace_builder import SubspaceBuilder
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from pnpp_model import PatchClassifier
        self.torch, self.device = torch, device
        self.dentnet = load_model(DENTNET_CKPT, device=device)
        self.extractor = DentNetFeatureExtractor(model_path=DENTNET_CKPT, device=device)
        tr = np.load(FEATURES_TRAIN)
        X, y = tr["features"], tr["labels"].astype(np.int8)
        print("fitting rules on synthetic train features", X.shape, "dent", int(y.sum()), flush=True)
        self.pbss = SubspaceBuilder(n_components=50)
        self.pbss.build_subspaces(X[y == 1], X[y == 0])
        self.mu_d, self.mu_n = X[y == 1].mean(0), X[y == 0].mean(0)
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(max_iter=2000, C=1.0).fit(self.sc.transform(X), y)
        self.pnpp = PatchClassifier(norm_radius=0.3).to(device)
        self.pnpp.load_state_dict(torch.load(PNPP_CKPT, map_location=device))
        self.pnpp.eval()

    def score(self, R, Xf, Yf, rng):
        torch = self.torch
        n = len(R)
        out = {k: np.full(n, np.nan) for k in ("dentnet_mean", "dentnet_max", "pbss_k50",
                                               "nearest_mean", "logistic", "pnpp")}
        if n == 0:
            return out
        with torch.no_grad():
            for i in range(0, n, 256):
                x = torch.from_numpy(R[i:i + 256]).float().unsqueeze(1).to(self.device)
                pr = torch.softmax(self.dentnet(x), dim=1)[:, 1].cpu().numpy()
                out["dentnet_mean"][i:i + 256] = pr.reshape(len(pr), -1).mean(1)
                out["dentnet_max"][i:i + 256] = pr.reshape(len(pr), -1).max(1)
        feats = self.extractor.extract_features_batch([r for r in R], batch_size=256)
        _, conf = self.pbss.classify_features(feats)
        out["pbss_k50"] = np.asarray(conf, float)
        out["nearest_mean"] = (np.linalg.norm(feats - self.mu_n, axis=1) -
                               np.linalg.norm(feats - self.mu_d, axis=1))
        out["logistic"] = self.lr.decision_function(self.sc.transform(feats))
        pts = np.empty((n, N_POINTS, 3), np.float32)
        for k in range(n):
            sel = rng.choice(PATCH * PATCH, N_POINTS, replace=False)
            pts[k] = normalize_patch(np.c_[Xf[sel], Yf[sel], R[k].ravel()[sel]])
        with torch.no_grad():
            for i in range(0, n, 256):
                x = torch.from_numpy(pts[i:i + 256]).to(self.device)
                out["pnpp"][i:i + 256] = torch.sigmoid(self.pnpp(x)).cpu().numpy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    outdir = os.path.join(C.OUT, "items_learned_s%g" % a.scale)
    os.makedirs(outdir, exist_ok=True)
    items = C.enumerate_test_items()
    if a.limit:
        keep = []
        for cat in C.CATS:
            ci = [i for i in items if i[0] == cat]
            keep += [i for i in ci if i[2] == "good"][: a.limit] + [i for i in ci if i[2] != "good"][: a.limit]
        items = keep
    items = [i for i in items if not os.path.exists(os.path.join(outdir, "%s__%s__r%.1f.npz" % (i[0], i[1], SEED_RADIUS)))]
    print("scale %g: %d items to do" % (a.scale, len(items)), flush=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = Models(device)
    t0 = time.time()
    with Pool(a.procs, initializer=_init_worker) as pool:
        for k, w in enumerate(pool.imap(build_windows, [(c, s, l, a.scale) for (c, s, l) in items], chunksize=1), 1):
            rng = np.random.default_rng(abs(hash((w["cat"], w["stem"], a.scale))) % (2 ** 32))
            sc = models.score(w["R"], w["Xf"], w["Yf"], rng)
            z = np.load(w["src"])
            m = len(w["valid"])
            stats = {}
            for key, v in sc.items():
                full = np.full(m, np.nan); full[w["keep"]] = v; stats[key] = full
            cov = np.full(m, np.nan); cov[w["keep"]] = w["cov"]
            hptp = np.full(m, np.nan); hptp[w["keep"]] = w["hptp"]
            np.savez_compressed(os.path.join(outdir, "%s__%s__r%.1f.npz" % (w["cat"], w["stem"], SEED_RADIUS)),
                                cat=w["cat"], stem=w["stem"], label=w["label"], radius=SEED_RADIUS,
                                scale=a.scale, window_mm=PATCH * PITCH_MM / a.scale,
                                seeds=z["seeds"], valid=w["valid"], p1_check=np.nan,
                                n_points=z["n_points"], n_gt=z["n_gt"], n_defect=z["n_defect"],
                                pt_label=z["pt_label"], pt_seed=z["pt_seed"],
                                seed_defect_dist=z["seed_defect_dist"],
                                ref_med=json.dumps({}), ref_mad=json.dumps({}),
                                h_ptp=hptp, cov_max=cov, **stats)
            print("[%d/%d] %s %s %s: %d windows, dentnet_mean med %.3f max %.3f | pbss>0 %.2f | pnpp med %.3f  (%.0fs)" % (
                k, len(items), w["cat"], w["stem"], w["label"], len(w["keep"]),
                np.nanmedian(sc["dentnet_mean"]), np.nanmax(sc["dentnet_mean"]),
                np.nanmean(sc["pbss_k50"] > 0), np.nanmedian(sc["pnpp"]), time.time() - t0), flush=True)
    print("ALL_DONE_R3_LEARNED", flush=True)


if __name__ == "__main__":
    main()
