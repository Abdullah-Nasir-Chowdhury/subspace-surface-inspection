"""Shared helpers: BLAS caps, paths, the weighted-AUC cluster bootstrap and the decision rules.

Import this first: it caps BLAS threads before numpy loads, puts the repo's method
modules on sys.path, and chdir's to the repo root so relative paths in the legacy
modules resolve.

Everything here reads cached artefacts of the earlier pipelines and never refits RANSAC:
the real-scan residuals are the exact arrays in ~/accv_results/pbss/real_<S>_masks.npz
that the paper's masks were built from.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT = os.path.expanduser("~/accv_results_rev2")
os.makedirs(OUT, exist_ok=True)

PBSS_RES = os.path.expanduser("~/accv_results/pbss")
SHAPE_RES = os.path.expanduser("~/accv_results/shape_subspace")
FPR_JSON = os.path.expanduser("~/accv_results/fpr_real.json")
P1AUDIT = os.path.expanduser("~/accv_results/p1_audit")
PNPP_DIR = os.path.expanduser("~/accv_results_real/pointnetpp_fuselage_r0.3")
PNPP_CKPT = os.path.join(PNPP_DIR, "best.pth")
PNPP_REAL = os.path.expanduser("~/accv_results_real/pointnetpp_real")
PNPP_RADIUS = 0.3
PNPP_DATA = os.path.expanduser("~/data/fuselage_pnpp")
SYN = os.path.expanduser("~/data/pointcloud_db_real_scale")
DENTNET_CKPT = "best_dentnet_real_scale.pth"

SAMPLES = ["A", "B", "C", "D"]
KNOWN_A = (-63.84, -16.80)
R_ZONE_MM = 40.0          # the paper's damage-zone radius
R_ZONE_RES_MM = 20.0      # the resolution table's zone radius
SMOOTH_SIGMA_PX = 5
BORDER_FRAC = 0.1
PATCH, STRIDE, LABEL_THR, N_POINTS = 64, 32, 0.1, 1024
SEED = 0
N_BOOT = 2000


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def save_json(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
    log("wrote", p)
    return p


def write_md(name, lines):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    log("wrote", p)


# ------------------------------------------------------------------ metrics
def auc(scores, labels):
    """Rank AUC with ties averaged."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return WeightedAUC(scores, labels)(np.ones(len(scores)))


class WeightedAUC:
    """AUC under per-item multiplicity weights, O(n) per call after one sort.

    The bootstrap resamples clusters (samples, items); a cluster drawn m times gives every
    window in it weight m, and the weighted Mann-Whitney statistic is the AUC of the
    resampled set.
    """

    def __init__(self, scores, labels):
        u, inv = np.unique(np.asarray(scores, float), return_inverse=True)
        self.inv, self.n_u = inv, len(u)
        self.pos = np.asarray(labels, bool)

    def __call__(self, w):
        wp = np.bincount(self.inv, weights=w * self.pos, minlength=self.n_u)
        wn = np.bincount(self.inv, weights=w * (~self.pos), minlength=self.n_u)
        below = np.cumsum(wn) - wn
        sp, sn = wp.sum(), wn.sum()
        if sp == 0 or sn == 0:
            return float("nan")
        return float(np.sum(wp * (below + 0.5 * wn)) / (sp * sn))


def weighted_prf(pred, y, w):
    pred = np.asarray(pred, bool)
    y = np.asarray(y, bool)
    tp = float(np.sum(w * (pred & y)))
    fp = float(np.sum(w * (pred & ~y)))
    fn = float(np.sum(w * (~pred & y)))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    return p, r, f1


def cluster_weights(groups, n_boot=N_BOOT, seed=SEED):
    """Yield multiplicity weight vectors for a cluster bootstrap over `groups`."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        yield cnt[inv].astype(np.float64)


def ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [None, None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mean_ci(values, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    return float(v.mean()), ci(v[idx].mean(axis=1))


def patch_metrics_ci(scores, labels, groups, pred=None, thr=None, n_boot=1000,
                     seed=SEED, weights_iter=None):
    """AUC, precision, recall, F1 with cluster-bootstrap CIs.

    `pred` is the binary decision; if None it is scores > thr. Pass `weights_iter` (a
    list of weight vectors) to share draws across methods for paired differences.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    if pred is None:
        pred = scores > thr
    pred = np.asarray(pred, bool)
    wa = WeightedAUC(scores, labels)
    one = np.ones(len(scores))
    p0, r0, f0 = weighted_prf(pred, labels, one)
    out = {"auc": wa(one), "precision": p0, "recall": r0, "f1": f0,
           "n": int(len(labels)), "n_pos": int(labels.sum())}
    draws = {"auc": [], "f1": [], "precision": [], "recall": []}
    ws = weights_iter if weights_iter is not None else list(
        cluster_weights(groups, n_boot, seed))
    for w in ws:
        draws["auc"].append(wa(w))
        p, r, f = weighted_prf(pred, labels, w)
        draws["precision"].append(p)
        draws["recall"].append(r)
        draws["f1"].append(f)
    for k, v in draws.items():
        out[k + "_ci"] = ci(v)
    out["_draws"] = {k: np.asarray(v) for k, v in draws.items()}
    return out


# ------------------------------------------------------------------ real scans
def load_scan(name):
    from util_grid import detect_grid, load_ply_points, to_grids
    pts = load_ply_points(f"realSamples/{name}.ply")
    h, w = detect_grid(pts)
    Xg, Yg, Zg = to_grids(pts, h, w)
    return h, w, Xg, Yg, Zg


def stored_masks(name):
    return np.load(f"{PBSS_RES}/real_{name}_masks.npz")


def window_geometry(h, w, Xg, Yg):
    from patch_utils import PatchExtractor, PatchReconstructor
    pe = PatchExtractor(patch_size=PATCH, stride=STRIDE, ignore_edge_patches=True)
    pr = PatchReconstructor(patch_size=PATCH, stride=STRIDE)
    geom = pe.extract_patches(np.zeros((h, w), dtype=np.float32))
    valid = np.array(geom["valid_mask"])
    vpos = [p for k, p in enumerate(geom["positions"]) if valid[k]]
    cx = np.array([Xg[y:y + PATCH, x:x + PATCH].mean() for y, x in vpos])
    cy = np.array([Yg[y:y + PATCH, x:x + PATCH].mean() for y, x in vpos])
    return pe, pr, vpos, cx, cy


def window_patches(res, pe):
    info = pe.extract_patches(res)
    valid = np.array(info["valid_mask"])
    return [p for k, p in enumerate(info["patches"]) if valid[k]]


def window_mean(field, vpos):
    return np.array([field[y:y + PATCH, x:x + PATCH].mean() for y, x in vpos])


def residual_extremum_centre(res, Xg, Yg):
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(np.abs(res.astype(np.float64)), SMOOTH_SIGMA_PX)
    h, w = sm.shape
    mh, mw = int(BORDER_FRAC * h), int(BORDER_FRAC * w)
    m = np.full_like(sm, -np.inf)
    m[mh:h - mh, mw:w - mw] = sm[mh:h - mh, mw:w - mw]
    iy, ix = np.unravel_index(int(np.argmax(m)), sm.shape)
    return float(Xg[iy, ix]), float(Yg[iy, ix])


def outside(cx, cy, centre, r):
    return np.hypot(cx - centre[0], cy - centre[1]) > r


def pnpp_pointsets(Xg, Yg, res, vpos, rng):
    """Exactly p3_real_scans' construction: 1024-point seeded draw per window, centred
    and divided by mean radius. `rng` is shared across scans in A-D order as p3 did."""
    from patches import normalize_patch
    arr = np.empty((len(vpos), N_POINTS, 3), dtype=np.float32)
    for k, (y0, x0) in enumerate(vpos):
        sl = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
        q = np.stack([Xg[sl].ravel(), Yg[sl].ravel(), res[sl].ravel()], 1)
        sel = rng.choice(q.shape[0], N_POINTS, replace=False)
        arr[k] = normalize_patch(q[sel])
    return arr


# ------------------------------------------------------------------ models
def load_pnpp(device):
    import torch
    from pnpp_model import PatchClassifier
    model = PatchClassifier(norm_radius=PNPP_RADIUS).to(device)
    model.load_state_dict(torch.load(PNPP_CKPT, map_location=device))
    model.eval()
    return model


def pnpp_forward(model, arr, device, batch=512):
    """Returns (prob, feat128): the head's probability and the 128-D global feature
    after the second set abstraction, i.e. the frozen encoder output before fc1."""
    import torch
    probs, feats = [], []
    with torch.no_grad():
        for i0 in range(0, len(arr), batch):
            b = torch.from_numpy(np.asarray(arr[i0:i0 + batch], dtype=np.float32)).to(device)
            l1_xyz, l1_pts = model.sa1(b, None)
            _, l2_pts = model.sa2(l1_xyz, l1_pts)
            f = l2_pts.view(l2_pts.shape[0], -1)
            x = torch.nn.functional.relu(model.bn1(model.fc1(f)))
            x = model.drop(x)
            logit = model.fc2(x).squeeze(-1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            feats.append(f.cpu().numpy())
    return np.concatenate(probs), np.concatenate(feats)


def dentnet_extractor(device):
    from feature_extractor import DentNetFeatureExtractor
    return DentNetFeatureExtractor(model_path=DENTNET_CKPT, device=device)


# ------------------------------------------------------------------ decision rules
class Rule:
    name = ""

    def fit(self, X, y):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError

    def predict(self, X):
        return self.score(X) > 0


class PBSS(Rule):
    def __init__(self, K=50):
        self.K = K
        self.name = f"pbss_k{K}"

    def fit(self, X, y):
        from subspace_builder import SubspaceBuilder
        self.b = SubspaceBuilder(n_components=self.K)
        self.b.build_subspaces(X[y == 1], X[y == 0])

    def score(self, X):
        _, conf = self.b.classify_features(X)
        return np.asarray(conf, float)


class NearestMean(Rule):
    name = "nearest_mean"

    def fit(self, X, y):
        self.mu_d = X[y == 1].mean(0)
        self.mu_n = X[y == 0].mean(0)

    def score(self, X):
        return (np.linalg.norm(X - self.mu_n, axis=1)
                - np.linalg.norm(X - self.mu_d, axis=1))


class Logistic(Rule):
    name = "logistic"

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(max_iter=2000, C=1.0)
        self.lr.fit(self.sc.transform(X), y)

    def score(self, X):
        return self.lr.decision_function(self.sc.transform(X))


# ------------------------------------------------------------------ pixel IoU
def per_sample_iou(sample_idx, pred, pr, vpos, shape=(640, 760)):
    """Per-sample dent-class IoU of the mask rebuilt from window decisions, as
    s2_pbss.pixel_iou_per_sample does, but returning the per-sample arrays."""
    ids = np.unique(sample_idx)
    ious = np.zeros(len(ids))
    dented = np.zeros(len(ids), bool)
    pred = np.asarray(pred, np.float32)
    for n, i in enumerate(ids):
        m = sample_idx == i
        mask = pr.reconstruct_from_patches(pred[m], vpos, shape)
        gt = np.load(f"{SYN}/{i:05d}_gt.npy").astype(np.uint8)
        inter = int(np.logical_and(mask, gt).sum())
        union = int(np.logical_or(mask, gt).sum())
        ious[n] = inter / union if union else 1.0
        dented[n] = gt.sum() > 0
    return ids, ious, dented


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def fmt_ci(x, c, nd=3):
    if c is None or c[0] is None:
        return fmt(x, nd)
    return f"{fmt(x, nd)} [{fmt(c[0], nd)}, {fmt(c[1], nd)}]"
