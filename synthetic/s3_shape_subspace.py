"""Stage 3: training-free shape-subspace detection (canonical angles).

Runs the training-free shape subspace (subspace/shape_subspace.py)
on the real scans A-D and a slice of fresh synthetic validation samples, under
the unified protocol: same samples as the learned methods, patch-level scores,
pooled AUC against ground truth (synthetic only), per-sample z-scores.

Method orientation: for all five detectors (sv2, magd/ds neighbor,
magd/ds reference), higher score = more anomalous.

Synthetic surfaces are (X, Y, residual) points — the detrended surface, i.e.
the same input family the thesis' "image-based" variants used; raw real scans
are used untouched (the method's selling point: no preprocessing, no model).

Outputs (~/accv_results/shape_subspace/):
  - real_<S>_<nx>x<ny>.json / .png   score maps + z-scores per real sample
  - synthetic_metrics.json           pooled AUC + top-patch hit rate
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import json
import os
import sys

import numpy as np

PIPE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PIPE)
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from shape_subspace import compute_all_detection_methods_with_grid
from util_grid import detect_grid, load_ply_points

DATA = os.path.expanduser("~/data/pointcloud_db_real_scale")
RES = os.path.expanduser("~/accv_results/shape_subspace")
CONFIGS = [(6, 5), (10, 10)]  # (n_patches_x, n_patches_y)
METHODS = ["sv2", "magd_neighbor", "ds_neighbor",
           "magd_reference", "ds_reference"]
REAL_SAMPLES = ["A", "B", "C", "D"]
SYN_IDS = list(range(24000, 24050))
MAX_POINTS = 120_000  # subsample cap (thesis exp. 2: sparse sampling is fine)
GRID_SIZE = 10
LABEL_THR = 0.1
SEED = 0


def zscores(s):
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean()) / s.std() if s.std() > 0 else np.zeros_like(s)


def run_surface(points, nx, ny):
    out = compute_all_detection_methods_with_grid(
        points, n_patches_x=nx, n_patches_y=ny,
        use_grid_resampling=True, grid_size=GRID_SIZE,
        reference_method="edges")
    return {m: np.asarray(out[m], dtype=np.float64) for m in METHODS}, out


def patch_labels_from_gt(X, Y, gt, bounds):
    """Fraction of dent pixels inside each patch's metric bounds -> 0/1."""
    xf, yf, gf = X.ravel(), Y.ravel(), gt.ravel().astype(bool)
    labels = []
    for (x0, x1, y0, y1) in bounds:
        m = (xf >= x0) & (xf < x1) & (yf >= y0) & (yf < y1)
        frac = gf[m].mean() if m.any() else 0.0
        labels.append(1 if frac > LABEL_THR else 0)
    return np.array(labels, dtype=np.int8)


def real_pass():
    for name in REAL_SAMPLES:
        path = f"realSamples/{name}.ply"
        if not os.path.exists(path):
            print(f"[real] missing {path}", flush=True)
            continue
        pts = load_ply_points(path)
        h, w = detect_grid(pts)
        print(f"[real] {name}: {len(pts)} pts, grid {h}x{w}", flush=True)
        for nx, ny in CONFIGS:
            scores, _ = run_surface(pts, nx, ny)
            rec = {"sample": name, "nx": nx, "ny": ny, "grid": [h, w]}
            fig, ax = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4))
            for a, m in zip(ax, METHODS):
                z = zscores(scores[m])
                rec[m] = scores[m].tolist()
                rec[m + "_z"] = z.tolist()
                im = a.imshow(z.reshape(ny, nx), cmap="viridis")
                a.set_title(f"{m} (z)")
                plt.colorbar(im, ax=a, fraction=0.046)
            fig.suptitle(f"{name} — shape subspace, {nx}x{ny} patches")
            fig.tight_layout()
            fig.savefig(f"{RES}/real_{name}_{nx}x{ny}.png", dpi=120)
            plt.close(fig)
            with open(f"{RES}/real_{name}_{nx}x{ny}.json", "w") as f:
                json.dump(rec, f)
            print(f"[real] {name} {nx}x{ny} done", flush=True)


def synthetic_pass():
    rng = np.random.default_rng(SEED)
    pooled = {(nx, ny): {m: ([], []) for m in METHODS} for nx, ny in CONFIGS}
    top_hit = {(nx, ny): [] for nx, ny in CONFIGS}
    n_used = 0
    for i in SYN_IDS:
        base = f"{DATA}/{i:05d}"
        try:
            X = np.load(base + "_X.npy")
            Y = np.load(base + "_Y.npy")
            R = np.load(base + "_input.npy")
            gt = np.load(base + "_gt.npy").astype(np.uint8)
        except FileNotFoundError:
            continue
        pts = np.column_stack([X.ravel(), Y.ravel(), R.ravel()])
        if len(pts) > MAX_POINTS:
            pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
        for nx, ny in CONFIGS:
            try:
                scores, out = run_surface(pts, nx, ny)
            except Exception as e:  # keep the stage alive per-sample
                print(f"[syn] {i} {nx}x{ny} FAILED: {e}", flush=True)
                continue
            labels = patch_labels_from_gt(X, Y, gt, out["bounds"])
            for m in METHODS:
                pooled[(nx, ny)][m][0].append(labels)
                pooled[(nx, ny)][m][1].append(scores[m])
            if labels.any():
                for m in METHODS:
                    top_hit[(nx, ny)].append(
                        (m, bool(labels[int(np.argmax(scores[m]))])))
        n_used += 1
        if n_used % 10 == 0:
            print(f"[syn] {n_used} samples processed", flush=True)

    summary = {"n_samples": n_used, "label_threshold": LABEL_THR,
               "max_points": MAX_POINTS, "configs": {}}
    for nx, ny in CONFIGS:
        cfg = {}
        for m in METHODS:
            labs, scs = pooled[(nx, ny)][m]
            if not labs:
                continue
            labs = np.concatenate(labs)
            scs = np.concatenate(scs)
            try:
                auc = float(roc_auc_score(labs, scs))
            except ValueError:
                auc = None
            hits = [h for mm, h in top_hit[(nx, ny)] if mm == m]
            cfg[m] = {
                "pooled_auc": auc,
                "n_patches": int(len(labs)),
                "n_dent_patches": int(labs.sum()),
                "top_patch_hit_rate":
                    float(np.mean(hits)) if hits else None,
            }
        summary["configs"][f"{nx}x{ny}"] = cfg
        print(f"[syn] {nx}x{ny}: {json.dumps(cfg)}", flush=True)
    with open(f"{RES}/synthetic_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    os.makedirs(RES, exist_ok=True)
    np.random.seed(SEED)
    real_pass()
    synthetic_pass()
    print("ALL_DONE_S3", flush=True)


if __name__ == "__main__":
    main()
