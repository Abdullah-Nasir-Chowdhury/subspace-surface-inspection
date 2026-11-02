"""Stage 2: PBSS — frozen DentNet encoder + PCA nearest-subspace classifier.

Honest re-run replacing the untraceable tables in ACCV2026_paper.tex, using
the modules in subspace/ (feature_extractor, subspace_builder, patch_utils).

Deviations from the legacy scripts, both documented in design.md:
  - subspaces are built from the TRAIN split (the legacy builder drew its
    "training" features from validation IDs 24000+, leaking the eval set);
  - patch label threshold is the code default 0.1 (the old draft's prose
    claimed 0.5; the code that produced its numbers used 0.1).

Outputs (~/accv_results/pbss/):
  - features_train.npz / features_val.npz     cached encoder features
  - pbss_val_metrics.json                     patch + pixel metrics per K
  - dentnet_val_patch_metrics.json            patch-level baseline
  - real_<S>.json / real_<S>_masks.npz / .png real samples A-D
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
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

from feature_extractor import DentNetFeatureExtractor
from inference_ransac import fit_quadric_ransac, load_model, predict_dents
from patch_utils import PatchExtractor, PatchReconstructor
from subspace_builder import SubspaceBuilder
from util_grid import detect_grid, load_ply_points, to_grids

DATA = os.path.expanduser("~/data/pointcloud_db_real_scale")
RES = os.path.expanduser("~/accv_results/pbss")
CKPT = "best_dentnet_real_scale.pth"
PATCH, STRIDE, LABEL_THR = 64, 32, 0.1
K_SWEEP = [10, 25, 50, 100]
K_DEFAULT = 50
TRAIN_IDS = list(range(0, 500))
VAL_IDS = list(range(24000, 24500))
REAL_SAMPLES = ["A", "B", "C", "D"]
SEED = 0


def sample_paths(i):
    return f"{DATA}/{i:05d}_input.npy", f"{DATA}/{i:05d}_gt.npy"


def extract_split(ids, extractor, pe, tag, model=None):
    """Extract encoder features (+ labels, + optional DentNet patch scores)
    for a list of synthetic samples. Cached to features_<tag>.npz."""
    cache = f"{RES}/features_{tag}.npz"
    if os.path.exists(cache):
        print(f"[cache] {cache}", flush=True)
        return dict(np.load(cache, allow_pickle=False))
    feats, labels, sample_idx, dn_scores = [], [], [], []
    for n, i in enumerate(ids):
        rp, gp = sample_paths(i)
        if not (os.path.exists(rp) and os.path.exists(gp)):
            continue
        residuals = np.load(rp)
        gt = np.load(gp).astype(np.uint8)
        info = pe.extract_patches(residuals)
        valid = np.array(info["valid_mask"])
        lab = pe.extract_patch_labels(gt, info["positions"],
                                      threshold=LABEL_THR)[valid]
        patches = [p for k, p in enumerate(info["patches"]) if valid[k]]
        f = extractor.extract_features_batch(patches, batch_size=256)
        feats.append(f.astype(np.float32))
        labels.append(lab.astype(np.int8))
        sample_idx.append(np.full(len(lab), i, dtype=np.int32))
        if model is not None:
            _, prob = predict_dents(model, residuals, fill_holes=False)
            s = [prob[y:y + PATCH, x:x + PATCH].mean()
                 for k, (y, x) in enumerate(info["positions"]) if valid[k]]
            dn_scores.append(np.array(s, dtype=np.float32))
        if (n + 1) % 50 == 0:
            print(f"[{tag}] {n+1}/{len(ids)} samples", flush=True)
    out = {
        "features": np.vstack(feats),
        "labels": np.concatenate(labels),
        "sample_idx": np.concatenate(sample_idx),
    }
    if model is not None:
        out["dentnet_scores"] = np.concatenate(dn_scores)
    np.savez_compressed(cache, **out)
    return out


def patch_metrics(labels, preds, scores):
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "n_patches": int(len(labels)), "n_dent_patches": int(labels.sum()),
    }
    try:
        out["auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        out["auc"] = None
    return out


def pixel_iou_per_sample(val, preds, pe, pr):
    """Reconstruct masks from patch predictions, IoU against GT per sample."""
    ious, dented = [], []
    # patch geometry is identical for every (640, 760) sample
    geom = pe.extract_patches(np.zeros((640, 760), dtype=np.float32))
    valid = np.array(geom["valid_mask"])
    vpos = [p for k, p in enumerate(geom["positions"]) if valid[k]]
    for i in np.unique(val["sample_idx"]):
        m = val["sample_idx"] == i
        mask = pr.reconstruct_from_patches(
            preds[m].astype(np.float32), vpos, (640, 760))
        gt = np.load(sample_paths(i)[1]).astype(np.uint8)
        inter = int(np.logical_and(mask, gt).sum())
        union = int(np.logical_or(mask, gt).sum())
        ious.append(inter / union if union else 1.0)
        dented.append(gt.sum() > 0)
    ious, dented = np.array(ious), np.array(dented)
    return {
        "mean_iou_all": float(ious.mean()),
        "mean_iou_dented": float(ious[dented].mean()) if dented.any() else None,
        "n_samples": int(len(ious)), "n_dented": int(dented.sum()),
    }


def run_real_samples(builder, extractor, model, pe, pr):
    for name in REAL_SAMPLES:
        path = f"realSamples/{name}.ply"
        if not os.path.exists(path):
            print(f"[real] missing {path}", flush=True)
            continue
        pts = load_ply_points(path)
        h, w = detect_grid(pts)
        Xg, Yg, Zg = to_grids(pts, h, w)
        residuals, inliers = fit_quadric_ransac(Xg, Yg, Zg,
                                                residual_threshold=0.3)
        residuals = -residuals  # negate: real dents appear as positive Z
        dn_mask, dn_prob = predict_dents(model, residuals)
        info = pe.extract_patches(residuals)
        valid = np.array(info["valid_mask"])
        patches = [p for k, p in enumerate(info["patches"]) if valid[k]]
        feats = extractor.extract_features_batch(patches, batch_size=256)
        preds, conf = builder.classify_features(feats)
        vpos = [p for k, p in enumerate(info["positions"]) if valid[k]]
        pb_mask = pr.reconstruct_from_patches(
            preds.astype(np.float32), vpos, (h, w))
        stats = {
            "sample": name, "grid": [h, w],
            "ransac_inlier_pct": float(100 * inliers.mean()),
            "dentnet_dent_fraction": float(dn_mask.mean()),
            "pbss_dent_fraction": float(pb_mask.mean()),
            "pbss_dent_patches": int(preds.sum()),
            "pbss_total_patches": int(len(preds)),
            "pbss_mean_confidence": float(conf.mean()),
        }
        print(f"[real] {name}: {json.dumps(stats)}", flush=True)
        with open(f"{RES}/real_{name}.json", "w") as f:
            json.dump(stats, f, indent=2)
        np.savez_compressed(f"{RES}/real_{name}_masks.npz",
                            residuals=residuals.astype(np.float32),
                            dentnet_mask=dn_mask.astype(np.uint8),
                            dentnet_prob=dn_prob.astype(np.float32),
                            pbss_mask=pb_mask.astype(np.uint8),
                            pbss_patch_conf=conf.astype(np.float32))
        fig, ax = plt.subplots(1, 4, figsize=(22, 5))
        ax[0].imshow(residuals, cmap="RdBu_r", vmin=-2, vmax=2)
        ax[0].set_title(f"{name}: residuals (RANSAC, negated)")
        ax[1].imshow(dn_mask, cmap="Reds")
        ax[1].set_title(f"DentNet mask ({100*dn_mask.mean():.1f}%)")
        ax[2].imshow(pb_mask, cmap="Greens")
        ax[2].set_title(f"PBSS mask K={builder.n_components} "
                        f"({100*pb_mask.mean():.1f}%)")
        ax[3].imshow(dn_prob, cmap="hot", vmin=0, vmax=1)
        ax[3].set_title("DentNet probability")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(f"{RES}/real_{name}.png", dpi=120)
        plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DentNetFeatureExtractor(model_path=CKPT, device=device)
    model = load_model(CKPT, device=device)
    pe = PatchExtractor(patch_size=PATCH, stride=STRIDE,
                        ignore_edge_patches=True)
    pr = PatchReconstructor(patch_size=PATCH, stride=STRIDE)

    train = extract_split(TRAIN_IDS, extractor, pe, "train")
    val = extract_split(VAL_IDS, extractor, pe, "val", model=model)

    dent_tr = train["features"][train["labels"] == 1]
    norm_tr = train["features"][train["labels"] == 0]
    print(f"train patches: dent={len(dent_tr)} normal={len(norm_tr)}",
          flush=True)

    # DentNet patch-level baseline on the same val patches
    dn_scores = val["dentnet_scores"]
    dn_preds = (dn_scores > 0.5).astype(np.int8)
    dn_patch = patch_metrics(val["labels"], dn_preds, dn_scores)
    with open(f"{RES}/dentnet_val_patch_metrics.json", "w") as f:
        json.dump(dn_patch, f, indent=2)
    print("[dentnet patch]", json.dumps(dn_patch), flush=True)

    # PBSS: K sweep on cached features
    results = {"train_dent_patches": int(len(dent_tr)),
               "train_normal_patches": int(len(norm_tr)),
               "label_threshold": LABEL_THR,
               "patch": PATCH, "stride": STRIDE, "K": {}}
    default_builder = None
    for K in K_SWEEP:
        builder = SubspaceBuilder(n_components=K)
        builder.build_subspaces(dent_tr, norm_tr)
        preds, conf = builder.classify_features(val["features"])
        results["K"][str(K)] = {
            "patch": patch_metrics(val["labels"], preds, conf),
            "pixel": pixel_iou_per_sample(val, preds, pe, pr),
            "explained_var_dent":
                float(builder.dent_subspace["explained_variance"]),
            "explained_var_normal":
                float(builder.normal_subspace["explained_variance"]),
        }
        print(f"[PBSS K={K}]", json.dumps(results["K"][str(K)]), flush=True)
        if K == K_DEFAULT:
            default_builder = builder
    with open(f"{RES}/pbss_val_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    run_real_samples(default_builder, extractor, model, pe, pr)
    print("ALL_DONE_S2", flush=True)


if __name__ == "__main__":
    main()
