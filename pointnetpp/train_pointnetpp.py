"""Stage P2: train PointNet++ on the PRIMARY fuselage task and report it
under the paper's unified protocol.

Same architecture, optimizer, loss, and normalization as the ModelNet runs
(pnpp_model.py), so the two PointNet++ numbers in the paper are
produced by one recipe. Model selection uses a held-out slice of the TRAIN
sample IDs, never the reported validation split.

Reported: patch ROC-AUC, F1, precision, recall on the same validation patches
PBSS and DentNet are scored on, plus pixel IoU reconstructed from patch
predictions by the same overlap voting PBSS uses, so PointNet++ can take a
row in the master comparison table.

Outputs (~/accv_results_real/pointnetpp_fuselage/):
  training_log.json, metrics.json, best.pth
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)

PIPE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from patch_utils import PatchExtractor, PatchReconstructor
from pnpp_model import PatchClassifier, NORM_RADIUS

DATA = os.path.expanduser("~/data/fuselage_pnpp")
SYN = os.path.expanduser("~/data/pointcloud_db_real_scale")
RES = os.path.expanduser("~/accv_results_real/pointnetpp_fuselage")
LOG = os.path.expanduser("~/accv_logs/pnpp_fuselage.log")

BATCH_SIZE = 256
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 8
VAL_FRACTION = 0.15
SEED = 0
PATCH, STRIDE = 64, 32


def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


class Patches(torch.utils.data.Dataset):
    def __init__(self, path, labels, idx=None):
        self.pts = np.load(path, mmap_mode="r")
        self.labels = labels.astype(np.float32)
        self.idx = np.arange(len(self.labels)) if idx is None else idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.asarray(self.pts[j], dtype=np.float32)),
                torch.tensor(self.labels[j]))


def run_epoch(model, loader, device, opt=None, pos_weight=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, n, P, L = 0.0, 0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for pts, labs in loader:
            pts, labs = pts.to(device, non_blocking=True), labs.to(device)
            logits = model(pts)
            loss = F.binary_cross_entropy_with_logits(
                logits, labs, pos_weight=pos_weight)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(labs)
            n += len(labs)
            P.append(torch.sigmoid(logits).detach().cpu().numpy())
            L.append(labs.detach().cpu().numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    try:
        auc = float(roc_auc_score(L, P))
    except ValueError:
        auc = None
    return tot / n, auc, P, L


def pixel_iou(val_probs, val_sample, thr=0.5):
    """Reconstruct masks from patch predictions with the same overlap voting
    PBSS uses, then per-sample dent-class IoU against ground truth."""
    pe = PatchExtractor(patch_size=PATCH, stride=STRIDE,
                        ignore_edge_patches=True)
    pr = PatchReconstructor(patch_size=PATCH, stride=STRIDE)
    geom = pe.extract_patches(np.zeros((640, 760), dtype=np.float32))
    valid = np.array(geom["valid_mask"])
    vpos = [p for k, p in enumerate(geom["positions"]) if valid[k]]
    preds = (val_probs >= thr).astype(np.float32)
    ious, dented = [], []
    for i in np.unique(val_sample):
        m = val_sample == i
        mask = pr.reconstruct_from_patches(preds[m], vpos, (640, 760))
        gt = np.load(f"{SYN}/{i:05d}_gt.npy").astype(np.uint8)
        inter = int(np.logical_and(mask, gt).sum())
        union = int(np.logical_or(mask, gt).sum())
        ious.append(inter / union if union else 1.0)
        dented.append(gt.sum() > 0)
    ious, dented = np.array(ious), np.array(dented)
    return {"mean_iou_all": float(ious.mean()),
            "mean_iou_dented": float(ious[dented].mean()),
            "n_samples": int(len(ious)), "n_dented": int(dented.sum())}


def main():
    os.makedirs(RES, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[p2] device={device}")

    tr_lab = np.load(f"{DATA}/train_labels.npy")
    tr_smp = np.load(f"{DATA}/train_sample.npy")
    va_lab = np.load(f"{DATA}/val_labels.npy")
    va_smp = np.load(f"{DATA}/val_sample.npy")

    # hold out whole samples, never individual patches, so the inner
    # validation cannot see patches overlapping a training patch
    ids = np.unique(tr_smp)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    n_hold = max(1, int(round(len(ids) * VAL_FRACTION)))
    hold, fit = set(ids[:n_hold].tolist()), set(ids[n_hold:].tolist())
    fit_idx = np.flatnonzero(np.isin(tr_smp, list(fit)))
    hold_idx = np.flatnonzero(np.isin(tr_smp, list(hold)))
    log(f"[p2] train patches {len(fit_idx)}, inner-val {len(hold_idx)}, "
        f"reported val {len(va_lab)}")

    pts_tr = f"{DATA}/train_points.npy"
    train_ds = Patches(pts_tr, tr_lab, fit_idx)
    inner_ds = Patches(pts_tr, tr_lab, hold_idx)
    test_ds = Patches(f"{DATA}/val_points.npy", va_lab)

    npos = float(tr_lab[fit_idx].sum())
    nneg = len(fit_idx) - npos
    pos_weight = torch.tensor([nneg / max(npos, 1.0)], dtype=torch.float32,
                              device=device)
    log(f"[p2] class balance pos={int(npos)} neg={int(nneg)} "
        f"pos_weight={pos_weight.item():.3f}")

    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=sh, num_workers=6,
        pin_memory=True, drop_last=sh, persistent_workers=True)
    train_loader, inner_loader, test_loader = (
        mk(train_ds, True), mk(inner_ds, False), mk(test_ds, False))

    model = PatchClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt = f"{RES}/best.pth"
    best, best_ep, bad, hist = -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        trl, tra, _, _ = run_epoch(model, train_loader, device, opt, pos_weight)
        vl, va, _, _ = run_epoch(model, inner_loader, device, None, pos_weight)
        hist.append({"epoch": ep, "train_loss": trl, "train_auc": tra,
                     "val_loss": vl, "val_auc": va})
        log(f"[p2] epoch {ep:3d} train_loss={trl:.4f} train_auc={tra:.4f} "
            f"inner_val_loss={vl:.4f} inner_val_auc={va:.4f} "
            f"({time.time()-t0:.0f}s)")
        if va is not None and va > best:
            best, best_ep, bad = va, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[p2] early stop at epoch {ep}")
                break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, auc, probs, labels = run_epoch(model, test_loader, device, None,
                                      pos_weight)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "task": "primary fuselage benchmark, same 64x64 stride-32 patches and "
                "0.1 dent-fraction label rule as DentNet and PBSS",
        "input": "(X, Y, residual) points, 1024 per patch, centred and "
                 "scaled by mean radius",
        "norm_radius": NORM_RADIUS,
        "n_train_patches": int(len(fit_idx)),
        "n_val_patches": int(len(va_lab)),
        "n_val_dent_patches": int(va_lab.sum()),
        "best_epoch": best_ep, "inner_val_auc": best,
        "train_time_sec": round(train_time, 1),
        "patch": {
            "auc": auc,
            "f1": float(f1_score(labels, preds)),
            "precision": float(precision_score(labels, preds,
                                               zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        },
        "pixel": pixel_iou(probs, va_smp),
    }
    np.savez_compressed(f"{RES}/val_scores.npz", probs=probs,
                        labels=labels, sample=va_smp)
    with open(f"{RES}/training_log.json", "w") as f:
        json.dump({"history": hist, "best_epoch": best_ep,
                   "inner_val_auc": best}, f, indent=2)
    with open(f"{RES}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("[p2] " + json.dumps(metrics["patch"]) + " " +
        json.dumps(metrics["pixel"]))
    log("P2_DONE")


if __name__ == "__main__":
    main()
