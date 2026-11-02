"""Stage 1: evaluate best_dentnet_real_scale.pth on the full fresh val split.

Fresh draws from the same generative distribution (per-sample seeded), so this
is an honest re-measurement of the checkpoint's in-domain IoU claim (98.97% in
the research log, measured on the friend's lost original dataset). Mirrors
test_synthetic_A.py's preprocessing exactly: raw residuals in, pad to /16 with
edge mode, softmax channel 1 > 0.5.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import csv
import json
import os
import sys

import numpy as np
import torch

PIPE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PIPE)
os.chdir(ROOT)

from model import DentNet

DATA = os.path.expanduser("~/data/pointcloud_db_real_scale")
RES = os.path.expanduser("~/accv_results/dentnet")
START, END = 24000, 30000
CKPT = "best_dentnet_real_scale.pth"


def load_model(device):
    model = DentNet().to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()
    return model


def main():
    os.makedirs(RES, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    ids = [i for i in range(START, END)
           if os.path.exists(f"{DATA}/{i:05d}_gt.npy")]
    print(f"Evaluating {len(ids)} val samples on {device}", flush=True)
    ious, rows = [], []
    with torch.no_grad():
        for n, i in enumerate(ids):
            r = np.load(f"{DATA}/{i:05d}_input.npy")
            gt = np.load(f"{DATA}/{i:05d}_gt.npy").astype(np.uint8)
            h, w = r.shape
            ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
            rp = np.pad(r, ((0, ph), (0, pw)), mode="edge")
            t = torch.from_numpy(rp).float()[None, None].to(device)
            probs = torch.softmax(model(t), dim=1)
            pred = (probs[0, 1, :h, :w] > 0.5).byte().cpu().numpy()
            inter = int(np.logical_and(pred, gt).sum())
            union = int(np.logical_or(pred, gt).sum())
            iou = inter / union if union else 1.0  # no dent, none predicted
            ious.append(iou)
            rows.append((i, f"{iou:.6f}", int(gt.sum()), int(pred.sum())))
            if (n + 1) % 1000 == 0:
                print(f"{n+1}/{len(ids)} mean IoU so far {np.mean(ious):.4f}",
                      flush=True)
    ious = np.array(ious)
    dent = np.array([int(r[2]) > 0 for r in rows])
    summary = {
        "checkpoint": CKPT,
        "n_samples": len(ious),
        "mean_iou_all": float(ious.mean()),
        "median_iou_all": float(np.median(ious)),
        "std_iou_all": float(ious.std()),
        "n_dented": int(dent.sum()),
        "mean_iou_dented": float(ious[dent].mean()) if dent.any() else None,
    }
    print("FINAL:", json.dumps(summary), flush=True)
    with open(f"{RES}/val_iou_per_sample.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["id", "iou", "gt_pixels", "pred_pixels"])
        cw.writerows(rows)
    with open(f"{RES}/val_iou_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("ALL_DONE_S1", flush=True)


if __name__ == "__main__":
    main()
