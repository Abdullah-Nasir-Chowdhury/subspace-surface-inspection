"""Stage P2c: training-seed repeats of the three PointNet++ encoders (2026-09-24).

The paper keeps three PointNet++ encoders (grouping radius 2.0, 0.6, 0.3), each trained
once from scratch (seed 0). To check whether the in-domain differences between the
radii are intrinsic or partly initialization, this script retrains each radius with two
further torch seeds under the recipe of its original run (radius 2.0: 40 epochs, patience 8,
as train_pointnetpp.py; 0.6 and 0.3: 20 epochs, patience 6, as radius_sweep.py).
Only the network initialization and the batch order change: the held-out slice of training
sample IDs is drawn with numpy seed 0 exactly as before, so every seed fits the same samples
and selects on the same held-out samples.

Output: ~/accv_results_real/pointnetpp_fuselage_r<radius>_s<seed>/{metrics.json, best.pth},
summary ~/accv_results_real/pointnetpp_seeds.json with mean and std over the three seeds
(seed 0 taken from the original run).
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import argparse
import json
import os
import sys

import numpy as np
import torch

PIPE = os.path.dirname(os.path.abspath(__file__))
import train_pointnetpp as P2
from pnpp_model import PatchClassifier

RESROOT = os.path.expanduser("~/accv_results_real")
RECIPE = {2.0: (40, 8), 0.6: (20, 6), 0.3: (20, 6)}
ORIG = {2.0: "pointnetpp_fuselage", 0.6: "pointnetpp_fuselage_r0.6", 0.3: "pointnetpp_fuselage_r0.3"}


def train_one(radius, seed):
    epochs, patience = RECIPE[radius]
    res = f"{RESROOT}/pointnetpp_fuselage_r{radius:g}_s{seed}"
    marker = f"{res}/metrics.json"
    if os.path.exists(marker):
        with open(marker) as f:
            return json.load(f)
    os.makedirs(res, exist_ok=True)
    P2.RES, P2.MAX_EPOCHS, P2.PATIENCE = res, epochs, patience
    P2.LOG = os.path.expanduser(f"~/accv_logs/pnpp_fuselage_r{radius:g}_s{seed}.log")
    orig_cls, orig_seed = P2.PatchClassifier, torch.manual_seed
    P2.PatchClassifier = lambda: PatchClassifier(norm_radius=radius)
    torch.manual_seed = lambda s: orig_seed(seed)      # P2.main seeds torch with SEED=0; redirect it
    try:
        P2.main()
    finally:
        P2.PatchClassifier, torch.manual_seed = orig_cls, orig_seed
    with open(marker) as f:
        m = json.load(f)
    m["norm_radius"], m["torch_seed"] = radius, seed
    with open(marker, "w") as f:
        json.dump(m, f, indent=2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[0.3, 0.6, 2.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    summary = {}
    for r in a.radii:
        runs = {}
        with open(f"{RESROOT}/{ORIG[r]}/metrics.json") as f:
            runs[0] = json.load(f)
        for s in a.seeds:
            runs[s] = train_one(r, s)
        auc = np.array([runs[s]["patch"]["auc"] for s in sorted(runs)])
        f1 = np.array([runs[s]["patch"]["f1"] for s in sorted(runs)])
        iou = np.array([runs[s]["pixel"]["mean_iou_dented"] for s in sorted(runs)])
        inner = np.array([runs[s]["inner_val_auc"] for s in sorted(runs)])
        summary[f"r{r:g}"] = {
            "seeds": sorted(runs), "recipe_epochs_patience": RECIPE[r],
            "val_auc": auc.tolist(), "val_auc_mean": float(auc.mean()), "val_auc_std": float(auc.std(ddof=1)),
            "val_f1": f1.tolist(), "val_f1_mean": float(f1.mean()), "val_f1_std": float(f1.std(ddof=1)),
            "pixel_iou_dented": iou.tolist(), "pixel_iou_mean": float(iou.mean()), "pixel_iou_std": float(iou.std(ddof=1)),
            "inner_val_auc": inner.tolist(), "best_epoch": [runs[s]["best_epoch"] for s in sorted(runs)],
            "train_time_sec": [runs[s]["train_time_sec"] for s in sorted(runs)]}
        print(json.dumps({f"r{r:g}": summary[f"r{r:g}"]}), flush=True)
        with open(f"{RESROOT}/pointnetpp_seeds.json", "w") as f:
            json.dump(summary, f, indent=2)
    print("P2C_DONE", flush=True)


if __name__ == "__main__":
    main()
