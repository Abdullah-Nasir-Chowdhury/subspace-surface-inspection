"""Stage P2b: give PointNet++ a fair shake on the fuselage task by sweeping
the set-abstraction grouping radius.

The ModelNet runs use NORM_RADIUS = 2.0, chosen there so the query ball
covers the whole normalized patch. On a 64x64 fuselage window that choice
removes the locality the hierarchy is supposed to provide: after centring
and dividing by mean radius the farthest point sits at about 1.85, so a
2.0 ball contains every point of every patch and the first set-abstraction
layer degenerates to 32 copies of a global feature. A dent covers only part
of a patch, so local grouping should matter here even though it did not on
100-point ModelNet patches.

We therefore train the same architecture at several radii and select on the
held-out slice of TRAIN sample IDs, never on the reported validation split.

Output: ~/accv_results_real/pointnetpp_fuselage_r<radius>/
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


def train_one(radius, epochs, patience):
    tag = f"r{radius:g}"
    res = f"{RESROOT}/pointnetpp_fuselage_{tag}"
    os.makedirs(res, exist_ok=True)
    marker = f"{res}/metrics.json"
    if os.path.exists(marker):
        with open(marker) as f:
            return json.load(f)

    P2.RES = res
    P2.MAX_EPOCHS = epochs
    P2.PATIENCE = patience
    orig = P2.PatchClassifier
    P2.PatchClassifier = lambda: PatchClassifier(norm_radius=radius)
    try:
        P2.main()
    finally:
        P2.PatchClassifier = orig
    with open(marker) as f:
        m = json.load(f)
    m["norm_radius"] = radius
    with open(marker, "w") as f:
        json.dump(m, f, indent=2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[0.3, 0.6])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    rows = []
    for r in a.radii:
        m = train_one(r, a.epochs, a.patience)
        rows.append({"radius": r, "inner_val_auc": m["inner_val_auc"],
                     "val_patch_auc": m["patch"]["auc"],
                     "val_f1": m["patch"]["f1"],
                     "pixel_iou_dented": m["pixel"]["mean_iou_dented"]})
        print("[p2b] " + json.dumps(rows[-1]), flush=True)
    with open(f"{RESROOT}/pointnetpp_radius_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("P2B_DONE", flush=True)


if __name__ == "__main__":
    main()
