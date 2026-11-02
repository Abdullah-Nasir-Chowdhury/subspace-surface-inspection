"""Stage P1: build PointNet++ training patches for the PRIMARY fuselage task.

The paper's protocol labels a patch dent when its dent-pixel fraction exceeds
0.1, over 64x64 windows at stride 32 with the boundary band excluded, and
both learned methods read those same windows. We hand PointNet++ exactly
those windows, as raw 3-D point sets (X, Y, residual) rather than as an
image, so the only thing that differs between it and DentNet is how the
patch is represented and consumed, not which patch it sees or how it is
labelled.

Each 64x64 window is 4{,}096 grid points, subsampled to 1{,}024 with a
per-patch seeded draw, then centred and divided by its mean radius, the same
normalization used for the ModelNet PointNet++ runs.

Output: ~/data/fuselage_pnpp/{train,val}_points.npy (memmapped float32),
        {train,val}_labels.npy, {train,val}_index.json
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from patch_utils import PatchExtractor

DATA = os.path.expanduser("~/data/pointcloud_db_real_scale")
OUT = os.path.expanduser("~/data/fuselage_pnpp")

PATCH, STRIDE, LABEL_THR = 64, 32, 0.1
N_POINTS = 1024
TRAIN_IDS = list(range(0, 500))     # same split PBSS builds its subspaces from
VAL_IDS = list(range(24000, 24500))  # same validation patches PBSS reports on
SEED = 0


def patch_geometry():
    """Patch positions are identical for every 640x760 sample."""
    pe = PatchExtractor(patch_size=PATCH, stride=STRIDE,
                        ignore_edge_patches=True)
    geom = pe.extract_patches(np.zeros((640, 760), dtype=np.float32))
    valid = np.array(geom["valid_mask"])
    pos = [p for k, p in enumerate(geom["positions"]) if valid[k]]
    return pe, pos


def normalize_patch(p):
    c = p.mean(axis=0, keepdims=True)
    q = p - c
    s = np.linalg.norm(q, axis=1).mean()
    return (q / (s if s > 1e-8 else 1.0)).astype(np.float32)


def build(ids, pos, tag):
    n_per = len(pos)
    total = len(ids) * n_per
    pts_path = f"{OUT}/{tag}_points.npy"
    arr = np.lib.format.open_memmap(pts_path, mode="w+", dtype=np.float32,
                                    shape=(total, N_POINTS, 3))
    labels = np.zeros(total, dtype=np.int8)
    sample_of = np.zeros(total, dtype=np.int32)
    rng = np.random.default_rng(SEED)
    w = 0
    kept_ids = []
    for n, i in enumerate(ids):
        base = f"{DATA}/{i:05d}"
        try:
            X = np.load(base + "_X.npy").astype(np.float32)
            Y = np.load(base + "_Y.npy").astype(np.float32)
            R = np.load(base + "_input.npy").astype(np.float32)
            gt = np.load(base + "_gt.npy").astype(np.uint8)
        except FileNotFoundError:
            continue
        kept_ids.append(int(i))
        for (y0, x0) in pos:
            sl = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
            p = np.stack([X[sl].ravel(), Y[sl].ravel(), R[sl].ravel()], 1)
            sel = rng.choice(p.shape[0], N_POINTS, replace=False)
            arr[w] = normalize_patch(p[sel])
            labels[w] = 1 if gt[sl].mean() > LABEL_THR else 0
            sample_of[w] = i
            w += 1
        if (n + 1) % 50 == 0:
            print(f"[{tag}] {n+1}/{len(ids)} samples, {w} patches", flush=True)
    arr.flush()
    del arr
    # trim to what was actually written
    if w < total:
        full = np.load(pts_path, mmap_mode="r")
        trimmed = np.lib.format.open_memmap(
            pts_path + ".tmp", mode="w+", dtype=np.float32,
            shape=(w, N_POINTS, 3))
        trimmed[:] = full[:w]
        trimmed.flush()
        del trimmed, full
        os.replace(pts_path + ".tmp", pts_path)
    np.save(f"{OUT}/{tag}_labels.npy", labels[:w])
    np.save(f"{OUT}/{tag}_sample.npy", sample_of[:w])
    with open(f"{OUT}/{tag}_index.json", "w") as f:
        json.dump({"n_patches": int(w), "n_samples": len(kept_ids),
                   "patches_per_sample": n_per, "n_points": N_POINTS,
                   "patch": PATCH, "stride": STRIDE,
                   "label_threshold": LABEL_THR,
                   "n_dent_patches": int(labels[:w].sum()),
                   "sample_ids": kept_ids}, f)
    print(f"[{tag}] {w} patches, {int(labels[:w].sum())} dent "
          f"({100*labels[:w].mean():.2f}%)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    pe, pos = patch_geometry()
    print(f"{len(pos)} valid patches per sample", flush=True)
    build(TRAIN_IDS, pos, "train")
    build(VAL_IDS, pos, "val")
    print("P1_DONE", flush=True)


if __name__ == "__main__":
    main()
