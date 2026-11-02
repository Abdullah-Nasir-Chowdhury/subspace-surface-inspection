"""Parallel regeneration of the real-scale synthetic dataset.

Wraps main_real_scale.py's generation functions unchanged (same numerical
pipeline as the friend's original run), but:
  - seeds each sample with its ID -> fully reproducible
  - runs worker processes in parallel
  - resumable: skips samples whose files already exist

Usage: python gen_parallel.py [start] [end]
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import main_real_scale as gen

OUTPUT_DIR = os.path.expanduser("~/data/pointcloud_db_real_scale")
TOTAL = 30000
WORKERS = 18


def make_sample(i):
    np.random.seed(i)
    X_noisy, Y_noisy, residuals, gt_mask = gen.generate_single_sample_data(i)
    base = os.path.join(OUTPUT_DIR, f"{i:05d}")
    np.save(f"{base}_input.npy", residuals.astype(np.float32))
    np.save(f"{base}_gt.npy", gt_mask)
    np.save(f"{base}_X.npy", X_noisy.astype(np.float32))
    # _Y saved last; its existence marks the sample complete (resume check)
    np.save(f"{base}_Y.npy", Y_noisy.astype(np.float32))
    return i


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else TOTAL
    todo = [i for i in range(start, end)
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{i:05d}_Y.npy"))]
    print(f"Generating {len(todo)} samples into {OUTPUT_DIR}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(make_sample, todo, chunksize=8):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(todo)} done", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
