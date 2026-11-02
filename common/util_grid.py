"""Grid utilities shared by pipeline stages.

Real scans are row-major grids serialized as (N, 3) point lists. N = 486,400
factors as both 640x760 (A.ply) and 608x800 (B/C/D), so the grid cannot be
inferred from the point count alone; the legacy scripts picked whichever
divisor came first. Here the candidate factorization is chosen by row
structure instead: within a true scan row, Y is nearly constant (jitter
~0.01 mm), while a wrong factorization mixes rows and inflates the spread.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import numpy as np


def load_ply_points(path):
    import open3d as o3d  # real open3d, or the shim in common/open3d.py
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


def detect_grid(points, dim_range=(400, 1200)):
    """Return (h, w) minimizing the mean within-row spread of Y."""
    n = len(points)
    Y = points[:, 1]
    lo, hi = dim_range
    best, best_score = None, np.inf
    for h in range(lo, hi + 1):
        if n % h:
            continue
        w = n // h
        if not (lo <= w <= hi):
            continue
        score = np.mean(np.std(Y[:h * w].reshape(h, w), axis=1))
        if score < best_score:
            best, best_score = (h, w), score
    if best is None:
        raise ValueError(f"no grid factorization found for {n} points")
    return best


def to_grids(points, h, w):
    """Reshape an (N, 3) point list into X, Y, Z grids of shape (h, w)."""
    return (points[:h * w, 0].reshape(h, w),
            points[:h * w, 1].reshape(h, w),
            points[:h * w, 2].reshape(h, w))
