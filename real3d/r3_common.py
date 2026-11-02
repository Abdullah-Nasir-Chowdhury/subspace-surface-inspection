"""Shared machinery for the Real3D-AD defect-versus-good benchmark of the training-free
shape-subspace statistics. Self-contained: the PCD reader, tangent-plane resampling and the
canonical grid are lifted from the (unpublished) idea-B harness so that this directory needs
nothing outside the paper repo.

BLAS thread caps are set here, before numpy is imported. Import this module first.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, re
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.spatial import cKDTree

R3D_ROOT = os.path.expanduser("~/data/ext/real3d_ad/Real3D-AD-PCD")
OUT = os.path.expanduser("~/accv_results/real3d")
CATS = ["airplane", "candybar", "car", "chicken", "diamond", "duck",
        "fish", "gemstone", "seahorse", "shell", "starfish", "toffees"]

GRID = 10                       # the paper's 10x10 canonical lattice, n = 100
RADII_MM = [2.5, 3.6, 5.0]      # patch half-extent. 3.6 = 1.4x the measured median defect
                                # half-extent on these scans (5.17 mm extent), fixed before
                                # any score was looked at. The other two report the axis.
PRIMARY_RADIUS = 3.6
K_NEIGHBOURS = 8                # paper: mean Mag(D) against the 8-neighbourhood
MIN_BALL_PTS = 200
N_POINT_SUBSAMPLE = 10000       # points per item kept for the point-level AUROC


# ------------------------------------------------------------------------------ I/O

def read_pcd(path):
    with open(path, "rb") as f:
        fields = size = typ = count = npts = data = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no DATA header in %s" % path)
            s = line.decode("ascii", "replace").strip()
            k = s.split()[0].upper() if s else ""
            if k == "FIELDS":
                fields = s.split()[1:]
            elif k == "SIZE":
                size = [int(x) for x in s.split()[1:]]
            elif k == "TYPE":
                typ = s.split()[1:]
            elif k == "COUNT":
                count = [int(x) for x in s.split()[1:]]
            elif k == "POINTS":
                npts = int(s.split()[1])
            elif k == "DATA":
                data = s.split()[1].lower()
                break
        if count is None:
            count = [1] * len(fields)
        if data == "ascii":
            arr = np.loadtxt(f, dtype=np.float64).reshape(npts, -1)
            return arr[:, [fields.index(c) for c in ("x", "y", "z")]]
        dt = []
        for fl, sz, tp, ct in zip(fields, size, typ, count):
            base = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
                    ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}[(tp, sz)]
            dt.append((fl, base, ct) if ct > 1 else (fl, base))
        rec = np.frombuffer(f.read(np.dtype(dt).itemsize * npts), dtype=np.dtype(dt), count=npts)
        return np.c_[rec["x"], rec["y"], rec["z"]].astype(np.float64)


def read_gt(path):
    a = np.loadtxt(path)
    return a[:, :3], a[:, 3].astype(int)


def enumerate_test_items():
    """(cat, stem, label) for every uncut, uncopied test item. label: good / bulge / sink."""
    out = []
    for cat in CATS:
        td = os.path.join(R3D_ROOT, cat, "test")
        for f in sorted(os.listdir(td)):
            m = re.match(r"^(\d+)_([a-z]+)\.pcd$", f)
            if not m:
                continue
            w = m.group(2)
            if w == "good":
                lab = "good"
            elif w.startswith("bulge"):
                lab = "bulge"
            elif w.startswith("sink"):
                lab = "sink"
            else:
                continue
            if lab != "good" and not os.path.exists(os.path.join(R3D_ROOT, cat, "gt", f[:-4] + ".txt")):
                continue
            out.append((cat, f[:-4], lab))
    return out


def enumerate_templates(cat):
    td = os.path.join(R3D_ROOT, cat, "train")
    return [f[:-4] for f in sorted(os.listdir(td)) if f.endswith(".pcd")]


# ------------------------------------------------------------------------ canonical grid

class Grid:
    """Fixed canonical lattice shared by every patch (the premise of Proposition 1)."""

    def __init__(self, g=GRID, ann=0.5):
        self.g = g
        t = np.linspace(-1.0, 1.0, g)
        Y, X = np.meshgrid(t, t, indexing="ij")
        self.X, self.Y = X.ravel(), Y.ravel()
        A = np.c_[self.X, self.Y, np.ones(g * g)]
        self.P = np.eye(g * g) - A @ np.linalg.pinv(A)              # plane residual projector
        ring = np.maximum(np.abs(self.X), np.abs(self.Y)) > ann
        S = np.eye(g * g)[ring]
        Aq = np.c_[A, self.X ** 2, self.X * self.Y, self.Y ** 2]
        self.P_quad_ann = np.eye(g * g) - Aq @ np.linalg.pinv(Aq[ring]) @ S   # ring quadric


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def local_frame(P, c, obj_centre):
    """Right-handed frame at c, third axis the surface normal oriented away from the object."""
    Q = P - c
    C = Q.T @ Q / max(len(Q) - 1, 1)
    w, V = np.linalg.eigh(C)
    n = V[:, 0]
    if n @ (c - obj_centre) < 0:
        n = -n
    a = np.array([1.0, 0.0, 0.0])
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(n, a))
    e2 = np.cross(n, e1)
    return np.c_[e1, e2, n]


def resample(P, c, F, grid, radius, kk=8, slab=0.5, return_cov=False):
    """Height field over the canonical grid, sampled from the tangent plane at c (IDW, k=8).

    Only points within |w| <= slab * radius of the tangent plane take part. The Real3D-AD
    training templates are complete objects and carry a second sheet a few millimetres
    behind the front face inside every neighbourhood ball, while the test scans are
    single-sided; without the slab the 2-D nearest-neighbour interpolation averages both
    sheets on the template alone and the comparison is meaningless (found 2026-09-11).
    """
    L = (P - c) @ F
    keep = np.abs(L[:, 2]) <= slab * radius
    if keep.sum() >= kk:
        L = L[keep]
    tree = cKDTree(L[:, :2])
    q = np.c_[grid.X * radius, grid.Y * radius]
    d, i = tree.query(q, k=min(kk, len(L)))
    d, i = np.atleast_2d(d), np.atleast_2d(i)
    w = 1.0 / (d + 1e-9)
    h = (L[i, 2] * w).sum(1) / w.sum(1)
    if return_cov:
        # coverage: the largest gap between a grid node and its nearest sample. A window
        # that runs off the object or over a hole has a large gap and its height field is
        # extrapolated rather than sampled.
        return h, float(d[:, 0].max())
    return h


def voxel_seeds(P, spacing, rng):
    """One seed per occupied voxel of side `spacing`: the voxel's centroid. Deterministic
    given the cloud; rng only shuffles the order."""
    key = np.floor((P - P.min(0)) / spacing).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = inv.max() + 1
    cnt = np.bincount(inv, minlength=m).astype(np.float64)
    cen = np.stack([np.bincount(inv, weights=P[:, k], minlength=m) for k in range(3)], 1) / cnt[:, None]
    order = rng.permutation(m)
    return cen[order]


# ------------------------------------------------------------------------ subspace ops

def shape_subspace(M):
    Mc = M - M.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Mc, full_matrices=False)
    return U, S


def mag_d(A, B):
    """Mag(D) = sum_i 2(1 - sigma_i), Fukui and Maki 2015, Def. 2.6."""
    s = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0.0, 1.0)
    return float(np.sum(2.0 * (1.0 - s)))


def stat_skew(r):
    s = r.std()
    return 0.0 if s < 1e-15 else float((((r - r.mean()) / s) ** 3).mean())
