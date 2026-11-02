"""Shared pieces for the MVTec 3D-AD synthetic-to-real stage (2026-09-23).

MVTec 3D-AD (Bergmann et al., VISAPP 2022): ten categories of real structured-light scans
delivered as organized xyz grids (3-channel float32 TIFF, background pixels zero) with
per-pixel defect masks for the anomalous test scans. It replaces the four fuselage scans
as the real target of the synthetic-to-real experiment: the same synthetic fuselage
training set, the same frozen encoders and rules, evaluated zero-shot on a real dataset
that is not aircraft and has real ground truth.

Layout expected at MV_ROOT: <cat>/{train,validation,test}/<type>/{rgb,xyz,gt}/NNN.*
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, glob, json

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import core as C  # noqa: E402  (caps BLAS, sets sys.path, chdir ROOT)

MV_ROOT = os.path.expanduser(os.environ.get("MV_ROOT", "~/data/ext/mvtec_3d_ad"))
OUT = os.path.expanduser(os.environ.get("MV_OUT", "~/accv_results_mvtec"))
os.makedirs(OUT, exist_ok=True)
CATS = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]
PATCH, STRIDE, LABEL_THR = 64, 32, 0.1
PITCH_MM = 0.1875                 # the synthetic benchmark's pixel pitch
TARGET_EXTENT_MM = 34.0           # the paper maps the median defect extent onto the smallest synthetic dents
N_TRAIN_GOOD = 10                 # train/good scans per category used for the shift measurement
MIN_VALID = 0.95                  # window fraction of valid foreground pixels required


def log(*a):
    C.log(*a)


def read_xyz(path):
    import tifffile
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.moveaxis(a, 0, -1)
    return a


def read_gt(path):
    from PIL import Image
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def enumerate_scans(cats=CATS):
    """Yields dict(cat, split, dtype, stem, xyz, gt) for test scans and N_TRAIN_GOOD train scans."""
    for cat in cats:
        for split in ("test", "train"):
            base = os.path.join(MV_ROOT, cat, split)
            if not os.path.isdir(base):
                continue
            for dtype in sorted(os.listdir(base)):
                xs = sorted(glob.glob(os.path.join(base, dtype, "xyz", "*.tiff")))
                if split == "train":
                    xs = xs[:N_TRAIN_GOOD]
                for x in xs:
                    stem = os.path.splitext(os.path.basename(x))[0]
                    gt = os.path.join(base, dtype, "gt", stem + ".png")
                    yield dict(cat=cat, split=split, dtype=dtype, stem=stem, xyz=x,
                               gt=gt if os.path.exists(gt) else None)


def scan_id(s):
    return "%s__%s__%s__%s" % (s["cat"], s["split"], s["dtype"], s["stem"])


def units_to_mm(xyz):
    """MVTec's xyz values are auto-detected: if the object's z spread is below 5 the unit is
    metres and the array is scaled to millimetres."""
    v = xyz[..., 2] != 0
    if v.sum() == 0:
        return xyz, 1.0
    z = xyz[..., 2][v]
    span = np.percentile(z, 99) - np.percentile(z, 1)
    return (xyz * 1000.0, 1000.0) if span < 5.0 else (xyz, 1.0)


def native_pitch(xyz):
    v = xyz[..., 2] != 0
    dx = np.abs(np.diff(xyz[..., 0], axis=1))[v[:, 1:] & v[:, :-1]]
    dy = np.abs(np.diff(xyz[..., 1], axis=0))[v[1:] & v[:-1]]
    return float(np.median(np.r_[dx, dy]))


def save_json(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)
    log("wrote", os.path.join(OUT, name))


def write_md(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("wrote", os.path.join(OUT, name))
