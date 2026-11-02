"""Stage 1. Detrended 64x64 windows from every MVTec 3D-AD scan, two protocols.

The organized xyz grid is a height field z over an (x, y) grid of native pitch p0. As on
Real3D-AD, windows are taken at two pre-declared pitches on the object: the synthetic
pitch (PITCH_MM, scale 1) and the similarity scale s from mv_inventory.json, which maps
the probe's median defect extent onto TARGET_EXTENT_MM (pitch PITCH_MM / s on the object,
residual multiplied by s). The height field is resampled to the target pitch (invalid
pixels filled from the nearest valid one before interpolation and masked after), 64x64
windows at stride 32 with at least MIN_VALID valid pixels are kept, each is detrended by a
quadric fitted on its outer ring (r3_learned.ring_quadric_residual), and the sign is the
sensor convention fixed in stage 0 so that a hole gives a positive residual, as a dent
does on the synthetic benchmark. A window is a defect window when more than LABEL_THR of
its pixels are annotated, and a normal window when none is.

Output: ~/accv_results_mvtec/windows_<proto>/<scan id>.npz with R (n, 64, 64) float16,
frac, pos (row, col in the resampled grid), the resampled gt and valid masks (packed) and
the scan metadata.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import mv_common as M
import os, json, argparse
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from r3_learned import ring_quadric_residual

_INV = None


def _init():
    global _INV
    _INV = json.load(open(os.path.join(M.OUT, "mv_inventory.json")))


def resample(z, valid, zoom):
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    zf = z[tuple(idx)]
    zs = ndimage.zoom(zf, zoom, order=1)
    vs = ndimage.zoom(valid.astype(np.float32), zoom, order=1) > 0.5
    return zs, vs


def windows(scan, proto, scale, sign):
    xyz = M.read_xyz(scan["xyz"]); xyz, _ = M.units_to_mm(xyz)
    valid = xyz[..., 2] != 0
    p0 = M.native_pitch(xyz)
    pitch = M.PITCH_MM / scale
    zoom = p0 / pitch
    z = xyz[..., 2] * sign
    zs, vs = resample(z, valid, zoom)
    gt = M.read_gt(scan["gt"]) if scan["gt"] else np.zeros(valid.shape, np.uint8)
    gs = ndimage.zoom(gt.astype(np.float32), zoom, order=0) > 0.5
    H, W = zs.shape
    ax = (np.arange(M.PATCH) - (M.PATCH - 1) / 2.0) * M.PITCH_MM
    Yg, Xg = np.meshgrid(ax, ax, indexing="ij"); Xf, Yf = Xg.ravel(), Yg.ravel()
    half = ax[-1]; ring = np.maximum(np.abs(Xf), np.abs(Yf)) > 0.5 * half
    R, frac, pos = [], [], []
    for y in range(0, H - M.PATCH + 1, M.STRIDE):
        for x in range(0, W - M.PATCH + 1, M.STRIDE):
            v = vs[y:y + M.PATCH, x:x + M.PATCH]
            if v.mean() < M.MIN_VALID:
                continue
            h = zs[y:y + M.PATCH, x:x + M.PATCH].ravel().astype(np.float64)
            r = ring_quadric_residual(h, Xf, Yf, ring & v.ravel()) * scale
            r[~v.ravel()] = 0.0
            R.append(r.astype(np.float16)); frac.append(float(gs[y:y + M.PATCH, x:x + M.PATCH].mean())); pos.append((y, x))
    return dict(R=np.array(R, np.float16).reshape(-1, M.PATCH, M.PATCH), frac=np.array(frac, np.float32),
                pos=np.array(pos, np.int32).reshape(-1, 2), zoom=zoom, scale=scale, native_pitch_mm=p0,
                shape_small=np.array(zs.shape), gt_small=np.packbits(gs), valid_small=np.packbits(vs),
                cat=scan["cat"], split=scan["split"], dtype=scan["dtype"], stem=scan["stem"], defective=scan["dtype"] != "good")


def work(args):
    scan, proto, scale = args
    dst = os.path.join(M.OUT, "windows_" + proto, M.scan_id(scan) + ".npz")
    if os.path.exists(dst):
        return None
    w = windows(scan, proto, scale, _INV["sign"])
    np.savez_compressed(dst, **w)
    return (M.scan_id(scan), proto, len(w["R"]), int((w["frac"] > M.LABEL_THR).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=M.CATS)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    _init()
    protos = [("metric", 1.0), ("sim", float(_INV["similarity_factor"]))]
    for proto, _ in protos:
        os.makedirs(os.path.join(M.OUT, "windows_" + proto), exist_ok=True)
    jobs = [(s, p, sc) for s in M.enumerate_scans(a.cats) for p, sc in protos]
    M.log("%d scan-protocol jobs, sign %+d, similarity factor %.2f" % (len(jobs), _INV["sign"], protos[1][1]))
    with Pool(a.procs, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
            if r and k % 50 == 0:
                M.log("[%d/%d]" % (k, len(jobs)), *r)
    print("ALL_DONE_M1", flush=True)


if __name__ == "__main__":
    main()
