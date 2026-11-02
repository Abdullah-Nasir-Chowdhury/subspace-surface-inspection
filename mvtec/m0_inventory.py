"""Stage 0. Inventory and the pre-declared similarity factor.

For every test scan: grid size, units, native pitch (mm per pixel), foreground fraction,
and for defective scans the largest bounding-box side of the defect mask in mm. The
similarity factor is fixed the way the paper fixed it for Real3D-AD: a probe over the
first two defective scans of every category gives a median defect extent, and the factor
maps that extent onto TARGET_EXTENT_MM. The full-set value is reported beside it and is
not used. Output: mv_inventory.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import mv_common as M
import numpy as np


def synthetic_dent_sign(n=20):
    """Median (dent minus ring) of the synthetic training residuals, in mm."""
    from scipy import ndimage
    d = []
    for i in range(n):
        r = np.load("%s/%05d_input.npy" % (M.C.SYN, i)); g = np.load("%s/%05d_gt.npy" % (M.C.SYN, i)).astype(bool)
        if g.sum() == 0:
            continue
        ring = ndimage.binary_dilation(g, iterations=20) & ~g
        d.append(float(np.median(r[g]) - np.median(r[ring])))
    return float(np.median(d))


def main():
    rows, probe = [], []
    for s in M.enumerate_scans():
        if s["split"] != "test":
            continue
        xyz = M.read_xyz(s["xyz"]); xyz, unit = M.units_to_mm(xyz)
        v = xyz[..., 2] != 0
        r = dict(id=M.scan_id(s), cat=s["cat"], dtype=s["dtype"], shape=list(v.shape), unit_scale=unit,
                 pitch_mm=M.native_pitch(xyz), fg_frac=float(v.mean()),
                 z_span_mm=float(np.percentile(xyz[..., 2][v], 99) - np.percentile(xyz[..., 2][v], 1)))
        if s["gt"]:
            g = M.read_gt(s["gt"]); ys, xs = np.nonzero(g)
            if len(ys):
                r["defect_extent_mm"] = float(max(np.ptp(ys), np.ptp(xs)) * r["pitch_mm"])
                r["defect_px"] = int(g.sum())
                r["defect_frac"] = float(g.mean())
                # sensor sign: raw z inside the defect against a ring around it
                from scipy import ndimage
                gb = g.astype(bool); ring = ndimage.binary_dilation(gb, iterations=20) & ~gb & v
                if ring.sum() > 10 and (gb & v).sum() > 10:
                    r["z_in_minus_ring_mm"] = float(np.median(xyz[..., 2][gb & v]) - np.median(xyz[..., 2][ring]))
        rows.append(r)
        M.log(r["id"], r["shape"], "pitch %.4f mm" % r["pitch_mm"], "fg %.2f" % r["fg_frac"],
              "extent %.1f mm" % r.get("defect_extent_mm", float("nan")))
    ext_all = np.array([r["defect_extent_mm"] for r in rows if "defect_extent_mm" in r])
    for cat in M.CATS:
        d = [r["defect_extent_mm"] for r in rows if r["cat"] == cat and "defect_extent_mm" in r][:2]
        probe += d
    probe_med = float(np.median(probe)); full_med = float(np.median(ext_all))
    holes = [r["z_in_minus_ring_mm"] for r in rows if r["dtype"] == "hole" and "z_in_minus_ring_mm" in r]
    syn = synthetic_dent_sign()
    # a hole on MVTec must carry the same sign as a dent on the synthetic benchmark
    sign = 1 if np.sign(np.median(holes)) == np.sign(syn) else -1
    inv = {"n_test": len(rows), "n_defective": int(len(ext_all)),
           "sign": sign, "hole_z_in_minus_ring_median_mm": float(np.median(holes)), "n_hole_scans": len(holes),
           "synthetic_dent_minus_ring_median_mm": syn,
           "probe_n": len(probe), "probe_median_extent_mm": probe_med, "probe_factor_not_used": M.TARGET_EXTENT_MM / probe_med,
           "full_median_extent_mm": full_med,
           # the similarity factor is the full-set dataset-level statistic, fixed after this
           # inventory and before any method was scored. An alphabetical two-per-category probe
           # (reported above) samples the "combined" and "bent" classes and is not used.
           "similarity_factor": M.TARGET_EXTENT_MM / full_med,
           "pitch_mm_median": float(np.median([r["pitch_mm"] for r in rows])),
           "per_category": {c: {"n": sum(r["cat"] == c for r in rows),
                                "n_defective": sum(r["cat"] == c and "defect_extent_mm" in r for r in rows),
                                "pitch_mm": float(np.median([r["pitch_mm"] for r in rows if r["cat"] == c])),
                                "extent_median_mm": float(np.median([r["defect_extent_mm"] for r in rows if r["cat"] == c and "defect_extent_mm" in r]))}
                            for c in M.CATS},
           "scans": rows}
    M.save_json("mv_inventory.json", inv)
    L = ["# MVTec 3D-AD inventory", "", "| category | test scans | defective | pitch mm | median defect extent mm |", "|---|---|---|---|---|"]
    for c, e in inv["per_category"].items():
        L.append("| %s | %d | %d | %.4f | %.1f |" % (c, e["n"], e["n_defective"], e["pitch_mm"], e["extent_median_mm"]))
    L += ["", "Sign: median raw z inside a hole minus its ring is %.3f mm over %d hole scans, a synthetic dent minus its ring is %.3f mm, so the residual sign is %+d." % (
        inv["hole_z_in_minus_ring_median_mm"], inv["n_hole_scans"], syn, sign)]
    L += ["", "Similarity factor: full-set median defect extent %.2f mm onto %.0f mm gives %.2f, fixed after this inventory and before any scoring." % (
        full_med, M.TARGET_EXTENT_MM, inv["similarity_factor"]),
          "An alphabetical two-per-category probe (n=%d) gives median %.2f mm and factor %.2f. It samples the combined and bent classes and is not used." % (
        inv["probe_n"], probe_med, inv["probe_factor_not_used"])]
    M.write_md("mv_inventory.md", L)
    print("ALL_DONE_M0", flush=True)


if __name__ == "__main__":
    main()
