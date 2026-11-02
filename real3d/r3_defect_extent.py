"""Provenance of the 5.17 mm median defect extent on Real3D-AD.

The number is the median, over every ninth of the first 200 DEFECTIVE items (23 items,
category order then filename order, skipping "good" items and "_cut"/"_copy" duplicates),
of the largest bounding-box side of the ground-truth defect points. It comes from the TEST
annotations, as a dataset-level statistic, before any method was scored. The primary radius
3.6 mm = 1.4 x (5.17 / 2) and the similarity factor 6.6 = 34.1 / 5.17 (the synthetic dent
extent's tenth percentile, a training-set quantity) both derive from it.

r3_common.enumerate_test_items() interleaves "good" items with defective ones, so the list
is filtered to defective items before the stride is applied.

This script recomputes that statistic on the (corrected) same subsample and on all
defective items, under three extent definitions, so the paper can state exactly what was
used. Output: r3_defect_extent.{json,md}.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json
import r3_common as C
import numpy as np
from scipy.spatial import ConvexHull


def extents(d):
    bbox = np.ptp(d, axis=0)
    e = {"bbox_max_side": float(bbox.max()), "bbox_diagonal": float(np.linalg.norm(bbox))}
    try:
        h = ConvexHull(d); v = d[h.vertices]
        dd = np.linalg.norm(v[:, None] - v[None], axis=2)
        e["max_pairwise"] = float(dd.max())
    except Exception:
        e["max_pairwise"] = float("nan")
    return e


def main():
    # defective-only enumeration, category then filename order: matches
    # b2_real3d.enumerate_items() exactly (r3_common's regex already excludes _cut/_copy
    # files, since their filenames don't match ^\d+_[a-z]+\.pcd$).
    defective_items = [(cat, stem, lab) for (cat, stem, lab) in C.enumerate_test_items() if lab != "good"]
    rows = []
    for k, (cat, stem, lab) in enumerate(defective_items):
        g, gl = C.read_gt(os.path.join(C.R3D_ROOT, cat, "gt", stem + ".txt"))
        d = g[gl == 1]
        if len(d) == 0:
            continue
        e = extents(d); e.update(cat=cat, stem=stem, label=lab, n_defect_points=int(len(d)), enum_index=k)
        rows.append(e)
    sub_idx = set(range(0, 200, 9))
    R = {"n_defective": len(rows), "definitions": {}}
    L = ["# Median defect extent, Real3D-AD (provenance check)", "",
         "| definition | median, all %d defects | median, probe subsample (every 9th of first 200 items) | n subsample |" % len(rows), "|---|---|---|---|"]
    for key in ("bbox_max_side", "bbox_diagonal", "max_pairwise"):
        allv = np.array([r[key] for r in rows]); subv = np.array([r[key] for r in rows if r["enum_index"] in sub_idx])
        R["definitions"][key] = {"median_all": float(np.nanmedian(allv)), "median_sub": float(np.nanmedian(subv)) if len(subv) else None,
                                 "n_sub": int(len(subv)), "p10_all": float(np.nanpercentile(allv, 10)), "p90_all": float(np.nanpercentile(allv, 90))}
        L.append("| %s | %.2f mm (p10 %.2f, p90 %.2f) | %s | %d |" % (key, R["definitions"][key]["median_all"], R["definitions"][key]["p10_all"],
                 R["definitions"][key]["p90_all"], "%.2f mm" % R["definitions"][key]["median_sub"] if len(subv) else "--", len(subv)))
    R["per_item"] = rows
    with open(os.path.join(C.OUT, "r3_defect_extent.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, "r3_defect_extent.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ALL_DONE_R3_DEFECT_EXTENT", flush=True)


if __name__ == "__main__":
    main()
