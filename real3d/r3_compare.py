"""Comparison of our reference-based statistics with the published Real3D-AD object-level
results, on the same footing the literature uses: object AUROC per category, then the mean
over categories (not pooled over items).

Reads r3_objagg_nominal_reg_r<r>.json (geometric, all categories) and
r3_objagg_nominal_feat_reg.json (frozen features, all categories) plus the published table
in accv_results_real3d/baselines_real3d.md (PO3AD, CVPR 2025, Table 3). Writes
r3_compare.{json,md}: per-category rows for the published methods and ours, the 12-category
mean, and the mean over the five pre-registered categories.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, argparse
import r3_common as C
import numpy as np

PUBLISHED = {   # object-level AUC-ROC (%), Real3D-AD, from PO3AD Table 3 (column order confirmed)
    "cats": ["airplane", "car", "candybar", "chicken", "diamond", "duck", "fish", "gemstone",
             "seahorse", "shell", "starfish", "toffees"],
    "BTF (Raw)":            [73.0, 64.7, 53.9, 78.9, 70.7, 69.1, 60.2, 68.6, 59.6, 39.6, 53.0, 70.3],
    "PatchCore (FPFH)":     [88.2, 59.0, 54.1, 83.7, 57.4, 54.6, 67.5, 37.0, 50.5, 58.9, 44.1, 56.5],
    "CPMF":                 [70.1, 55.1, 55.2, 50.4, 52.3, 58.2, 55.8, 58.9, 72.9, 65.3, 70.0, 39.0],
    "Reg3D-AD":             [71.6, 69.7, 68.5, 85.2, 90.0, 58.4, 91.5, 41.7, 76.2, 58.3, 50.6, 82.7],
    "IMRNet":               [76.2, 71.1, 75.5, 78.0, 90.5, 51.7, 88.0, 67.4, 60.4, 66.5, 67.4, 77.4],
    "R3D-AD":               [77.2, 69.3, 71.3, 71.4, 68.5, 90.9, 69.2, 66.5, 72.0, 84.0, 70.1, 70.3],
    "Group3AD":             [74.4, 72.8, 84.7, 78.6, 93.2, 67.9, 97.6, 53.9, 84.1, 58.5, 56.2, 79.6],
    "PO3AD":                [80.4, 65.4, 78.5, 68.6, 80.1, 82.0, 85.9, 69.3, 75.6, 80.0, 75.8, 77.1],
}
REGISTERED5 = ["candybar", "car", "fish", "seahorse", "toffees"]
OURS = [  # (report tag, statistic, aggregation, label)
    ("r3_objagg_nominal_reg_r2.5", "nn_p95", "covered", "point-to-template distance (2.5 mm, covered)"),
    ("r3_objagg_nominal_reg_r2.5", "nom_rec_k6_pl_norm", "covered", "nominal subspace, height field, normalized (2.5 mm)"),
    ("r3_objagg_nominal_reg_r2.5", "magd_nom_min", "covered", "Mag(D) vs nearest nominal patch (2.5 mm)"),
    ("r3_objagg_nominal_feat_reg", "dentnet_nom_rec3", "covered", "local nominal subspace, DentNet features"),
    ("r3_objagg_nominal_feat_reg", "dentnet_nom_rec3_norm", "covered", "local nominal subspace, DentNet features, normalized"),
    ("r3_objagg_nominal_feat_reg", "pnpp_nom_rec3", "covered", "local nominal subspace, PointNet++ features"),
    ("r3_objagg_nominal_feat_reg", "pnpp_nom_rec3_norm", "covered", "local nominal subspace, PointNet++ features, normalized"),
]


def main():
    cats = PUBLISHED["cats"]
    rows = []
    for name, vals in PUBLISHED.items():
        if name == "cats":
            continue
        v = dict(zip(cats, vals))
        rows.append((name, v, "published"))
    for tag, stat, agg, label in OURS:
        p = os.path.join(C.OUT, tag + ".json")
        if not os.path.exists(p):
            print("missing", p); continue
        d = json.load(open(p))
        if stat not in d["stats"]:
            print("missing stat", stat, "in", tag); continue
        pc = d["stats"][stat]["object"][agg]["per_category"]
        rows.append((label, {c: 100 * pc[c] for c in cats if c in pc}, "ours"))
    L = ["# Real3D-AD object-level AUROC (%), per category and category means", "",
         "Published rows from PO3AD (CVPR 2025) Table 3. Ours: covered-window aggregation, "
         "item-clustered; airplane has 45 items after registration (5 rejected at 0.65 to 0.73 mm), "
         "gemstone 49.", "",
         "| method | " + " | ".join(cats) + " | mean 12 | mean 5 reg. |", "|---|" + "---|" * (len(cats) + 2)]
    R = {"cats": cats, "rows": {}}
    for name, v, kind in rows:
        m12 = float(np.mean([v[c] for c in cats if c in v]))
        m5 = float(np.mean([v[c] for c in REGISTERED5 if c in v]))
        R["rows"][name] = {"per_category": v, "mean12": m12, "mean5": m5, "kind": kind}
        L.append("| %s | " % name + " | ".join("%.1f" % v[c] if c in v else "--" for c in cats) + " | %.1f | %.1f |" % (m12, m5))
    with open(os.path.join(C.OUT, "r3_compare.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, "r3_compare.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
