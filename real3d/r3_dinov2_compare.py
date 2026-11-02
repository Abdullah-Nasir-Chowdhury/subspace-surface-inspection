"""Comparison table for r3_dinov2.py on the published footing (every one of the 597 items,
object score = maximum over all valid windows) beside the coverage rule, with the point
AUROC and localization of r3_object_agg.py. Category mean with the item-clustered
within-category bootstrap of r3_fair_compare.py.

Output: ~/accv_results/real3d/r3_dinov2_compare.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, argparse
import r3_common as C
import r3_object_agg as OA
import r3_compare as CMP
from r3_fair_compare import cat_mean_ci
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="items_dinov2_v2")
    ap.add_argument("--tag", default="r3_dinov2_compare_v2")
    a = ap.parse_args()
    cats = CMP.PUBLISHED["cats"]
    rows = OA.load_items(a.items_dir, 3.6)
    y = np.array([r["defective"] for r in rows]); catidx = np.array([r["cat"] for r in rows])
    stats = [s for s in sorted(rows[0]["stats"]) if s.startswith(("dino_", "dentnet_", "pnpp_"))]
    R = {"n_items": len(rows), "stats": {}}
    rng = np.random.default_rng(0)
    L = ["# SubspaceAD and GDS on Real3D-AD, every item (%d), 12-category mean object AUROC (%%)" % len(rows), "",
         "Published: PO3AD 76.6, Group3AD 75.1, Reg3D-AD 70.4. Paper's local nominal subspace on DentNet features: 72.1 (norm. 78.2).", "",
         "| statistic | max, 12 cat. [CI] | max, 5 pre-reg. | covered, 12 cat. | point AUROC [CI] | hit rate | median mm |", "|---|---|---|---|---|---|---|"]
    for st in stats:
        obj = OA.object_scores(rows, st, 3.6)
        e = {}
        for agg in ("max", "covered"):
            sc = obj[agg]
            pc = {c: OA.auc(sc[catidx == c], y[catidx == c]) for c in cats}
            e[agg] = {"per_category": {c: 100 * pc[c] for c in cats}, "mean12": 100 * float(np.nanmean([pc[c] for c in cats])),
                      "mean5": 100 * float(np.nanmean([pc[c] for c in CMP.REGISTERED5]))}
            if agg == "max":
                e[agg]["mean12_ci"] = [100 * v for v in cat_mean_ci(sc, y, catidx, cats)]
        S, Yp, I = OA.point_level(rows, st, rng)
        items = np.unique(I)
        pa = OA.auc(S, Yp)
        def pboot(ix):
            m = np.isin(I, items[ix]); return OA.auc(S[m], Yp[m])
        e["point"] = {"auc": pa, "ci": OA.boot(pboot, len(items), 300)}
        hits, dists = OA.localization(rows, st, 3.6)
        e["localization"] = {"hit_rate": float(hits.mean()), "median_dist_mm": float(np.median(dists))}
        R["stats"][st] = e
        L.append("| %s | %.1f [%.1f, %.1f] | %.1f | %.1f | %.3f [%.3f, %.3f] | %.2f | %.2f |" % (
            st, e["max"]["mean12"], *e["max"]["mean12_ci"], e["max"]["mean5"], e["covered"]["mean12"],
            pa, *e["point"]["ci"], hits.mean(), np.median(dists)))
        print(L[-1], flush=True)
    L += ["", "## Per-category object AUROC, max convention", "", "| statistic | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for st, e in R["stats"].items():
        L.append("| %s | " % st + " | ".join("%.1f" % e["max"]["per_category"][c] for c in cats) + " |")
    with open(os.path.join(C.OUT, a.tag + ".json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, a.tag + ".md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("ALL_DONE_R3_DINOV2_COMPARE", flush=True)


if __name__ == "__main__":
    main()
