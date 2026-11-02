"""The Real3D-AD comparison on the published footing (Table 2 of the paper).

The published methods score EVERY test item and take the object score as the maximum of
the point scores. A covered-window maximum is also reported, on on 591 items (six items
whose registration residual exceeded 0.6 mm were never scored). This script reports, for
each of our reference-based statistics:

  item sets     all597   every test item (the items directories built with --tol 1.0)
                strict   items with registration residual <= 0.6 mm (the round-3 set)
  aggregations  max      maximum over all valid windows   <- the published convention
                top5     mean of the five largest windows
                covered  maximum over windows whose grid nodes are all sampled 
                top5cov  top-5 mean over covered windows

with per-category object AUROC, the 12-category mean, the mean over the five pre-registered
categories, and a 95 % interval on each category mean from a within-category item bootstrap
(2,000 draws). The published rows (PO3AD, CVPR 2025, Table 3) are copied from r3_compare.py.

Output: r3_fair_compare.{json,md} in ~/accv_results/real3d/.
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
import numpy as np

STRICT_TOL = 0.6
OURS = [  # (items dir, radius, statistic, label)
    ("items_nominal_all", 2.5, "nn_p95", "point-to-template distance, 2.5 mm"),
    ("items_nominal_all", 2.5, "magd_nom_min", "Mag(D) vs nearest nominal patch, 2.5 mm"),
    ("items_nominal_all", 2.5, "nom_rec_k6_pl_norm", "nominal subspace, height field, normalized, 2.5 mm"),
    ("items_nominal_feat_all", 3.6, "dentnet_nom_nn", "nearest nominal feature, DentNet"),
    ("items_nominal_feat_all", 3.6, "dentnet_nom_rec3", "local nominal subspace, DentNet features"),
    ("items_nominal_feat_all", 3.6, "dentnet_nom_rec3_norm", "local nominal subspace, DentNet features, normalized"),
    ("items_nominal_feat_all", 3.6, "pnpp_nom_rec3", "local nominal subspace, PointNet++ features"),
    ("items_nominal_feat_all", 3.6, "pnpp_nom_rec3_norm", "local nominal subspace, PointNet++ features, normalized"),
]
AGGS = ["max", "top5", "covered", "top5cov"]


def cat_mean_ci(sc, y, catidx, cats, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = {c: np.flatnonzero(catidx == c) for c in cats}
    vals = []
    for _ in range(draws):
        per = []
        for c in cats:
            ix = idx[c][rng.integers(0, len(idx[c]), len(idx[c]))]
            per.append(OA.auc(sc[ix], y[ix]))
        vals.append(np.nanmean(per))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r3_fair_compare")
    a = ap.parse_args()
    cats = CMP.PUBLISHED["cats"]
    R = {"cats": cats, "published": {}, "ours": {}}
    for name, vals in CMP.PUBLISHED.items():
        if name == "cats":
            continue
        v = dict(zip(cats, vals))
        R["published"][name] = {"per_category": v, "mean12": float(np.mean(vals)),
                                "mean5": float(np.mean([v[c] for c in CMP.REGISTERED5]))}
    cache = {}
    for items_dir, radius, stat, label in OURS:
        key = (items_dir, radius)
        if key not in cache:
            cache[key] = OA.load_items(items_dir, radius)
        rows = cache[key]
        if not rows or stat not in rows[0]["stats"]:
            print("missing", items_dir, stat); continue
        obj = OA.object_scores(rows, stat, radius)
        y = np.array([r["defective"] for r in rows])
        catidx = np.array([r["cat"] for r in rows])
        regfit = np.array([r["reg_fit"] for r in rows])
        strict = np.isfinite(regfit) & (regfit <= STRICT_TOL) | ~np.isfinite(regfit)
        entry = {"items_dir": items_dir, "radius": radius, "n_all": len(rows), "n_strict": int(strict.sum()), "sets": {}}
        for setname, mask in (("all597", np.ones(len(rows), bool)), ("strict", strict)):
            e = {}
            for agg in AGGS:
                sc = obj[agg]
                pc = {c: OA.auc(sc[mask & (catidx == c)], y[mask & (catidx == c)]) for c in cats}
                m12 = float(np.nanmean([pc[c] for c in cats]))
                m5 = float(np.nanmean([pc[c] for c in CMP.REGISTERED5]))
                ci = cat_mean_ci(sc[mask], y[mask], catidx[mask], cats)
                e[agg] = {"per_category": {c: 100 * pc[c] for c in cats}, "mean12": 100 * m12, "mean5": 100 * m5,
                          "mean12_ci": [100 * ci[0], 100 * ci[1]], "pooled": OA.auc(sc[mask], y[mask])}
                print("%-60s %-7s %-8s mean12 %.1f [%.1f, %.1f] mean5 %.1f pooled %.3f" % (
                    label, setname, agg, e[agg]["mean12"], *e[agg]["mean12_ci"], e[agg]["mean5"], e[agg]["pooled"]), flush=True)
            entry["sets"][setname] = e
        R["ours"][label] = entry
    # markdown
    L = ["# Real3D-AD object-level AUROC (%): published footing (every item, max over windows) beside the round-3 covered-window protocol", "",
         "Published rows: PO3AD (CVPR 2025) Table 3. Ours: item-clustered, category mean with a 95 % within-category item bootstrap interval.", "",
         "## Table A: every item (%d), aggregation = max over all valid windows (the published convention)" % (
             next(iter(R["ours"].values()))["n_all"] if R["ours"] else 0), "",
         "| method | " + " | ".join(cats) + " | mean 12 [CI] | mean 5 reg. |", "|---|" + "---|" * (len(cats) + 2)]
    for name, e in R["published"].items():
        L.append("| %s | " % name + " | ".join("%.1f" % e["per_category"][c] for c in cats) + " | %.1f | %.1f |" % (e["mean12"], e["mean5"]))
    for label, e in R["ours"].items():
        x = e["sets"]["all597"]["max"]
        L.append("| %s | " % label + " | ".join("%.1f" % x["per_category"][c] for c in cats) + " | %.1f [%.1f, %.1f] | %.1f |" % (x["mean12"], *x["mean12_ci"], x["mean5"]))
    L += ["", "## Table B: our statistics under every item set and aggregation (12-category mean [CI]; mean over 5 pre-registered; pooled item AUROC)", "",
          "| statistic | set | n | " + " | ".join(AGGS) + " |", "|---|---|---|" + "---|" * len(AGGS)]
    for label, e in R["ours"].items():
        for setname in ("all597", "strict"):
            x = e["sets"][setname]
            L.append("| %s | %s | %d | " % (label, setname, e["n_all"] if setname == "all597" else e["n_strict"]) +
                     " | ".join("%.1f [%.1f, %.1f]; 5-reg %.1f; pooled %.3f" % (x[g]["mean12"], *x[g]["mean12_ci"], x[g]["mean5"], x[g]["pooled"]) for g in AGGS) + " |")
    L += ["", "## Table C: per-category, every item, covered-window aggregation (the round-3 protocol, now on all items)", "",
          "| method | " + " | ".join(cats) + " | mean 12 [CI] | mean 5 reg. |", "|---|" + "---|" * (len(cats) + 2)]
    for label, e in R["ours"].items():
        x = e["sets"]["all597"]["covered"]
        L.append("| %s | " % label + " | ".join("%.1f" % x["per_category"][c] for c in cats) + " | %.1f [%.1f, %.1f] | %.1f |" % (x["mean12"], *x["mean12_ci"], x["mean5"]))
    with open(os.path.join(C.OUT, a.tag + ".json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, a.tag + ".md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ALL_DONE_R3_FAIR_COMPARE", flush=True)


if __name__ == "__main__":
    main()
