"""Registration quality as a first-class variable.

Two views, one figure (fig_reg_sensitivity.png) and one report (r3_reg_sensitivity.{json,md}):

  A. Observational. From the all-item hybrid run: per-category object AUROC (max and
     covered aggregation) against the category's median registration residual after ICP,
     with the Spearman rank correlation over the twelve categories; and item-level AUROC
     pooled within residual bins [0, 0.05), [0.05, 0.15), [0.15, 0.3), [0.3, inf) mm, with
     item bootstrap intervals. Items of the pre-registered categories carry the ICP polish
     residual (0.03 to 0.05 mm).
  B. Interventional. Runs of r3_nominal_feat.py with --perturb-trans-mm / --perturb-rot-deg
     on the five pre-registered categories (where the baseline residual is 0.03 to 0.05 mm):
     object AUROC (5-category mean, max and covered), point AUROC and localization hit
     rate against the perturbation magnitude. A rotation of d degrees about the centroid
     displaces a point at distance r by about r*d*pi/180 (an object of 40 mm radius: 0.7 mm
     per degree at its edge).

Usage: r3_reg_sensitivity.py --items-dir items_nominal_feat_all --stats dentnet_nom_rec3 dentnet_nom_rec3_norm pnpp_nom_rec3_norm
         --pert items_nominal_feat_pert_t0.1:trans:0.1 items_nominal_feat_pert_r0.5:rot:0.5 ...
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, argparse
import r3_common as C
import r3_object_agg as OA
import numpy as np

BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, np.inf)]
REG5 = ["candybar", "car", "fish", "seahorse", "toffees"]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate(rows, stat, radius):
    y = np.array([r["defective"] for r in rows]); cats = sorted(set(r["cat"] for r in rows))
    catidx = np.array([r["cat"] for r in rows])
    obj = OA.object_scores(rows, stat, radius)
    out = {"per_category": {}, "cat_mean": {}, "pooled": {}}
    for agg in ("max", "covered"):
        pc = {c: OA.auc(obj[agg][catidx == c], y[catidx == c]) for c in cats}
        out["per_category"][agg] = pc
        out["cat_mean"][agg] = float(np.nanmean(list(pc.values())))
        out["pooled"][agg] = OA.auc(obj[agg], y)
    S, Yp, I = OA.point_level(rows, stat, np.random.default_rng(0))
    out["point_auc"] = OA.auc(S, Yp)
    hits, dists = OA.localization(rows, stat, radius)
    out["hit_rate"] = float(hits.mean()); out["median_dist_mm"] = float(np.median(dists))
    return out, obj, y, catidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="items_nominal_feat_all")
    ap.add_argument("--radius", type=float, default=3.6)
    ap.add_argument("--stats", nargs="*", default=["dentnet_nom_rec3", "dentnet_nom_rec3_norm", "pnpp_nom_rec3_norm"])
    ap.add_argument("--pert", nargs="*", default=[], help="items_dir:kind:value, kind in {trans, rot}")
    ap.add_argument("--tag", default="r3_reg_sensitivity")
    a = ap.parse_args()
    R = {"A": {}, "B": {}}
    L = ["# Registration sensitivity", ""]

    # ---- A. observational
    rows = OA.load_items(a.items_dir, a.radius)
    cats = sorted(set(r["cat"] for r in rows))
    regfit = np.array([r["reg_fit"] for r in rows])
    catidx = np.array([r["cat"] for r in rows])
    y = np.array([r["defective"] for r in rows])
    med = {c: float(np.nanmedian(regfit[catidx == c])) for c in cats}
    L += ["## A. Per-category object AUROC against the median registration residual (%s, %d items)" % (a.items_dir, len(rows)), "",
          "| category | median residual mm | n | " + " | ".join("%s max / covered" % s for s in a.stats) + " |", "|---|---|---|" + "---|" * len(a.stats)]
    evals = {}
    for st in a.stats:
        evals[st], obj, _, _ = evaluate(rows, st, a.radius)
        R["A"][st] = {"per_category": evals[st]["per_category"], "median_residual": med,
                      "spearman_max": spearman([med[c] for c in cats], [evals[st]["per_category"]["max"][c] for c in cats]),
                      "spearman_covered": spearman([med[c] for c in cats], [evals[st]["per_category"]["covered"][c] for c in cats])}
        # item bins
        rng = np.random.default_rng(0)
        bins = {}
        for lo, hi in BINS:
            m = (regfit >= lo) & (regfit < hi)
            if m.sum() < 10 or y[m].sum() == 0 or (~y[m]).sum() == 0:
                bins["[%.2f, %s)" % (lo, "inf" if np.isinf(hi) else "%.2f" % hi)] = {"n": int(m.sum())}
                continue
            ix = np.flatnonzero(m)
            sc = {agg: obj[agg][ix] for agg in ("max", "covered")}
            yy = y[ix]
            e = {"n": int(m.sum()), "n_defective": int(yy.sum())}
            for agg in ("max", "covered"):
                e[agg] = OA.auc(sc[agg], yy)
                e[agg + "_ci"] = OA.boot(lambda k: OA.auc(sc[agg][k], yy[k]), len(ix), 2000)
            bins["[%.2f, %s)" % (lo, "inf" if np.isinf(hi) else "%.2f" % hi)] = e
        R["A"][st]["bins"] = bins
    for c in sorted(cats, key=lambda c: med[c]):
        L.append("| %s | %.3f | %d | " % (c, med[c], int((catidx == c).sum())) +
                 " | ".join("%.2f / %.2f" % (evals[s]["per_category"]["max"][c], evals[s]["per_category"]["covered"][c]) for s in a.stats) + " |")
    L.append("| Spearman rho (12 categories) | | | " + " | ".join("%.2f / %.2f" % (R["A"][s]["spearman_max"], R["A"][s]["spearman_covered"]) for s in a.stats) + " |")
    L += ["", "### Item-level AUROC within residual bins (pooled over categories; max / covered, 95 % item bootstrap)", "",
          "| bin (mm) | n (defective) | " + " | ".join(a.stats) + " |", "|---|---|" + "---|" * len(a.stats)]
    for b in R["A"][a.stats[0]]["bins"]:
        cells = []
        for s in a.stats:
            e = R["A"][s]["bins"][b]
            cells.append("%.2f [%.2f, %.2f] / %.2f [%.2f, %.2f]" % (e["max"], *e["max_ci"], e["covered"], *e["covered_ci"]) if "max" in e else "--")
        e0 = R["A"][a.stats[0]]["bins"][b]
        L.append("| %s | %d (%s) | " % (b, e0["n"], e0.get("n_defective", "-")) + " | ".join(cells) + " |")

    # ---- B. interventional
    pert = []
    base5 = {}
    rows5 = [r for r in rows if r["cat"] in REG5]
    for st in a.stats:
        base5[st], _, _, _ = evaluate(rows5, st, a.radius)
    for spec in a.pert:
        d, kind, val = spec.split(":")
        prow = OA.load_items(d, a.radius)
        prow = [r for r in prow if r["cat"] in REG5]
        if not prow:
            print("missing", d); continue
        e = {"items_dir": d, "kind": kind, "value": float(val), "n": len(prow), "stats": {}}
        for st in a.stats:
            e["stats"][st], _, _, _ = evaluate(prow, st, a.radius)
        pert.append(e)
    R["B"] = {"baseline": {st: base5[st] for st in a.stats}, "perturbed": pert}
    if pert:
        L += ["", "## B. Perturbed registration on the five pre-registered categories (%d items): 5-category mean object AUROC max / covered, point AUROC, hit rate" % len(rows5), "",
              "| perturbation | " + " | ".join(a.stats) + " |", "|---|" + "---|" * len(a.stats)]
        L.append("| none (baseline) | " + " | ".join("%.3f / %.3f, pt %.3f, hit %.2f" % (
            base5[s]["cat_mean"]["max"], base5[s]["cat_mean"]["covered"], base5[s]["point_auc"], base5[s]["hit_rate"]) for s in a.stats) + " |")
        for e in sorted(pert, key=lambda e: (e["kind"], e["value"])):
            L.append("| %s %.2f %s | " % (e["kind"], e["value"], "mm" if e["kind"] == "trans" else "deg") +
                     " | ".join("%.3f / %.3f, pt %.3f, hit %.2f" % (
                         e["stats"][s]["cat_mean"]["max"], e["stats"][s]["cat_mean"]["covered"], e["stats"][s]["point_auc"], e["stats"][s]["hit_rate"]) for s in a.stats) + " |")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        st0 = a.stats[0]
        ncol = 1 + (2 if pert else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.4))
        axes = np.atleast_1d(axes)
        ax = axes[0]
        for agg, mk in (("max", "o"), ("covered", "s")):
            xs = [med[c] for c in cats]; ys = [evals[st0]["per_category"][agg][c] for c in cats]
            ax.scatter(xs, ys, marker=mk, label="%s aggregation" % agg)
            if agg == "covered":
                for c, x_, y_ in zip(cats, xs, ys):
                    ax.annotate(c, (x_, y_), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("median registration residual (mm)"); ax.set_ylabel("object AUROC, %s" % st0)
        ax.set_title("A. per category (rho max %.2f, covered %.2f)" % (R["A"][st0]["spearman_max"], R["A"][st0]["spearman_covered"]), fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if pert:
            for k, kind in enumerate(("trans", "rot")):
                ax = axes[1 + k]
                ps = sorted([e for e in pert if e["kind"] == kind], key=lambda e: e["value"])
                xs = [0.0] + [e["value"] for e in ps]
                for st in a.stats:
                    for agg, ls in (("max", "-"), ("covered", "--")):
                        ys = [base5[st]["cat_mean"][agg]] + [e["stats"][st]["cat_mean"][agg] for e in ps]
                        ax.plot(xs, ys, ls, marker="o", ms=3, label="%s, %s" % (st, agg))
                    ys = [base5[st]["point_auc"]] + [e["stats"][st]["point_auc"] for e in ps]
                    ax.plot(xs, ys, ":", marker="^", ms=3, label="%s, point" % st)
                ax.set_xlabel("translation (mm)" if kind == "trans" else "rotation (deg)"); ax.set_ylabel("AUROC, 5 pre-registered categories")
                ax.set_title("B. perturbed registration", fontsize=8); ax.grid(alpha=0.3)
                if k == 0:
                    ax.legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT, "fig_reg_sensitivity.png"), dpi=200)
        fig.savefig(os.path.join(C.OUT, "fig_reg_sensitivity.pdf"))
        print("figure written", flush=True)
    except Exception as ex:
        print("figure failed:", ex, flush=True)
    with open(os.path.join(C.OUT, a.tag + ".json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, a.tag + ".md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ALL_DONE_R3_REG_SENSITIVITY", flush=True)


if __name__ == "__main__":
    main()
