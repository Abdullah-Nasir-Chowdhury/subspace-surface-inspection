"""Report for stages 5 and 5b: one table of every reference-based statistic on the five
registered categories, and the bulge-versus-sink polarity test on the signed reference
difference.

Reads the aggregator's JSON reports (run r3_aggregate.py first for items_nominal and
items_nominal_feat) and the item files for the polarity test.
Output: r3_nominal_report.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, glob, json
import r3_common as C
import numpy as np


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def boot_auc(s, y, n=2000, seed=0):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    rng = np.random.default_rng(seed)
    d = [auc(s[i], y[i]) for i in (rng.integers(0, len(s), len(s)) for _ in range(n))]
    return auc(s, y), [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def polarity(items_dir, radius, stat, near_frac=0.5):
    rows = []
    for fp in sorted(glob.glob(os.path.join(C.OUT, items_dir, "*__r%.1f.npz" % radius))):
        z = np.load(fp)
        lab = str(z["label"])
        if lab == "good" or stat not in z.files:
            continue
        v = z["valid"]; d = z["seed_defect_dist"]
        near = v & (d <= near_frac * radius) & np.isfinite(z[stat])
        if not near.any():
            continue
        rows.append((lab == "bulge", float(np.mean(z[stat][near])), str(z["cat"])))
    if not rows:
        return None
    y = np.array([r[0] for r in rows]); s = np.array([r[1] for r in rows])
    a, ci = boot_auc(s, y)
    return {"n": len(rows), "n_bulge": int(y.sum()), "auc": a, "ci": ci,
            "sign_accuracy": float(((s > 0) == y).mean())}


def load_report(tag):
    p = os.path.join(C.OUT, tag + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    R = {"polarity": {}, "reports": {}}
    L = ["# Real3D-AD, subspace methods given the nominal reference (five registered categories)", ""]
    for tag in ("r3_report_items_nominal", "r3_report_items_nominal_feat"):
        rep = load_report(tag)
        if rep is None:
            L.append("(missing %s)" % tag); continue
        R["reports"][tag] = rep
        md = os.path.join(C.OUT, tag + ".md")
        if os.path.exists(md):
            L += ["## " + tag, "", open(md).read(), ""]
    for rad in C.RADII_MM:
        for stat in ("diff_mean",):
            pol = polarity("items_nominal", rad, stat)
            if pol:
                R["polarity"]["%s_r%.1f" % (stat, rad)] = pol
    L += ["## Polarity from the reference: bulge (positive) against sink, defective items, "
          "mean signed height difference test minus template over seeds within half a radius of the defect", "",
          "| radius | n (bulge) | AUROC [95% CI] | sign accuracy |", "|---|---|---|---|"]
    for k, p in R["polarity"].items():
        L.append("| %s | %d (%d) | %.3f [%.3f, %.3f] | %.3f |" % (k, p["n"], p["n_bulge"], p["auc"], p["ci"][0], p["ci"][1], p["sign_accuracy"]))
    with open(os.path.join(C.OUT, "r3_nominal_report.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, "r3_nominal_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-8:]))
    print("ALL_DONE_R3_NOMINAL_REPORT", flush=True)


if __name__ == "__main__":
    main()
