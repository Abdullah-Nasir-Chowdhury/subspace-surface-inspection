"""Flagged fractions of the learned methods at the paper's fixed decision thresholds
(DentNet p > 0.5 on the window mean, PBSS / nearest mean / logistic score > 0, PointNet++
p > 0.5), the zone-free statistic the paper reports on the fuselage scans. Reported on good
items (a pure false-positive rate, since nothing on them is a defect) and on defective items,
overall and per category, for each items_learned_s<scale> directory found.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, glob, json
import r3_common as C
import numpy as np

RULES = {"dentnet_mean": 0.5, "pbss_k50": 0.0, "nearest_mean": 0.0, "logistic": 0.0, "pnpp": 0.5}


def main():
    report = {}
    for d in sorted(glob.glob(os.path.join(C.OUT, "items_learned_s*"))):
        tag = os.path.basename(d)
        rows = []
        for fp in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(fp)
            v = z["valid"]
            r = dict(cat=str(z["cat"]), good=str(z["label"]) == "good")
            for k, thr in RULES.items():
                x = z[k][v]; x = x[np.isfinite(x)]
                r[k] = float((x > thr).mean()) if len(x) else np.nan
            rows.append(r)
        out = {"n_good": sum(r["good"] for r in rows), "n_defective": sum(not r["good"] for r in rows)}
        lines = ["### %s: window flagged fraction at fixed thresholds (%d good, %d defective items)" % (
            tag, out["n_good"], out["n_defective"]),
            "| rule | good items (FPR) | defective items | per-category FPR min / max |", "|---|---|---|---|"]
        for k in RULES:
            g = np.array([r[k] for r in rows if r["good"]]); dd = np.array([r[k] for r in rows if not r["good"]])
            pc = {c: float(np.nanmean([r[k] for r in rows if r["good"] and r["cat"] == c])) for c in C.CATS}
            out[k] = dict(fpr_good=float(np.nanmean(g)), flagged_defective=float(np.nanmean(dd)), per_cat_fpr=pc)
            lo, hi = min(pc, key=pc.get), max(pc, key=pc.get)
            lines.append("| %s | %.3f | %.3f | %s %.3f / %s %.3f |" % (k, out[k]["fpr_good"], out[k]["flagged_defective"], lo, pc[lo], hi, pc[hi]))
        report[tag] = out
        print("\n".join(lines) + "\n", flush=True)
    with open(os.path.join(C.OUT, "r3_learned_summary.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("ALL_DONE_R3_LEARNED_SUMMARY", flush=True)


if __name__ == "__main__":
    main()
