"""Object-level aggregation study for the nominal-reference stages.

r3_aggregate.py scores an object by the MAXIMUM window score, which windows that run off
the object hijack. This script evaluates, for every statistic in an items directory,
several pre-declared object aggregations side by side, with the same point-level AUROC and
localization as r3_aggregate.py, and per-category breakdowns:

  max        maximum over valid windows (the round-2 number)
  top5       mean of the five largest window scores
  covered    maximum over windows whose grid nodes are all sampled (cov_max <= 0.25 mm)
  top5cov    mean of the five largest scores among covered windows

Intervals: 95 % item-clustered bootstrap, 2,000 draws at object level, 300 draws over 800
points per item at point level. Higher score = more defective for every statistic here.

Output: <tag>.json and <tag>.md in ~/accv_results/real3d/.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, glob, json, argparse
import r3_common as C
import numpy as np

COV_TOL = 0.25


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    wp = np.bincount(inv, weights=y.astype(float), minlength=len(u))
    wn = np.bincount(inv, weights=(~y).astype(float), minlength=len(u))
    below = np.cumsum(wn) - wn
    return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


def boot(fn, n, draws, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(draws)]
    v = np.array(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def load_items(items_dir, radius):
    rows = []
    for fp in sorted(glob.glob(os.path.join(C.OUT, items_dir, "*__r%.1f.npz" % radius))):
        z = np.load(fp)
        rows.append(dict(cat=str(z["cat"]), stem=str(z["stem"]), defective=str(z["label"]) != "good",
                         reg_fit=float(z["template_fit_mm"]) if "template_fit_mm" in z.files else np.nan,
                         valid=z["valid"], cov=z["cov_max"] if "cov_max" in z.files else np.zeros(len(z["valid"])),
                         d=z["seed_defect_dist"], pt_label=z["pt_label"], pt_seed=z["pt_seed"],
                         stats={k: z[k] for k in z.files if z[k].ndim == 1 and z[k].dtype.kind == "f"
                                and len(z[k]) == len(z["valid"]) and k not in ("seed_defect_dist", "h_ptp", "cov_max")}))
    return rows


def object_scores(rows, stat, radius):
    out = {"max": [], "top5": [], "covered": [], "top5cov": []}
    for r in rows:
        s = r["stats"][stat].astype(float)
        v = r["valid"] & np.isfinite(s)
        cv = v & (np.nan_to_num(r["cov"], nan=np.inf) <= COV_TOL)
        for key, mask, k in (("max", v, 1), ("top5", v, 5), ("covered", cv, 1), ("top5cov", cv, 5)):
            x = np.sort(s[mask])[::-1]
            out[key].append(float(x[:k].mean()) if len(x) else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def point_level(rows, stat, rng):
    """Nearest-seed score per point, 800 points per item; returns (scores, labels, item index)."""
    S, Y, I = [], [], []
    for i, r in enumerate(rows):
        if not r["defective"]:
            continue
        s = r["stats"][stat].astype(float)
        seed = r["pt_seed"]; lab = r["pt_label"].astype(bool)
        ok = (seed >= 0) & np.isfinite(s[np.clip(seed, 0, len(s) - 1)]) & r["valid"][np.clip(seed, 0, len(s) - 1)]
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(800, len(idx)), replace=False)
        S.append(s[seed[pick]]); Y.append(lab[pick]); I.append(np.full(len(pick), i))
    return np.concatenate(S), np.concatenate(Y), np.concatenate(I)


def localization(rows, stat, radius):
    hits, dists = [], []
    for r in rows:
        if not r["defective"]:
            continue
        s = r["stats"][stat].astype(float)
        v = r["valid"] & np.isfinite(s)
        if not v.any():
            continue
        j = np.flatnonzero(v)[int(np.argmax(s[v]))]
        dists.append(float(r["d"][j])); hits.append(float(r["d"][j] <= radius))
    return np.array(hits), np.array(dists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--radius", type=float, default=C.PRIMARY_RADIUS)
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows = load_items(a.items_dir, a.radius)
    if not rows:
        print("no items"); return
    stats = a.stats or sorted(rows[0]["stats"].keys())
    y = np.array([r["defective"] for r in rows])
    cats = sorted(set(r["cat"] for r in rows))
    catidx = np.array([r["cat"] for r in rows])
    R = {"items_dir": a.items_dir, "radius": a.radius, "n_items": len(rows), "n_defective": int(y.sum()),
         "categories": cats, "stats": {}}
    L = ["# Object-level aggregation, %s, radius %.1f mm, %d items (%d defective), %d categories" % (
        a.items_dir, a.radius, len(rows), int(y.sum()), len(cats)), "",
         "| statistic | object max [CI] | object top5 | object covered | object top5cov | point AUROC [CI] | hit | median mm |",
         "|---|---|---|---|---|---|---|---|"]
    for st in stats:
        if not all(st in r["stats"] for r in rows):
            continue
        obj = object_scores(rows, st, a.radius)
        e = {"object": {}}
        for key, sc in obj.items():
            a0 = auc(sc, y)
            ci = boot(lambda ix: auc(sc[ix], y[ix]), len(rows), 2000)
            e["object"][key] = {"auc": a0, "ci": ci,
                                "per_category": {c: auc(sc[catidx == c], y[catidx == c]) for c in cats}}
        rng = np.random.default_rng(0)
        S, Yp, I = point_level(rows, st, rng)
        pa = auc(S, Yp)
        items = np.unique(I)
        def pboot(ix):
            m = np.isin(I, items[ix]); return auc(S[m], Yp[m])
        e["point"] = {"auc": pa, "ci": boot(pboot, len(items), 300)}
        hits, dists = localization(rows, st, a.radius)
        e["localization"] = {"hit_rate": float(hits.mean()), "hit_ci": boot(lambda ix: hits[ix].mean(), len(hits), 2000),
                             "median_dist_mm": float(np.median(dists))}
        R["stats"][st] = e
        o = e["object"]
        L.append("| %s | %.3f [%.3f, %.3f] | %.3f | %.3f | %.3f | %.3f [%.3f, %.3f] | %.2f | %.2f |" % (
            st, o["max"]["auc"], *o["max"]["ci"], o["top5"]["auc"], o["covered"]["auc"], o["top5cov"]["auc"],
            pa, *e["point"]["ci"], e["localization"]["hit_rate"], e["localization"]["median_dist_mm"]))
        print(L[-1], flush=True)
    # per-category table for the best aggregation of each statistic
    L += ["", "## Per-category object AUROC (aggregation = covered)", "",
          "| statistic | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    reg = {c: float(np.nanmedian([r["reg_fit"] for r in rows if r["cat"] == c])) for c in cats}
    R["median_template_fit_mm"] = reg
    L.append("| median template fit (mm) | " + " | ".join("%.2f" % reg[c] for c in cats) + " |")
    L.append("| n items (defective) | " + " | ".join("%d (%d)" % ((catidx == c).sum(), (y & (catidx == c)).sum()) for c in cats) + " |")
    for st, e in R["stats"].items():
        L.append("| %s | " % st + " | ".join("%.2f" % e["object"]["covered"]["per_category"][c] for c in cats) + " |")
    with open(os.path.join(C.OUT, a.tag + ".json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(C.OUT, a.tag + ".md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("ALL_DONE_R3_OBJECT_AGG", flush=True)


if __name__ == "__main__":
    main()
