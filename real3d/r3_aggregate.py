"""Stage 2. Metrics with uncertainty from the per-item files of r3_score.py.

For every radius and every statistic:
  object AUROC   item score = max over valid seeds (raw, and template-z-normalised using the
                 category's good training templates), defective vs good items
  point AUROC    per-point score = score of the nearest valid seed, pooled over the subsampled
                 points of all items (the Real3D-AD point-level protocol)
  localisation   on defective items: distance from the top-scoring seed to the nearest
                 ground-truth defect point, hit rate within one radius, and the rank fraction
                 at which the first hit occurs
All pooled numbers carry a 95% bootstrap interval over ITEMS (2000 draws), never over seeds or
points, since seeds of one scan are not independent. Per-category numbers are also written.

Scores are reported in the orientation "larger = more anomalous" for every statistic as
computed; AUROC below 0.5 means the statistic is inverted on this data, which Proposition 1
predicts for magd_nb. The flipped AUROC is 1 - AUROC; it is not reported separately so that
no sign is chosen after looking at the labels.

An edge filter (h_ptp <= radius, i.e. a window whose height span does not exceed its
half-extent) is reported as a secondary variant alongside the unfiltered one.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, glob, argparse
import r3_common as C
import numpy as np
from sklearn.metrics import roc_auc_score

STATS = ["sv2", "surfvar", "magd_nb", "ds_ref", "plane_rms", "rq_rms", "rq_absmax", "rq_skew"]
ITEMS_DIR = "items"
TAG = "r3_report"
N_BOOT = 2000


def load_items(radius):
    files = sorted(glob.glob(os.path.join(C.OUT, ITEMS_DIR, "*__r%.1f.npz" % radius)))
    items = []
    for fp in files:
        z = np.load(fp)
        med = json.loads(str(z["ref_med"]))
        mad = json.loads(str(z["ref_mad"]))
        valid = z["valid"]
        edge_ok = valid & (z["h_ptp"] <= radius)
        if "cov_max" in z.files:
            # covered windows only: every grid node within 0.25 mm of a sample in BOTH clouds
            edge_ok &= np.nan_to_num(z["cov_max"], nan=np.inf) <= 0.25
        it = dict(cat=str(z["cat"]), stem=str(z["stem"]), label=str(z["label"]),
                  defective=int(str(z["label"]) != "good"),
                  n_seeds=int(valid.sum()), n_edge_ok=int(edge_ok.sum()),
                  p1_check=float(z["p1_check"]), pt_label=z["pt_label"],
                  pt_seed=z["pt_seed"], dist=z["seed_defect_dist"],
                  valid=valid, edge_ok=edge_ok)
        for s in STATS:
            v = z[s].astype(float)
            it[s] = v
            it["z_" + s] = (v - med[s]) / mad[s] if s in med else v
        items.append(it)
    return items


def object_scores(items, stat, mask_key):
    y, sc = [], []
    for it in items:
        m = it[mask_key] & np.isfinite(it[stat])
        if not m.any():
            continue
        y.append(it["defective"])
        sc.append(np.nanmax(it[stat][m]))
    return np.array(y), np.array(sc)


def point_arrays(items, stat):
    ys, ss, owner = [], [], []
    for k, it in enumerate(items):
        s = it[stat][it["pt_seed"]]
        ok = np.isfinite(s)
        ys.append(it["pt_label"][ok]); ss.append(s[ok]); owner.append(np.full(ok.sum(), k))
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(owner)


def localisation(items, stat, radius, mask_key):
    """Per defective item: distance of the argmax seed to the defect, hit within one radius,
    and the rank fraction of the first seed within one radius."""
    d_top, hit, rankfrac = [], [], []
    for it in items:
        if not it["defective"]:
            continue
        m = it[mask_key] & np.isfinite(it[stat]) & np.isfinite(it["dist"])
        if m.sum() < 2:
            continue
        s, d = it[stat][m], it["dist"][m]
        order = np.argsort(-s)
        d_top.append(d[order[0]])
        hit.append(d[order[0]] <= radius)
        within = np.flatnonzero(d[order] <= radius)
        rankfrac.append((within[0] + 1) / len(order) if len(within) else 1.0)
    return np.array(d_top), np.array(hit, float), np.array(rankfrac)


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def boot_ci(fn, n_items, rng, n=N_BOOT):
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_items, n_items)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [np.nan, np.nan]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate(items, radius, rng):
    res = {}
    for stat in STATS:
        for zpref in ("", "z_"):
            key = zpref + stat
            row = {}
            for mask_key, tag in (("valid", "all"), ("edge_ok", "edgefilt")):
                y, sc = object_scores(items, key, mask_key)
                row["object_auroc_" + tag] = safe_auc(y, sc)
                row["object_auroc_%s_ci" % tag] = boot_ci(
                    lambda idx: safe_auc(y[idx], sc[idx]), len(y), rng)
            if zpref == "":
                py, ps, own = point_arrays(items, key)
                row["point_auroc"] = safe_auc(py, ps)
                # bootstrap over items on a fixed 800-point subsample per item, so that a
                # draw costs a sort of ~0.5M points rather than ~6M
                per_item = [rng.permutation(np.flatnonzero(own == k))[:800] for k in range(len(items))]

                def pfn(idx):
                    sel = np.concatenate([per_item[i] for i in idx])
                    return safe_auc(py[sel], ps[sel])
                row["point_auroc_ci"] = boot_ci(pfn, len(items), rng, n=300)
                d_top, hit, rf = localisation(items, key, radius, "valid")
                row["loc_median_dist_mm"] = float(np.median(d_top)) if len(d_top) else np.nan
                row["loc_hit_rate"] = float(hit.mean()) if len(hit) else np.nan
                row["loc_hit_rate_ci"] = boot_ci(lambda idx: float(hit[idx].mean()), len(hit), rng)
                row["loc_median_rank_frac"] = float(np.median(rf)) if len(rf) else np.nan
                row["n_defective_items"] = int(len(d_top))
            res[key] = row
    return res


def per_category(items, radius, rng):
    out = {}
    for cat in C.CATS:
        ci = [it for it in items if it["cat"] == cat]
        if not ci:
            continue
        out[cat] = dict(n_items=len(ci), n_defective=int(sum(it["defective"] for it in ci)))
        for stat in STATS:
            y, sc = object_scores(ci, stat, "valid")
            py, ps, _ = point_arrays(ci, stat)
            out[cat][stat] = dict(object_auroc=safe_auc(y, sc), point_auroc=safe_auc(py, ps))
    return out


def md_table(res, radius, items):
    lines = ["### radius %.1f mm, %d items (%d defective), %d seeds per item (median)" % (
        radius, len(items), sum(it["defective"] for it in items),
        int(np.median([it["n_seeds"] for it in items])))]
    lines.append("| statistic | object AUROC [95% CI] | object AUROC, template-z | object AUROC, edge-filtered | point AUROC [95% CI] | loc. hit rate [95% CI] | median dist mm | median rank frac |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in STATS:
        r, rz = res[s], res["z_" + s]
        ci = r["object_auroc_all_ci"]; pci = r["point_auroc_ci"]; hci = r["loc_hit_rate_ci"]
        lines.append("| %s | %.3f [%.3f, %.3f] | %.3f | %.3f | %.3f [%.3f, %.3f] | %.2f [%.2f, %.2f] | %.2f | %.3f |" % (
            s, r["object_auroc_all"], ci[0], ci[1], rz["object_auroc_all"],
            r["object_auroc_edgefilt"], r["point_auroc"], pci[0], pci[1],
            r["loc_hit_rate"], hci[0], hci[1], r["loc_median_dist_mm"], r["loc_median_rank_frac"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="*", default=C.RADII_MM)
    ap.add_argument("--items-dir", default="items")
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    global ITEMS_DIR, STATS, TAG
    ITEMS_DIR = a.items_dir
    if a.stats:
        STATS = a.stats
    TAG = a.tag or ("r3_report" if a.items_dir == "items" else "r3_report_" + a.items_dir)
    rng = np.random.default_rng(2026)
    report, md = {}, ["# Real3D-AD, training-free statistics, defect versus good\n"]
    for radius in a.radii:
        items = load_items(radius)
        if not items:
            print("no items for radius", radius); continue
        p1 = np.nanmax([it["p1_check"] for it in items])
        print("radius %.1f: %d items, Proposition 1 max |Mag(D) - closed form| = %.2e" % (radius, len(items), p1), flush=True)
        res = evaluate(items, radius, rng)
        report["r%.1f" % radius] = dict(
            n_items=len(items), n_defective=int(sum(it["defective"] for it in items)),
            n_good=int(sum(1 - it["defective"] for it in items)),
            n_bulge=int(sum(it["label"] == "bulge" for it in items)),
            n_sink=int(sum(it["label"] == "sink" for it in items)),
            median_seeds=float(np.median([it["n_seeds"] for it in items])),
            prop1_max_abs_diff=float(p1), pooled=res,
            per_category=per_category(items, radius, rng))
        t = md_table(res, radius, items)
        md.append(t + "\n")
        print(t, flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, TAG + ".json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    with open(os.path.join(C.OUT, TAG + ".md"), "w") as f:
        f.write("\n".join(md))
    print("ALL_DONE_R3_AGGREGATE", flush=True)


if __name__ == "__main__":
    main()
