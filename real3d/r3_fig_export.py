"""Export of per-item score maps for the qualitative figure of rev9 (2026-09-24).

For every defective item of the five pre-registered categories the script reads the
per-seed scores already produced by the earlier stages, nothing is recomputed:

  items_nominal_reg/<item>__r2.5.npz     nn_p95: raw point-to-template distance (2.5 mm lattice)
  items_nominal_feat_all/<item>__r3.6.npz dentnet_nom_rec3 / _norm: the local nominal subspace
                                           on DentNet features, raw and normalized (3.6 mm lattice)
  items_dinov2_v2/<item>__r3.6.npz         pnpp_gds_r1_rec3_norm: uncentred + GDS R=1 on
                                           PointNet++ features

It ranks the items by whether the raw distance's argmax misses the defect while the
normalized local subspace's argmax hits it, and exports the top candidates (the point
cloud with its per-point label, subsampled, plus the seeds and scores of every statistic)
to fig_export/<cat>__<stem>.npz together with fig_candidates.json. The figure itself is
drawn locally.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, argparse
import r3_common as C
import numpy as np

CATS5 = ["seahorse", "candybar", "fish", "toffees", "car"]
STATS3 = ["dentnet_nom_rec3", "dentnet_nom_rec3_norm", "pnpp_nom_rec3_norm"]
STATSG = ["pnpp_gds_r1_rec3_norm", "dentnet_gds_r1_rec3_norm", "pnpp_gds_r0_rec3"]
ODIR = os.path.join(C.OUT, "fig_export")


def argmax_dist(s, valid, d):
    s = np.asarray(s, float); v = valid & np.isfinite(s)
    if not v.any():
        return float("nan")
    return float(d[np.flatnonzero(v)[int(np.argmax(s[v]))]])


def seed_auc(s, valid, d, radius):
    s = np.asarray(s, float); v = valid & np.isfinite(s)
    y = d[v] <= radius
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    return float(C_auc(s[v], y))


def C_auc(scores, labels):
    from scipy.stats import rankdata
    r = rankdata(scores); n1 = labels.sum(); n0 = len(labels) - n1
    return (r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--npts", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(ODIR, exist_ok=True)
    cands = []
    for cat, stem, lab in C.enumerate_test_items():
        if cat not in CATS5 or lab == "good":
            continue
        f2 = os.path.join(C.OUT, "items_nominal_reg", "%s__%s__r2.5.npz" % (cat, stem))
        f3 = os.path.join(C.OUT, "items_nominal_feat_all", "%s__%s__r3.6.npz" % (cat, stem))
        fg = os.path.join(C.OUT, "items_dinov2_v2", "%s__%s__r3.6.npz" % (cat, stem))
        if not (os.path.exists(f2) and os.path.exists(f3) and os.path.exists(fg)):
            continue
        z2, z3, zg = np.load(f2), np.load(f3), np.load(fg)
        d2, d3 = z2["seed_defect_dist"], z3["seed_defect_dist"]
        e = dict(cat=cat, stem=stem, label=lab, n_defect=int(z3["n_defect"]), n_points=int(z3["n_points"]),
                 raw_argmax_mm=argmax_dist(z2["nn_p95"], z2["valid"], d2),
                 raw_seed_auc=seed_auc(z2["nn_p95"], z2["valid"], d2, 2.5))
        for st in STATS3:
            e[st + "_argmax_mm"] = argmax_dist(z3[st], z3["valid"], d3)
            e[st + "_seed_auc"] = seed_auc(z3[st], z3["valid"], d3, 3.6)
        for st in STATSG:
            e[st + "_argmax_mm"] = argmax_dist(zg[st], zg["valid"], d3)
            e[st + "_seed_auc"] = seed_auc(zg[st], zg["valid"], d3, 3.6)
        # the case the figure should show: raw distance misses, normalized subspace hits
        e["score"] = (float(e["raw_argmax_mm"] > 2.5) * 2 + float(e["dentnet_nom_rec3_norm_argmax_mm"] <= 3.6) * 2
                      + float(e["pnpp_gds_r1_rec3_norm_argmax_mm"] <= 3.6)
                      + (e["dentnet_nom_rec3_norm_seed_auc"] - e["raw_seed_auc"]))
        cands.append(e)
    cands.sort(key=lambda e: -e["score"])
    with open(os.path.join(C.OUT, "fig_candidates.json"), "w") as fh:
        json.dump(cands, fh, indent=1)
    print("%d defective items in the five categories" % len(cands))
    for e in cands[:a.top]:
        print("%-9s %-12s %5s n_def %6d | argmax mm: raw %5.2f sub %5.2f subn %5.2f gds %5.2f | seed AUC raw %.2f subn %.2f gds %.2f" % (
            e["cat"], e["stem"], e["label"], e["n_defect"], e["raw_argmax_mm"], e["dentnet_nom_rec3_argmax_mm"],
            e["dentnet_nom_rec3_norm_argmax_mm"], e["pnpp_gds_r1_rec3_norm_argmax_mm"],
            e["raw_seed_auc"], e["dentnet_nom_rec3_norm_seed_auc"], e["pnpp_gds_r1_rec3_norm_seed_auc"]))
        cat, stem = e["cat"], e["stem"]
        P, lab = C.read_gt(os.path.join(C.R3D_ROOT, cat, "gt", stem + ".txt"))
        if len(P) > a.npts:
            keep = np.zeros(len(P), bool)
            rng = np.random.default_rng(0)
            pos = np.flatnonzero(lab > 0); neg = np.flatnonzero(lab == 0)
            keep[pos[rng.choice(len(pos), min(len(pos), a.npts // 4), replace=False)]] = True
            keep[neg[rng.choice(len(neg), min(len(neg), a.npts - keep.sum()), replace=False)]] = True
            P, lab = P[keep], lab[keep]
        z2 = np.load(os.path.join(C.OUT, "items_nominal_reg", "%s__%s__r2.5.npz" % (cat, stem)))
        z3 = np.load(os.path.join(C.OUT, "items_nominal_feat_all", "%s__%s__r3.6.npz" % (cat, stem)))
        zg = np.load(os.path.join(C.OUT, "items_dinov2_v2", "%s__%s__r3.6.npz" % (cat, stem)))
        out = dict(cat=cat, stem=stem, label=e["label"], P=P.astype(np.float32), lab=lab.astype(np.int8),
                   seeds2=z2["seeds"], valid2=z2["valid"], d2=z2["seed_defect_dist"], nn_p95=z2["nn_p95"],
                   nom_rec_k6_pl_norm=z2["nom_rec_k6_pl_norm"],
                   seeds3=z3["seeds"], valid3=z3["valid"], d3=z3["seed_defect_dist"], cov3=z3["cov_max"],
                   meta=json.dumps(e))
        for st in STATS3:
            out[st] = z3[st]
        for st in STATSG:
            out[st] = zg[st]
        np.savez_compressed(os.path.join(ODIR, "%s__%s.npz" % (cat, stem)), **out)
    print("ALL_DONE_R3_FIG_EXPORT", flush=True)


if __name__ == "__main__":
    main()
