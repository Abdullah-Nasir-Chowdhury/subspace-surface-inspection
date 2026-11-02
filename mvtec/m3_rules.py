"""Stage 3. The paper's Table 2 on MVTec 3D-AD: five frozen encoders, the rules, the shift.

For every encoder the rules are fitted on the synthetic training windows only (PBSS,
nearest mean, logistic head, and the uncentred SM and GDS of rv8 with K, R, Q selected on
held-out synthetic training samples) and applied zero-shot to every MVTec window. The
DINOv2 synthetic features are the per-window ones of m2 --synthetic.

Per category and protocol:
  window AUC     defect windows (frac > LABEL_THR) against normal windows (frac == 0) of
                 the test scans, scan-clustered bootstrap
  image AUROC    max margin over a scan's windows, defective against good test scans
  pixel AUROC    every pixel of a defective test scan scored by the max margin of the
                 windows covering it, against the gt mask, 20,000 pixels per scan
  good flagged   fraction of windows of the good test scans above the a-priori threshold
                 (margin 0)
  shift          delta = mean feature of the train/good windows minus the synthetic
                 validation background mean; margin shift under delta in in-domain sigma,
                 flagged synthetic background before and after, energy of delta inside
                 the class subspaces (PBSS) and inside the GDS

A scan is defective when its defect type is not "good" (MVTec ships all-black gt masks
for the good test scans too). The end-to-end scores of stage 2 (DentNet probability,
PointNet++ probability) are evaluated on the same windows as controls.
Output: mv_rules_<proto>.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import mv_common as M
import os, glob, argparse, json
import numpy as np
from mv_common import C
import gds as G8
from gds import SM, GDS, unit

ENCODERS = ["dentnet_1024", "pnpp_r0.3_128", "pnpp_r0.6_128", "pnpp_r2.0_128", "dinov2_1024"]


def load_synthetic():
    reps = G8.load_reps()
    sd = os.path.expanduser("~/accv_results_mvtec")
    tr = np.load(os.path.join(sd, "syn_dinov2_train.npz")); va = np.load(os.path.join(sd, "syn_dinov2_val.npz"))
    reps["dinov2_1024"] = {"train": dict(features=tr["features"].astype(np.float32), labels=tr["labels"].astype(np.int8)),
                           "val": dict(features=va["features"].astype(np.float32), labels=va["labels"].astype(np.int8), sample_idx=va["sample_idx"])}
    return reps


def pixel_lookup(shape, pos, gt_packed, valid_packed, rng, per_scan=20000):
    """A seeded sample of covered valid pixels, each with the (up to four) windows that
    cover it. Windows of size PATCH at stride STRIDE cover a pixel at most once per parity
    of (y0 // STRIDE, x0 // STRIDE), which gives the slot."""
    H, W = shape
    gt = np.unpackbits(gt_packed)[:H * W].reshape(H, W).astype(bool)
    valid = np.unpackbits(valid_packed)[:H * W].reshape(H, W).astype(bool)
    widx = np.full((H, W, 4), -1, np.int32)
    for k, (y, x) in enumerate(pos):
        slot = ((y // M.STRIDE) & 1) * 2 + ((x // M.STRIDE) & 1)
        widx[y:y + M.PATCH, x:x + M.PATCH, slot] = k
    ok = np.flatnonzero(((widx >= 0).any(2) & valid).ravel())
    if len(ok) == 0:
        return None
    pick = rng.choice(ok, min(per_scan, len(ok)), replace=False)
    return widx.reshape(-1, 4)[pick], gt.ravel()[pick]


def load_scans(proto, rng):
    rows = []
    for fp in sorted(glob.glob(os.path.join(M.OUT, "feats_" + proto, "*.npz"))):
        z = np.load(fp)
        w = np.load(os.path.join(M.OUT, "windows_" + proto, os.path.basename(fp)))
        r = dict(id=os.path.basename(fp)[:-4], cat=str(z["cat"]), split=str(z["split"]), dtype=str(z["dtype"]),
                 defective=str(z["dtype"]) != "good", frac=z["frac"], pos=z["pos"],
                 feats={e: z[e] for e in ENCODERS}, e2e={k: z[k] for k in ("dentnet_prob_mean", "dentnet_prob_max", "pnpp_prob")},
                 pix=None)
        if r["split"] == "test" and r["defective"] and len(r["pos"]):
            r["pix"] = pixel_lookup(tuple(int(v) for v in w["shape_small"]), r["pos"], w["gt_small"], w["valid_small"], rng)
        rows.append(r)
    return rows


def boot_auc(scores, labels, groups, n=1000, seed=0):
    """Scan-clustered bootstrap of the AUC: each draw resamples scans with replacement,
    which is a weighted AUC with the draw's multiplicity as the weight (C.WeightedAUC keeps
    the ranking fixed, so a draw costs one pass)."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    g, inv = np.unique(groups, return_inverse=True)
    wa = C.WeightedAUC(scores, labels); vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[inv].astype(float)
        if (w * labels).sum() == 0 or (w * ~labels).sum() == 0:
            continue
        vals.append(wa(w))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def pixel_auc(rows, margins):
    S, Y = [], []
    for r, m in zip(rows, margins):
        if r["pix"] is None:
            continue
        pw, pg = r["pix"]
        if pg.sum() == 0:
            continue
        sc = np.where(pw >= 0, m[np.maximum(pw, 0)], -np.inf).max(1)
        S.append(sc); Y.append(pg)
    if not S:
        return float("nan")
    return C.auc(np.concatenate(S), np.concatenate(Y))


def evaluate(rows, margins, cat):
    """margins: list aligned with rows. Returns the per-category metrics for one scorer."""
    idx = [i for i, r in enumerate(rows) if r["cat"] == cat]
    test = [i for i in idx if rows[i]["split"] == "test"]
    ws, wl, wg = [], [], []
    for i in test:
        f = rows[i]["frac"]; keep = (f > M.LABEL_THR) | (f == 0)
        ws.append(margins[i][keep]); wl.append(f[keep] > M.LABEL_THR); wg.append(np.full(int(keep.sum()), i))
    ws, wl, wg = np.concatenate(ws), np.concatenate(wl), np.concatenate(wg)
    img_s = np.array([margins[i].max() if len(margins[i]) else np.nan for i in test])
    img_y = np.array([rows[i]["defective"] for i in test])
    good = [i for i in test if not rows[i]["defective"]]
    bad = [i for i in test if rows[i]["defective"]]
    return {"n_test": len(test), "n_defective": int(img_y.sum()), "n_windows": int(len(ws)), "n_defect_windows": int(wl.sum()),
            "window_auc": C.auc(ws, wl), "window_auc_ci": boot_auc(ws, wl, wg),
            "image_auroc": C.auc(img_s, img_y), "image_auroc_ci": boot_auc(img_s, img_y, np.arange(len(test))),
            "pixel_auroc": pixel_auc([rows[i] for i in test], [margins[i] for i in test]),
            "good_flagged": float(np.mean(np.concatenate([margins[i] > 0 for i in good]))) if good else float("nan"),
            "defective_flagged": float(np.mean(np.concatenate([margins[i] > 0 for i in bad]))) if bad else float("nan")}


def cat_mean(e, k):
    return float(np.nanmean([v[k] for v in e.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="sim")
    ap.add_argument("--cats", nargs="*", default=M.CATS)
    a = ap.parse_args()
    reps = load_synthetic()
    rng = np.random.default_rng(0)
    rows = load_scans(a.proto, rng)
    a.cats = [c for c in a.cats if any(r["cat"] == c and r["split"] == "test" for r in rows)]
    M.log(a.proto, len(rows), "scans, categories", a.cats)
    R = {"proto": a.proto, "encoders": {}, "end_to_end": {}}
    for k in ("dentnet_prob_mean", "dentnet_prob_max", "pnpp_prob"):
        margins = [r["e2e"][k].astype(np.float32) - 0.5 for r in rows]
        R["end_to_end"][k] = {c: evaluate(rows, margins, c) for c in a.cats}
        e = R["end_to_end"][k]
        M.log("%s: cat-mean window %.3f image %.3f pixel %.3f good-flagged %.1f%%" % (
            k, cat_mean(e, "window_auc"), cat_mean(e, "image_auroc"), cat_mean(e, "pixel_auroc"), 100 * cat_mean(e, "good_flagged")))
    for enc in ENCODERS:
        rep = reps[enc]
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        big = Xtr.shape[1] >= 256
        bsm, bgds, grid = G8.select(Xtr, ytr, (10, 25, 50) if big else (5, 10, 25), M.log)
        rules = [C.PBSS(50 if big else 25), C.NearestMean(), C.Logistic(), SM(bsm["K"]), GDS(bgds["K"], bgds["R"], bgds["Q"])]
        out = {"dim": int(Xtr.shape[1]), "selected": {"sm": bsm, "gds": bgds}, "rules": {}}
        bgva = Xva[yva == 0]
        tg = np.concatenate([r["feats"][enc] for r in rows if r["split"] == "train"]).astype(np.float32)
        delta = tg.mean(0) - bgva.mean(0)
        delta_u = unit(tg).mean(0) - unit(bgva).mean(0)
        out["delta_norm_over_bg_feature_std"] = float(np.linalg.norm(delta) / np.sqrt(bgva.var(0).sum()))
        for r_ in rules:
            r_.fit(Xtr, ytr)
            sva = r_.score(Xva); sig = float(sva.std())
            e = {"auc_val": C.auc(sva, yva), "f1_val": C.weighted_prf(sva > 0, yva, np.ones(len(yva)))[2], "sigma_val": sig}
            margins = [r_.score(r["feats"][enc].astype(np.float32)) if len(r["frac"]) else np.zeros(0) for r in rows]
            e["categories"] = {c: evaluate(rows, margins, c) for c in a.cats}
            s0 = r_.score(bgva); s1 = r_.score(bgva + delta)
            e["shift_of_margin_in_sigma"] = float((s1 - s0).mean() / sig)
            e["flagged_synthetic_bg_before"] = float((s0 > 0).mean()); e["flagged_synthetic_bg_after_shift"] = float((s1 > 0).mean())
            e["train_good_flagged"] = float(np.mean(np.concatenate([m > 0 for m, r in zip(margins, rows) if r["split"] == "train"])))
            if isinstance(r_, C.PBSS):
                Vn = np.asarray(r_.b.normal_subspace["components"]); Vd = np.asarray(r_.b.dent_subspace["components"])
                e["delta_energy_in_normal_subspace"] = float(np.sum((Vn @ delta) ** 2) / np.sum(delta ** 2))
                e["delta_energy_in_dent_subspace"] = float(np.sum((Vd @ delta) ** 2) / np.sum(delta ** 2))
            if isinstance(r_, GDS):
                tot = float(np.sum(delta_u ** 2))
                e["delta_u_energy_in_removed_principal"] = float(np.sum((r_.principal @ delta_u) ** 2) / tot)
                e["delta_u_energy_in_gds"] = float(np.sum((r_.D @ delta_u) ** 2) / tot)
                e["delta_u_energy_outside_sumspace"] = float(1 - np.sum((r_.sumspace @ delta_u) ** 2) / tot)
            out["rules"][r_.name] = e
            ce = e["categories"]
            M.log("%s %s: val %.3f | cat-mean window %.3f image %.3f pixel %.3f good-flagged %.1f%% | shift %+.2f syn-bg %.1f -> %.1f%%" % (
                enc, r_.name, e["auc_val"], cat_mean(ce, "window_auc"), cat_mean(ce, "image_auroc"), cat_mean(ce, "pixel_auroc"),
                100 * cat_mean(ce, "good_flagged"), e["shift_of_margin_in_sigma"],
                100 * e["flagged_synthetic_bg_before"], 100 * e["flagged_synthetic_bg_after_shift"]))
        R["encoders"][enc] = out
        M.save_json("mv_rules_%s.json" % a.proto, R)
    print("ALL_DONE_M3", flush=True)


if __name__ == "__main__":
    main()
