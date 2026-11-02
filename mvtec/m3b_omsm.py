"""Stage 3b. GDS variants and OMSM on MVTec 3D-AD (supervisor's questions, 2026-09-23).

Same harness as m3_rules.py on the DentNet and DINOv2 encoders: the paper's GDS with
refitted PCA, the GDS with the original bases projected and re-orthonormalized, and OMSM
(Lambda^{-1/2} whitening of the sum space, bases re-orthonormalized), all selected on
held-out synthetic training samples. Output: mv_omsm_<proto>.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import mv_common as M
import os, argparse
import numpy as np
from mv_common import C
import gds as G8
from gds import SM, GDS
import omsm as O
import m3_rules as M3

ENCODERS = ["dentnet_1024", "pnpp_r0.3_128", "pnpp_r0.6_128", "pnpp_r2.0_128", "dinov2_1024"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="sim")
    a = ap.parse_args()
    reps = M3.load_synthetic()
    rng = np.random.default_rng(0)
    rows = M3.load_scans(a.proto, rng)
    cats = [c for c in M.CATS if any(r["cat"] == c and r["split"] == "test" for r in rows)]
    R = {"proto": a.proto, "encoders": {}}
    L = ["# GDS refit, GDS with projected bases, and OMSM on MVTec 3D-AD, protocol %s" % a.proto, "",
         "| encoder | rule | val AUC | window AUC | pixel AUROC | image AUROC | good test flagged % | margin shift (sigma) | syn bg before -> after % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for enc in ENCODERS:
        rep = reps[enc]
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        Ks = (10, 25, 50) if Xtr.shape[1] >= 256 else (5, 10, 25)
        bsm, bgds, _ = G8.select(Xtr, ytr, Ks, M.log)
        best, grid = O.select(Xtr, ytr, Ks, M.log)
        M.log(enc, "selected", {"sm": bsm, "gds_refit": bgds, **best})
        rules = [SM(bsm["K"]), GDS(bgds["K"], bgds["R"], bgds["Q"]),
                 O.GDSBasis(best["gdsbasis"]["K"], best["gdsbasis"]["R"]), O.OMSM(best["omsm"]["K"], best["omsm"]["R"]),
                 O.OMSM(bgds["K"], 0), O.OMSM(bgds["K"], bgds["R"])]
        seen = set(); rules = [r for r in rules if not (r.name in seen or seen.add(r.name))]
        out = {"selected": {"sm": bsm, "gds_refit": bgds, **best}, "rules": {}}
        bgva = Xva[yva == 0]
        tg = np.concatenate([r["feats"][enc] for r in rows if r["split"] == "train"]).astype(np.float32)
        delta = tg.mean(0) - bgva.mean(0)
        for r_ in rules:
            r_.fit(Xtr, ytr)
            sva = r_.score(Xva); sig = float(sva.std())
            e = {"auc_val": C.auc(sva, yva)}
            margins = [r_.score(r["feats"][enc].astype(np.float32)) if len(r["frac"]) else np.zeros(0) for r in rows]
            e["categories"] = {c: M3.evaluate(rows, margins, c) for c in cats}
            s0 = r_.score(bgva); s1 = r_.score(bgva + delta)
            e["shift_of_margin_in_sigma"] = float((s1 - s0).mean() / sig)
            e["flagged_synthetic_bg_before"] = float((s0 > 0).mean()); e["flagged_synthetic_bg_after_shift"] = float((s1 > 0).mean())
            out["rules"][r_.name] = e
            ce = e["categories"]
            line = "| %s | %s | %.3f | %.3f | %.3f | %.3f | %.1f | %+.2f | %.1f -> %.1f |" % (
                enc, r_.name, e["auc_val"], M3.cat_mean(ce, "window_auc"), M3.cat_mean(ce, "pixel_auroc"), M3.cat_mean(ce, "image_auroc"),
                100 * M3.cat_mean(ce, "good_flagged"), e["shift_of_margin_in_sigma"], 100 * e["flagged_synthetic_bg_before"], 100 * e["flagged_synthetic_bg_after_shift"])
            L.append(line); M.log(line)
        R["encoders"][enc] = out
        M.save_json("mv_omsm_%s.json" % a.proto, R)
    M.write_md("mv_omsm_%s.md" % a.proto, L)
    print("ALL_DONE_M3B", flush=True)


if __name__ == "__main__":
    main()
