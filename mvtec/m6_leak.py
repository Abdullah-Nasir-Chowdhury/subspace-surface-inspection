"""Stage 6. The leakage diagnostic per MVTec class (2026-09-24, rev11).

The proposition of the paper bounds the move of the nearest-subspace margin under a
feature shift delta by the part of delta outside the two class subspaces (the leak),
divided by the margin's in-domain standard deviation. m5_bound.py checked the bound on the
pooled shift of every encoder. This stage asks whether the bound, computed from UNLABELED
good scans only, predicts what happens on unseen good scans, class by class:

  per encoder, protocol and class c
    delta_c        mean feature of the good TRAIN windows of class c minus the synthetic
                   validation background mean (label-free: only good scans enter)
    leak / bound   ||(I - P_n) delta_c|| + ||(I - P_d) delta_c|| over sigma_margin
    simulated      margin move when delta_c is added to every synthetic background window
    observed       mean margin of the good TEST windows of class c minus the synthetic
                   background mean margin, in sigma_margin (the displacement the threshold
                   actually meets on unseen scans of that class)
    spread         std of the good TEST margins over sigma_margin (the second-order part)
    domain AUC     AUC of good TEST margins against synthetic background margins
    good flagged   fraction of good TEST windows above the a-priori threshold (margin 0)
    covariance     ||Sigma_real - Sigma_syn||_F / ||Sigma_syn||_F on the good TEST windows,
                   and the fraction of their variance inside the normal class subspace
                   against the same fraction for the synthetic background

  for nearest subspace, nearest mean (bound with P = 0), logistic (no bound) and the GDS
  (no bound). Spearman correlations across the classes and encoders close the loop:
  bound -> observed, simulated -> observed, observed -> good flagged.
Output: mv_leak_<proto>.{json,md}
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
import m3_rules as M3
from gds import GDS


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="sim")
    a = ap.parse_args()
    reps = M3.load_synthetic()
    sel = json.load(open(os.path.join(M.OUT, "mv_rules_%s.json" % a.proto)))["encoders"]
    # real windows, grouped per encoder later; only the keys needed are kept
    scans = []
    for fp in sorted(glob.glob(os.path.join(M.OUT, "feats_" + a.proto, "*.npz"))):
        z = np.load(fp)
        if not len(z["frac"]):
            continue
        scans.append(dict(cat=str(z["cat"]), split=str(z["split"]), dtype=str(z["dtype"]),
                          feats={e: z[e] for e in M3.ENCODERS}))
    M.log(a.proto, len(scans), "scans")
    R = {"proto": a.proto, "encoders": {}, "correlations": {}}
    L = ["# Leakage diagnostic per class, protocol %s" % a.proto, "",
         "| encoder | rule | class | shift/sigma_f | in normal | in dent | bound | simulated | observed | spread | domain AUC | good flagged | cov shift |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pooled = {"bound": [], "simulated": [], "observed": [], "flagged": [], "rule": [], "enc": [], "cat": []}
    for enc in M3.ENCODERS:
        rep = reps[enc]
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        big = Xtr.shape[1] >= 256
        bgva = Xva[yva == 0].astype(np.float32)
        mu_bg = bgva.mean(0)
        sig_f = float(np.sqrt(bgva.var(0).sum()))
        Sig_bg = np.cov(bgva, rowvar=False)
        g = sel[enc]["selected"]["gds"]
        rules = [C.PBSS(50 if big else 25), C.NearestMean(), C.Logistic(), GDS(g["K"], g["R"], g["Q"])]
        for r_ in rules:
            r_.fit(Xtr, ytr)
        sig = {r_.name: float(r_.score(Xva).std()) for r_ in rules}
        s_bg = {r_.name: r_.score(bgva) for r_ in rules}
        Vn = np.asarray(rules[0].b.normal_subspace["components"]); Vd = np.asarray(rules[0].b.dent_subspace["components"])
        var_in_n_bg = float(np.trace(Vn @ Sig_bg @ Vn.T) / np.trace(Sig_bg))
        out = {"feature_sigma": sig_f, "sigma_margin": sig, "classes": {}}
        for cat in M.CATS:
            tr = [s["feats"][enc] for s in scans if s["cat"] == cat and s["split"] == "train"]
            te = [s["feats"][enc] for s in scans if s["cat"] == cat and s["split"] == "test" and s["dtype"] == "good"]
            if not tr or not te:
                continue
            tr = np.concatenate(tr).astype(np.float32); te = np.concatenate(te).astype(np.float32)
            delta = tr.mean(0) - mu_bg
            Sig_te = np.cov(te, rowvar=False)
            ln = float(np.linalg.norm(delta - Vn.T @ (Vn @ delta))); ld = float(np.linalg.norm(delta - Vd.T @ (Vd @ delta)))
            e = {"n_train_windows": int(len(tr)), "n_good_test_windows": int(len(te)),
                 "delta_norm_over_sigma_f": float(np.linalg.norm(delta) / sig_f),
                 "energy_in_normal": float(np.sum((Vn @ delta) ** 2) / np.sum(delta ** 2)),
                 "energy_in_dent": float(np.sum((Vd @ delta) ** 2) / np.sum(delta ** 2)),
                 "leak_normal_over_sigma_f": ln / sig_f, "leak_dent_over_sigma_f": ld / sig_f,
                 "cov_shift_frobenius": float(np.linalg.norm(Sig_te - Sig_bg) / np.linalg.norm(Sig_bg)),
                 "trace_ratio_real_over_syn": float(np.trace(Sig_te) / np.trace(Sig_bg)),
                 "var_in_normal_subspace_real": float(np.trace(Vn @ Sig_te @ Vn.T) / np.trace(Sig_te)),
                 "var_in_normal_subspace_syn": var_in_n_bg,
                 "test_mean_shift_over_train_mean_shift": float(np.linalg.norm(te.mean(0) - mu_bg) / max(np.linalg.norm(delta), 1e-12)),
                 "rules": {}}
            for r_ in rules:
                nm = r_.name; s0 = s_bg[nm]
                sim_ = float((r_.score(bgva + delta) - s0).mean() / sig[nm])
                ste = r_.score(te)
                obs = float((ste.mean() - s0.mean()) / sig[nm])
                q = {"simulated_shift_sigma": sim_, "observed_shift_sigma": obs,
                     "observed_spread_sigma": float(ste.std() / sig[nm]),
                     "domain_auc": float(C.auc(np.r_[ste, s0], np.r_[np.ones(len(ste), bool), np.zeros(len(s0), bool)])),
                     "good_flagged": float((ste > 0).mean())}
                if isinstance(r_, C.PBSS):
                    q["bound_sigma"] = (ln + ld) / sig[nm]
                elif isinstance(r_, C.NearestMean):
                    q["bound_sigma"] = 2 * float(np.linalg.norm(delta)) / sig[nm]
                else:
                    q["bound_sigma"] = float("nan")
                e["rules"][nm] = q
                pooled["bound"].append(q["bound_sigma"]); pooled["simulated"].append(sim_); pooled["observed"].append(obs)
                pooled["flagged"].append(q["good_flagged"]); pooled["rule"].append(nm); pooled["enc"].append(enc); pooled["cat"].append(cat)
                L.append("| %s | %s | %s | %.2f | %.2f | %.2f | %s | %+.2f | %+.2f | %.2f | %.3f | %.2f | %.2f |" % (
                    enc, nm, cat, e["delta_norm_over_sigma_f"], e["energy_in_normal"], e["energy_in_dent"],
                    ("%.2f" % q["bound_sigma"]) if np.isfinite(q["bound_sigma"]) else "", sim_, obs,
                    q["observed_spread_sigma"], q["domain_auc"], q["good_flagged"], e["cov_shift_frobenius"]))
            out["classes"][cat] = e
        R["encoders"][enc] = out
        M.log(enc, "done")
    # correlations across classes and encoders, per rule
    rule_names = sorted(set(pooled["rule"]))
    for nm in rule_names:
        idx = [i for i, r in enumerate(pooled["rule"]) if r == nm]
        b = [pooled["bound"][i] for i in idx]; s = [pooled["simulated"][i] for i in idx]
        o = [pooled["observed"][i] for i in idx]; f = [pooled["flagged"][i] for i in idx]
        R["correlations"][nm] = {
            "n": len(idx),
            "spearman_bound_observed": spearman(b, o), "pearson_bound_observed": pearson(b, o),
            "spearman_simulated_observed": spearman(s, o), "pearson_simulated_observed": pearson(s, o),
            "spearman_observed_flagged": spearman(o, f),
            "frac_observed_within_bound": float(np.mean([abs(oo) <= bb for oo, bb in zip(o, b) if np.isfinite(bb)])) if np.isfinite(b).any() else float("nan"),
            "median_observed_over_bound": float(np.nanmedian([abs(oo) / bb for oo, bb in zip(o, b) if np.isfinite(bb) and bb > 0])) if np.isfinite(b).any() else float("nan"),
            "median_abs_observed_minus_simulated": float(np.median(np.abs(np.array(o) - np.array(s))))}
        # per-encoder Spearman across the ten classes
        per_enc = {}
        for enc in M3.ENCODERS:
            j = [i for i in idx if pooled["enc"][i] == enc]
            per_enc[enc] = {"spearman_bound_observed": spearman([pooled["bound"][i] for i in j], [pooled["observed"][i] for i in j]),
                            "spearman_simulated_observed": spearman([pooled["simulated"][i] for i in j], [pooled["observed"][i] for i in j]),
                            "spearman_observed_flagged": spearman([pooled["observed"][i] for i in j], [pooled["flagged"][i] for i in j])}
        R["correlations"][nm]["per_encoder"] = per_enc
    R["points"] = pooled
    L += ["", "## Correlations across classes and encoders", "",
          "| rule | n | Spearman bound vs observed | Pearson | Spearman simulated vs observed | Pearson | Spearman observed vs good flagged | within bound | median observed/bound | median abs(observed - simulated) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for nm in rule_names:
        c = R["correlations"][nm]
        L.append("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            nm, c["n"], c["spearman_bound_observed"], c["pearson_bound_observed"], c["spearman_simulated_observed"],
            c["pearson_simulated_observed"], c["spearman_observed_flagged"], c["frac_observed_within_bound"],
            c["median_observed_over_bound"], c["median_abs_observed_minus_simulated"]))
    M.save_json("mv_leak_%s.json" % a.proto, R)
    with open(os.path.join(M.OUT, "mv_leak_%s.md" % a.proto), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[-len(rule_names) - 4:]))
    print("ALL_DONE_M6", flush=True)


if __name__ == "__main__":
    main()
