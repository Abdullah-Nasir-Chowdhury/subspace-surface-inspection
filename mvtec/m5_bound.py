"""The margin-shift bound of rev9, checked on the measured MVTec shift (2026-09-24).

For the nearest-subspace rule e_c(f) = ||(I - P_c)(f - mu_c)|| the triangle inequality gives
|e_c(f + delta) - e_c(f)| <= ||(I - P_c) delta||, so the margin e_n - e_d moves by at most
||(I - P_n) delta|| + ||(I - P_d) delta|| under any shift delta, and not at all when delta
lies in both class subspaces. The logistic margin moves by exactly w' delta. This script
evaluates both sides on the sim-protocol shift of m3_rules.py, in the margin's in-domain
standard deviation, so the paper can state the bound and the measured value in one unit.
Output: mv_bound_<proto>.{json,md}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="sim")
    a = ap.parse_args()
    reps = M3.load_synthetic()
    R = {"proto": a.proto, "encoders": {}}
    L = ["# Margin-shift bound on the measured shift, protocol %s" % a.proto, "",
         "| encoder | rule | sigma (feature) | shift norm / sigma_f | leak normal | leak dent | bound (margin sigma) | measured (margin sigma) |",
         "|---|---|---|---|---|---|---|---|"]
    for enc in M3.ENCODERS:
        rep = reps[enc]
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        big = Xtr.shape[1] >= 256
        bgva = Xva[yva == 0]
        tg = []
        for fp in sorted(glob.glob(os.path.join(M.OUT, "feats_" + a.proto, "*.npz"))):
            z = np.load(fp)
            if str(z["split"]) == "train":
                tg.append(z[enc])
        tg = np.concatenate(tg).astype(np.float32)
        delta = tg.mean(0) - bgva.mean(0)
        sig_f = float(np.sqrt(bgva.var(0).sum()))
        out = {"delta_norm": float(np.linalg.norm(delta)), "feature_sigma": sig_f,
               "delta_norm_over_feature_sigma": float(np.linalg.norm(delta) / sig_f)}
        rules = [C.PBSS(50 if big else 25), C.NearestMean(), C.Logistic()]
        for r_ in rules:
            r_.fit(Xtr, ytr)
            sva = r_.score(Xva); sig = float(sva.std())
            s0 = r_.score(bgva); s1 = r_.score(bgva + delta)
            e = {"sigma_val": sig, "measured_shift_sigma": float((s1 - s0).mean() / sig),
                 "measured_shift_max_sigma": float(np.abs(s1 - s0).max() / sig)}
            if isinstance(r_, C.PBSS):
                Vn = np.asarray(r_.b.normal_subspace["components"]); Vd = np.asarray(r_.b.dent_subspace["components"])
                ln = float(np.linalg.norm(delta - Vn.T @ (Vn @ delta))); ld = float(np.linalg.norm(delta - Vd.T @ (Vd @ delta)))
                e.update(leak_normal=ln, leak_dent=ld, leak_normal_over_sigma_f=ln / sig_f, leak_dent_over_sigma_f=ld / sig_f,
                         bound_sigma=(ln + ld) / sig)
                L.append("| %s | nearest subspace | %.3f | %.2f | %.2f | %.2f | %.2f | %+.2f (max %.2f) |" % (
                    enc, sig_f, out["delta_norm_over_feature_sigma"], ln / sig_f, ld / sig_f, e["bound_sigma"],
                    e["measured_shift_sigma"], e["measured_shift_max_sigma"]))
            elif isinstance(r_, C.NearestMean):
                e.update(bound_sigma=2 * float(np.linalg.norm(delta)) / sig)
                L.append("| %s | nearest mean | | | | | %.2f | %+.2f (max %.2f) |" % (
                    enc, e["bound_sigma"], e["measured_shift_sigma"], e["measured_shift_max_sigma"]))
            else:
                w = np.asarray(getattr(r_, "w", getattr(r_, "coef_", None)))
                L.append("| %s | logistic | | | | | exact w'delta | %+.2f (max %.2f) |" % (
                    enc, e["measured_shift_sigma"], e["measured_shift_max_sigma"]))
            out[r_.name] = e
        R["encoders"][enc] = out
        M.log(enc, json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}),
              json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out.items() if isinstance(v, dict)}))
    M.save_json("mv_bound_%s.json" % a.proto, R)
    with open(os.path.join(M.OUT, "mv_bound_%s.md" % a.proto), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ALL_DONE_M5", flush=True)


if __name__ == "__main__":
    main()
