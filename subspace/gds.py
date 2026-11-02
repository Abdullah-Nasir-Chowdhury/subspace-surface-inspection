"""Generalized difference subspace (GDS) projection on frozen features (rev5).

The conventional rule of the paper (PBSS) builds centred class subspaces on frozen
features. Here the class subspaces are built by PCA WITHOUT centring (the eigenvectors of
the class autocorrelation matrix), as the GDS requires, and three rules are compared on
the same four frozen encoders, windows, labels and real scans as rv7:

  sm_k{K}            subspace method, uncentred, similarity ||V_c x||^2 / ||x||^2,
                     margin = sim_dent - sim_normal. The control for "no centring".
  gds_k{K}_r{R}_q{Q} the same after projecting every feature onto the GDS: the sum
                     space of the class subspaces minus the R principal eigenvectors of
                     G = sum_c P_c. Class subspaces are rebuilt (uncentred, Q dims) in
                     the GDS and the same similarity margin is used.
  pbss_k{K}          the paper's centred rule, re-run as a sanity anchor against rv7.

(K, R, Q) are selected on held-out TRAINING samples (the last 20 percent of the training
windows, which are whole samples because windows are stored sample by sample), never on
the validation split. The reported columns are those of the paper's encoder table, plus
the fraction of the measured scan-A mean shift's energy that falls in the removed
principal space, in the GDS, and outside the sum space.

Output: ~/accv_results_rev2/gds.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import core as C
import os, json
import numpy as np

import shift_analysis as RV7


def ubasis(X, K):
    """Top-K eigenvectors of the autocorrelation matrix, rows orthonormal (K x D)."""
    X = np.asarray(X, np.float64)
    A = X.T @ X / len(X)
    w, v = np.linalg.eigh(A)
    return v[:, ::-1][:, :K].T.copy()


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class SM(C.Rule):
    """Uncentred subspace method on unit-norm features."""

    def __init__(self, K):
        self.K = K
        self.name = f"sm_k{K}"

    def fit(self, X, y):
        X = unit(X)
        self.Vd = ubasis(X[y == 1], self.K)
        self.Vn = ubasis(X[y == 0], self.K)

    def score(self, X):
        X = unit(X)
        return ((X @ self.Vd.T) ** 2).sum(1) - ((X @ self.Vn.T) ** 2).sum(1)


class GDS(C.Rule):
    def __init__(self, K, R, Q):
        self.K, self.R, self.Q = K, R, Q
        self.name = f"gds_k{K}_r{R}_q{Q}"

    def fit(self, X, y, bases=None):
        X = unit(X)
        Vd, Vn = bases if bases is not None else (ubasis(X[y == 1], self.K), ubasis(X[y == 0], self.K))
        W = np.vstack([Vd, Vn])                      # G = W^T W = P_d + P_n
        _, s, vt = np.linalg.svd(W, full_matrices=False)
        keep = s > 1e-6
        self.eig = (s[keep] ** 2)
        self.sumspace = vt[keep]                     # rows, descending eigenvalue of G
        self.principal = self.sumspace[:self.R]
        self.D = self.sumspace[self.R:]              # the GDS, rows orthonormal
        Z = unit(X @ self.D.T)
        self.Vd = ubasis(Z[y == 1], self.Q)
        self.Vn = ubasis(Z[y == 0], self.Q)

    def score(self, X):
        Z = unit(unit(X) @ self.D.T)
        return ((Z @ self.Vd.T) ** 2).sum(1) - ((Z @ self.Vn.T) ** 2).sum(1)


def f1_at0(s, y):
    return C.weighted_prf(s > 0, y, np.ones(len(y)))[2]


def select(Xtr, ytr, Ks, log):
    """Pick K for SM and (K, R, Q) for GDS on held-out training samples, by AUC."""
    n = len(ytr); cut = int(0.8 * n)
    Xa, ya, Xb, yb = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]
    Xau = unit(Xa)
    grid = {"sm": [], "gds": []}
    for K in Ks:
        Vd, Vn = ubasis(Xau[ya == 1], K), ubasis(Xau[ya == 0], K)
        sm = SM(K); sm.Vd, sm.Vn = Vd, Vn
        s = sm.score(Xb)
        grid["sm"].append({"K": K, "auc": C.auc(s, yb), "f1": f1_at0(s, yb)})
        for R in sorted({1, 2, 5, 10, K // 2, K, K + K // 2} - {0}):
            if R >= 2 * K - 2:
                continue
            for Q in (1, 3, 5, 10, 25):
                if Q >= 2 * K - R:
                    continue
                g = GDS(K, R, Q); g.fit(Xa, ya, bases=(Vd, Vn))
                s = g.score(Xb)
                grid["gds"].append({"K": K, "R": R, "Q": Q, "auc": C.auc(s, yb), "f1": f1_at0(s, yb)})
        log(f"    held-out grid K={K} done")
    best_sm = max(grid["sm"], key=lambda e: e["auc"])
    best_gds = max(grid["gds"], key=lambda e: e["auc"])
    return best_sm, best_gds, grid


def scan_features(device):
    cache = os.path.join(C.OUT, "rv8_scan_feats.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["scans"].item()
    import torch
    models = {}
    for name, (tag, rad) in RV7.PNPP_RUNS.items():
        from pnpp_model import PatchClassifier
        m = PatchClassifier(norm_radius=rad).to(device)
        m.load_state_dict(torch.load(os.path.expanduser(f"~/accv_results_real/{tag}/best.pth"), map_location=device)); m.eval()
        models[name] = m
    extractor = C.dentnet_extractor(device)
    scans = {}
    rngs = {name: np.random.default_rng(C.SEED) for name in RV7.PNPP_RUNS}
    for s in C.SAMPLES:
        h, w, Xg, Yg, _ = C.load_scan(s)
        m = C.stored_masks(s); res = m["residuals"].astype(np.float32)
        pe, pr, vpos, cx, cy = C.window_geometry(h, w, Xg, Yg)
        rec = {"feat": {}}
        rec["feat"]["dentnet_1024"] = np.asarray(extractor.extract_features_batch(C.window_patches(res, pe), batch_size=256), np.float32)
        for name in RV7.PNPP_RUNS:
            arr = C.pnpp_pointsets(Xg, Yg, res, vpos, rngs[name])
            _, f = C.pnpp_forward(models[name], arr, device)
            rec["feat"][name] = f.astype(np.float32)
        if s == "A":
            rec["bg40"] = C.outside(cx, cy, C.KNOWN_A, C.R_ZONE_MM)
            rec["inz20"] = ~C.outside(cx, cy, C.KNOWN_A, C.R_ZONE_RES_MM)
        scans[s] = rec
    np.savez_compressed(cache, scans=np.array(scans, dtype=object))
    return scans


def load_reps():
    reps = {}
    tr = np.load(f"{C.PBSS_RES}/features_train.npz"); va = np.load(f"{C.PBSS_RES}/features_val.npz")
    reps["dentnet_1024"] = {"train": dict(features=tr["features"], labels=tr["labels"].astype(np.int8)),
                            "val": dict(features=va["features"], labels=va["labels"].astype(np.int8), sample_idx=va["sample_idx"])}
    for name, (tag, rad) in RV7.PNPP_RUNS.items():
        reps[name] = {}
        for split in ("train", "val"):
            reps[name][split] = dict(np.load(os.path.join(C.OUT, f"feats_{tag}_{split}.npz")))
    return reps


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = load_reps()
    scans = scan_features(device)
    R = {}
    for name, rep in reps.items():
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        grp = rep["val"]["sample_idx"]
        big = Xtr.shape[1] >= 256
        Ks = (10, 25, 50) if big else (5, 10, 25)
        C.log("encoder:", name, Xtr.shape)
        best_sm, best_gds, grid = select(Xtr, ytr, Ks, C.log)
        C.log(f"  selected sm {best_sm}  gds {best_gds}")
        Kp = 50 if big else 25
        rules = [C.PBSS(Kp), SM(best_sm["K"]), SM(Kp),
                 GDS(best_gds["K"], best_gds["R"], best_gds["Q"])]
        seen = set(); rules = [r for r in rules if not (r.name in seen or seen.add(r.name))]
        out = {"dim": int(Xtr.shape[1]), "selected_sm": best_sm, "selected_gds": best_gds,
               "heldout_grid": grid, "rules": {}}
        bgva = Xva[yva == 0]
        fA = scans["A"]["feat"][name]; bg = scans["A"]["bg40"]
        delta = fA[bg].mean(0) - bgva.mean(0)
        # the same shift on the unit sphere, where the uncentred rules live
        delta_u = unit(fA[bg]).mean(0) - unit(bgva).mean(0)
        for r in rules:
            r.fit(Xtr, ytr)
            sva = r.score(Xva); sig = float(sva.std())
            m = C.patch_metrics_ci(sva, yva, grp, thr=0.0, n_boot=1000)
            e = {"auc_val": m["auc"], "auc_val_ci": m["auc_ci"], "f1_val": m["f1"], "f1_val_ci": m["f1_ci"], "sigma_val": sig}
            for s in C.SAMPLES:
                sc = r.score(scans[s]["feat"][name])
                e[f"flagged_windows_{s}"] = float((sc > 0).mean())
                if s == "A":
                    e["auc_vs_known_20"] = C.auc(sc, scans[s]["inz20"])
                    e["flagged_bg40_A"] = float((sc[bg] > 0).mean())
            s0 = r.score(bgva); s1 = r.score(bgva + delta)
            e["shift_of_margin_in_sigma"] = float((s1 - s0).mean() / sig)
            e["flagged_synthetic_bg_before"] = float((s0 > 0).mean())
            e["flagged_synthetic_bg_after_shift"] = float((s1 > 0).mean())
            if isinstance(r, GDS):
                tot = float(np.sum(delta_u ** 2))
                e["gds_dim"] = int(len(r.D)); e["sumspace_dim"] = int(len(r.sumspace))
                e["G_eigenvalues_head"] = [float(x) for x in r.eig[:12]]
                e["delta_u_energy_in_removed_principal"] = float(np.sum((r.principal @ delta_u) ** 2) / tot)
                e["delta_u_energy_in_gds"] = float(np.sum((r.D @ delta_u) ** 2) / tot)
                e["delta_u_energy_outside_sumspace"] = float(1 - np.sum((r.sumspace @ delta_u) ** 2) / tot)
            out["rules"][r.name] = e
            C.log(f"{name} {r.name}: auc {e['auc_val']:.3f} f1 {e['f1_val']:.3f} A flagged {100*e['flagged_windows_A']:.1f}% "
                  f"auc-zone {e['auc_vs_known_20']:.3f} shift {e['shift_of_margin_in_sigma']:+.2f} sigma "
                  f"syn-bg {100*e['flagged_synthetic_bg_before']:.1f} -> {100*e['flagged_synthetic_bg_after_shift']:.1f}%")
        R[name] = out
    C.save_json("gds.json", R)
    L = ["# GDS projection on four frozen encoders (uncentred class subspaces)", "",
         "| encoder (dim) | rule | val AUC [CI] | val F1 [CI] | A flagged windows % | A AUC vs 20 mm zone | margin shift (sigma_val) | synthetic bg flagged before -> after % | B / C / D flagged % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, out in R.items():
        for rn, e in out["rules"].items():
            L.append(f"| {name} ({out['dim']}) | {rn} | {C.fmt_ci(e['auc_val'], e['auc_val_ci'])} | {C.fmt_ci(e['f1_val'], e['f1_val_ci'])} | {100*e['flagged_windows_A']:.1f} | "
                     f"{e['auc_vs_known_20']:.3f} | {e['shift_of_margin_in_sigma']:+.2f} | {100*e['flagged_synthetic_bg_before']:.1f} -> {100*e['flagged_synthetic_bg_after_shift']:.1f} | "
                     f"{100*e['flagged_windows_B']:.0f} / {100*e['flagged_windows_C']:.0f} / {100*e['flagged_windows_D']:.0f} |")
    L += ["", "Where the measured scan-A mean shift (on the unit sphere) lies relative to the GDS:", "",
          "| encoder | rule | sum-space dim | GDS dim | energy in removed principal space | energy in GDS | energy outside sum space |", "|---|---|---|---|---|---|---|"]
    for name, out in R.items():
        for rn, e in out["rules"].items():
            if "gds_dim" in e:
                L.append(f"| {name} | {rn} | {e['sumspace_dim']} | {e['gds_dim']} | {e['delta_u_energy_in_removed_principal']:.3f} | "
                         f"{e['delta_u_energy_in_gds']:.3f} | {e['delta_u_energy_outside_sumspace']:.3f} |")
    C.write_md("gds.md", L)
    print("ALL_DONE_RV8", flush=True)


if __name__ == "__main__":
    main()
