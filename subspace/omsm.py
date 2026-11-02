"""GDS variants and OMSM on the paper's five frozen encoders (supervisor's questions,
2026-09-23).

All rules work on unit-norm features x and uncentred class bases U_c (K x d, PCA without
centering). G = sum_c U_c^T U_c = Phi Lambda Phi^T over the sum space.

  gds_refit   the paper's rule: z = D x / ||D x|| with D the eigenvectors of G after the R
              largest are removed, class subspaces REFITTED by uncentred PCA on z (Q dims)
  gds_basis   the literal GDS operation: the original bases are projected, U_c D^T, and
              re-orthonormalized by SVD (rank kept), z as above, similarity ||W_c z||^2
  omsm        orthogonal mutual subspace method: O = Lambda^{-1/2} Phi^T over the whole sum
              space (optionally after removing the R largest), W_c = orth(U_c O^T),
              z = O x / ||O x||, similarity ||W_c z||^2. With a single input vector the
              canonical angle is the angle to the subspace, i.e. the projection energy.
  sm          the uncentred subspace method without any transform (control)

The margin is sim_dent - sim_normal. K, R (and Q) are selected on held-out synthetic
training samples by AUC, as in rv8. The DINOv2 encoder uses the whole-image token-pooled
window features of rv10, as in Table 2. Output: ~/accv_results_rev2/omsm.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import core as C
import os
import numpy as np

import gds as G8
from gds import SM, GDS, ubasis, unit, f1_at0


def orth(W):
    """Orthonormal rows spanning the row space of W (rank kept)."""
    U, s, Vt = np.linalg.svd(np.asarray(W, np.float64), full_matrices=False)
    return Vt[s > 1e-8 * s[0]]


def sumspace(Vd, Vn, R):
    W = np.vstack([Vd, Vn])
    _, s, vt = np.linalg.svd(W, full_matrices=False)
    keep = s > 1e-6
    return vt[keep][R:], (s[keep] ** 2)[R:]          # rows (eigvecs of G), eigenvalues


class GDSBasis(C.Rule):
    def __init__(self, K, R):
        self.K, self.R = K, R
        self.name = f"gdsbasis_k{K}_r{R}"

    def fit(self, X, y, bases=None):
        X = unit(X)
        Vd, Vn = bases if bases is not None else (ubasis(X[y == 1], self.K), ubasis(X[y == 0], self.K))
        self.D, _ = sumspace(Vd, Vn, self.R)
        self.Wd, self.Wn = orth(Vd @ self.D.T), orth(Vn @ self.D.T)

    def score(self, X):
        Z = unit(unit(X) @ self.D.T)
        return ((Z @ self.Wd.T) ** 2).sum(1) - ((Z @ self.Wn.T) ** 2).sum(1)


class OMSM(C.Rule):
    def __init__(self, K, R=0):
        self.K, self.R = K, R
        self.name = f"omsm_k{K}_r{R}"

    def fit(self, X, y, bases=None):
        X = unit(X)
        Vd, Vn = bases if bases is not None else (ubasis(X[y == 1], self.K), ubasis(X[y == 0], self.K))
        D, lam = sumspace(Vd, Vn, self.R)
        self.O = D / np.sqrt(lam)[:, None]                 # Lambda^{-1/2} Phi^T
        self.Wd, self.Wn = orth(Vd @ self.O.T), orth(Vn @ self.O.T)

    def score(self, X):
        Z = unit(unit(X) @ self.O.T)
        return ((Z @ self.Wd.T) ** 2).sum(1) - ((Z @ self.Wn.T) ** 2).sum(1)


def select(Xtr, ytr, Ks, log):
    n = len(ytr); cut = int(0.8 * n)
    Xa, ya, Xb, yb = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]
    Xau = unit(Xa)
    grid = {"gdsbasis": [], "omsm": []}
    for K in Ks:
        Vd, Vn = ubasis(Xau[ya == 1], K), ubasis(Xau[ya == 0], K)
        for R in sorted({0, 1, 2, 5, 10, K // 2, K} - {0} | {0}):
            if R >= 2 * K - 2:
                continue
            for cls, key in ((GDSBasis, "gdsbasis"), (OMSM, "omsm")):
                if key == "gdsbasis" and R == 0:
                    continue
                r = cls(K, R); r.fit(Xa, ya, bases=(Vd, Vn))
                s = r.score(Xb)
                grid[key].append({"K": K, "R": R, "auc": C.auc(s, yb), "f1": f1_at0(s, yb)})
        log(f"    held-out grid K={K} done")
    return {k: max(v, key=lambda e: e["auc"]) for k, v in grid.items()}, grid


def main():
    reps = G8.load_reps()
    scans = G8.scan_features("cuda")
    # DINOv2 (whole-image token pooling, as in Table 2)
    D = os.path.join(C.OUT, "dinov2")
    trm = dict(np.load(f"{D}/train_meta.npz")); vam = dict(np.load(f"{D}/val_meta.npz"))
    reps["dinov2_1024"] = {"train": dict(features=trm["features"], labels=trm["labels"].astype(np.int8)),
                           "val": dict(features=vam["features"], labels=vam["labels"].astype(np.int8), sample_idx=vam["sample_idx"])}
    dsc = np.load(f"{D}/scans.npz", allow_pickle=True)["scans"].item()
    for s in C.SAMPLES:
        scans[s]["feat"]["dinov2_1024"] = dsc[s]["features"]
    R = {}
    for name, rep in reps.items():
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        big = Xtr.shape[1] >= 256
        Ks = (10, 25, 50) if big else (5, 10, 25)
        C.log("encoder:", name, Xtr.shape)
        bsm, bgds, g8 = G8.select(Xtr, ytr, Ks, C.log)
        best, grid = select(Xtr, ytr, Ks, C.log)
        C.log(f"  selected sm {bsm} gds_refit {bgds} gds_basis {best['gdsbasis']} omsm {best['omsm']}")
        rules = [C.PBSS(50 if big else 25), SM(bsm["K"]), GDS(bgds["K"], bgds["R"], bgds["Q"]),
                 GDSBasis(best["gdsbasis"]["K"], best["gdsbasis"]["R"]), OMSM(best["omsm"]["K"], best["omsm"]["R"]),
                 OMSM(bgds["K"], 0), OMSM(bgds["K"], bgds["R"])]
        seen = set(); rules = [r for r in rules if not (r.name in seen or seen.add(r.name))]
        out = {"dim": int(Xtr.shape[1]), "selected": {"sm": bsm, "gds_refit": bgds, **best}, "heldout_grid": grid, "rules": {}}
        bgva = Xva[yva == 0]
        fA = scans["A"]["feat"][name]; bg = scans["A"]["bg40"]; inz = scans["A"]["inz20"]
        delta = fA[bg].mean(0) - bgva.mean(0)
        for r in rules:
            r.fit(Xtr, ytr)
            sva = r.score(Xva); sig = float(sva.std())
            e = {"auc_val": C.auc(sva, yva), "f1_val": f1_at0(sva, yva)}
            for s in C.SAMPLES:
                sc = r.score(scans[s]["feat"][name])
                e[f"flagged_windows_{s}"] = float((sc > 0).mean())
                if s == "A":
                    e["auc_vs_known_20"] = C.auc(sc, inz)
            s0 = r.score(bgva); s1 = r.score(bgva + delta)
            e["shift_of_margin_in_sigma"] = float((s1 - s0).mean() / sig)
            e["flagged_synthetic_bg_before"] = float((s0 > 0).mean()); e["flagged_synthetic_bg_after_shift"] = float((s1 > 0).mean())
            out["rules"][r.name] = e
            C.log(f"{name} {r.name}: auc {e['auc_val']:.3f} f1 {e['f1_val']:.3f} A flagged {100*e['flagged_windows_A']:.1f}% "
                  f"auc-zone {e['auc_vs_known_20']:.3f} shift {e['shift_of_margin_in_sigma']:+.2f} syn-bg {100*e['flagged_synthetic_bg_before']:.1f} -> {100*e['flagged_synthetic_bg_after_shift']:.1f}%")
        R[name] = out
        C.save_json("omsm.json", R)
    L = ["# GDS refit, GDS with projected bases, and OMSM on five frozen encoders (fuselage task)", "",
         "| encoder | rule | val AUC | val F1 | A flagged % | A AUC vs zone | margin shift (sigma) | syn bg before -> after % | B / C / D % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, out in R.items():
        for rn, e in out["rules"].items():
            L.append(f"| {name} | {rn} | {e['auc_val']:.3f} | {e['f1_val']:.3f} | {100*e['flagged_windows_A']:.1f} | {e['auc_vs_known_20']:.3f} | "
                     f"{e['shift_of_margin_in_sigma']:+.2f} | {100*e['flagged_synthetic_bg_before']:.1f} -> {100*e['flagged_synthetic_bg_after_shift']:.1f} | "
                     f"{100*e['flagged_windows_B']:.0f} / {100*e['flagged_windows_C']:.0f} / {100*e['flagged_windows_D']:.0f} |")
    C.write_md("omsm.md", L)
    print("ALL_DONE_RV13", flush=True)


if __name__ == "__main__":
    main()
