"""Frozen encoders and their feature read-out, plus the margin-shift decomposition.

Representations, all frozen, all on the same 170,000 training and 170,000 validation
windows and the same four real scans from the stored residuals:

  dentnet_1024      DentNet bottleneck, GAP (the paper's PBSS features)
  pnpp_r0.3_128     fuselage PointNet++ at the selected grouping radius
  pnpp_r0.6_128     the same architecture at radius 0.6 (a different trained encoder)
  pnpp_r2.0_128     the same at the inherited radius 2.0 (the first stage collapsed)

Per representation the three rules are fitted on the training windows and reported in
domain (AUC, F1), on scan A at the a-priori threshold (window flagged fraction), and as a
ranking against the known dent (window AUC vs the 20 mm zone), so the calibration effect
and the ranking are separated for every encoder.

The mechanism: let delta be the mean feature shift from synthetic validation background
windows to scan A's background windows (outside the known 40 mm zone). Each rule's margin
is evaluated on the synthetic background features before and after adding delta, in units
of the rule's synthetic standard deviation; the flagged fraction that shift alone induces
is reported. delta is also decomposed into the fraction of its energy inside the normal
subspace, inside the dent subspace, and along the logistic weight direction.

Output: ~/accv_results_rev2/shift_analysis.{json,md}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import core as C
import os, json
import numpy as np

PNPP_RUNS = {"pnpp_r0.3_128": ("pointnetpp_fuselage_r0.3", 0.3),
             "pnpp_r0.6_128": ("pointnetpp_fuselage_r0.6", 0.6),
             "pnpp_r2.0_128": ("pointnetpp_fuselage", 2.0)}


def pnpp_features(tag, radius, device):
    import torch
    from pnpp_model import PatchClassifier
    ck = os.path.expanduser(f"~/accv_results_real/{tag}/best.pth")
    model = PatchClassifier(norm_radius=radius).to(device)
    model.load_state_dict(torch.load(ck, map_location=device)); model.eval()
    out = {}
    for split in ("train", "val"):
        cache = os.path.join(C.OUT, f"feats_{tag}_{split}.npz")
        if os.path.exists(cache):
            out[split] = dict(np.load(cache)); continue
        pts = np.load(f"{C.PNPP_DATA}/{split}_points.npy", mmap_mode="r")
        lab = np.load(f"{C.PNPP_DATA}/{split}_labels.npy").astype(np.int8)
        smp = np.load(f"{C.PNPP_DATA}/{split}_sample.npy")
        feats = []
        for i0 in range(0, len(lab), 8192):
            _, f = C.pnpp_forward(model, pts[i0:i0 + 8192], device, batch=1024)
            feats.append(f.astype(np.float32))
        out[split] = {"features": np.concatenate(feats), "labels": lab, "sample_idx": smp}
        np.savez_compressed(cache, **out[split])
        C.log(f"  {tag} {split} features cached")
    return model, out


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = {}
    tr = np.load(f"{C.PBSS_RES}/features_train.npz"); va = np.load(f"{C.PBSS_RES}/features_val.npz")
    reps["dentnet_1024"] = {"train": dict(features=tr["features"], labels=tr["labels"].astype(np.int8)),
                            "val": dict(features=va["features"], labels=va["labels"].astype(np.int8), sample_idx=va["sample_idx"])}
    models = {}
    for name, (tag, rad) in PNPP_RUNS.items():
        C.log("features:", name)
        models[name], reps[name] = pnpp_features(tag, rad, device)
    extractor = C.dentnet_extractor(device)
    # real scans: window features per representation
    scans = {}
    rngs = {name: np.random.default_rng(C.SEED) for name in PNPP_RUNS}
    for s in C.SAMPLES:
        h, w, Xg, Yg, _ = C.load_scan(s)
        m = C.stored_masks(s); res = m["residuals"].astype(np.float32)
        pe, pr, vpos, cx, cy = C.window_geometry(h, w, Xg, Yg)
        rec = {"n": len(vpos), "feat": {}}
        rec["feat"]["dentnet_1024"] = np.asarray(extractor.extract_features_batch(C.window_patches(res, pe), batch_size=256), np.float32)
        for name in PNPP_RUNS:
            arr = C.pnpp_pointsets(Xg, Yg, res, vpos, rngs[name])
            _, f = C.pnpp_forward(models[name], arr, device)
            rec["feat"][name] = f.astype(np.float32)
        if s == "A":
            rec["bg40"] = C.outside(cx, cy, C.KNOWN_A, C.R_ZONE_MM)
            rec["inz20"] = ~C.outside(cx, cy, C.KNOWN_A, C.R_ZONE_RES_MM)
        scans[s] = rec
    R = {}
    for name, rep in reps.items():
        Xtr, ytr = rep["train"]["features"], rep["train"]["labels"]
        Xva, yva = rep["val"]["features"], rep["val"]["labels"]
        rules = [C.PBSS(50 if Xtr.shape[1] >= 256 else 25), C.NearestMean(), C.Logistic()]
        out = {"dim": int(Xtr.shape[1]), "rules": {}}
        bgva = Xva[yva == 0]
        # feature shift from synthetic background to scan A background
        fA = scans["A"]["feat"][name]; bg = scans["A"]["bg40"]
        delta = fA[bg].mean(0) - bgva.mean(0)
        out["delta_norm"] = float(np.linalg.norm(delta))
        out["delta_norm_over_bg_feature_std"] = float(np.linalg.norm(delta) / np.sqrt(bgva.var(0).sum()))
        for r in rules:
            r.fit(Xtr, ytr)
            sva = r.score(Xva); sig = float(sva.std())
            e = {"auc_val": C.auc(sva, yva), "f1_val": C.weighted_prf(sva > 0, yva, np.ones(len(yva)))[2], "sigma_val": sig}
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
            if isinstance(r, C.PBSS):
                Vn = np.asarray(r.b.normal_subspace["components"]) if "components" in r.b.normal_subspace else None
                Vd = np.asarray(r.b.dent_subspace["components"]) if "components" in r.b.dent_subspace else None
                if Vn is not None:
                    e["delta_energy_in_normal_subspace"] = float(np.sum((Vn @ delta) ** 2) / np.sum(delta ** 2))
                    e["delta_energy_in_dent_subspace"] = float(np.sum((Vd @ delta) ** 2) / np.sum(delta ** 2))
            if isinstance(r, C.Logistic):
                wdir = r.lr.coef_.ravel() / np.linalg.norm(r.lr.coef_)
                dz = delta / r.sc.scale_
                e["delta_cos_with_logistic_weight"] = float(abs(wdir @ dz) / np.linalg.norm(dz))
            out["rules"][r.name] = e
            C.log(f"{name} {r.name}: auc {e['auc_val']:.3f} A flagged {100*e['flagged_windows_A']:.1f}% "
                  f"auc-zone {e['auc_vs_known_20']:.3f} shift {e['shift_of_margin_in_sigma']:+.2f} sigma "
                  f"syn-bg flagged {100*e['flagged_synthetic_bg_before']:.1f} -> {100*e['flagged_synthetic_bg_after_shift']:.1f}%")
        R[name] = out
    C.save_json("shift_analysis.json", R)
    L = ["# Four frozen encoders, three rules, and the feature shift that moves the threshold", "",
         "| encoder (dim) | rule | val AUC | val F1 | A flagged windows % | A AUC vs 20 mm zone | margin shift (sigma_val) | synthetic bg flagged before -> after shift % | B / C / D flagged % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, out in R.items():
        for rn, e in out["rules"].items():
            L.append(f"| {name} ({out['dim']}) | {rn} | {e['auc_val']:.3f} | {e['f1_val']:.3f} | {100*e['flagged_windows_A']:.1f} | "
                     f"{e['auc_vs_known_20']:.3f} | {e['shift_of_margin_in_sigma']:+.2f} | {100*e['flagged_synthetic_bg_before']:.1f} -> {100*e['flagged_synthetic_bg_after_shift']:.1f} | "
                     f"{100*e['flagged_windows_B']:.0f} / {100*e['flagged_windows_C']:.0f} / {100*e['flagged_windows_D']:.0f} |")
    L += ["", "Shift decomposition (scan A background minus synthetic background mean):", "",
          "| encoder | |delta| / bg feature std | energy in normal subspace | energy in dent subspace | cos(delta, logistic weight) |", "|---|---|---|---|---|"]
    for name, out in R.items():
        pb = [e for k, e in out["rules"].items() if k.startswith("pbss")][0]
        lg = out["rules"]["logistic"]
        L.append(f"| {name} | {out['delta_norm_over_bg_feature_std']:.2f} | {pb.get('delta_energy_in_normal_subspace', float('nan')):.2f} | "
                 f"{pb.get('delta_energy_in_dent_subspace', float('nan')):.2f} | {lg['delta_cos_with_logistic_weight']:.3f} |")
    C.write_md("shift_analysis.md", L)
    print("ALL_DONE_RV7", flush=True)


if __name__ == "__main__":
    main()
