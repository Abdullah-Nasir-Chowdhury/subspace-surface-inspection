"""Scale sweep of the nominal-reference stage on Real3D-AD (2026-09-24, rev9).

The similarity factor 6.6 of the paper's nominal stage is derived from the median extent
of the annotated test defects (a dataset-level statistic, fixed before scoring, but still a
test-annotation statistic). This script reruns the nominal-feature stage of
r3_nominal_feat.py at a scale S that is fixed WITHOUT any annotation: S = 1 is the metric
pitch (the synthetic scanner's own 0.1875 mm per pixel on the object), and S in {2, 4, 8}
is the pre-declared power-of-two sweep around it. Everything else is the paper's: the
3.6 mm seed lattice of stage 1, the registration of stage 6, the 20 jittered nominal
windows per seed, the frozen DentNet and PointNet++ encoders, K = 3.

Per encoder and seed it writes the statistics the paper reports on all 597 items:
  <enc>_nom_rec3[_norm]     centred local nominal subspace, raw and normalized (the paper's)
  <enc>_gds_r0_rec3[_norm]  uncentred local subspace (R = 0)
  <enc>_gds_r1_rec3[_norm]  uncentred, projected onto the GDS of the category's local
                            subspaces with the pre-declared R = 1
  <enc>_cmean_rec3[_norm]   centred on the category-wide nominal mean
  <enc>_cat_rec50           category-level one-class subspace control
Features are cached in items_scale_s<S>_feats/, items in items_scale_s<S>/ (stage-4
format, the seeds and labels copied from items_nominal_feat_all), aggregated by
r3_dinov2_compare.py --items-dir items_scale_s<S> --tag r3_scale_s<S>. Resumable.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, glob, json, time, argparse
import r3_common as C
import numpy as np
from multiprocessing import Pool
import r3_learned as L
import r3_nominal_feat as NF
import r3_dinov2 as D2

ITEMS_REF = os.path.join(C.OUT, "items_nominal_feat_all")
R_SWEEP = [0, 1]


def _score_item(args):
    fp, cat_sub, gdsD = args
    z = np.load(fp)
    out = {}
    for enc, f, G in (("dentnet", z["fd"], z["gd"]), ("pnpp", z["fp"], z["gp"])):
        nnom = z["nnom"]
        nn, mean, rec3, norm = NF.local_scores(f, G, nnom)
        out["%s_nom_rec3" % enc] = rec3; out["%s_nom_rec3_norm" % enc] = norm
        mu, V = cat_sub[enc]
        d = f - mu; out["%s_cat_rec50" % enc] = np.linalg.norm(d - (d @ V.T) @ V, axis=1)
        rec, nrm = D2.local_uncentred(f - mu, G - mu, nnom, None)
        out["%s_cmean_rec3" % enc] = rec; out["%s_cmean_rec3_norm" % enc] = nrm
        for R in R_SWEEP:
            rec, nrm = D2.local_uncentred(f, G, nnom, gdsD[enc][R])
            out["%s_gds_r%d_rec3" % (enc, R)] = rec; out["%s_gds_r%d_rec3_norm" % (enc, R)] = nrm
    return str(z["cat"]), str(z["stem"]), z["keep"], out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--cats", nargs="*", default=C.CATS)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    tag = ("%g" % a.scale)
    fdir = os.path.join(C.OUT, "items_scale_s%s_feats" % tag)
    odir = os.path.join(C.OUT, "items_scale_s%s" % tag)
    os.makedirs(fdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    NF.SCALE = a.scale                                   # read by build_pair at call time
    NF._ARGS.update(reg=True, tol=1.0, pert_rot=0.0, pert_trans=0.0, pert_seed=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = NF.Encoders(device)
    rng = np.random.default_rng(0)
    skipped = []
    print("scale %g: pitch %.4f mm, window %.2f mm on the object -> %s" % (
        a.scale, L.PITCH_MM / a.scale, L.PATCH * L.PITCH_MM / a.scale, odir), flush=True)
    for cat in a.cats:
        t0 = time.time()
        items = [(c, s, l, fdir) for (c, s, l) in C.enumerate_test_items() if c == cat
                 and os.path.exists(os.path.join(ITEMS_REF, "%s__%s__r3.6.npz" % (c, s)))]
        todo = [it for it in items if not os.path.exists(os.path.join(fdir, "%s__%s.npz" % (it[0], it[1])))]
        # build_pair skips when <outdir>/<cat>__<stem>__r3.6.npz exists; fdir holds the
        # feature files under a different name, so nothing is skipped there by accident
        with Pool(a.procs, initializer=NF._init, initargs=(dict(NF._ARGS),)) as pool:
            for b in pool.imap(NF.build_pair, todo, chunksize=2):
                if b is None:
                    continue
                if "skipped" in b:
                    skipped.append((b["cat"], b["stem"], b["skipped"])); continue
                if len(b["keep"]) == 0:
                    skipped.append((b["cat"], b["stem"], "no-valid-window")); continue
                n = len(b["keep"])
                fd = enc.dentnet(b["R"])
                gd = enc.dentnet(b["RN"].reshape(-1, L.PATCH, L.PATCH)).reshape(n, NF.N_NOM, -1)
                fp = enc.pointnet(b["R"], b["Xf"], b["Yf"], rng)
                gp = enc.pointnet(b["RN"].reshape(-1, L.PATCH, L.PATCH), b["Xf"], b["Yf"], rng).reshape(n, NF.N_NOM, -1)
                np.savez_compressed(os.path.join(fdir, "%s__%s.npz" % (b["cat"], b["stem"])),
                                    cat=b["cat"], stem=b["stem"], label=b["label"], keep=b["keep"], nnom=b["nnom"],
                                    cov=b["cov"], n_nominal=b["n_nominal"], reg_fit=b["reg_fit"],
                                    template_fit=b["template_fit"], valid=b["valid"], fd=fd, gd=gd, fp=fp, gp=gp)
        fps = sorted(glob.glob(os.path.join(fdir, "%s__*.npz" % cat)))
        print("[%s] %d items encoded (%.0fs)" % (cat, len(fps), time.time() - t0), flush=True)
        if not fps:
            continue
        Gs = {"dentnet": [], "pnpp": []}; nnoms = []
        for fp in fps:
            z = np.load(fp)
            Gs["dentnet"].append(z["gd"]); Gs["pnpp"].append(z["gp"]); nnoms.append(z["nnom"])
        cat_sub, gdsD, info = {}, {}, {}
        for e_ in Gs:
            Gall = np.concatenate([G[i][:int(nn[i])] for G, nn in zip(Gs[e_], nnoms) for i in range(len(G))])
            mu = Gall.mean(0); _, _, Vt = np.linalg.svd(Gall - mu, full_matrices=False); cat_sub[e_] = (mu, Vt[:50])
            sums, eig = D2.gds_basis(Gs[e_], nnoms, 0)
            gdsD[e_] = {r: sums[r:] for r in R_SWEEP}
            info[e_] = {"n_nominal_windows": int(len(Gall)), "sumspace_dim": int(len(sums)),
                        "G_eigenvalues_head": [float(x) for x in eig[:10]]}
        del Gs
        with Pool(a.procs) as pool:
            for c, stem, keep, out in pool.imap(_score_item, [(fp, cat_sub, gdsD) for fp in fps]):
                z = np.load(os.path.join(fdir, "%s__%s.npz" % (c, stem)))
                src = dict(np.load(os.path.join(ITEMS_REF, "%s__%s__r3.6.npz" % (c, stem))))
                m = len(src["seeds"])
                for k in list(src):
                    if k.startswith(("dentnet_", "pnpp_")):
                        del src[k]
                src["scale"] = np.float64(a.scale); src["valid"] = z["valid"]
                cov = np.full(m, np.nan); cov[keep] = z["cov"]; src["cov_max"] = cov
                src["template_fit_mm"] = z["template_fit"]; src["registration_fit_mm"] = z["reg_fit"]
                for k, v in out.items():
                    arr = np.full(m, np.nan); arr[keep] = v; src[k] = arr
                np.savez_compressed(os.path.join(odir, "%s__%s__r%.1f.npz" % (c, stem, L.SEED_RADIUS)), **src)
        print("[%s] scored (%.0fs), G top eigenvalues dentnet %s" % (
            cat, time.time() - t0, np.round(info["dentnet"]["G_eigenvalues_head"][:4], 3)), flush=True)
        with open(os.path.join(C.OUT, "r3_scale_s%s_info.json" % tag), "a") as fh:
            fh.write(json.dumps({cat: info}) + "\n")
    with open(os.path.join(C.OUT, "items_scale_s%s_skipped.json" % tag), "w") as fh:
        json.dump(skipped, fh, indent=1)
    print("skipped %d" % len(skipped), skipped[:10], flush=True)
    print("ALL_DONE_R3_SCALE_SWEEP", flush=True)


if __name__ == "__main__":
    main()
