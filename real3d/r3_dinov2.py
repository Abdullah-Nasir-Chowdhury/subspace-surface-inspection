"""SubspaceAD and GDS on Real3D-AD (2026-09-22, for the discussion with the supervisor).

Two stages, both on the 597-item footing of items_nominal_feat_all (registration on,
--tol 1.0), with the identical seeds, windows and nominal sets of r3_nominal_feat.py.

encode   Rebuilds the test windows R and the nominal windows RN of every item with
         r3_nominal_feat.build_pair and encodes them with a frozen DINOv2 following the
         SubspaceAD implementation (facebook/dinov2-with-registers-large, hidden layers -12
         to -18 averaged, CLS and register tokens dropped). A 64x64 residual window (in the
         synthetic scale, factor 6.6) is clipped to [-2, 2] mm, mapped to [0, 1], replicated
         to three channels and resized to 224x224, and the 16x16 patch tokens are averaged
         into one 1024-D window feature. Only the first 13 transformer blocks are run, since
         hidden state -12 is the output of block 13. Cache: items_dinov2_feats/<cat>__<stem>.npz
         with fdino (n, 1024) and gdino (n, 20, 1024), asserted against the cached DentNet
         features' seed set.

score    For DINOv2 and (from the round-4 cache) DentNet features, per seed:
  <enc>_nom_rec3[_norm]      the paper's local nominal subspace, centred PCA, K=3
  <enc>_sad_ev99             SubspaceAD as published: one centred PCA on every nominal window
                             feature of the category (the templates are the normal images),
                             components to 99 percent explained variance, squared residual
  <enc>_cat_rec50            the paper's category-level CLAFIC control
  <enc>_gds_r<R>_rec3[_norm] GDS over the SET of local nominal subspaces. Each seed's nominal
                             windows give an uncentred K=3 subspace U_i. G = sum_i U_i^T U_i
                             over every seed of every item in the category (templates only,
                             nothing from a test scan). The GDS is the sum space minus the R
                             principal eigenvectors of G. Test and nominal features are
                             projected onto it and the uncentred local residual is taken
                             there. R = 0 is the uncentred control. Pre-declared default R = 1,
                             which removes the direction common to every location, i.e. it
                             centres on the mean of all local regions rather than on each
                             region's own mean. R in {0, 1, 2, 5, 10, 20} reported as a sweep.
         Output: items_dinov2/<cat>__<stem>__r3.6.npz in stage-4 format (the existing item
         file with the new statistics added), aggregated by r3_object_agg.py.
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

FEATS_REF = os.path.join(C.OUT, "items_nominal_feat_all_feats")
ITEMS_REF = os.path.join(C.OUT, "items_nominal_feat_all")
FDIR = os.path.join(C.OUT, "items_dinov2_feats")
ODIR = os.path.join(C.OUT, "items_dinov2_v2")
CKPT = "facebook/dinov2-with-registers-large"
LAYERS = [-12, -13, -14, -15, -16, -17, -18]
N_BLOCKS = 13
CLIP_MM = 2.0
RES = 224
K_LOCAL = 3
R_SWEEP = [0, 1, 2, 5, 10, 20]
R_DEFAULT = 1


# ------------------------------------------------------------------ encode
class Dino:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        m = AutoModel.from_pretrained(CKPT)
        m.encoder.layer = m.encoder.layer[:N_BLOCKS]          # hidden_states[-12] of 25 is block 13's output
        self.m = m.to(device).eval().half()
        self.drop = 1 + m.config.num_register_tokens
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half()
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half()
        self.idx = [25 + l for l in LAYERS]   # -12..-18 of the full 25 hidden states = 13..7, unchanged by truncation

    def __call__(self, R, batch=256):
        torch, F = self.torch, self.torch.nn.functional
        if len(R) == 0:
            return np.zeros((0, 1024), np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(R), batch):
                x = torch.from_numpy(np.clip(np.asarray(R[i:i + batch], np.float32), -CLIP_MM, CLIP_MM)).to(self.device).half()
                x = ((x + CLIP_MM) / (2 * CLIP_MM))[:, None].repeat(1, 3, 1, 1)
                x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
                x = (x - self.mean) / self.std
                hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
                f = torch.stack([hs[j][:, self.drop:] for j in self.idx]).float().mean(0).mean(1)
                out.append(f.cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def encode(a):
    os.makedirs(FDIR, exist_ok=True)
    NF._ARGS.update(reg=True, tol=1.0, pert_rot=0.0, pert_trans=0.0, pert_seed=0)
    dino = Dino()
    assert dino.idx == [13, 12, 11, 10, 9, 8, 7], dino.idx
    tmp = os.path.join(C.OUT, "items_dinov2_tmpbuild")
    for cat in a.cats:
        items = [(c, s, l, tmp) for (c, s, l) in C.enumerate_test_items() if c == cat
                 and not os.path.exists(os.path.join(FDIR, "%s__%s.npz" % (c, s)))
                 and os.path.exists(os.path.join(FEATS_REF, "%s__%s.npz" % (c, s)))]
        if not items:
            continue
        t0 = time.time()
        with Pool(a.procs, initializer=NF._init, initargs=(dict(NF._ARGS),)) as pool:
            for b in pool.imap(NF.build_pair, items, chunksize=2):
                if b is None or "skipped" in b or len(b["keep"]) == 0:
                    continue
                ref = np.load(os.path.join(FEATS_REF, "%s__%s.npz" % (b["cat"], b["stem"])))
                assert np.array_equal(ref["keep"], b["keep"]) and np.array_equal(ref["nnom"], b["nnom"]), b["stem"]
                n = len(b["keep"])
                fd = dino(b["R"])
                gd = dino(b["RN"].reshape(-1, L.PATCH, L.PATCH)).reshape(n, NF.N_NOM, -1)
                np.savez_compressed(os.path.join(FDIR, "%s__%s.npz" % (b["cat"], b["stem"])),
                                    cat=b["cat"], stem=b["stem"], label=b["label"], keep=b["keep"], nnom=b["nnom"],
                                    fdino=fd, gdino=gd)
        print("[%s] encoded %d items in %.0fs" % (cat, len(items), time.time() - t0), flush=True)
    print("ALL_DONE_R3_DINOV2_ENCODE", flush=True)


# ------------------------------------------------------------------ score
def ubasis(X, K):
    _, _, Vt = np.linalg.svd(np.asarray(X, np.float64), full_matrices=False)
    return Vt[:min(K, len(Vt))]


def uresid(x, G, K):
    V = ubasis(G, K)
    return float(np.linalg.norm(x - V.T @ (V @ x)))


def local_uncentred(f, G, nnom, D=None):
    """Uncentred local residual, optionally after projection onto the rows of D (the GDS)."""
    rec = np.empty(len(f)); norm = np.empty(len(f))
    rng = np.random.default_rng(0)
    for i in range(len(f)):
        Gi = G[i][:max(int(nnom[i]), 2)]; x = f[i]
        if D is not None:
            Gi = Gi @ D.T; x = D @ x
        rec[i] = uresid(x, Gi, K_LOCAL)
        loo = [uresid(Gi[k], np.delete(Gi, k, axis=0), min(K_LOCAL, len(Gi) - 1))
               for k in rng.choice(len(Gi), min(8, len(Gi)), replace=False)]
        norm[i] = rec[i] / max(float(np.median(loo)), 1e-6)
    return rec, norm


def sad_pca(Gall, ev=0.99):
    mu = Gall.mean(0)
    _, s, Vt = np.linalg.svd(Gall - mu, full_matrices=False)
    var = s ** 2; cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, ev)) + 1
    return mu, Vt[:k]


def gds_basis(Gs, nnoms, R):
    """GDS of the set of uncentred K_LOCAL subspaces, one per seed of every item."""
    d = Gs[0].shape[-1]
    Gsum = np.zeros((d, d))
    n = 0
    for G, nnom in zip(Gs, nnoms):
        for i in range(len(G)):
            V = ubasis(G[i][:max(int(nnom[i]), 2)], K_LOCAL)
            Gsum += V.T @ V; n += 1
    w, v = np.linalg.eigh(Gsum)
    w, v = w[::-1], v[:, ::-1]
    keep = w > 1e-8 * w[0]
    sums = v[:, keep].T
    return sums[R:], w[keep] / n


def _score_item(args):
    fp, gsub, cat_sub, gdsD = args
    z = np.load(fp)
    ref = np.load(os.path.join(FEATS_REF, os.path.basename(fp)))
    out = {}
    for enc, f, G in (("dino", z["fdino"], z["gdino"]), ("dentnet", ref["fd"], ref["gd"]), ("pnpp", ref["fp"], ref["gp"])):
        nnom = z["nnom"]
        nn, mean, rec3, norm = NF.local_scores(f, G, nnom)
        out["%s_nom_rec3" % enc] = rec3; out["%s_nom_rec3_norm" % enc] = norm
        mu, V = gsub[enc]
        d = f - mu; out["%s_sad_ev99" % enc] = ((d - (d @ V.T) @ V) ** 2).sum(1)
        mu, V = cat_sub[enc]
        d = f - mu; out["%s_cat_rec50" % enc] = np.linalg.norm(d - (d @ V.T) @ V, axis=1)
        # centering on the mean of multiple local regions (the category-wide nominal mean)
        # instead of each region's own mean, then the K=3 local subspace
        rec, nrm = local_uncentred(f - mu, G - mu, nnom, None)
        out["%s_cmean_rec3" % enc] = rec; out["%s_cmean_rec3_norm" % enc] = nrm
        for R in R_SWEEP:
            rec, norm = local_uncentred(f, G, nnom, gdsD[enc][R])
            out["%s_gds_r%d_rec3" % (enc, R)] = rec; out["%s_gds_r%d_rec3_norm" % (enc, R)] = norm
    return str(z["cat"]), str(z["stem"]), z["keep"], out


def score(a):
    os.makedirs(ODIR, exist_ok=True)
    R = {}
    for cat in a.cats:
        fps = sorted(glob.glob(os.path.join(FDIR, "%s__*.npz" % cat)))
        if not fps:
            continue
        t0 = time.time()
        Gs = {"dino": [], "dentnet": [], "pnpp": []}; nnoms = []
        for fp in fps:
            z = np.load(fp); ref = np.load(os.path.join(FEATS_REF, os.path.basename(fp)))
            Gs["dino"].append(z["gdino"]); Gs["dentnet"].append(ref["gd"]); Gs["pnpp"].append(ref["gp"]); nnoms.append(z["nnom"])
        gsub, cat_sub, gdsD, info = {}, {}, {}, {}
        for enc in Gs:
            Gall = np.concatenate([G[i][:int(nn[i])] for G, nn in zip(Gs[enc], nnoms) for i in range(len(G))])
            gsub[enc] = sad_pca(Gall, 0.99)
            mu = Gall.mean(0); _, _, Vt = np.linalg.svd(Gall - mu, full_matrices=False); cat_sub[enc] = (mu, Vt[:50])
            sums, eig = gds_basis(Gs[enc], nnoms, 0)
            gdsD[enc] = {r: sums[r:] for r in R_SWEEP}
            info[enc] = {"n_nominal_windows": int(len(Gall)), "sad_components": int(len(gsub[enc][1])),
                         "sumspace_dim": int(len(sums)), "G_eigenvalues_head": [float(x) for x in eig[:25]]}
        print("[%s] %d items, SAD comps dino %d dentnet %d pnpp %d, G top eigenvalues dino %s" % (
            cat, len(fps), info["dino"]["sad_components"], info["dentnet"]["sad_components"], info["pnpp"]["sad_components"],
            np.round(info["dino"]["G_eigenvalues_head"][:6], 3)), flush=True)
        with Pool(a.procs) as pool:
            for c, stem, keep, out in pool.imap(_score_item, [(fp, gsub, cat_sub, gdsD) for fp in fps]):
                src = dict(np.load(os.path.join(ITEMS_REF, "%s__%s__r%.1f.npz" % (c, stem, L.SEED_RADIUS))))
                m = len(src["seeds"])
                for k, v in out.items():
                    arr = np.full(m, np.nan); arr[keep] = v; src[k] = arr
                np.savez_compressed(os.path.join(ODIR, "%s__%s__r%.1f.npz" % (c, stem, L.SEED_RADIUS)), **src)
        R[cat] = info
        print("[%s] scored in %.0fs" % (cat, time.time() - t0), flush=True)
    with open(os.path.join(C.OUT, "r3_dinov2_info.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    print("ALL_DONE_R3_DINOV2_SCORE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "score"])
    ap.add_argument("--cats", nargs="*", default=C.CATS)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    encode(a) if a.stage == "encode" else score(a)
