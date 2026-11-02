"""Ablations of the local nominal subspace on the cached
features written by `r3_nominal_feat.py --save-feats`. No encoder is run here.

For each encoder (dentnet, pnpp) and every kept seed, from the nominal feature block
G (up to 20 windows: four templates, each at the seed and at four in-plane jitters, in
template-major order, five per template), three families of reconstruction-error scores:

  <enc>_k{0,1,2,3,4,6,9}      subspace dimension K over all available nominal windows
                              (K=0 is the distance to the nominal mean, K=3 is the paper's rule)
  <enc>_t{1,2,3,4}_k3         number of templates: the first n_t templates (five windows each,
                              jitters kept), K=3; and <enc>_t{n}_k3_norm, the leave-one-out
                              normalized form of the paper
  <enc>_nojit_k{0,1,2,3}      the un-jittered window of each template only (at most four
                              windows), K in {0,1,2,3}: the literal four-template subspace

Templates enter in the fixed order of `C.enumerate_templates`, never chosen by a score.
Output: <out-dir>/<cat>__<stem>__r3.6.npz with every key of the source item file plus the
new statistics, so `r3_object_agg.py --items-dir <out-dir> --stats ...` aggregates them.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, glob, argparse, time
import r3_common as C
import numpy as np
from multiprocessing import Pool

K_ALL = [0, 1, 2, 3, 4, 6, 9]
T_LIST = [1, 2, 3, 4]
K_NOJIT = [0, 1, 2, 3]
PER_TEMPLATE = 5
_A = {}


def residual(x, G, K):
    mu = G.mean(0)
    d = x - mu
    if K <= 0 or len(G) < 2:
        return float(np.linalg.norm(d))
    _, _, Vt = np.linalg.svd(G - mu, full_matrices=False)
    K = min(K, len(Vt))
    return float(np.linalg.norm(d - Vt[:K].T @ (Vt[:K] @ d)))


def loo_norm(x, G, K, rng):
    r = residual(x, G, K)
    loo = [residual(G[k], np.delete(G, k, axis=0), min(K, len(G) - 2))
           for k in rng.choice(len(G), min(8, len(G)), replace=False)]
    return r / max(float(np.median(loo)), 1e-6)


def one_item(fp):
    z = np.load(fp)
    cat, stem = str(z["cat"]), str(z["stem"])
    src = os.path.join(C.OUT, _A["items_dir"], "%s__%s__r3.6.npz" % (cat, stem))
    dst = os.path.join(C.OUT, _A["out_dir"], "%s__%s__r3.6.npz" % (cat, stem))
    if os.path.exists(dst) or not os.path.exists(src):
        return dst, "skip"
    s = np.load(src)
    keep, nnom = z["keep"], z["nnom"]
    m = len(s["valid"])
    out = {}
    rng = np.random.default_rng(0)
    for enc, f, G in (("dentnet", z["fd"], z["gd"]), ("pnpp", z["fp"], z["gp"])):
        cols = {}
        for K in K_ALL:
            cols["%s_k%d" % (enc, K)] = np.full(m, np.nan)
        for nt in T_LIST:
            cols["%s_t%d_k3" % (enc, nt)] = np.full(m, np.nan)
            cols["%s_t%d_k3_norm" % (enc, nt)] = np.full(m, np.nan)
        for K in K_NOJIT:
            cols["%s_nojit_k%d" % (enc, K)] = np.full(m, np.nan)
        for i, j in enumerate(keep):
            n = max(int(nnom[i]), 2)
            Gi = G[i][:n]
            for K in K_ALL:
                cols["%s_k%d" % (enc, K)][j] = residual(f[i], Gi, K)
            for nt in T_LIST:
                Gt = Gi[:min(nt * PER_TEMPLATE, n)]
                cols["%s_t%d_k3" % (enc, nt)][j] = residual(f[i], Gt, 3)
                cols["%s_t%d_k3_norm" % (enc, nt)][j] = loo_norm(f[i], Gt, 3, rng)
            Gn = Gi[0:n:PER_TEMPLATE]
            for K in K_NOJIT:
                cols["%s_nojit_k%d" % (enc, K)][j] = residual(f[i], Gn, K) if len(Gn) >= max(K + 1, 1) else np.nan
        out.update(cols)
    base = {k: s[k] for k in s.files}
    base.update(out)
    np.savez_compressed(dst, **base)
    return dst, "ok"


def _init(a):
    _A.update(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", default="items_nominal_feat_all")
    ap.add_argument("--out-dir", default="items_nominal_feat_ablate")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    _A["items_dir"], _A["out_dir"] = a.items_dir, a.out_dir
    os.makedirs(os.path.join(C.OUT, a.out_dir), exist_ok=True)
    files = sorted(glob.glob(os.path.join(C.OUT, a.items_dir + "_feats", "*.npz")))
    print("feature files:", len(files), flush=True)
    t0 = time.time()
    with Pool(a.procs, initializer=_init, initargs=(dict(_A),)) as pool:
        for k, (dst, st) in enumerate(pool.imap_unordered(one_item, files, chunksize=4)):
            if k % 50 == 0:
                print(k, os.path.basename(dst), st, "%.0fs" % (time.time() - t0), flush=True)
    stats = []
    for enc in ("dentnet", "pnpp"):
        stats += ["%s_k%d" % (enc, K) for K in K_ALL]
        stats += ["%s_t%d_k3" % (enc, n) for n in T_LIST] + ["%s_t%d_k3_norm" % (enc, n) for n in T_LIST]
        stats += ["%s_nojit_k%d" % (enc, K) for K in K_NOJIT]
    print("STATS " + " ".join(stats))
    print("ALL_DONE_R3_NOMINAL_ABLATE", flush=True)


if __name__ == "__main__":
    main()
