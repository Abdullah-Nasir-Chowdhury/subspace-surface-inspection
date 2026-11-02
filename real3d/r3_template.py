"""Stage 3. Nominal reference at the SAME location: the paper's reference-based variant done
with a registered template instead of boundary patches.

Real3D-AD ships four good training templates per category. In five categories the test
scans are already registered to a template (median nearest-neighbour distance 0.03 to
0.05 mm, checked in r3_diag / the alignment probe), so a nominal patch is available at every
seed without any fitting. For those items, at every seed of the r3_score.py lattice (same
voxel seeds, same tangent frame, same canonical grid) the template surface is resampled in
the identical frame and compared with the test patch:

  magd_tmpl    Mag(D) between the test patch's shape subspace and the template patch's
               (direction-only by Proposition 1; the canonical-angle comparison the paper
               uses, now against a nominal patch rather than a neighbour)
  diff_rms     RMS of (h_test - h_template) on the grid: the amplitude-aware baseline that
               Proposition 1 says the subspace comparison discards
  diff_absmax  max |h_test - h_template| on the grid
  nn_p95       95th percentile of point-to-template nearest-neighbour distance over the
               window's raw points: the plain registration baseline, no grid at all

Template choice per item: the training template with the smallest median NN distance on a
4000-point subsample of the test scan. Items whose best template is farther than 0.3 mm are
skipped and listed, never scored. Nothing touches a test label.

Writes ~/accv_results/real3d/items_tmpl/<cat>__<stem>__r<radius>.npz with the same keys the
aggregator needs (valid, pt_seed, pt_label, seed_defect_dist, ...), so r3_aggregate.py
--items-dir items_tmpl --stats magd_tmpl diff_rms diff_absmax nn_p95 evaluates it.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, time, argparse
import r3_common as C
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool

ALIGN_TOL_MM = 0.3
_TREES = {}


def template_trees(cat):
    if cat not in _TREES:
        d = {}
        for t in C.enumerate_templates(cat):
            T = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "train", t + ".pcd"))
            d[t] = (T, cKDTree(T))
        _TREES[cat] = d
    return _TREES[cat]


def score_item(args):
    cat, stem, lab, radius = args
    src = os.path.join(C.OUT, "items", "%s__%s__r%.1f.npz" % (cat, stem, radius))
    dst = os.path.join(C.OUT, "items_tmpl", "%s__%s__r%.1f.npz" % (cat, stem, radius))
    if os.path.exists(dst):
        return dst, "skip"
    if not os.path.exists(src):
        return dst, "no-stage1-file"
    t0 = time.time()
    z = np.load(src)
    P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "test", stem + ".pcd"))
    trees = template_trees(cat)
    sub = P[:: max(1, len(P) // 4000)]
    fit = {t: float(np.median(tr.query(sub)[0])) for t, (T, tr) in trees.items()}
    tbest = min(fit, key=fit.get)
    if fit[tbest] > ALIGN_TOL_MM:
        return dst, "unaligned %.2fmm" % fit[tbest]
    T, ttree = trees[tbest]

    grid = C.Grid()
    seeds = z["seeds"]
    valid = z["valid"].copy()
    tree = cKDTree(P)
    obj_c = P.mean(0)
    XY = np.c_[grid.X * radius, grid.Y * radius]
    m = len(seeds)
    f = {k: np.full(m, np.nan) for k in ("magd_tmpl", "diff_rms", "diff_absmax", "nn_p95", "cov_max")}
    for j in np.flatnonzero(valid):
        c = seeds[j]
        idx = tree.query_ball_point(c, radius * 1.6)
        tidx = ttree.query_ball_point(c, radius * 1.6)
        if len(idx) < C.MIN_BALL_PTS or len(tidx) < C.MIN_BALL_PTS:
            valid[j] = False
            continue
        nb = P[idx]
        F = C.local_frame(nb, c, obj_c)                  # frame from the TEST neighbourhood
        h, cov = C.resample(nb, c, F, grid, radius, return_cov=True)
        ht, covt = C.resample(T[tidx], c, F, grid, radius, return_cov=True)   # identical frame
        f["cov_max"][j] = max(cov, covt)
        U, _ = C.shape_subspace(np.c_[XY, h])
        Ut, _ = C.shape_subspace(np.c_[XY, ht])
        d = h - ht
        f["magd_tmpl"][j] = C.mag_d(U, Ut)
        f["diff_rms"][j] = np.sqrt(np.mean(d ** 2))
        f["diff_absmax"][j] = np.abs(d).max()
        inwin = nb[np.max(np.abs((nb - c) @ F[:, :2]), axis=1) <= radius]
        f["nn_p95"][j] = np.percentile(ttree.query(inwin)[0], 95) if len(inwin) else np.nan

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, cat=cat, stem=stem, label=lab, radius=radius,
                        seeds=seeds, valid=valid, p1_check=z["p1_check"],
                        template=tbest, template_fit_mm=fit[tbest],
                        n_points=len(P), n_gt=z["n_gt"], n_defect=z["n_defect"],
                        pt_label=z["pt_label"], pt_seed=z["pt_seed"],
                        seed_defect_dist=z["seed_defect_dist"],
                        ref_med=json.dumps({}), ref_mad=json.dumps({}),
                        h_ptp=z["h_ptp"], **f)
    return dst, "%.1fs %d valid tmpl=%s fit=%.3f" % (time.time() - t0, int(valid.sum()), tbest, fit[tbest])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=C.CATS)
    ap.add_argument("--radii", type=float, nargs="*", default=C.RADII_MM)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    items = [i for i in C.enumerate_test_items() if i[0] in a.cats]
    # group jobs by category so each worker builds a category's template trees once
    jobs = sorted([(c, s, l, r) for (c, s, l) in items for r in a.radii])
    print("jobs:", len(jobs), "cats:", a.cats, flush=True)
    skipped = []
    with Pool(a.procs) as pool:
        for k, (p, msg) in enumerate(pool.imap(score_item, jobs, chunksize=4), 1):
            if msg.startswith("unaligned") or msg.startswith("no-"):
                skipped.append((os.path.basename(p), msg))
            elif msg != "skip":
                print("[%d/%d] %s %s" % (k, len(jobs), os.path.basename(p), msg), flush=True)
    print("skipped %d:" % len(skipped), skipped[:20], flush=True)
    with open(os.path.join(C.OUT, "items_tmpl_skipped.json"), "w") as fh:
        json.dump(skipped, fh, indent=1)
    print("ALL_DONE_R3_TEMPLATE", flush=True)


if __name__ == "__main__":
    main()
