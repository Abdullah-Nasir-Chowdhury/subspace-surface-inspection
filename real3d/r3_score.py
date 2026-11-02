"""Stage 1. Per-item training-free statistics on every Real3D-AD test item, good and defective.

For each item and each patch radius, seeds are placed on the surface by voxel centroids at
spacing = radius / 2 (a sliding window with 75% overlap, the lattice the paper argues for;
the stride was fixed for localisation resolution before any label was scored). At every
seed the local surface is resampled onto the shared 10x10 canonical grid from its tangent
plane, and the paper's statistics are computed:

  sv2        raw third singular value of the centred (100,3) patch matrix        (mm)
  surfvar    S2^2 / (S0^2 + S1^2 + S2^2), the scale-normalised replacement
  magd_nb    mean Mag(D) against the 8 nearest seeds' shape subspaces (Proposition 1
             predicts this is INVERTED on a shared grid: clean neighbours look dissimilar)
  ds_ref     min Mag(D) against the nominal reference set built from the category's four
             good training templates (the paper's reference-based variant)
  plane_rms  RMS residual after a full-patch plane fit                             (mm)
  rq_rms     RMS residual after a ring-fitted quadric                              (mm)
  rq_absmax  largest |residual| after the ring quadric                              (mm)
  rq_skew    signed skewness of the ring-quadric residual (sign-aware, for bulge/sink)
  h_ptp      height range of the raw window, to flag seeds straddling an object edge

Nothing is fitted to any test label. Writes one .npz per (item, radius) to
~/accv_results/real3d/items/ and the per-category reference to ~/accv_results/real3d/ref/.
Resumable: existing files are skipped.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, time, argparse
import r3_common as C          # sets BLAS caps before numpy
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool


def seed_features(P, tree, obj_c, seeds, grid, radius):
    """Every per-seed statistic, plus U (m,100,3) and the plane-residual direction rdir (m,100)."""
    m = len(seeds)
    n = grid.g * grid.g
    XY = np.c_[grid.X * radius, grid.Y * radius]
    valid = np.zeros(m, bool)
    U = np.zeros((m, n, 3))
    rdir = np.zeros((m, n))
    f = {k: np.full(m, np.nan) for k in
         ("sv2", "surfvar", "plane_rms", "rq_rms", "rq_absmax", "rq_skew", "h_ptp")}
    for j, c in enumerate(seeds):
        idx = tree.query_ball_point(c, radius * 1.6)
        if len(idx) < C.MIN_BALL_PTS:
            continue
        nb = P[idx]
        F = C.local_frame(nb, c, obj_c)
        h = C.resample(nb, c, F, grid, radius)
        Uj, S = C.shape_subspace(np.c_[XY, h])
        U[j] = Uj
        r = grid.P @ h
        rdir[j] = C.unit(r)
        rq = grid.P_quad_ann @ h
        f["sv2"][j] = S[2]
        f["surfvar"][j] = S[2] ** 2 / max(float(np.sum(S ** 2)), 1e-30)
        f["plane_rms"][j] = np.sqrt(np.mean(r ** 2))
        f["rq_rms"][j] = np.sqrt(np.mean(rq ** 2))
        f["rq_absmax"][j] = np.abs(rq).max()
        f["rq_skew"][j] = C.stat_skew(rq)
        f["h_ptp"][j] = np.ptp(h)
        valid[j] = True
    return f, U, rdir, valid


def neighbour_magd(seeds, U, rdir, valid, k, rng):
    """Mean Mag(D) against the k nearest valid seeds, plus a Proposition 1 self-check."""
    m = len(seeds)
    out = np.full(m, np.nan)
    vi = np.flatnonzero(valid)
    if len(vi) < k + 1:
        return out, np.nan
    t = cKDTree(seeds[vi])
    kk = min(k + 1, len(vi))
    _, nn = t.query(seeds[vi], k=kk)
    nn = np.atleast_2d(nn)
    checks = []
    for a, row in zip(vi, nn):
        nbrs = [vi[b] for b in row[1:]]
        vals = [C.mag_d(U[a], U[b]) for b in nbrs]
        out[a] = np.mean(vals)
        if rng.random() < 0.02:
            b = nbrs[0]
            checks.append(abs(vals[0] - 2.0 * (1.0 - abs(rdir[a] @ rdir[b]))))
    return out, (float(max(checks)) if checks else np.nan)


def reference_magd(rdir, valid, R):
    """min over the reference set of Mag(D) = 2(1 - |<r, r_ref>|), exact by Proposition 1."""
    out = np.full(len(rdir), np.nan)
    if R is None or len(R) == 0:
        return out
    dots = np.abs(rdir[valid] @ R.T)
    out[valid] = 2.0 * (1.0 - dots.max(1))
    return out


# ---------------------------------------------------------------------------- reference

def build_reference(args):
    cat, radius = args
    path = os.path.join(C.OUT, "ref", "%s_r%.1f.npz" % (cat, radius))
    if os.path.exists(path):
        return path
    grid = C.Grid()
    rng = np.random.default_rng(1234)
    feats, rdirs = [], []
    for stem in C.enumerate_templates(cat):
        P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "train", stem + ".pcd"))
        tree = cKDTree(P)
        seeds = C.voxel_seeds(P, radius, rng)            # coarser than the test lattice
        f, U, rdir, valid = seed_features(P, tree, P.mean(0), seeds, grid, radius)
        rdirs.append(rdir[valid])
        feats.append({k: v[valid] for k, v in f.items()})
    R = np.concatenate(rdirs)
    stats = {k: np.concatenate([d[k] for d in feats]) for k in feats[0]}
    med = {k: float(np.median(v)) for k, v in stats.items()}
    mad = {k: float(1.4826 * np.median(np.abs(v - med[k])) + 1e-12) for k, v in stats.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, R=R, med=json.dumps(med), mad=json.dumps(mad), n=len(R))
    return path


# ---------------------------------------------------------------------------- items

def score_item(args):
    cat, stem, lab, radius = args
    path = os.path.join(C.OUT, "items", "%s__%s__r%.1f.npz" % (cat, stem, radius))
    if os.path.exists(path):
        return path, "skip"
    t0 = time.time()
    grid = C.Grid()
    rng = np.random.default_rng(abs(hash((cat, stem, radius))) % (2 ** 32))
    P = C.read_pcd(os.path.join(C.R3D_ROOT, cat, "test", stem + ".pcd"))
    tree = cKDTree(P)
    obj_c = P.mean(0)
    seeds = C.voxel_seeds(P, radius / 2.0, rng)       # stride = half the half-extent
    f, U, rdir, valid = seed_features(P, tree, obj_c, seeds, grid, radius)
    f["magd_nb"], p1_check = neighbour_magd(seeds, U, rdir, valid, C.K_NEIGHBOURS, rng)
    ref = np.load(os.path.join(C.OUT, "ref", "%s_r%.1f.npz" % (cat, radius)))
    f["ds_ref"] = reference_magd(rdir, valid, ref["R"])

    # point-level ground truth in the ground-truth file's own coordinates
    if lab == "good":
        gx, gl = P, np.zeros(len(P), int)
    else:
        gx, gl = C.read_gt(os.path.join(C.R3D_ROOT, cat, "gt", stem + ".txt"))
    sub = rng.choice(len(gx), size=min(C.N_POINT_SUBSAMPLE, len(gx)), replace=False)
    vi = np.flatnonzero(valid)
    if len(vi) == 0:
        return path, "novalid"
    st = cKDTree(seeds[vi])
    _, nn = st.query(gx[sub])
    pt_seed = vi[nn]
    defect = gx[gl == 1]
    # distance from every valid seed to the nearest defect point (for localisation)
    if len(defect):
        dtree = cKDTree(defect)
        seed_defect_dist = np.full(len(seeds), np.nan)
        seed_defect_dist[vi] = dtree.query(seeds[vi])[0]
    else:
        seed_defect_dist = np.full(len(seeds), np.inf)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, cat=cat, stem=stem, label=lab, radius=radius,
                        seeds=seeds, valid=valid, p1_check=p1_check,
                        n_points=len(P), n_gt=len(gx), n_defect=int(gl.sum()),
                        pt_label=gl[sub].astype(np.int8), pt_seed=pt_seed,
                        seed_defect_dist=seed_defect_dist,
                        ref_med=ref["med"], ref_mad=ref["mad"],
                        **f)
    return path, "%.1fs %d seeds %d valid" % (time.time() - t0, len(seeds), int(valid.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="items per class per category, 0 = all")
    ap.add_argument("--radii", type=float, nargs="*", default=C.RADII_MM)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    items = C.enumerate_test_items()
    if a.limit:
        keep = []
        for cat in C.CATS:
            ci = [i for i in items if i[0] == cat]
            g = [i for i in ci if i[2] == "good"][: a.limit]
            d = [i for i in ci if i[2] != "good"][: a.limit]
            keep += g + d
        items = keep
    n = {l: sum(i[2] == l for i in items) for l in ("good", "bulge", "sink")}
    print("items:", n, "radii:", a.radii, flush=True)

    with Pool(a.procs) as pool:
        for p in pool.imap_unordered(build_reference, [(c, r) for c in C.CATS for r in a.radii]):
            print("ref", os.path.basename(p), flush=True)
        jobs = [(c, s, l, r) for r in a.radii for (c, s, l) in items]
        done = 0
        for p, msg in pool.imap_unordered(score_item, jobs, chunksize=1):
            done += 1
            if msg != "skip":
                print("[%d/%d] %s %s" % (done, len(jobs), os.path.basename(p), msg), flush=True)
    print("ALL_DONE_R3_SCORE", flush=True)


if __name__ == "__main__":
    main()
