"""Qualitative Real3D-AD figure (Fig. 2): per item, the object colored by the nearest seed's
score under the raw point-to-template distance, the local nominal subspace on DentNet
features (raw and normalized), the uncentred + GDS rule on PointNet++ features, and the
ground truth. Data from results/real3d/fig_export/ (real3d/r3_fig_export.py).

  python figures/make_fig_qualitative.py            all candidates as PNG previews
  python figures/make_fig_qualitative.py car__624_sink toffees__538_sink --pdf   the paper figure
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "results", "real3d", "fig_export")
PANELS = [("nn_p95", "seeds2", "valid2", "d2", "raw point-to-template distance"),
          ("dentnet_nom_rec3", "seeds3", "valid3", "d3", "local nominal subspace"),
          ("dentnet_nom_rec3_norm", "seeds3", "valid3", "d3", "normalized by template disagreement"),
          ("pnpp_gds_r1_rec3_norm", "seeds3", "valid3", "d3", "uncentered, GDS $R{=}1$, normalized")]


def view(P):
    """Project the single-sided scan onto its two principal in-plane axes."""
    c = P.mean(0); U, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    Q = (P - c) @ Vt[:2].T
    return Q


def draw(fp, ax_row, cmap="viridis", ms=1.2, titles=True):
    z = np.load(fp)
    P, lab = z["P"], z["lab"]
    Q = view(P)
    meta = json.loads(str(z["meta"]))
    for ax, (st, sk, vk, dk, title) in zip(ax_row[:4], PANELS):
        s = z[st].astype(float); seeds = z[sk]; valid = z[vk] & np.isfinite(s)
        tree = cKDTree(seeds[valid])
        _, j = tree.query(P)
        sc = s[valid][j]
        lo, hi = np.nanpercentile(sc, [2, 99.5])
        sc = np.clip((sc - lo) / max(hi - lo, 1e-9), 0, 1)
        ax.scatter(Q[:, 0], Q[:, 1], c=sc, s=ms, cmap=cmap, linewidths=0, rasterized=True)
        jm = np.flatnonzero(valid)[int(np.argmax(s[valid]))]
        qm = view_point(P, seeds[jm])
        ax.plot(qm[0], qm[1], marker="o", ms=9, mfc="none", mec="red", mew=1.5)
        ax.set_aspect("equal"); ax.set_axis_off()
        if titles:
            d = float(z[dk][jm])
            ax.set_title("%s\nargmax %.2f mm from the defect" % (title, d), fontsize=7)
    ax = ax_row[4]
    ax.scatter(Q[lab == 0, 0], Q[lab == 0, 1], c="0.8", s=ms, linewidths=0, rasterized=True)
    ax.scatter(Q[lab > 0, 0], Q[lab > 0, 1], c="red", s=ms, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_axis_off()
    if titles:
        ax.set_title("ground truth (%s, %d points)" % (meta["label"], meta["n_defect"]), fontsize=7)
    return meta


def view_point(P, p):
    c = P.mean(0); U, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    return (p - c) @ Vt[:2].T


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pdf = "--pdf" in sys.argv
    files = sorted(glob.glob(os.path.join(EXP, "*.npz")))
    if args:
        files = [f for f in files if any(a in os.path.basename(f) for a in args)]
    if pdf:
        # LNCS text width is 12.2 cm, so the page draws this at about 0.67 of its size
        fig, axes = plt.subplots(len(files), 5, figsize=(7.2, 1.45 * len(files)), squeeze=False)
        labels = ["(a) raw distance", "(b) local subspace", "(c) normalized", "(d) uncentered + GDS", "(e) ground truth"]
        for i, (row, fp) in enumerate(zip(axes, files)):
            draw(fp, row, ms=0.35, titles=False)
            if i == 0:
                for ax, lab in zip(row, labels):
                    ax.set_title(lab, fontsize=8, pad=2)
        plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
        out = os.path.join(HERE, "fig_qualitative.pdf")
        plt.savefig(out, dpi=200); plt.savefig(out[:-4] + ".png", dpi=150)
        print(out)
        return
    for fp in files:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
        meta = draw(fp, axes)
        fig.suptitle("%s %s  (%s, seed AUC raw %.2f, norm. subspace %.2f, GDS %.2f)" % (
            meta["cat"], meta["stem"], meta["label"], meta["raw_seed_auc"], meta["dentnet_nom_rec3_norm_seed_auc"],
            meta["pnpp_gds_r1_rec3_norm_seed_auc"]), fontsize=9)
        plt.tight_layout()
        out = os.path.join(HERE, "qual_" + os.path.basename(fp)[:-4] + ".png")
        plt.savefig(out, dpi=110); plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
