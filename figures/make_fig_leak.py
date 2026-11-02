"""The leakage diagnostic figure of rev11: the label-free bound L of eq. (leak) against the
displacement the margin shows on unseen good scans, one point per encoder and MVTec class,
under both window scales. Data: results/mvtec/mv_leak_{metric,sim}.json (mvtec/m6_leak.py).

  python figures/make_fig_leak.py          -> figures/fig_leak.{pdf,png}
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
from paper_style import use_paper_style, OKABE_ITO

RES = os.path.join(HERE, "..", "results", "mvtec")
ENC = [("dentnet_1024", "DentNet", OKABE_ITO["blue"]),
       ("pnpp_r0.3_128", "PointNet++ r0.3", OKABE_ITO["vermillion"]),
       ("pnpp_r0.6_128", "PointNet++ r0.6", OKABE_ITO["orange"]),
       ("pnpp_r2.0_128", "PointNet++ r2.0", OKABE_ITO["green"]),
       ("dinov2_1024", "DINOv2", OKABE_ITO["purple"])]
PROTO = [("metric", "o", "synthetic pitch"), ("sim", "s", "defect scale")]


def points(proto, rule_prefix):
    d = json.load(open(os.path.join(RES, "mv_leak_%s.json" % proto)))
    out = []
    for enc, lab, col in ENC:
        for cat, c in d["encoders"][enc]["classes"].items():
            r = c["rules"][[k for k in c["rules"] if k.startswith(rule_prefix)][0]]
            out.append((enc, cat, r["bound_sigma"], r["observed_shift_sigma"], r["observed_spread_sigma"], r["good_flagged"]))
    return out


def main():
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45))
    ax = axes[0]
    allx, ally = [], []
    for proto, mk, plab in PROTO:
        for enc, lab, col in ENC:
            P = [p for p in points(proto, "pbss") if p[0] == enc]
            x = np.array([p[2] for p in P]); y = np.array([p[3] for p in P])
            allx += list(x); ally += list(y)
            ax.scatter(x, y, s=16, marker=mk, facecolors=col if proto == "metric" else "none", edgecolors=col,
                       linewidths=0.8, label=lab if proto == "metric" else None, zorder=3)
    lo, hi = 0.15, 60
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, ls="-", zorder=1)
    ax.plot([lo, hi], [lo / 4, hi / 4], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.text(1.55, 1.9, "observed = L", fontsize=6.5, color="0.3", rotation=32, ha="left", va="bottom")
    ax.text(3.6, 0.62, "observed = L/4", fontsize=6.5, color="0.5", rotation=32, ha="left", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("leakage $L$ from unlabeled good scans ($\\sigma_\\Delta$)", fontsize=7.5)
    ax.set_ylabel("observed margin displacement ($\\sigma_\\Delta$)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    leg = ax.legend(fontsize=6.2, loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=8, va="top")
    # filled = synthetic pitch, hollow = defect scale, one note
    ax.text(0.97, 0.03, "filled: synthetic pitch\nhollow: defect scale", transform=ax.transAxes, fontsize=6.2, va="bottom", ha="right", color="0.25")

    ax = axes[1]
    for proto, mk, plab in PROTO:
        P = points(proto, "pbss")
        x = np.array([p[3] for p in P]); y = np.array([p[4] for p in P]); f = np.array([p[5] for p in P])
        sc = ax.scatter(x, y, c=f, s=16, marker=mk, cmap="magma", vmin=0, vmax=1, edgecolors="0.3" if proto == "sim" else "none",
                        linewidths=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.15, 20); ax.set_ylim(0.15, 60)
    ax.plot([0.15, 20], [0.15, 20], color="0.3", lw=0.8, zorder=1)
    ax.set_xlabel("observed margin displacement ($\\sigma_\\Delta$)", fontsize=7.5)
    ax.set_ylabel("spread of the good margins ($\\sigma_\\Delta$)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("good windows flagged", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(HERE, "fig_leak.pdf")); fig.savefig(os.path.join(HERE, "fig_leak.png"), dpi=200)
    print("bound holds in", np.mean(np.abs(ally) <= np.array(allx)), "of", len(allx), "cases; median observed/L", np.median(np.abs(ally) / np.array(allx)))


if __name__ == "__main__":
    main()
