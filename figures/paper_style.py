"""Shared figure styling, so every figure in the paper reads as one system.

Type is Computer Modern Roman, the LNCS body face, taken from the font matplotlib bundles
so no system font needs installing.

Colour follows one rule. The Okabe-Ito palette is *categorical*, so it is used for
categorical marks only: one fixed colour per method, plus one pair for ground truth against
method output. Score and residual fields keep perceptually uniform or properly diverging
colour maps, because a categorical palette stretched into a ramp is neither.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}

# One colour per method, used identically in every figure so a reader can carry the
# association from one to the next.
METHOD = {
    "dentnet":     OKABE_ITO["blue"],
    "pbss":        OKABE_ITO["orange"],
    "pointnetpp":  OKABE_ITO["vermillion"],
    "shape":       OKABE_ITO["green"],
}

# Ground truth against method output. The two cool entries, chosen because they stay
# legible against a warm score ramp such as magma or hot, where vermillion or yellow
# would collide with the map's own mid and high range.
C_TRUTH = OKABE_ITO["green"]
C_METHOD_MARK = OKABE_ITO["sky"]

# Score and residual fields keep their established maps.
CMAP_SCORE = "magma"       # sequential, perceptually uniform
CMAP_RESIDUAL = "RdBu_r"   # diverging, signed residuals about zero
CMAP_PROB = "hot"          # sequential, probabilities on [0, 1]


def mask_cmap(hex_colour):
    """White to a method colour, for binary masks laid over a light background."""
    return LinearSegmentedColormap.from_list("mask", ["#FFFFFF", hex_colour])


def use_paper_style():
    """Computer Modern Roman at paper sizes."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # cmr10 carries no Unicode minus glyph, so fall back to the ASCII hyphen
        "axes.unicode_minus": False,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.formatter.use_mathtext": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
