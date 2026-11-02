"""Stage 4. Tables from mv_rules_<proto>.json: the Table 2 analogue on MVTec 3D-AD.

Category means over the ten categories, with the per-category breakdown for the image
AUROC. Output: mv_report_<proto>.md
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import mv_common as M
import os, json, argparse
import numpy as np


def cm(e, k):
    return float(np.nanmean([v[k] for v in e["categories"].values() if v is not None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default="sim")
    a = ap.parse_args()
    R = json.load(open(os.path.join(M.OUT, "mv_rules_%s.json" % a.proto)))
    L = ["# Synthetic fuselage to MVTec 3D-AD, protocol %s" % a.proto, "",
         "Category means over %s. Rules fitted on synthetic training windows only, applied zero-shot." % ", ".join(M.CATS), "",
         "## Frozen encoders and rules", "",
         "| encoder | rule | val AUC | window AUC | image AUROC | pixel AUROC | good test flagged % | train good flagged % | margin shift (sigma) | syn bg before -> after % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for enc, out in R["encoders"].items():
        for rn, e in out["rules"].items():
            L.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.1f | %.1f | %+.2f | %.1f -> %.1f |" % (
                enc, rn, e["auc_val"], cm(e, "window_auc"), cm(e, "image_auroc"), cm(e, "pixel_auroc"),
                100 * cm(e, "good_flagged"), 100 * e["train_good_flagged"], e["shift_of_margin_in_sigma"],
                100 * e["flagged_synthetic_bg_before"], 100 * e["flagged_synthetic_bg_after_shift"]))
    L += ["", "## End-to-end controls (threshold 0.5)", "", "| scorer | window AUC | image AUROC | pixel AUROC | good test flagged % |", "|---|---|---|---|---|"]
    for k, cats in R["end_to_end"].items():
        e = {"categories": cats}
        L.append("| %s | %.3f | %.3f | %.3f | %.1f |" % (k, cm(e, "window_auc"), cm(e, "image_auroc"), cm(e, "pixel_auroc"), 100 * cm(e, "good_flagged")))
    L += ["", "## Shift decomposition", "", "| encoder | |delta| / bg std | in normal subspace | in dent subspace | GDS: removed / in GDS / outside sum space |", "|---|---|---|---|---|"]
    for enc, out in R["encoders"].items():
        pb = [e for k, e in out["rules"].items() if k.startswith("pbss")][0]
        gd = [e for k, e in out["rules"].items() if k.startswith("gds")][0]
        L.append("| %s | %.2f | %.2f | %.2f | %.2f / %.2f / %.2f |" % (
            enc, out["delta_norm_over_bg_feature_std"], pb.get("delta_energy_in_normal_subspace", float("nan")),
            pb.get("delta_energy_in_dent_subspace", float("nan")), gd["delta_u_energy_in_removed_principal"],
            gd["delta_u_energy_in_gds"], gd["delta_u_energy_outside_sumspace"]))
    L += ["", "## Per-category image AUROC", "", "| encoder | rule | " + " | ".join(M.CATS) + " |", "|---|---|" + "---|" * len(M.CATS)]
    for enc, out in R["encoders"].items():
        for rn, e in out["rules"].items():
            L.append("| %s | %s | " % (enc, rn) + " | ".join("%.2f" % e["categories"][c]["image_auroc"] if c in e["categories"] else "" for c in M.CATS) + " |")
    M.write_md("mv_report_%s.md" % a.proto, L)
    print("ALL_DONE_M4", flush=True)


if __name__ == "__main__":
    main()
