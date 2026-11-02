#!/bin/bash
# Real3D-AD (Sec. 5.3 to 5.5, Table 2, Fig. 2). Resumable: stages skip existing items.
# Needs the DentNet and PointNet++ checkpoints and Real3D-AD at ~/data/ext/real3d_ad/Real3D-AD-PCD.
# Outputs go to ~/accv_results/real3d, logs to ~/accv_logs.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
LOG=~/accv_logs
mkdir -p $LOG
run() { name=$1; shift; echo "=== $name $(date)"; "$@" > $LOG/r3_$name.log 2>&1; }

FEAT_STATS="dentnet_nom_nn dentnet_nom_rec3 dentnet_nom_rec3_norm dentnet_cat_rec50 dentnet_pbss_k50 pnpp_nom_nn pnpp_nom_rec3 pnpp_nom_rec3_norm pnpp_cat_rec50"
GEOM_STATS="nom_rec_k6 nom_rec_k6_norm nom_rec_k6_pl nom_rec_k6_pl_norm nom_nn diff_rms magd_tmpl magd_nom_min magd_nom_sub nn_p95"

# reference-free statistics and point-to-template distance
run score          $PY r3_score.py
run aggregate      $PY r3_aggregate.py
run template       $PY r3_template.py
run agg_template   $PY r3_aggregate.py --items-dir items_tmpl --stats magd_tmpl diff_rms diff_absmax nn_p95

# the subspace as a classifier on Real3D-AD (Sec. 5.3)
run learned_s1     $PY r3_learned.py --scale 1
run learned_s6.6   $PY r3_learned.py --scale 6.6
run learned_sum    $PY r3_learned_summary.py

# the subspace as a model of the good parts (Sec. 5.4)
run defect_extent  $PY r3_defect_extent.py
run nominal_5      $PY r3_nominal.py --procs 12          # five pre-registered categories
run nominal_feat_5 $PY r3_nominal_feat.py --procs 10     # point AUROC 0.957
run agg_nominal_5  $PY r3_aggregate.py --items-dir items_nominal --tag r3_report_items_nominal \
                     --stats nom_rec_k0 nom_rec_k1 nom_rec_k3 nom_rec_k6 nom_nn diff_rms diff_absmean magd_tmpl magd_nom_min magd_nom_sub nn_p95
run agg_nom_feat_5 $PY r3_aggregate.py --items-dir items_nominal_feat --radii 3.6 --tag r3_report_items_nominal_feat \
                     --stats dentnet_nom_nn dentnet_nom_mean dentnet_nom_rec3 dentnet_cat_rec50 dentnet_pbss_k50 pnpp_nom_nn pnpp_nom_mean pnpp_nom_rec3 pnpp_cat_rec50
run register       $PY r3_register.py --procs 12
run nominal_feat   $PY r3_nominal_feat.py --procs 10 --items-dir items_nominal_feat_all --tol 1.0 --save-feats
run nominal        $PY r3_nominal.py --procs 12 --items-dir items_nominal_all --tol 1.0 --radii 2.5
run objagg_feat    $PY r3_object_agg.py --items-dir items_nominal_feat_all --radius 3.6 --tag r3_objagg_nominal_feat_all --stats $FEAT_STATS
run objagg_geom    $PY r3_object_agg.py --items-dir items_nominal_all --radius 2.5 --tag r3_objagg_nominal_all_r2.5 --stats $GEOM_STATS
run nominal_report $PY r3_nominal_report.py
run fair_compare   $PY r3_fair_compare.py

# DINOv2 features, uncentered subspaces and GDS rows of Table 2
run dinov2_encode  $PY r3_dinov2.py encode
run dinov2_score   $PY r3_dinov2.py score --procs 12
run dinov2_compare $PY r3_dinov2_compare.py

# ablations of the local subspace (K, template count)
run ablate         $PY r3_nominal_ablate.py --items-dir items_nominal_feat_all --out-dir items_nominal_feat_ablate --procs 12
ABL_STATS=$(grep "^STATS " $LOG/r3_ablate.log | sed 's/^STATS //')
run objagg_ablate  $PY r3_object_agg.py --items-dir items_nominal_feat_ablate --radius 3.6 --tag r3_objagg_nominal_feat_ablate --stats $ABL_STATS

# window scale without annotation (Sec. 5.4)
for S in 1 2 4 8; do
  run scale_s$S         $PY r3_scale_sweep.py --scale $S --procs 12
  run scale_s${S}_cmp   $PY r3_dinov2_compare.py --items-dir items_scale_s$S --tag r3_scale_s$S
done

# registration sensitivity (Sec. 5.5)
CATS5="candybar car fish seahorse toffees"
PERT=""
for t in 0.1 0.25 0.5 1.0; do
  run pert_t$t $PY r3_nominal_feat.py --procs 10 --cats $CATS5 --items-dir items_nominal_feat_pert_t$t --tol 5.0 --perturb-trans-mm $t
  PERT="$PERT items_nominal_feat_pert_t$t:trans:$t"
done
for r in 0.25 0.5 1.0 2.0; do
  run pert_r$r $PY r3_nominal_feat.py --procs 10 --cats $CATS5 --items-dir items_nominal_feat_pert_r$r --tol 5.0 --perturb-rot-deg $r
  PERT="$PERT items_nominal_feat_pert_r$r:rot:$r"
done
run reg_sensitivity $PY r3_reg_sensitivity.py --items-dir items_nominal_feat_all --stats dentnet_nom_rec3 dentnet_nom_rec3_norm pnpp_nom_rec3_norm --pert $PERT

# data for the qualitative figure (figures/make_fig_qualitative.py)
run fig_export     $PY r3_fig_export.py
echo "done"
