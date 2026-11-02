#!/bin/bash
# Synthetic training to MVTec 3D-AD (Table 1, Eq. 2, Fig. 1). Resumable: stages skip existing outputs.
# Needs the DentNet and PointNet++ checkpoints and MVTec 3D-AD at ~/data/ext/mvtec_3d_ad.
# "metric" is the annotation-free synthetic pitch, "sim" the defect scale (factor 3.19).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
LOG=~/accv_logs
mkdir -p $LOG
run() { name=$1; shift; echo "=== $name $(date)"; "$@" > $LOG/mvtec_$name.log 2>&1; }
run m0_inventory   $PY m0_inventory.py
run m1_windows     $PY m1_windows.py --procs 12
run m2_synthetic   $PY m2_features.py --synthetic
run m2_features    $PY m2_features.py
for P in metric sim; do
  run m3_$P        $PY m3_rules.py --proto $P      # decision rules, Table 1
  run m4_$P        $PY m4_report.py --proto $P
  run m5_$P        $PY m5_bound.py --proto $P      # margin bound, Eq. 2
  run m6_$P        $PY m6_leak.py --proto $P       # leakage diagnostic, Fig. 1
done
run m3b_omsm_sim   $PY m3b_omsm.py --proto sim     # whitened sum space (OMSM) comparison
echo "done"
