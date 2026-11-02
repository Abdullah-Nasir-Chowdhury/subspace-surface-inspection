#!/bin/bash
# PointNet++ on the synthetic windows: patches, radius 2.0, the radius sweep (0.3, 0.6), seed repeats.
# Needs the synthetic dataset from synthetic/run.sh. Outputs and checkpoints go to ~/accv_results_real.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p ~/accv_logs
$PY pointnetpp/patches.py              > ~/accv_logs/pnpp_patches.log 2>&1
$PY pointnetpp/train_pointnetpp.py     > ~/accv_logs/pnpp_r2.0.log 2>&1
$PY pointnetpp/radius_sweep.py         > ~/accv_logs/pnpp_sweep.log 2>&1
$PY pointnetpp/seed_repeats.py         > ~/accv_logs/pnpp_seeds.log 2>&1
echo "done"
