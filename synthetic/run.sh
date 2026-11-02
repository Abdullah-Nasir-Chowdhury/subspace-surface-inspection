#!/bin/bash
# Synthetic benchmark: generate the data, then evaluate DentNet, PBSS and the shape subspace in domain.
# Needs best_dentnet_real_scale.pth in the repository root (or run dentnet/train.py after step 1).
# Outputs go to ~/accv_results, logs to ~/accv_logs.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p ~/accv_logs
export MPLBACKEND=Agg
$PY synthetic/gen_parallel.py          > ~/accv_logs/gen.log 2>&1   # 30,000 samples into ~/data/pointcloud_db_real_scale
$PY synthetic/s1_eval_val_iou.py       > ~/accv_logs/s1.log 2>&1    # DentNet IoU on the validation split
$PY synthetic/s2_pbss.py               > ~/accv_logs/s2.log 2>&1    # nearest subspace on DentNet features
$PY synthetic/s3_shape_subspace.py     > ~/accv_logs/s3.log 2>&1    # training-free shape subspace
echo "done"
