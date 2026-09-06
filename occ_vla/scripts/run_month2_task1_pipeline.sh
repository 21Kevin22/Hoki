#!/bin/bash
set -e
cd /home/ubuntu/slocal/Hoki/occ_vla
export PYTHONPATH=/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/LIBERO
export LIBERO_CONFIG_PATH=/home/ubuntu/.libero
export HOME=/home/ubuntu
PY=/home/ubuntu/slocal/Hoki/occ_vla/.venv_openvla_oft/bin/python3
CKPT=/home/ubuntu/slocal/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa
OFT=/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/openvla-oft
# NOTE (bug found+fixed 2026-09-06): both scripts os.chdir() into $OFT at
# import time, and train_object_centric_adapter.py separately re-bases any
# RELATIVE --data-dir/--out-adapter onto scripts/, not $OFT -- the two
# scripts disagree on relative-path conventions. Using absolute paths
# everywhere below to remove all ambiguity.

echo "[pipeline] Starting Month2 data collection for task1 (n=20, CBF-teacher rollout)..."

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids 1 --n-episodes 20 --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions proactive_avoidance_depth \
  --save-distillation-pairs-dir $OFT/month2_collect_task1_n20 \
  --results-dir $OFT/month2_collect_task1_n20_meta \
  > logs/month2_task1_collect.log 2>&1

echo "[pipeline] data collection done. Training Month2 adapter for task1 (300 steps)..."

CUDA_VISIBLE_DEVICES=0 $PY scripts/train_object_centric_adapter.py \
  --checkpoint $CKPT \
  --data-dir $OFT/month2_collect_task1_n20 \
  --n-steps 300 \
  --out-adapter $OFT/object_centric_adapter_task1 \
  > logs/month2_task1_train.log 2>&1

echo "[pipeline] training done. Running full eval (baseline / CBF / Month2 / Month2+CBF) for task1, n=20..."

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids 1 --n-episodes 20 --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions baseline proactive_avoidance_depth \
  --load-object-centric-adapter $OFT/object_centric_adapter_task1 \
  --results-dir $OFT/month2_plus_cbf_task1_n20 \
  > logs/month2_task1_eval.log 2>&1

echo "[pipeline] ALL DONE for task1 Month2+CBF."
