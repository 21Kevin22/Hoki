#!/bin/bash
# Generic Month2 (data-collect -> train -> eval) pipeline for a single task,
# at "proper" scale matching task2's own originally-validated recipe:
# 30-episode CBF-teacher data collection, 300 training steps, n=40 paired eval.
# Usage: run_month2_pipeline.sh <TASK_ID> <GPU_ID> <WAIT_PID_OR_0>
set -e
TASK_ID=$1
GPU_ID=$2
WAIT_PID=$3
N_COLLECT=30
N_EVAL=40
cd /home/ubuntu/slocal/Hoki/occ_vla
export PYTHONPATH=/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/LIBERO
export LIBERO_CONFIG_PATH=/home/ubuntu/.libero
export HOME=/home/ubuntu
PY=/home/ubuntu/slocal/Hoki/occ_vla/.venv_openvla_oft/bin/python3
CKPT=/home/ubuntu/slocal/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa
OFT=/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/openvla-oft
# NOTE: both scripts os.chdir() into $OFT at import time, and
# train_object_centric_adapter.py separately re-bases any RELATIVE
# --data-dir/--out-adapter onto scripts/, not $OFT. Using absolute paths
# everywhere below to remove all ambiguity (bug found+fixed 2026-09-06).

if [ "$WAIT_PID" != "0" ]; then
  echo "[pipeline task${TASK_ID}] waiting for PID ${WAIT_PID} to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi

echo "[pipeline task${TASK_ID}] Starting Month2 data collection (n=${N_COLLECT}, CBF-teacher rollout) on GPU${GPU_ID}..."

CUDA_VISIBLE_DEVICES=$GPU_ID $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids $TASK_ID --n-episodes $N_COLLECT --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions proactive_avoidance_depth \
  --save-distillation-pairs-dir $OFT/month2_collect_task${TASK_ID}_n${N_COLLECT} \
  --results-dir $OFT/month2_collect_task${TASK_ID}_n${N_COLLECT}_meta \
  > logs/month2_task${TASK_ID}_collect.log 2>&1

echo "[pipeline task${TASK_ID}] data collection done. Training Month2 adapter (300 steps)..."

CUDA_VISIBLE_DEVICES=$GPU_ID $PY scripts/train_object_centric_adapter.py \
  --checkpoint $CKPT \
  --data-dir $OFT/month2_collect_task${TASK_ID}_n${N_COLLECT} \
  --n-steps 300 \
  --out-adapter $OFT/object_centric_adapter_task${TASK_ID}_v2 \
  > logs/month2_task${TASK_ID}_train.log 2>&1

echo "[pipeline task${TASK_ID}] training done. Running full paired eval (baseline / CBF / Month2 / Month2+CBF), n=${N_EVAL}..."

# PRIORITY: Month2-alone + Month2+CBF first (adapter loaded) -- this is the
# result that actually matters; the plain baseline/CBF-alone re-measurement
# below is only a same-batch ablation reference, run second/lower-priority
# per explicit user request (2026-09-06).
CUDA_VISIBLE_DEVICES=$GPU_ID $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids $TASK_ID --n-episodes $N_EVAL --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions baseline proactive_avoidance_depth \
  --load-object-centric-adapter $OFT/object_centric_adapter_task${TASK_ID}_v2 \
  --results-dir $OFT/month2_v2_adapter_task${TASK_ID}_n${N_EVAL} \
  > logs/month2_task${TASK_ID}_eval_adapter.log 2>&1

echo "[pipeline task${TASK_ID}] Month2 eval done. Running ablation reference (plain baseline/CBF, no adapter), n=${N_EVAL}..."

# ablation reference: true (no-Month2) baseline+CBF at the SAME n/episode-offset
CUDA_VISIBLE_DEVICES=$GPU_ID $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids $TASK_ID --n-episodes $N_EVAL --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions baseline proactive_avoidance_depth \
  --results-dir $OFT/month2_v2_noadapter_task${TASK_ID}_n${N_EVAL} \
  > logs/month2_task${TASK_ID}_eval_noadapter.log 2>&1

echo "[pipeline task${TASK_ID}] ALL DONE (proper scale: collect n=${N_COLLECT}, eval n=${N_EVAL})."
