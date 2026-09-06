#!/bin/bash
# Eval-only stage (Month2 first, ablation reference second) for a task whose
# adapter is ALREADY trained. Usage: run_month2_eval_only.sh <TASK_ID> <GPU_ID>
set -e
TASK_ID=$1
GPU_ID=$2
N_EVAL=40
cd /home/ubuntu/slocal/Hoki/occ_vla
export PYTHONPATH=/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/LIBERO
export LIBERO_CONFIG_PATH=/home/ubuntu/.libero
export HOME=/home/ubuntu
PY=/home/ubuntu/slocal/Hoki/occ_vla/.venv_openvla_oft/bin/python3
CKPT=/home/ubuntu/slocal/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa
OFT=/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/openvla-oft

echo "[eval task${TASK_ID}] Running Month2 eval (priority) on GPU${GPU_ID}..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids $TASK_ID --n-episodes $N_EVAL --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions baseline proactive_avoidance_depth \
  --load-object-centric-adapter $OFT/object_centric_adapter_task${TASK_ID}_v2 \
  --results-dir $OFT/month2_v2_adapter_task${TASK_ID}_n${N_EVAL} \
  > logs/month2_task${TASK_ID}_eval_adapter.log 2>&1

echo "[eval task${TASK_ID}] Month2 eval done. Running ablation reference (no adapter)..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PY scripts/run_libero_occluded_oracle_headroom.py \
  --task-ids $TASK_ID --n-episodes $N_EVAL --episode-offset 0 --suite 10 \
  --checkpoint $CKPT \
  --conditions baseline proactive_avoidance_depth \
  --results-dir $OFT/month2_v2_noadapter_task${TASK_ID}_n${N_EVAL} \
  > logs/month2_task${TASK_ID}_eval_noadapter.log 2>&1

echo "[eval task${TASK_ID}] ALL DONE."
