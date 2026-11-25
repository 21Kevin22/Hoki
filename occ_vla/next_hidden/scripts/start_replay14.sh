#!/usr/bin/env bash
# PREREG (14): 12 CPU-only replay shards (no GPU).
cd "$(dirname "$0")/.."
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh >/dev/null 2>&1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
for k in $(seq 0 11); do SHARD=$k/12 setsid nohup python main/replay14.py C0_side9 VIS_side9 LOOK_main10 > results/logs/replay14_$k.log 2>&1 & done
