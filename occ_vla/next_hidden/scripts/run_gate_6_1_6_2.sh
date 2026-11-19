#!/usr/bin/env bash
# 6-1 + 6-2 end to end (no policy). Rendering only on one GPU via EGL.
set -e
cd "$(dirname "$0")/.."
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh
export CUDA_VISIBLE_DEVICES=${RENDER_GPU:-2}
python scripts/build_init_states.py --n 10 --max-tries 150 | tail -1
python scripts/verify_init_states.py | tail -1
python scripts/rgb_visibility.py | tail -3
for k in 0 1 2 3 4 5 6 7 8 9; do python scripts/scripted_expert.py --state $k 2>/dev/null | grep '^{"state'; done > results/expert/all_states.jsonl
python scripts/summarize_expert.py
