#!/usr/bin/env bash
# 6-1 + 6-2 end to end (no policy). Rendering only on one GPU via EGL.
# Any failing step stops the gate (pipefail). verify_init_states.py exits 1 if a reloaded state fails a check;
# set ALLOW_VERIFY_FAIL=1 to continue anyway (known solver warm-start drift, RESULTS_next.md) -- it is then logged.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh
export CUDA_VISIBLE_DEVICES=${RENDER_GPU:-2}
python scripts/build_init_states.py --n 10 --max-tries 150 | tail -1
if ! python scripts/verify_init_states.py > results/init/verify.log; then
  tail -1 results/init/verify.log
  if [ "${ALLOW_VERIFY_FAIL:-0}" != 1 ]; then echo "verify failed; stopping (ALLOW_VERIFY_FAIL=1 to continue)"; exit 1; fi
  echo "WARNING: verify failed, continuing because ALLOW_VERIFY_FAIL=1"
fi
tail -1 results/init/verify.log
python scripts/rgb_visibility.py | tail -3
: > results/expert/all_states.jsonl
for k in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/scripted_expert.py --state $k > results/expert/state${k}.stdout 2> results/expert/state${k}.stderr
  line=$(grep '^{"state' results/expert/state${k}.stdout || true)
  [ -n "$line" ] || { echo "scripted_expert failed on state $k (see results/expert/state${k}.stderr)"; exit 1; }
  echo "$line" >> results/expert/all_states.jsonl
done
python scripts/summarize_expert.py
