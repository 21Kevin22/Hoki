#!/usr/bin/env bash
# Run a list of closed-loop jobs with ONE pi0.5 server. GPU rule (PREREG_main §8): server + rendering on GPU 0 only.
# Usage: bash main/run_jobs.sh <jobfile> <name>    jobfile lines: COND TASK SPLIT IDX TAG [PARAMS_JSON]
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=$1; NAME=$2; LOG=results/logs; mkdir -p "$LOG" main/runs
bash scripts/monitor.sh results/monitor_main.log & MON=$!
bash scripts/gpu_peak_sampler.sh "main/runs/${NAME}_gpu_1s.csv" & SAMP=$!
( cd /home/ubuntu/slocal1/third_party/openpi && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    exec .venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000 ) > "$LOG/${NAME}_server.log" 2>&1 & SRV=$!
cleanup() { kill $SRV $MON $SAMP 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
until grep -q "server listening" "$LOG/${NAME}_server.log"; do kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ubuntu/slocal1/third_party/openpi_client_site:$PYTHONPATH
while read -r COND TASK SPLIT IDX TAG PARAMS; do
  [ -z "${COND:-}" ] && continue; [[ "$COND" == \#* ]] && continue
  python main/run_closed_loop.py --cond "$COND" --task "$TASK" --split "$SPLIT" --idx "$IDX" --tag "$TAG" \
     --params "${PARAMS:-{\}}" 2>> "$LOG/${NAME}_${COND}_${TASK}.err" | grep '^{' | tee -a "$LOG/${NAME}_client.log"
done < "$JOBS"
echo "${NAME}_DONE" | tee -a "$LOG/${NAME}_client.log"
