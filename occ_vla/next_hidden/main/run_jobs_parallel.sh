#!/usr/bin/env bash
# Like run_jobs.sh but N env workers share ONE pi0.5 server (PREREG_main §7-§8): server + rendering on GPU 0 only.
# Usage: bash main/run_jobs_parallel.sh <jobfile> <name> [N=4]   jobfile lines: COND TASK SPLIT IDX TAG [PARAMS_JSON]
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=$1; NAME=$2; N=${3:-4}; LOG=results/logs; mkdir -p "$LOG" main/runs
bash scripts/monitor.sh results/monitor_main.log & MON=$!
bash scripts/gpu_peak_sampler.sh "main/runs/${NAME}_gpu_1s.csv" & SAMP=$!
( cd /home/ubuntu/slocal1/third_party/openpi && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    exec .venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000 ) > "$LOG/${NAME}_server.log" 2>&1 & SRV=$!
cleanup() { kill $SRV $MON $SAMP 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
until grep -q "server listening" "$LOG/${NAME}_server.log"; do kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ubuntu/slocal1/third_party/openpi_client_site:$PYTHONPATH
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NO_BRANCH=${NO_BRANCH:-0} SCRIPT=${SCRIPT:-main/run_closed_loop.py}
run_one() {
  read -r COND TASK SPLIT IDX TAG PARAMS <<< "$1"
  python "${SCRIPT:-main/run_closed_loop.py}" --cond "$COND" --task "$TASK" --split "$SPLIT" --idx "$IDX" --tag "$TAG" \
     --params "${PARAMS:-{\}}" 2>> "results/logs/${NAME}_${COND}_${TASK}_${TAG}.err" | grep '^{'
}
export -f run_one; export NAME
grep -v -E '^\s*(#|$)' "$JOBS" | xargs -d '\n' -P "$N" -I{} bash -c 'run_one "$@"' _ {} | tee -a "$LOG/${NAME}_client.log"
echo "${NAME}_DONE" | tee -a "$LOG/${NAME}_client.log"
