#!/usr/bin/env bash
# Week 1 dev pilot (PREREG_main.md §2): C0 on 10 dev init states x 7 candidate tasks, ONE worker.
# GPU rule (§8): policy server AND rendering on GPU 0 only; GPU 1 unused; GPU 2 not used.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/logs; mkdir -p "$LOG" main/runs
bash scripts/monitor.sh results/monitor_main.log & MON=$!
bash scripts/gpu_peak_sampler.sh main/runs/pilot_gpu_1s.csv & SAMP=$!
( cd /home/ubuntu/slocal1/third_party/openpi && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    exec .venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000 ) > "$LOG/pilot_server.log" 2>&1 & SRV=$!
cleanup() { kill $SRV $MON $SAMP 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
until grep -q "server listening" "$LOG/pilot_server.log"; do
  kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
source /home/ubuntu/slocal1/third_party/stowsafe_env.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ubuntu/slocal1/third_party/openpi_client_site:$PYTHONPATH
for T in T1 T2 T3 T4 T5 T6 T7; do
  python main/run_closed_loop.py --cond C0 --task $T --split dev --idx 0-9 2>> "$LOG/pilot_client_$T.err" | tee -a "$LOG/pilot_client.log"
done
echo PILOT_DONE | tee -a "$LOG/pilot_client.log"
