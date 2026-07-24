#!/usr/bin/env bash
# Starts pi05_worker, waits for it to load, runs the given command, then
# stops the worker (freeing its GPU) when the command exits -- whether
# it succeeds, fails, or is interrupted. Standing preference (per user,
# 2026-07-16): don't leave GPU workers running after an experiment
# finishes; use this wrapper for future GPU-backed runs instead of
# manually starting/stopping workers around each one.
#
# Usage: scripts/run_with_gpu_cleanup.sh <command...>
# Example: scripts/run_with_gpu_cleanup.sh python3 scripts/run_self_occlusion_pipeline.py --episodes 10
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

(
  source third_party/openpi/.venv/bin/activate
  # 2026-07-23: JAX defaults to preallocating memory on every visible
  # GPU even though pi0.5 only computes on one -- confirmed via
  # /proc/<pid>/environ this was silently starving other GPUs (SD/SEINE
  # workers) of VRAM. Pin to one device and disable the preallocate.
  export CUDA_VISIBLE_DEVICES="${PI05_WORKER_CUDA_DEVICE:-0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  exec python3 scripts/_workers/pi05_worker.py
) > logs/pi05_worker_wrapper.log 2>&1 &
WORKER_PID=$!

echo "[wrapper] started pi05_worker (pid=$WORKER_PID), waiting for it to load..."
until grep -q "policy loaded" logs/pi05_worker_wrapper.log 2>/dev/null; do
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "[wrapper] pi05_worker died before loading -- see logs/pi05_worker_wrapper.log"
    exit 1
  fi
  sleep 2
done
echo "[wrapper] pi05_worker ready"

"$@"
EXIT_CODE=$?

echo "[wrapper] command exited (code=$EXIT_CODE), stopping pi05_worker (pid=$WORKER_PID)"
kill "$WORKER_PID" 2>/dev/null
wait "$WORKER_PID" 2>/dev/null

exit "$EXIT_CODE"
