#!/usr/bin/env bash
# pi0.5 policy server for 6-4: one GPU, no JAX preallocation (rule 5).  Usage: PI05_GPU=0 bash scripts/serve_pi05.sh
cd /home/ubuntu/slocal1/third_party/openpi
export CUDA_VISIBLE_DEVICES=${PI05_GPU:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python scripts/serve_policy.py --env LIBERO --port ${PORT:-8000}
