#!/usr/bin/env bash
# 1-second GPU memory sampler (the 60-s monitor is too coarse for a peak). Usage: bash scripts/gpu_peak_sampler.sh out.csv
while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$1"; sleep 1; done
