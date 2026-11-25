#!/usr/bin/env bash
# After (10): (12) H3 visibility diag, then (11) with visibility logging. One job file at a time (GPU 0 only),
# each with its own pi0.5 server (run_jobs_parallel.sh), under the load guard.
cd "$(dirname "$0")/.."
until grep -q main10_DONE results/logs/main10_client.log 2>/dev/null; do sleep 60; done
sleep 20
for J in vis12 main11; do
  rm -f results/STOP_NEW_EPISODES
  ( while ! grep -q "${J}_DONE" results/logs/${J}_client.log 2>/dev/null; do
      read -r used total temp <<< "$(nvidia-smi -i 0 --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | tr -d ',')"
      host=$(free | awk '/Mem:/{printf "%d", ($2-$7)*100/$2}'); g=$(( used * 100 / total ))
      echo "$(date -Is) [$J] gpu0_mem=${g}% temp=${temp}C host=${host}%" >> results/logs/gpu_guard.log
      if [ "$g" -gt 90 ] || [ "$temp" -gt 87 ] || [ "$host" -gt 90 ]; then echo gpu_guard > results/STOP_NEW_EPISODES; echo "$(date -Is) STOP" >> results/logs/gpu_guard.log; fi
      sleep 30; done ) &
  NO_BRANCH=1 SCRIPT=main/run_vislog.py bash main/run_jobs_parallel.sh main/jobs_${J}.txt $J 4 > results/logs/${J}_launch.log 2>&1
  wait
done
echo CHAIN_DONE >> results/logs/main11_client.log
