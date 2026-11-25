#!/usr/bin/env bash
# Load guard for our runs (Kevin, 2026-11-25: "GPUへの負荷を考えて実行して").
# Every 30 s: if GPU 0 memory > 90 %, GPU 0 temperature > 87 C, or host memory > 90 %, create results/STOP_NEW_EPISODES
# (run_closed_loop.py then finishes the current episode and starts no new one). Removes the file once all are back
# below 80 % / 82 C for 5 minutes. Never touches GPU 1/2 or other users' processes. Log: results/logs/gpu_guard.log
cd "$(dirname "$0")/.."
ok=0
while true; do
  read -r used total temp <<< "$(nvidia-smi -i 0 --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | tr -d ',')"
  host=$(free | awk '/Mem:/{printf "%d", ($2-$7)*100/$2}')
  g=$(( used * 100 / total ))
  echo "$(date -Is) gpu0_mem=${g}% temp=${temp}C host=${host}%" >> results/logs/gpu_guard.log
  if [ "$g" -gt 90 ] || [ "$temp" -gt 87 ] || [ "$host" -gt 90 ]; then
    [ -f results/STOP_NEW_EPISODES ] || echo "$(date -Is) STOP (gpu0_mem=${g}% temp=${temp}C host=${host}%)" >> results/logs/gpu_guard.log
    touch results/STOP_NEW_EPISODES; ok=0
  elif [ "$g" -lt 80 ] && [ "$temp" -lt 82 ] && [ "$host" -lt 80 ]; then
    ok=$((ok + 1))
    if [ $ok -ge 10 ] && [ -f results/STOP_NEW_EPISODES ] && grep -q gpu_guard results/STOP_NEW_EPISODES 2>/dev/null; then
      rm -f results/STOP_NEW_EPISODES; echo "$(date -Is) resume" >> results/logs/gpu_guard.log
    fi
  fi
  [ -f results/STOP_NEW_EPISODES ] && [ ! -s results/STOP_NEW_EPISODES ] && echo gpu_guard > results/STOP_NEW_EPISODES
  grep -q main10_DONE results/logs/main10_client.log 2>/dev/null && break
  sleep 30
done
