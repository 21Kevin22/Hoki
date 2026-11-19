#!/usr/bin/env bash
# Section 5 watchdog: logs nvidia-smi + host memory every 60 s; writes STOP_NEW_EPISODES when GPU or host
# memory > 95 %; warns if no episode finished for 30 min (episode runners touch results/last_episode_done).
# Usage: bash scripts/monitor.sh results/monitor.log &
LOG=${1:-results/monitor.log}; DIR=$(dirname "$LOG"); STOP=$DIR/STOP_NEW_EPISODES; DONE=$DIR/last_episode_done
while true; do
  ts=$(date -Is)
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | while IFS=, read -r i u t g; do
        echo "$ts gpu$i used=${u// /}MiB total=${t// /}MiB util=${g// /}%" >> "$LOG"
        [ $(( ${u// /} * 100 / ${t// /} )) -gt 95 ] && echo "$ts gpu$i >95%" > "$STOP"
      done
  cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null); max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
  if [[ "$max" =~ ^[0-9]+$ ]]; then
    pct=$(( cur * 100 / max )); echo "$ts host_mem=${pct}% ($((cur>>20))MiB/$((max>>20))MiB)" >> "$LOG"
    [ "$pct" -gt 95 ] && echo "$ts host >95%" > "$STOP"
  else
    free -m | awk -v ts="$ts" '/Mem:/{print ts" host_mem used="$3"MiB total="$2"MiB"}' >> "$LOG"
  fi
  if [ -f "$DONE" ] && [ $(( $(date +%s) - $(stat -c %Y "$DONE") )) -gt 1800 ]; then
    echo "$ts WARNING no episode finished for 30 min" >> "$LOG"
  fi
  sleep 60
done
