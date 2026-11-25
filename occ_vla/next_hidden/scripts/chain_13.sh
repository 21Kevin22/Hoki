#!/usr/bin/env bash
# PREREG (13), after (11): smoke (1 ep each, not counted; stop if any real exception), H3b gate (C0 dev 0-9),
# then H3 eval LOOKFB / LOOK32 / LOOK32FB and, if the gate passes (box moved in >= 2/10), H3b eval C0 + LOOK32FB.
cd "$(dirname "$0")/.."
until grep -q CHAIN_DONE results/logs/main11_client.log 2>/dev/null && [ -f main/init_states_hidden/H3b.npz ]; do sleep 60; done
guard() { while ! grep -q "${1}_DONE" results/logs/${1}_client.log 2>/dev/null; do
    read -r used total temp <<< "$(nvidia-smi -i 0 --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | tr -d ',')"
    host=$(free | awk '/Mem:/{printf "%d", ($2-$7)*100/$2}'); g=$(( used * 100 / total ))
    echo "$(date -Is) [$1] gpu0_mem=${g}% temp=${temp}C host=${host}%" >> results/logs/gpu_guard.log
    if [ "$g" -gt 90 ] || [ "$temp" -gt 87 ] || [ "$host" -gt 90 ]; then echo gpu_guard > results/STOP_NEW_EPISODES; echo "$(date -Is) STOP" >> results/logs/gpu_guard.log; fi
    sleep 30; done; }
run() { rm -f results/STOP_NEW_EPISODES; guard $1 & NO_BRANCH=1 SCRIPT=main/run_vislog.py bash main/run_jobs_parallel.sh main/jobs_$1.txt $1 ${2:-4} > results/logs/$1_launch.log 2>&1; wait; }
run smoke13 3
n=$(ls main/runs/*_smoke13/T1_hid_dev00.json 2>/dev/null | wc -l)
if [ "$n" -lt 3 ]; then echo "smoke13 incomplete ($n/3), stopping" >> results/logs/chain_13.log; exit 1; fi
run gate13
moved=$(python3 -c "
import json,glob;print(sum(json.load(open(f))['collision']['hidden']['box_moved_1mm'] for f in glob.glob('main/runs/C0_gate13/T1_hidb_dev*.json')))")
echo "gate13 box moved $moved/10" >> results/logs/chain_13.log
cp main/jobs_main13.txt main/jobs_run13.txt
if [ "$moved" -ge 2 ]; then
  for c in C0 LOOK32FB; do for s in 0 5 10 15 20 25 30 35; do echo "$c T1@hidb eval $s-$((s+4)) main13 {}" >> main/jobs_run13.txt; done; done
fi
run run13
echo CHAIN13_DONE >> results/logs/chain_13.log
