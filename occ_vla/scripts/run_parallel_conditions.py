"""Generic multi-GPU launcher for this project's condition-comparison
scripts (run_t08_soft_pipeline.py, run_self_occlusion_pipeline.py).

Per user preference (2026-07-16): before starting, check whether other
GPUs are actually idle (no one else's job holding power/memory) -- if
at least 2 are free, run each condition as its own OS process on its
own GPU with its own pi0.5 worker, concurrently, instead of
serializing everything through one GPU/worker. Falls back to the
existing single-worker sequential path
(scripts/run_with_gpu_cleanup.sh) if fewer than 2 GPUs are free, so
this never assumes exclusive access to a shared machine.

"Free" = memory.used < FREE_MEM_MIB and utilization.gpu < FREE_UTIL_PCT,
checked immediately before launching -- a live check, not a
reservation, so it only avoids *assuming* GPUs are free; it doesn't
lock them against a real concurrent user appearing after the check.

Usage:
  python3 scripts/run_parallel_conditions.py \
      --script scripts/run_self_occlusion_pipeline.py \
      --conditions baseline soft_pipeline \
      -- --episodes 10 --max-steps 300

Everything after `--` is passed through to the target script unchanged
(minus --conditions/--results-path, which this launcher manages).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FREE_MEM_MIB = 500
FREE_UTIL_PCT = 5
WORKER_LOAD_TIMEOUT_S = 180


def free_gpu_indices() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    free = []
    for line in out.strip().splitlines():
        idx, mem_used, util = (int(x.strip()) for x in line.split(","))
        if mem_used < FREE_MEM_MIB and util < FREE_UTIL_PCT:
            free.append(idx)
    return free


def start_worker(gpu: int, rpc_dir: Path, log_path: Path) -> subprocess.Popen:
    rpc_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"source {_ROOT}/third_party/openpi/.venv/bin/activate && "
        f"CUDA_VISIBLE_DEVICES={gpu} PI05_WORKER_RPC_DIR={rpc_dir} "
        f"python3 {_ROOT}/scripts/_workers/pi05_worker.py"
    )
    log_file = log_path.open("w")
    return subprocess.Popen(["bash", "-c", cmd], stdout=log_file, stderr=subprocess.STDOUT, cwd=str(_ROOT))


def wait_for_worker(log_path: Path, proc: subprocess.Popen, timeout_s: float = WORKER_LOAD_TIMEOUT_S) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"worker process exited early (code={proc.returncode}) -- see {log_path}")
        if log_path.exists() and "policy loaded" in log_path.read_text():
            return
        time.sleep(2)
    raise TimeoutError(f"worker did not report 'policy loaded' within {timeout_s}s -- see {log_path}")


def run_parallel(script: str, conditions: list[str], extra_args: list[str]) -> None:
    free = free_gpu_indices()
    print(f"[launcher] free GPUs: {free}", flush=True)
    assigned_gpus = free[: len(conditions)]
    print(f"[launcher] running {len(conditions)} condition(s) in parallel on GPUs {assigned_gpus}", flush=True)

    workers = []
    per_condition_results = {}
    for condition, gpu in zip(conditions, assigned_gpus):
        rpc_dir = _ROOT / ".rpc" / f"pi05_parallel_{gpu}"
        worker_log = _ROOT / "logs" / f"pi05_worker_parallel_{gpu}.log"
        proc = start_worker(gpu, rpc_dir, worker_log)
        workers.append((gpu, rpc_dir, worker_log, proc))

    procs = []
    try:
        for condition, (gpu, rpc_dir, worker_log, worker_proc) in zip(conditions, workers):
            wait_for_worker(worker_log, worker_proc)
            print(f"[launcher] pi05 worker ready on GPU {gpu} for condition={condition!r}", flush=True)
            results_path = _ROOT / f".parallel_results_{condition}.json"
            per_condition_results[condition] = results_path
            env_prefix = f"OCC_VLA_PI05_RPC_DIR={rpc_dir}"
            cmd = f"{env_prefix} python3 {script} --conditions {condition} --results-path {results_path} " + " ".join(extra_args)
            script_log = _ROOT / "logs" / f"parallel_{condition}.log"
            log_file = script_log.open("w")
            procs.append((condition, subprocess.Popen(["bash", "-c", cmd], stdout=log_file, stderr=subprocess.STDOUT, cwd=str(_ROOT))))

        for condition, proc in procs:
            rc = proc.wait()
            print(f"[launcher] condition={condition!r} finished (exit={rc})", flush=True)
    finally:
        for gpu, rpc_dir, worker_log, worker_proc in workers:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker_proc.kill()
        print("[launcher] all workers stopped, GPUs released", flush=True)

    merged = []
    for condition, path in per_condition_results.items():
        if path.exists():
            merged.extend(json.loads(path.read_text()))
    merged.sort(key=lambda r: (r["condition"], r["episode"]))
    print(json.dumps(merged, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Target condition-comparison script, e.g. scripts/run_self_occlusion_pipeline.py")
    parser.add_argument("--conditions", nargs="+", required=True)
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    free = free_gpu_indices()
    if len(free) >= 2 and len(args.conditions) >= 2:
        run_parallel(args.script, args.conditions, extra_args)
    else:
        print(
            f"[launcher] only {len(free)} free GPU(s) -- falling back to serial single-worker run "
            "(scripts/run_with_gpu_cleanup.sh)",
            flush=True,
        )
        cmd = [
            "bash",
            str(_ROOT / "scripts" / "run_with_gpu_cleanup.sh"),
            "python3",
            args.script,
            "--conditions",
            *args.conditions,
            *extra_args,
        ]
        sys.exit(subprocess.run(cmd, cwd=str(_ROOT)).returncode)


if __name__ == "__main__":
    main()
