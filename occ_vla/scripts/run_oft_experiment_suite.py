"""
run_oft_experiment_suite.py

Driver for the real A1/A2/A4/B1/B3 experiment runs (2026-08-17 discussion),
once the pipeline itself is validated (occ_vla, 2026-08-18: confirmed
end-to-end on real Kaggle infra -- see kaggle/README.md for the full fix
chain that got there). Runs run_oft_camera_dropout_eval.py once per task
(so the model is loaded fresh per task but only once per task, not once
per condition), with a fixed condition set, live-streamed output (a
buffered subprocess.run() call is indistinguishable from a hang for a
real multi-minute rollout -- confirmed the hard way this session), and
per-task resumability via --start-episode so a run can be split across
several GPU-quota-limited Kaggle sessions.

Must run with venv_oft's own python (needs LIBERO/torch/transformers) --
this script itself is stdlib-only (subprocess orchestration), so it could
technically run from either python, but there's no reason to: just invoke
it the same way as run_oft_camera_dropout_eval.py.

Usage (inside venv_oft, from occ_vla/):
    python scripts/run_oft_experiment_suite.py \
        --checkpoint /root/oft_work/checkpoints/openvla-7b-oft-libero10-vjepa \
        --tasks libero_10:8:moka_pots libero_10:9:mug_in_microwave \
        --num-trials 10 \
        --out-dir /root/oft_work/experiment_results \
        --load-in-4bit

Resuming (e.g. a fresh Kaggle session after the GPU quota reset, same
--out-dir): just re-run the SAME command. Each task's own results.json is
inspected first; if it already has >= --num-trials episodes for every
condition, that task is skipped entirely (fast). If it has SOME episodes
recorded, --start-episode is set to the minimum count found across
conditions and only the remainder is run for that task, per
run_oft_camera_dropout_eval.py's own --start-episode support -- no manual
bookkeeping needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_CONDITIONS = ["baseline", "wrist_partial", "wrist_partial_vjepa_gated", "wrist_partial_prevframe"]


def parse_task_spec(spec: str) -> tuple[str, int, str]:
    """"libero_10:8:moka_pots" -> ("libero_10", 8, "moka_pots"). The name
    is just for directory naming -- run_oft_camera_dropout_eval.py itself
    resolves the real task name from (suite, task_id) via the benchmark."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--tasks entries must be 'suite:task_id:name', got: {spec!r}")
    suite, task_id, name = parts
    return suite, int(task_id), name


def episodes_already_done(results_path: str, conditions: list[str]) -> int:
    """Minimum episode count across all requested conditions found in an
    existing results.json (0 if the file doesn't exist or a condition is
    missing entirely) -- the safe --start-episode value: only resuming
    from the condition that's furthest behind guarantees every condition
    ends up with the same episode count, matching run_oft_camera_dropout_eval.py's
    own single --num-trials/--start-episode applying uniformly to every
    --conditions entry in one invocation."""
    if not os.path.exists(results_path):
        return 0
    with open(results_path) as f:
        data = json.load(f)
    results = data.get("results", {})
    counts = [len(results.get(c, [])) for c in conditions]
    if not counts or any(c == 0 for c in counts) and len(results) < len(conditions):
        # some condition never ran at all yet -- fall back to 0 (rerun everything for this task)
        # rather than a partial/misleading resume.
        missing = [c for c in conditions if c not in results]
        if missing:
            return 0
    return min(counts) if counts else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="'suite:task_id:name' entries, e.g. libero_10:8:moka_pots")
    parser.add_argument("--num-trials", type=int, default=10, help="per condition, per task -- this project's own established n=10 convention (see occ_vla/CLAUDE.md's own repeated 'small-n results have not reliably replicated' caution for why not less)")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--out-dir", required=True, help="base directory -- results.json and step logs are written under out-dir/<task_name>/")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--debounce-k", type=int, default=3)
    parser.add_argument("--s-occ-source", default="oracle", choices=["oracle", "probe"])
    parser.add_argument("--measure-latency", action="store_true", default=True, help="on by default -- A1 needs it and the CUDA-sync overhead is small relative to a whole rollout; pass --no-measure-latency to disable")
    parser.add_argument("--no-measure-latency", dest="measure_latency", action="store_false")
    parser.add_argument("--occ-vla-dir", default=".", help="cwd to run run_oft_camera_dropout_eval.py from (must contain scripts/)")
    parser.add_argument("--python", default=sys.executable, help="interpreter to run run_oft_camera_dropout_eval.py with -- defaults to whatever ran THIS script, override if invoking from a different env than venv_oft")
    parser.add_argument("--dry-run", action="store_true", help="print what would run for each task without launching anything -- useful for sanity-checking --tasks/--conditions/resume state before spending GPU quota")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for spec in args.tasks:
        suite, task_id, name = parse_task_spec(spec)
        task_dir = os.path.join(args.out_dir, name)
        os.makedirs(task_dir, exist_ok=True)
        results_path = os.path.join(task_dir, "results.json")
        log_steps_dir = os.path.join(task_dir, "steplogs")

        start_episode = episodes_already_done(results_path, args.conditions)
        if start_episode >= args.num_trials:
            print(f"=== {name} ({suite} task {task_id}): already has {start_episode}/{args.num_trials} episodes for every condition -- skipping ===")
            continue

        print(f"\n=== {name} ({suite} task {task_id}): running episodes [{start_episode}, {args.num_trials}) ===")

        cmd = [
            args.python, "scripts/run_oft_camera_dropout_eval.py",
            "--task-suite", suite,
            "--task-id", str(task_id),
            "--num-trials", str(args.num_trials),
            "--start-episode", str(start_episode),
            "--checkpoint", args.checkpoint,
            "--conditions", *args.conditions,
            "--log-steps-dir", log_steps_dir,
            "--s-occ-source", args.s_occ_source,
            "--debounce-k", str(args.debounce_k),
            "--results-path", results_path,
        ]
        if args.load_in_4bit:
            cmd.append("--load-in-4bit")
        if args.load_in_8bit:
            cmd.append("--load-in-8bit")
        if args.measure_latency:
            cmd.append("--measure-latency")

        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Jupyter's own MPLBACKEND leaks into subprocess env otherwise -- see kaggle/README.md
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd, cwd=args.occ_vla_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            print(f"  !! {name}: exited {proc.returncode} -- moving on to the next task rather than aborting the whole suite. Re-run this same command later to retry/resume this task specifically (its partial results, if any, are preserved).")

    print("\n=== Suite complete ===")
    for spec in args.tasks:
        suite, task_id, name = parse_task_spec(spec)
        results_path = os.path.join(args.out_dir, name, "results.json")
        if not os.path.exists(results_path):
            print(f"  {name}: no results (never completed a single episode)")
            continue
        with open(results_path) as f:
            data = json.load(f)
        summary = {c: f"{sum(1 for e in eps if e['success'])}/{len(eps)}" for c, eps in data.get("results", {}).items()}
        print(f"  {name}: {summary}")


if __name__ == "__main__":
    main()
