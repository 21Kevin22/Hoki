#!/usr/bin/env python3
"""occ_vla addition (2026-08-27): queued post-training analysis pipeline for the
v3 distillation-imitation LoRA run (distillation_lora_task1_n1000_v3), covering the
user's 3 requested milestones:

  Milestone 1 -- final loss-curve analysis: parse train_loss/held_out_loss from the
    training log, identify the held_out_loss minima (golden-checkpoint candidates)
    vs. any late-training re-divergence (overfitting boundary).
  Milestone 2 -- per-checkpoint rollout (n=10) success rate + avg episode length
    (done_step among successes), condition="baseline" (skill/capacity check --
    is grasping still intact, is the policy taking a more direct/shorter path).
  Milestone 3 -- per-checkpoint rollout (n=10), condition="proactive_avoidance_depth"
    (the same zero-privileged CBF teacher used to collect distillation data),
    counting n_correction_applied to see whether the distilled policy needs LESS
    external CBF intervention than an undistilled reference run of the same
    condition (autonomous-avoidance emergence check).

This script:
  1. Waits for the training process (PID passed via --train-pid) to exit.
  2. Parses the training log for every "step N: train_loss=... held_out_loss=..."
     line, ranks checkpoint-aligned steps (multiples of --checkpoint-every) by
     held_out_loss, and writes analysis_candidates.json.
  3. Builds a job queue: for each of the top-K candidates (by held_out_loss) plus
     the final step, run (a) condition=baseline n=10 and (b) condition=
     proactive_avoidance_depth n=10. Also runs ONE undistilled reference job
     (no --load-distillation-lora) under proactive_avoidance_depth as the
     milestone-3 control.
  4. Runs the queue with up to --max-parallel concurrent subprocesses, one GPU
     each (round-robin over --gpus).
  5. Aggregates every completed job's task1.json into a final markdown report.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OFT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "thirdparty", "openvla-oft")
VENV_PY = os.path.join(os.path.dirname(SCRIPT_DIR), ".venv_openvla_oft", "bin", "python3")
LIBERO_PP = os.path.join(os.path.dirname(SCRIPT_DIR), "thirdparty", "LIBERO")
CKPT = os.path.join(os.path.dirname(SCRIPT_DIR), "checkpoints", "openvla-7b-oft-libero10-vjepa")

STEP_RE = re.compile(r"^step (\d+): train_loss=([\d.]+)")
HELD_RE = re.compile(r"held_out_loss=([\d.]+)")


def parse_log(log_path):
    """Returns list of dicts: {step, train_loss, held_out_loss (or None)}.

    occ_vla note (2026-08-27): the original single-regex version anchored
    held_out_loss's capture group immediately before end-of-line ($), but
    every real log line has trailing "(uid=...)" text AFTER held_out_loss's
    value -- so that group could never actually match on real data, and
    every row silently came back with held_out_loss=None. Fixed by matching
    the step/train_loss prefix and the (optional, anywhere-in-line)
    held_out_loss value as two independent regexes instead of one anchored
    pattern.
    """
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = STEP_RE.match(line)
            if m:
                step = int(m.group(1))
                train_loss = float(m.group(2))
                hm = HELD_RE.search(line)
                held_out = float(hm.group(1)) if hm else None
                rows.append({"step": step, "train_loss": train_loss, "held_out_loss": held_out})
    return rows


def analyze(rows, checkpoint_every, top_k):
    ckpt_steps = sorted(s for s in glob.glob(
        os.path.join(os.path.dirname(SCRIPT_DIR), "scripts", "distillation_lora_task1_n1000_v3", "step*")
    ))
    # Map step -> held_out_loss at exactly that step (if logged)
    held = {r["step"]: r["held_out_loss"] for r in rows if r["held_out_loss"] is not None}
    max_step = max((r["step"] for r in rows), default=0)
    candidate_steps = [s for s in range(checkpoint_every, max_step + 1, checkpoint_every) if s in held]
    ranked = sorted(candidate_steps, key=lambda s: held[s])
    summary = {
        "max_step_logged": max_step,
        "n_rows": len(rows),
        "held_out_loss_by_checkpoint_step": {s: held[s] for s in candidate_steps},
        "ranked_by_held_out_loss_ascending": [(s, held[s]) for s in ranked],
        "top_k_candidates": ranked[:top_k],
        "final_step": candidate_steps[-1] if candidate_steps else None,
        # simple overfitting-boundary heuristic: last local minimum before a
        # sustained (3+ consecutive checkpoint-steps) rise
        "late_divergence_detected": False,
    }
    if len(candidate_steps) >= 4:
        vals = [held[s] for s in candidate_steps]
        for i in range(len(vals) - 3):
            if vals[i] < vals[i + 1] < vals[i + 2] < vals[i + 3]:
                summary["late_divergence_detected"] = True
                summary["divergence_starts_after_step"] = candidate_steps[i]
                break
    return summary


def gpu_is_free(gpu, threshold_mib=3000):
    """occ_vla note (2026-08-27): real bug found -- the orchestrator previously
    trusted its own `running` dict as the sole source of truth for which GPUs
    were busy, but leftover standalone jobs launched OUTSIDE this process
    (e.g. by a prior, killed instance of this same script) occupy GPU memory
    the orchestrator doesn't know about. Blindly launching a second 7B-model
    rollout onto an already-occupied 24GB GPU OOM'd twice in a row (step450_*,
    step950_* both crashed before writing even run_config.json). Always check
    real nvidia-smi memory before launching, not just internal bookkeeping."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)]
        )
        used = int(out.decode().strip())
        return used < threshold_mib
    except Exception as e:
        print(f"[queue] WARNING: nvidia-smi check failed for GPU{gpu}: {e} -- assuming NOT free", flush=True)
        return False


def run_rollout(label, checkpoint_step, condition, n_episodes, gpu, results_dir, log_path,
                 lora_dir=None):
    cmd = [
        VENV_PY, "-u", os.path.join(SCRIPT_DIR, "run_libero_occluded_oracle_headroom.py"),
        "--task-ids", "1", "--n-episodes", str(n_episodes), "--conditions", condition,
        "--checkpoint", CKPT, "--results-dir", results_dir,
    ]
    if lora_dir is not None:
        cmd += ["--load-distillation-lora", lora_dir]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = LIBERO_PP
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=OFT_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc


def collect_result(results_dir):
    path = os.path.join(OFT_DIR, results_dir, "task1.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    out = {}
    # occ_vla note (2026-08-27): real task1.json shape is
    # {"task_id":int, "task_description":str, "occluder_names":[str,...],
    #  "results": {condition: [episode_dict, ...]}} -- NOT a flat
    # {condition: [episodes]} dict at the top level (the original version here
    # iterated d.items() directly and crashed on the "occluder_names" list of
    # strings). Also: the per-episode CBF-intervention count field is
    # "proactive_correction_applied_count" (steps within replanned chunks where
    # the CBF safety correction actually fired), not "n_correction_applied"
    # (a different, always-0-in-this-data reactive-trigger counter).
    results = d.get("results", {}) if isinstance(d, dict) else {}
    for cond, eps in results.items():
        if not isinstance(eps, list):
            continue
        n_succ = sum(1 for e in eps if isinstance(e, dict) and e.get("success"))
        succ_steps = [e.get("done_step") for e in eps if isinstance(e, dict) and e.get("success")]
        n_corrections = sum((e.get("proactive_correction_applied_count", 0) or 0) for e in eps if isinstance(e, dict))
        out[cond] = {
            "n_episodes": len(eps),
            "n_success": n_succ,
            "success_rate": n_succ / len(eps) if eps else None,
            "avg_episode_len_success": (sum(succ_steps) / len(succ_steps)) if succ_steps else None,
            "total_n_correction_applied": n_corrections,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pid", type=int, required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="post_training_analysis")
    ap.add_argument("--run-tag", default=None,
                     help="occ_vla fix (2026-08-27): a real bug in the first version of this script "
                          "reused bare results_dir names like 'post_analysis_step450_baseline' across "
                          "DIFFERENT training runs (e.g. v3 and v4_interleave both have a 'step450') -- "
                          "the dedup-reuse logic then silently served the WRONG run's old results for any "
                          "step number two runs happened to share. Every results_dir is now namespaced by "
                          "this tag. Defaults to the basename of --adapter-dir, which is unique per run "
                          "in every invocation so far, but pass explicitly if that's ever not true.")
    args = ap.parse_args()
    run_tag = args.run_tag or os.path.basename(os.path.normpath(args.adapter_dir))

    print(f"[queue] waiting for training pid {args.train_pid} to exit...", flush=True)
    while True:
        try:
            os.kill(args.train_pid, 0)
        except OSError:
            break
        time.sleep(20)
    print("[queue] training process exited. Parsing log...", flush=True)

    rows = parse_log(args.train_log)
    summary = analyze(rows, args.checkpoint_every, args.top_k)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "loss_curve_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[milestone-1] held_out_loss ranking (ascending, best first): {summary['ranked_by_held_out_loss_ascending']}", flush=True)
    print(f"[milestone-1] late_divergence_detected={summary['late_divergence_detected']}", flush=True)

    candidates = sorted(set(summary["top_k_candidates"] + ([summary["final_step"]] if summary["final_step"] else [])))
    print(f"[queue] selected candidate steps for rollout: {candidates}", flush=True)

    # Build job queue: (label, results_dir, condition, lora_dir_or_None)
    jobs = []
    # occ_vla note (2026-08-27): generalized pre-existing-job dedup -- ANY job
    # (not just the undistilled reference) whose results_dir already exists on
    # disk (from a prior, now-superseded launch of this script) is skipped from
    # the launch queue; if it already has task1.json it's collected immediately,
    # otherwise it's tracked in `preexisting_pending` to be waited on at the end.
    preexisting_pending = []  # list of (label, results_dir)

    def maybe_queue_job(label, results_dir, condition, lora_dir):
        d = os.path.join(OFT_DIR, results_dir)
        j = os.path.join(d, "task1.json")
        if os.path.exists(j):
            done_results[label] = collect_result(results_dir)
            print(f"[queue] {label}: results_dir already has task1.json -- reusing, not re-launching", flush=True)
        elif os.path.isdir(d):
            preexisting_pending.append((label, results_dir))
            print(f"[queue] {label}: results_dir already exists (in-flight) -- will wait, not re-launching", flush=True)
        else:
            jobs.append((label, results_dir, condition, lora_dir))

    done_results = {}
    for step in candidates:
        lora_dir = os.path.join(args.adapter_dir, f"step{step}")
        maybe_queue_job(f"{run_tag}_step{step}_baseline", f"post_analysis_{run_tag}_step{step}_baseline", "baseline", lora_dir)
        maybe_queue_job(f"{run_tag}_step{step}_proactive_depth", f"post_analysis_{run_tag}_step{step}_proactive_depth", "proactive_avoidance_depth", lora_dir)
    # milestone-3 control: undistilled reference under proactive_avoidance_depth.
    # occ_vla note: deliberately NOT namespaced by run_tag -- this condition has
    # no --load-distillation-lora at all, so it's identical regardless of which
    # training run is being analyzed; reusing one result across runs (rather
    # than re-running an identical control every time) is correct, not a bug.
    maybe_queue_job("undistilled_reference_proactive_depth", "post_analysis_undistilled_reference", "proactive_avoidance_depth", None)

    print(f"[queue] total jobs to launch: {len(jobs)}, pre-existing to wait on: {len(preexisting_pending)}, already collected: {len(done_results)}", flush=True)

    running = {}  # gpu -> (proc, label, results_dir)
    pending = list(jobs)
    gpu_cycle = list(args.gpus)

    def launch_next(gpu):
        if not pending:
            return False
        if gpu in running:
            return False  # occ_vla fix: never double-launch onto a GPU already tracked as busy
        if not gpu_is_free(gpu):
            print(f"[queue] GPU{gpu} not actually free yet (nvidia-smi) -- deferring launch, will retry", flush=True)
            return False
        label, results_dir, condition, lora_dir = pending.pop(0)
        log_path = os.path.join("/tmp", f"post_analysis_{label}.log")
        print(f"[queue] launching {label} (condition={condition}, lora={lora_dir}) on GPU{gpu}", flush=True)
        proc = run_rollout(label, None, condition, args.n_episodes, gpu, results_dir, log_path, lora_dir)
        running[gpu] = (proc, label, results_dir)
        return True

    for gpu in gpu_cycle:
        launch_next(gpu)

    while running or pending:
        time.sleep(20)
        # occ_vla fix: also retry launching onto any gpu_cycle GPU not currently
        # tracked as running (covers the "deferred, GPU was busy" case above,
        # and the case where all gpu_cycle slots got consumed by pending==[]
        # at startup but a GPU frees up later for a job that arrived after).
        for gpu in gpu_cycle:
            if gpu not in running and pending:
                launch_next(gpu)
        for gpu in list(running.keys()):
            proc, label, results_dir = running[gpu]
            if proc.poll() is not None:
                result = collect_result(results_dir)
                done_results[label] = result
                print(f"[queue] job {label} finished (exit={proc.returncode}): {result}", flush=True)
                del running[gpu]
                launch_next(gpu)

    for label, results_dir in preexisting_pending:
        if label in done_results:
            continue
        json_path = os.path.join(OFT_DIR, results_dir, "task1.json")
        print(f"[queue] waiting for pre-existing job {label} ({results_dir}) to finish...", flush=True)
        while not os.path.exists(json_path):
            time.sleep(20)
        result = collect_result(results_dir)
        done_results[label] = result
        print(f"[queue] pre-existing job {label} finished: {result}", flush=True)

    print("[queue] ALL JOBS DONE. Writing final report...", flush=True)
    with open(os.path.join(args.out_dir, "all_job_results.json"), "w") as f:
        json.dump(done_results, f, indent=2)

    # Build markdown report
    lines = ["# Post-training analysis report", ""]
    lines.append("## Milestone 1: loss curve")
    lines.append(f"- held_out_loss ranking (best first): {summary['ranked_by_held_out_loss_ascending']}")
    lines.append(f"- late_divergence_detected: {summary['late_divergence_detected']}")
    lines.append("")
    lines.append("## Milestone 2+3: per-checkpoint rollout results")
    lines.append("| job | condition | n_success/n_episodes | avg_episode_len(success) | total_n_correction_applied |")
    lines.append("|---|---|---|---|---|")
    for label, result in done_results.items():
        if not result:
            lines.append(f"| {label} | - | (no data) | - | - |")
            continue
        for cond, stats in result.items():
            lines.append(
                f"| {label} | {cond} | {stats['n_success']}/{stats['n_episodes']} | "
                f"{stats['avg_episode_len_success']} | {stats['total_n_correction_applied']} |"
            )
    with open(os.path.join(args.out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print("[queue] report written to " + os.path.join(args.out_dir, "REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
