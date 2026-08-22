"""occ_vla addition (2026-08-22), per user's Method/Experiments metrics
request: Trigger Rate / Recovery Success Rate / Failure Mode
classification from existing scripted_recovery_after_contact logs
(baseline + scripted_recovery_after_contact conditions, paired by
episode index / init_state).

Definitions (stated explicitly since none of these are built-in fields):
  - "baseline would fail": the PAIRED baseline episode (same episode
    index) has success=False. Trigger Rate is measured only over this
    subset -- an episode baseline already succeeds on says nothing
    about whether the recovery mechanism is needed/working.
  - Trigger Rate: among baseline-would-fail episodes, fraction where
    scripted_recovery_after_contact's own `reactive_triggered=True`
    (the anomalous-arm-link-contact trigger actually fired).
  - Recovery Success Rate: among TRIGGERED episodes only, fraction with
    scripted_recovery_after_contact's own `success=True`.
  - Failure Mode (HEURISTIC, not ground truth -- no direct object-
    position tracking exists in these logs): for triggered-but-failed
    episodes, classify via:
      "timeout" -- termination_reason == "timeout" AND neither of the
          two heuristics below fire (the default/residual bucket).
      "stuck" -- the last N replan-steps' eef_speed_since_last_replan
          are ALL below a small threshold (near-zero motion plateau
          right up to timeout) -- approximates an IK/motion stall.
      "dropped (approx.)" -- gripper_qpos transitions from a closed
          state to a clearly-open state at some point BEFORE the final
          quarter of the episode, and stays open through the end
          (approximates "released the object early, then failed to
          recover it") -- this is a real proxy, not verified against
          ground-truth object position; stated as approximate
          throughout.
    An episode can only receive one label; checked in the order
    stuck -> dropped -> timeout (residual).
"""
import argparse
import json

STUCK_SPEED_THRESHOLD = 0.005  # meters/replan-step, near-zero motion
STUCK_WINDOW = 8  # last N replan-step log entries
GRIPPER_OPEN_THRESHOLD = 0.03  # qpos sum threshold, open vs closed (both fingers)
DROPPED_BEFORE_FRAC = 0.75  # "early" = before this fraction of the episode


def classify_failure_mode(episode_result):
    plog = episode_result["proprio_log"]
    if not plog:
        return "unknown (no proprio_log)"
    termination = episode_result["termination_reason"]

    # "stuck": last STUCK_WINDOW entries all near-zero eef speed
    tail = plog[-STUCK_WINDOW:]
    speeds = [p.get("eef_speed_since_last_replan") for p in tail]
    if len(speeds) == STUCK_WINDOW and all(s is not None and s < STUCK_SPEED_THRESHOLD for s in speeds):
        return "stuck (heuristic: near-zero motion, last {} replan-steps)".format(STUCK_WINDOW)

    # "dropped (approx.)": gripper goes from closed to open before the
    # final quarter of the episode and stays open
    total_steps = plog[-1]["t"]
    cutoff_t = total_steps * DROPPED_BEFORE_FRAC
    was_closed = False
    opened_early = False
    for p in plog:
        gq = p.get("gripper_qpos")
        if gq is None:
            continue
        is_open = sum(abs(x) for x in gq) > GRIPPER_OPEN_THRESHOLD
        if not is_open:
            was_closed = True
        elif was_closed and p["t"] < cutoff_t:
            opened_early = True
    if opened_early:
        # confirm it STAYS open through the end (not a brief re-grasp)
        final_gq = plog[-1].get("gripper_qpos")
        if final_gq is not None and sum(abs(x) for x in final_gq) > GRIPPER_OPEN_THRESHOLD:
            return "dropped (approx.: gripper opened before t={:.0f}/{}, stayed open)".format(cutoff_t, total_steps)

    return f"timeout (residual, termination_reason={termination})"


def analyze_task(path, task_json_name):
    d = json.load(open(f"{path}/{task_json_name}"))
    baseline = {r["episode"]: r for r in d["results"]["baseline"]}
    recovery = {r["episode"]: r for r in d["results"]["scripted_recovery_after_contact"]}
    common_eps = sorted(set(baseline) & set(recovery))

    baseline_fail_eps = [ep for ep in common_eps if not baseline[ep]["success"]]
    triggered_eps = [ep for ep in baseline_fail_eps if recovery[ep]["reactive_triggered"]]
    triggered_and_succeeded = [ep for ep in triggered_eps if recovery[ep]["success"]]
    triggered_and_failed = [ep for ep in triggered_eps if not recovery[ep]["success"]]

    trigger_rate = len(triggered_eps) / len(baseline_fail_eps) if baseline_fail_eps else float("nan")
    recovery_rate = len(triggered_and_succeeded) / len(triggered_eps) if triggered_eps else float("nan")

    failure_modes = {}
    for ep in triggered_and_failed:
        mode = classify_failure_mode(recovery[ep])
        failure_modes.setdefault(mode, []).append(ep)

    return dict(
        n_episodes=len(common_eps),
        n_baseline_success=sum(1 for ep in common_eps if baseline[ep]["success"]),
        n_baseline_fail=len(baseline_fail_eps),
        baseline_fail_episodes=baseline_fail_eps,
        n_triggered=len(triggered_eps),
        triggered_episodes=triggered_eps,
        trigger_rate=trigger_rate,
        n_triggered_and_succeeded=len(triggered_and_succeeded),
        n_triggered_and_failed=len(triggered_and_failed),
        recovery_success_rate=recovery_rate,
        failure_modes={k: v for k, v in failure_modes.items()},
        # also report the naive aggregate for reference, WITH the
        # explicit warning this project has repeatedly needed: not
        # attributable to the mechanism alone, confounded by cross-
        # launch non-determinism unless independently re-verified.
        naive_baseline_sr=sum(1 for ep in common_eps if baseline[ep]["success"]) / len(common_eps),
        naive_recovery_sr=sum(1 for ep in common_eps if recovery[ep]["success"]) / len(common_eps),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="e.g. task1:scripted_recovery_task1_n20_full:task1.json")
    args = ap.parse_args()
    all_results = {}
    for spec in args.tasks:
        name, path, json_name = spec.split(":")
        res = analyze_task(path, json_name)
        all_results[name] = res
        print(f"\n=== {name} ===")
        print(f"  n_episodes={res['n_episodes']}  baseline: {res['n_baseline_success']} success / {res['n_baseline_fail']} fail")
        print(f"  Trigger Rate (of baseline-fail episodes): {res['n_triggered']}/{res['n_baseline_fail']} = {res['trigger_rate']:.1%}" if res['n_baseline_fail'] else "  Trigger Rate: n/a (0 baseline failures)")
        print(f"    triggered episode indices: {res['triggered_episodes']}")
        print(f"  Recovery Success Rate (of triggered episodes): {res['n_triggered_and_succeeded']}/{res['n_triggered']} = {res['recovery_success_rate']:.1%}" if res['n_triggered'] else "  Recovery Success Rate: n/a (0 triggered)")
        print(f"  Failure modes among {res['n_triggered_and_failed']} triggered-but-failed episodes:")
        for mode, eps in res["failure_modes"].items():
            print(f"    {mode}: {len(eps)} episode(s) {eps}")
        print(f"  [reference only, NOT the recommended metric] naive baseline SR={res['naive_baseline_sr']:.1%} vs naive recovery SR={res['naive_recovery_sr']:.1%}")

    with open("scripted_recovery_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nsaved scripted_recovery_metrics.json")


if __name__ == "__main__":
    main()
