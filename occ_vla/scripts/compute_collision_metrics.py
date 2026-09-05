#!/usr/bin/env python3
"""Computes contact_frac / anomalous_contact_frac (real MuJoCo contact-pair
based collision metrics, established 2026-08-24 in CLAUDE.md) from already-
collected task1.json result files for every checkpoint tested this session
(v3 and v4_interleave), plus the undistilled reference and old/unshuffled
baselines -- answers "did the policy actually avoid physical collision",
not just "did the task succeed"."""
import json
import glob
import os

OFT_DIR = "/home/ubuntu/slocal1/Hoki/occ_vla/thirdparty/openvla-oft"

GRIPPER_BODY_NAMES = {"gripper0_right_gripper", "gripper0_leftfinger", "gripper0_rightfinger"}


def compute_metrics(eps):
    n_steps_total = 0
    n_contact = 0
    n_anomalous = 0
    n_succ = sum(1 for e in eps if isinstance(e, dict) and e.get("success"))
    succ_steps = [e["done_step"] for e in eps if isinstance(e, dict) and e.get("success")]
    for e in eps:
        if not isinstance(e, dict):
            continue
        for row in e.get("proprio_log", []):
            n_steps_total += 1
            if row.get("occluder_contact"):
                n_contact += 1
                names = set(row.get("contact_robot_body_names", []))
                if names - GRIPPER_BODY_NAMES:
                    n_anomalous += 1
    return {
        "n_episodes": len(eps),
        "n_success": n_succ,
        "success_rate": n_succ / len(eps) if eps else None,
        "avg_len_success": sum(succ_steps) / len(succ_steps) if succ_steps else None,
        "n_proprio_steps": n_steps_total,
        "contact_frac": n_contact / n_steps_total if n_steps_total else None,
        "anomalous_contact_frac": n_anomalous / n_steps_total if n_steps_total else None,
        "n_contact_steps": n_contact,
        "n_anomalous_steps": n_anomalous,
    }


# (label, glob_pattern_for_task1.json, condition_key_in_results)
TARGETS = [
    # v3 run
    ("v3_undistilled_reference", "post_analysis_undistilled_reference/task1.json", "proactive_avoidance_depth"),
    ("v3_step200_baseline", "post_analysis_step200_baseline/task1.json", "baseline"),
    ("v3_step200_proactive_depth", "post_analysis_step200_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v3_step350_baseline", "post_analysis_step350_baseline/task1.json", "baseline"),
    ("v3_step350_proactive_depth", "post_analysis_step350_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v3_step450_baseline", "post_analysis_step450_baseline/task1.json", "baseline"),
    ("v3_step450_proactive_depth", "post_analysis_step450_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v3_step950_baseline", "post_analysis_step950_baseline/task1.json", "baseline"),
    ("v3_step950_proactive_depth", "post_analysis_step950_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v3_step450_n20_baseline", "step450_n20_baseline/task1.json", "baseline"),
    ("v3_step450_n20_proactive_depth", "step450_n20_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v3_old_unshuffled_step50_baseline", "distillation_rollout_step50/task1.json", "baseline"),
    ("v3_step50_baseline", "distillation_rollout_v3_step50/task1.json", "baseline"),
    # v4 interleave run
    ("v4_step250_baseline", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step250_baseline/task1.json", "baseline"),
    ("v4_step250_proactive_depth", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step250_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v4_step450_baseline", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step450_baseline/task1.json", "baseline"),
    ("v4_step450_proactive_depth", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step450_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v4_step700_baseline", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step700_baseline/task1.json", "baseline"),
    ("v4_step700_proactive_depth", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step700_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v4_step950_baseline", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step950_baseline/task1.json", "baseline"),
    ("v4_step950_proactive_depth", "post_analysis_distillation_lora_task1_n1000_v4_interleave_step950_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v4_step100_baseline_n10", "planB_v4interleave_step100_baseline/task1.json", "baseline"),
    ("v4_step100_proactive_depth_n10", "planB_v4interleave_step100_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v4_step100_baseline_n20", "planB_v4interleave_step100_n20_baseline/task1.json", "baseline"),
    ("v4_step100_proactive_depth_n20", "planB_v4interleave_step100_n20_proactive_depth/task1.json", "proactive_avoidance_depth"),
    ("v4_step150_baseline", "planB_v4interleave_step150_baseline/task1.json", "baseline"),
    ("v4_step200_baseline", "planB_v4interleave_step200_baseline/task1.json", "baseline"),
    ("v4_step200_proactive_depth", "planB_v4interleave_step200_proactive_depth/task1.json", "proactive_avoidance_depth"),
]

results = {}
for label, relpath, cond in TARGETS:
    path = os.path.join(OFT_DIR, relpath)
    if not os.path.exists(path):
        print(f"{label}: MISSING ({relpath})")
        continue
    d = json.load(open(path))
    eps = d.get("results", {}).get(cond)
    if not eps:
        print(f"{label}: no data for condition {cond}")
        continue
    m = compute_metrics(eps)
    results[label] = m
    print(f"{label:45s} n={m['n_episodes']:2d} SR={m['success_rate']*100:5.1f}%  "
          f"contact_frac={m['contact_frac']*100:5.1f}%  anomalous_contact_frac={m['anomalous_contact_frac']*100:5.1f}%  "
          f"(proprio_steps={m['n_proprio_steps']})")

with open("/home/ubuntu/slocal1/Hoki/occ_vla/scripts/collision_metrics_results.json", "w") as f:
    json.dump(results, f, indent=2)
