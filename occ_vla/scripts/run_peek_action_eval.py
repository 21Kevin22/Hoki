"""
run_peek_action_eval.py

Phase 3 (quantitative evaluation) of the "Peek Action" investigation
(occ_vla/thirdparty/openvla-oft, 2026-08-11). Per the user's explicit
priority ("同一タスクでの n 数増加" over cross-task generalization):
n=N rollouts on mug_in_microwave (libero_10 task_id=9) with RANDOMIZED
(varied) init states, comparing:

  - `baseline`: the loaded checkpoint's standard closed-loop rollout under
    real, natural occlusion -- VLA called every step, no macro/inertia
    intervention at all. This is what the paper's headline number needs
    a real, freshly-measured baseline for -- not assumed to be 0% just
    because that's the framing used in conversation; this script measures
    it directly, on the identical init states used for peek_v3, in the
    same script/run to avoid any config drift between conditions.
  - `peek_v3`: the full 3-tier fallback validated at n=1 in
    run_peek_action_poc.py (retreat macro -> open-loop inertia -> VLA-
    under-occlusion last resort). Logic copied verbatim from that script
    (not reimplemented) and wrapped in a per-episode function + episode
    loop over varied init_state indices.

Both conditions load the SAME checkpoint (which already has the
VJEPA_LatentDynamicsPredictor spliced in per this project's own
established architecture) and call get_vla_action identically
(occlusion_mask=None) -- the ONLY difference between conditions is
whether the macro/inertia fallback logic can ever preempt a VLA call.
This isolates the Peek Action mechanism's own effect, not the
checkpoint's.

occluder_body_ids are resolved once (task-level MJCF structure is
constant across episodes of the same task -- only qpos/init_state
differs), not re-resolved every episode.
"""
import os
import sys
import json
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
OFT_ROOT = os.path.join(os.path.dirname(__file__), "..", "thirdparty", "openvla-oft")
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_wrist_image, get_libero_image, get_libero_dummy_action, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, process_action,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_vla_action, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_natural_occlusion_success_rate import derive_natural_mask_and_gt, find_occluder_body_ids  # noqa: E402

GRAY_FILL = 127  # matches train_vjepa_predictor_scaled.py's GRAY_FILL / run_natural_occlusion_success_rate.py's
                  # oracle-condition fill value, for the vjepa_oracle condition's gray-out-occluded-pixels step

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
TASK_ID = 9  # mug_in_microwave
MAX_STEPS = 520
OCCLUSION_TRIGGER_THRESHOLD = 0.15
MAX_MACRO_DISPLACEMENT_M = 0.05
MACRO_XY_ACTION_UNITS = 0.6
MACRO_DZ_ACTION_UNITS = 0.6
RETREAT_HISTORY_LEN = 5
INERTIA_BUDGET_STEPS = 12


def run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector,
                 init_state, occluder_body_ids, condition):
    env.reset()
    obs = env.set_init_state(init_state)
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    # sim MUST be re-fetched AFTER reset()/set_init_state() -- reset() rebuilds the
    # underlying MjSim, invalidating any reference captured before it (real bug hit
    # and fixed in run_peek_action_poc.py; reintroduced here by hoisting sim capture
    # to main(), outside the per-episode reset -- fixed by re-fetching every episode).
    sim = env.env.sim

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    n_vla_calls = 0
    n_macro_steps = 0
    n_macro_capped_steps = 0
    n_inertia_steps = 0
    n_macro_defeated_steps = 0
    n_vjepa_engaged = 0  # vjepa_oracle only: calls where occlusion_mask was actually passed (non-None)
    macro_start_eef_xy = None
    macro_defeated = False
    recent_real_eef_xy = deque(maxlen=RETREAT_HISTORY_LEN)
    recent_clear_actions = deque(maxlen=RETREAT_HISTORY_LEN)
    inertia_vector = None
    inertia_steps_used = 0
    last_real_gripper_raw = 1.0
    success = False
    occlusion_trace = []
    occluded_flags = []  # parallel to eef_trace -- was THIS env step occluded (s_occ > threshold)
    attn_entropy_trace = []  # per-VLA-call action-to-vision-patch attention entropy (Attention Collapse metric)
    attn_entropy_occ_trace = []  # same, but only calls made while occluded (Option B: local-slice, not
    attn_entropy_clear_trace = []  # diluted by whole-episode averaging -- 2026-08-12 follow-up)
    eef_trace = []  # per-env-step robot0_eef_pos, for path-length / jitter (Action Smoothness proxy)

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    while t < MAX_STEPS + cfg.num_steps_wait:
        wrist_occ, agent_occ, mask_px, mask_256 = derive_natural_mask_and_gt(env, sim, occluder_body_ids)
        s_occ = float(mask_256.mean())
        occlusion_trace.append(s_occ)
        eef_trace.append(np.array(obs["robot0_eef_pos"], dtype=float).copy())

        occluded = s_occ > OCCLUSION_TRIGGER_THRESHOLD
        occluded_flags.append(occluded)

        if condition in ("baseline", "vjepa_oracle"):
            mode = "vla_clear"  # never intervene via macro/inertia -- vjepa_oracle differs from baseline
                                 # only in whether occlusion_mask gets passed to get_vla_action below
        else:
            if not occluded:
                mode = "vla_clear"
                macro_start_eef_xy = None
                macro_defeated = False
                inertia_steps_used = 0
                inertia_vector = None
            elif not macro_defeated:
                mode = "macro"
            elif inertia_vector is not None and inertia_steps_used < INERTIA_BUDGET_STEPS:
                mode = "inertia"
            else:
                mode = "vla_occluded"

        if mode == "macro":
            cur_eef_xy = np.array(obs["robot0_eef_pos"][:2], dtype=float)
            if macro_start_eef_xy is None:
                macro_start_eef_xy = cur_eef_xy.copy()
            displacement_so_far = float(np.linalg.norm(cur_eef_xy - macro_start_eef_xy))

            if displacement_so_far < MAX_MACRO_DISPLACEMENT_M:
                if len(recent_real_eef_xy) >= 2:
                    move_dir = recent_real_eef_xy[-1] - recent_real_eef_xy[0]
                    norm = np.linalg.norm(move_dir)
                    if norm > 1e-6:
                        retreat_xy = -(move_dir / norm) * MACRO_XY_ACTION_UNITS
                        action = np.array([retreat_xy[0], retreat_xy[1], 0.0, 0.0, 0.0, 0.0, last_real_gripper_raw])
                    else:
                        action = np.array([0.0, 0.0, MACRO_DZ_ACTION_UNITS, 0.0, 0.0, 0.0, last_real_gripper_raw])
                else:
                    action = np.array([0.0, 0.0, MACRO_DZ_ACTION_UNITS, 0.0, 0.0, 0.0, last_real_gripper_raw])
                n_macro_steps += 1
            else:
                macro_defeated = True
                n_macro_capped_steps += 1
                if len(recent_clear_actions) > 0:
                    arr = np.stack(list(recent_clear_actions), axis=0)
                    inertia_vector = arr[:, :6].mean(axis=0)
                    inertia_vector = np.concatenate([inertia_vector, [arr[-1, 6]]])
                    inertia_steps_used = 0
                    mode = "inertia"
                else:
                    inertia_vector = None
                    mode = "vla_occluded"
            action_queue.clear()

        if mode == "inertia":
            action = inertia_vector.copy()
            inertia_steps_used += 1
            n_inertia_steps += 1
            action_queue.clear()

        if mode in ("vla_clear", "vla_occluded"):
            if len(action_queue) == 0:
                n_vla_calls += 1
                wrist_img = get_libero_wrist_image(obs).copy()
                occlusion_mask_t = None
                if condition == "vjepa_oracle" and mask_256.any():
                    # Real natural occlusion mask + trained VJEPA correction, matching
                    # run_natural_occlusion_success_rate.py's "oracle" condition exactly
                    # (gray-fill the occluded wrist pixels, pass the same mask so the
                    # spliced predictor's residual correction engages).
                    wrist_img[mask_px] = GRAY_FILL
                    occlusion_mask_t = torch.from_numpy(mask_256).to(
                        device=model.device, dtype=torch.bfloat16).reshape(1, -1, 1)
                    n_vjepa_engaged += 1
                observation = {
                    "full_image": get_libero_image(obs).copy(),
                    "wrist_image": wrist_img,
                    "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                }
                with torch.inference_mode():
                    actions, attn_entropy = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                        occlusion_mask=occlusion_mask_t, return_attn_entropy=True,
                    )
                if attn_entropy is not None:
                    attn_entropy_trace.append(float(attn_entropy))
                    (attn_entropy_occ_trace if occluded else attn_entropy_clear_trace).append(float(attn_entropy))
                action_queue.extend(actions)
            action = action_queue.popleft()
            if mode == "vla_occluded":
                n_macro_defeated_steps += 1

        action = np.array(action, dtype=float)
        if mode == "vla_clear":
            last_real_gripper_raw = float(action[-1])
            recent_clear_actions.append(action.copy())
            recent_real_eef_xy.append(np.array(obs["robot0_eef_pos"][:2], dtype=float))

        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            success = True
            break

    done_step = t - cfg.num_steps_wait
    eef_arr = np.stack(eef_trace, axis=0)
    step_disp = np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)  # (len(eef_trace)-1,) per-step displacement
    path_length = float(step_disp.sum())
    # Option B (2026-08-12): occluded-steps-ONLY per-step displacement, matching the sibling pi0.5
    # project's own occluded_jitter_mean definition exactly -- unlike jitter_per_step above (whole-episode
    # path_length/done_step, confounded by episode length/overall motion speed), this isolates the
    # quantity actually in question: does the arm move choppier specifically WHILE occluded.
    occ_disp = step_disp[np.array(occluded_flags[1:], dtype=bool)] if len(occluded_flags) > 1 else np.array([])
    return {
        "success": success,
        "done_step": done_step,
        "n_vla_calls": n_vla_calls,
        "n_macro_steps": n_macro_steps,
        "n_macro_capped_steps": n_macro_capped_steps,
        "n_inertia_steps": n_inertia_steps,
        "n_macro_defeated_steps": n_macro_defeated_steps,
        "n_vjepa_engaged": n_vjepa_engaged,
        "occ_min": float(min(occlusion_trace)),
        "occ_max": float(max(occlusion_trace)),
        "occ_mean": float(np.mean(occlusion_trace)),
        "n_steps_above_threshold": sum(1 for s in occlusion_trace if s > OCCLUSION_TRIGGER_THRESHOLD),
        "n_occ_steps_total": len(occlusion_trace),
        # Attention Collapse proxy (action-token -> vision-patch attention entropy, normalized [0,1];
        # higher = attention spread thinner across vision patches, i.e. less focused/grounded).
        "attn_entropy_mean": float(np.mean(attn_entropy_trace)) if attn_entropy_trace else None,
        "n_attn_entropy_calls": len(attn_entropy_trace),
        "attn_entropy_occ_mean": float(np.mean(attn_entropy_occ_trace)) if attn_entropy_occ_trace else None,
        "attn_entropy_clear_mean": float(np.mean(attn_entropy_clear_trace)) if attn_entropy_clear_trace else None,
        "n_attn_entropy_occ_calls": len(attn_entropy_occ_trace),
        "occluded_jitter_mean": float(occ_disp.mean()) if len(occ_disp) > 0 else None,
        "occluded_jitter_n": int(len(occ_disp)),
        # Action Smoothness proxy (path_length/compute_occlusion_trajectory_metrics.py's own convention):
        # total EEF path length and per-step displacement ("jitter") -- NOT the literal 2nd-difference
        # Jerk formula (would need raw per-step ACTION logging, not done here), but the same
        # already-validated "wandering/choppiness" family of proxy used elsewhere in this project.
        "path_length": path_length,
        "jitter_per_step": path_length / done_step if done_step > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "peek_v3", "vjepa_oracle"], required=True)
    parser.add_argument("--vjepa-checkpoint", type=str, default=None,
                         help="required for --condition vjepa_oracle -- path to a trained "
                              "vjepa_predictor_*.pt state dict (e.g. vjepa_predictor_multitask_3task_6000steps.pt)")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--results-path", type=str, required=True)
    args = parser.parse_args()
    if args.condition == "vjepa_oracle":
        assert args.vjepa_checkpoint, "--condition vjepa_oracle requires --vjepa-checkpoint"

    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    # REAL BUG, FOUND AND FIXED (2026-08-11): earlier revision of this script forced
    # `cfg.attn_implementation = "eager"` here, reasoning by analogy from
    # run_natural_occlusion_success_rate.py's attn_gated_oracle docstring warning about
    # SDPA<->eager fallback "perturbing behavior differently" -- but that warning is about
    # MIXING return_attn_entropy=True/False calls within one episode (gated_oracle only
    # requests attentions on SOME calls). This script requests return_attn_entropy=True on
    # EVERY call uniformly, so there is no mixing to guard against -- forcing eager was an
    # unnecessary "fix" for a problem that doesn't apply here, and it caused a REAL
    # regression: a full n=20 baseline run under this setting collapsed to ~0% success
    # (every episode timing out at 520 steps), a complete reversal from the un-forced
    # SDPA-default run's 90-100%. smoke_test_attn_entropy.py -- the project's own existing,
    # already-validated test for this exact plumbing -- never sets attn_implementation at
    # all, and explicitly checks return_attn_entropy=True produces a BYTE-IDENTICAL action
    # to return_attn_entropy=False on the same input; that's the validated, correct pattern.
    # Leaving attn_implementation unset (framework default / per-call fallback) here.
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    resize_size = get_image_resize_size(cfg)

    if args.condition == "vjepa_oracle":
        ckpt = torch.load(args.vjepa_checkpoint, map_location=model.device)
        model.vision_backbone.vjepa_predictor_dino.load_state_dict(ckpt["dino"])
        model.vision_backbone.vjepa_predictor_siglip.load_state_dict(ckpt["siglip"])
        print(f"[eval] Loaded vjepa predictor weights from {args.vjepa_checkpoint}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(TASK_ID)
    task_description = task.language
    init_states = task_suite.get_task_init_states(TASK_ID)
    n = min(args.n_episodes, len(init_states))
    print(f"[eval] condition={args.condition} n_episodes={n} (of {len(init_states)} available init states) task='{task_description}'")

    env, _ = get_libero_env(task, cfg.model_family, resolution=resize_size)
    env.reset()
    env.set_init_state(init_states[0])
    sim = env.env.sim
    occluder_body_ids = find_occluder_body_ids(sim, ["microwave_1"])
    assert occluder_body_ids, "no occluder bodies matched 'microwave_1'"
    print(f"[eval] occluder_body_ids: {occluder_body_ids}")

    results = []
    for i in range(n):
        res = run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector,
                           init_states[i], occluder_body_ids, args.condition)
        res["episode_idx"] = i
        results.append(res)
        occ_str = f"{res['attn_entropy_occ_mean']:.4f}" if res["attn_entropy_occ_mean"] is not None else "NA"
        clear_str = f"{res['attn_entropy_clear_mean']:.4f}" if res["attn_entropy_clear_mean"] is not None else "NA"
        oj_str = f"{res['occluded_jitter_mean']:.4f}" if res["occluded_jitter_mean"] is not None else "NA"
        print(f"[eval] ep{i} condition={args.condition} success={res['success']} done_step={res['done_step']} "
              f"n_macro_capped_steps={res['n_macro_capped_steps']} n_inertia_steps={res['n_inertia_steps']} "
              f"n_macro_defeated_steps={res['n_macro_defeated_steps']} n_vjepa_engaged={res['n_vjepa_engaged']} "
              f"occ_max={res['occ_max']:.3f} attn_entropy_occ={occ_str} attn_entropy_clear={clear_str} "
              f"occluded_jitter_mean={oj_str} (n={res['occluded_jitter_n']})")
        with open(args.results_path, "w") as f:
            json.dump({"condition": args.condition, "task_id": TASK_ID, "results": results}, f, indent=2)

    succ = sum(1 for r in results if r["success"])
    print(f"[eval] FINAL condition={args.condition} n={n} success_rate={succ}/{n} ({100.0*succ/n:.1f}%)")


if __name__ == "__main__":
    main()
