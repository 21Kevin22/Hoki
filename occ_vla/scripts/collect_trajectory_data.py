"""
collect_trajectory_data.py

Per-ENV-STEP trajectory logging (occ_vla, 2026-08-04), generalizing
run_dynamic_gating_eval.py's per-VLA-call episode loop for three
follow-up analyses (presentation figures):
  1. Drift-vs-elapsed-occlusion-steps: does a long-horizon task's
     uncorrected eef trajectory diverge irrecoverably from a clean
     reference while a short-horizon task doesn't have time to?
  2. Qualitative 3D trajectory comparison: clean / occluded_uncorrected /
     dynamic-corrected, one episode, geometric "does correction pull the
     trajectory back toward clean" plot.
  3. Clean-phase action jitter: always_on_unconditional vs dynamic --
     does forcing the correction mechanism to engage even when nothing
     is occluded introduce jitter run_dynamic_gating_eval.py's
     "always_corrected" (oracle, only ever engages once occlusion has
     actually started) can't measure?

Five modes:
  - clean: no occlusion, ever.
  - occluded_uncorrected: wrist_partial-style patch begins at
    --delay-steps, correction never engages (matches
    run_dynamic_gating_eval.py's "never_corrected").
  - always_corrected: same occlusion schedule, correction engages the
    instant occlusion starts (oracle; matches
    run_dynamic_gating_eval.py's "always_corrected").
  - always_on_unconditional: correction engages from t=0, REGARDLESS of
    whether pixels are actually occluded yet -- the occlusion_mask is
    built from the same fixed patch geometry _apply_partial_patch always
    uses, but the wrist image itself is only actually blanked once
    control_step >= delay_steps. Tests whether unconditionally running
    the correction on real, unoccluded content costs anything.
  - dynamic: matches run_dynamic_gating_eval.py -- occlusion_classifier's
    P(occluded) gates when correction engages (sticky once triggered).

Run with the openvla-oft conda env:
  python scripts/collect_trajectory_data.py \
    --task-suite libero_10 --task-id 8 \
    --checkpoint <path> --vjepa-checkpoint <path> \
    --classifier-path occlusion_classifier.npz \
    --delay-steps 30 --modes clean occluded_uncorrected dynamic \
    --num-trials 10 --out-dir trajectory_data_task8
"""

import argparse
import json
import os
import sys
import time
from collections import deque

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OCC_VLA_ROOT = os.path.dirname(SCRIPTS_DIR)
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402

import occlusion_classifier as oc  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, TaskSuite, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from run_oft_camera_dropout_eval import _apply_partial_patch, _build_patch_token_mask  # noqa: E402

MODES = ("clean", "occluded_uncorrected", "always_corrected", "always_on_unconditional", "dynamic")


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                 initial_state, max_steps, mode, delay_steps, classifier_params, threshold, num_images,
                 log_hidden_states=False):
    assert mode in MODES
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    n_calls = 0
    success = False
    triggered = False
    last_hidden = None
    log = []
    # one entry per VLA call (not per env step -- num_open_loop_steps=8x
    # smaller) for the stuck-vs-hold latent probe. Only populated if
    # log_hidden_states=True; every other caller/mode is unaffected.
    call_log = []

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            control_step = t - cfg.num_steps_wait
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
            eef_pos_now = obs["robot0_eef_pos"].tolist()

            pixel_occluded = (mode != "clean") and (control_step >= delay_steps)
            # pixel_bounds is deterministic (fixed geometry), needed even when
            # NOT actually occluding (always_on_unconditional's mask target)
            occluded_wrist, pixel_bounds = _apply_partial_patch(wrist_img_resized)
            if pixel_occluded:
                wrist_img_resized = occluded_wrist

            engage_correction = False
            if mode == "always_corrected" and pixel_occluded:
                engage_correction = True
            elif mode == "always_on_unconditional":
                engage_correction = True
            elif mode == "dynamic" and pixel_occluded:
                if triggered:
                    engage_correction = True
                elif last_hidden is not None:
                    p = oc.score(classifier_params, last_hidden)
                    if p >= threshold:
                        triggered = True
                        engage_correction = True

            occlusion_mask_np = None
            if engage_correction:
                occlusion_mask_np = _build_patch_token_mask(pixel_bounds, camera_block_index=1, num_images=num_images)

            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            occlusion_mask = None
            if occlusion_mask_np is not None:
                occlusion_mask = torch.from_numpy(occlusion_mask_np).to(
                    device=model.device, dtype=torch.bfloat16
                ).reshape(1, -1, 1)

            if len(action_queue) == 0:
                actions, hidden_state = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film,
                    occlusion_mask=occlusion_mask, return_hidden_states=True,
                )
                action_queue.extend(actions)
                last_hidden = hidden_state
                if log_hidden_states:
                    call_log.append({
                        "call_idx": n_calls,
                        "control_step": control_step,
                        "hidden_state": np.asarray(hidden_state).astype(np.float32).tolist(),
                    })
                n_calls += 1

            action = action_queue.popleft()
            action_processed = process_action(action, cfg.model_family)

            log.append({
                "t": control_step,
                "eef_pos": eef_pos_now,
                "action": np.asarray(action_processed).tolist(),
                "pixel_occluded": bool(pixel_occluded),
                "corrected": bool(engage_correction),
            })

            obs, reward, done, info = env.step(action_processed.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        print(f"  Episode error: {e}")

    return {"success": success, "done_step": t, "n_calls": n_calls, "mode": mode, "log": log, "call_log": call_log}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint", default=None,
                         help="required for modes other than clean/occluded_uncorrected")
    parser.add_argument("--classifier-path", default=None, help="required for mode=dynamic")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delay-steps", type=int, default=30)
    parser.add_argument("--modes", nargs="+", default=["clean", "occluded_uncorrected"], choices=MODES)
    parser.add_argument("--out-dir", default="trajectory_data")
    parser.add_argument("--log-hidden-states", action="store_true",
                         help="also save one hidden_state per VLA call (call_log) -- for the "
                              "stuck-vs-legitimate-hold latent probe, 2026-08-04")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=args.task_suite,
    )
    set_seed_everywhere(cfg.seed)

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model = get_model(cfg)
    if args.vjepa_checkpoint:
        state_dicts = torch.load(args.vjepa_checkpoint, map_location=model.device)
        model.vision_backbone.vjepa_predictor_dino.load_state_dict(state_dicts["dino"])
        model.vision_backbone.vjepa_predictor_dino.to(dtype=torch.bfloat16)
        model.vision_backbone.vjepa_predictor_siglip.load_state_dict(state_dicts["siglip"])
        model.vision_backbone.vjepa_predictor_siglip.to(dtype=torch.bfloat16)
        print(f"Loaded vjepa_predictor_dino/_siglip weights from {args.vjepa_checkpoint}")

    classifier_params = oc.load(args.classifier_path) if args.classifier_path else None

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    print(f"Task suite: {args.task_suite}, task_id={args.task_id}, name={task.name}")
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = TASK_MAX_STEPS[TaskSuite(args.task_suite)]
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    for mode in args.modes:
        print(f"\n=== Mode: {mode} (delay_steps={args.delay_steps}) ===")
        n_success = 0
        for ep in range(args.num_trials):
            t0 = time.time()
            data = run_episode(
                cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                initial_states[ep], max_steps, mode, args.delay_steps, classifier_params, args.threshold,
                cfg.num_images_in_input, log_hidden_states=args.log_hidden_states,
            )
            n_success += int(data["success"])
            out_path = os.path.join(args.out_dir, f"{mode}_episode_{ep:03d}.json")
            with open(out_path, "w") as f:
                json.dump(data, f)
            print(f"  ep{ep}: success={data['success']} done_step={data['done_step']} n_calls={data['n_calls']} "
                  f"wall={time.time()-t0:.1f}s -> {out_path}")
        print(f"  {mode}: {n_success}/{args.num_trials} success")


if __name__ == "__main__":
    main()
