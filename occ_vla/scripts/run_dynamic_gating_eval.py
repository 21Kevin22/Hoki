"""
run_dynamic_gating_eval.py

Dynamic occlusion-recovery gating (occ_vla, 2026-08-04): instead of the
"vjepa correction on for the whole episode" design every prior script in
this thread used, the wrist camera stays CLEAN for the first
--delay-steps control-loop steps of each episode, then a fixed
wrist_partial-style patch occlusion begins and persists for the rest of
the episode -- giving a genuine within-episode before/after transition
for a real-time gate to detect, which no earlier condition in this
project had (wrist_partial/_vjepa occlude from step 0, with no onset
event at all).

Three modes, same delayed-onset occlusion schedule and same init_states
(paired comparison):
  - never_corrected: correction never engages, even after occlusion starts.
  - always_corrected: correction engages the instant occlusion starts
    (oracle -- uses the KNOWN onset step, not a real trigger signal).
  - dynamic: correction engages only once occlusion_classifier.py's
    P(occluded) crosses --threshold on a live per-call hidden state
    (sticky -- once triggered, stays on for the rest of the episode).
    Detection latency is at least one VLA call (cfg.num_open_loop_steps
    env steps), since the trigger decision for call N is made from call
    N-1's hidden state (the state a call itself produces isn't available
    until after the occlusion_mask for that same call has already been
    decided).

Uses occlusion_classifier.py's trigger (a supervised clean-vs-occluded
logistic regression, val AUC=0.9997/1.0 in train_failure_probe.py /
fit_occlusion_classifier.py) rather than clean_manifold_detector.py's
PCA+Mahalanobis distance -- the latter was found to invert on this exact
failure mode (occlusion collapses activation variance toward the clean
centroid rather than pushing it away, so distance-from-clean scored
occluded activations as MORE typical, AUC 0.12-0.20) before this script
was written; kept in the codebase as a documented negative result, not
used here.

Run with the openvla-oft conda env:
  python scripts/run_dynamic_gating_eval.py \
    --task-suite libero_10 --task-id 8 \
    --checkpoint <path> --vjepa-checkpoint vjepa_predictor_multitask_3task_6000steps.pt \
    --classifier-path occlusion_classifier.npz --threshold 0.5 \
    --delay-steps 30 --modes never_corrected always_corrected dynamic \
    --num-trials 10 --results-path dynamic_gating_task8.json
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

MODES = ("never_corrected", "always_corrected", "dynamic")


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                 initial_state, max_steps, mode, delay_steps, classifier_params, threshold, num_images):
    assert mode in MODES
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    n_calls = 0
    success = False
    triggered = False        # 'dynamic' mode: has the classifier fired yet (sticky once on)
    trigger_call_idx = None  # which VLA call first engaged correction
    onset_call_idx = None    # which VLA call first saw occluded pixels
    scores = []               # 'dynamic' mode only: P(occluded) at each call, for diagnostics
    last_hidden = None

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

            occlusion_mask_np = None
            if control_step >= delay_steps:
                wrist_img_resized, pixel_bounds = _apply_partial_patch(wrist_img_resized)
                if onset_call_idx is None:
                    onset_call_idx = n_calls

                engage = False
                if mode == "always_corrected":
                    engage = True
                elif mode == "dynamic":
                    if triggered:
                        engage = True
                    elif last_hidden is not None:
                        p = oc.score(classifier_params, last_hidden)
                        scores.append(float(p))
                        if p >= threshold:
                            triggered = True
                            trigger_call_idx = n_calls
                            engage = True
                # mode == "never_corrected": engage stays False

                if engage:
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
                n_calls += 1

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        print(f"  Episode error: {e}")

    return {
        "success": success, "done_step": t, "n_calls": n_calls,
        "triggered": triggered, "trigger_call_idx": trigger_call_idx,
        "onset_call_idx": onset_call_idx,
        "detection_lag_calls": (trigger_call_idx - onset_call_idx) if trigger_call_idx is not None and onset_call_idx is not None else None,
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True,
                         help="required here (unlike other scripts) -- always_corrected/dynamic modes need real trained weights")
    parser.add_argument("--classifier-path", default="occlusion_classifier.npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delay-steps", type=int, default=30,
                         help="control-loop steps (post-wait) before occlusion begins")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--results-path", default=None)
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=args.task_suite,
    )
    set_seed_everywhere(cfg.seed)

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model = get_model(cfg)
    state_dicts = torch.load(args.vjepa_checkpoint, map_location=model.device)
    model.vision_backbone.vjepa_predictor_dino.load_state_dict(state_dicts["dino"])
    model.vision_backbone.vjepa_predictor_dino.to(dtype=torch.bfloat16)
    model.vision_backbone.vjepa_predictor_siglip.load_state_dict(state_dicts["siglip"])
    model.vision_backbone.vjepa_predictor_siglip.to(dtype=torch.bfloat16)
    print(f"Loaded vjepa_predictor_dino/_siglip weights from {args.vjepa_checkpoint}")

    classifier_params = oc.load(args.classifier_path)
    print(f"Loaded occlusion classifier from {args.classifier_path}")

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

    all_results = {}
    for mode in args.modes:
        print(f"\n=== Mode: {mode} (delay_steps={args.delay_steps}) ===")
        mode_results = []
        for ep in range(args.num_trials):
            t0 = time.time()
            data = run_episode(
                cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                initial_states[ep], max_steps, mode, args.delay_steps, classifier_params, args.threshold,
                cfg.num_images_in_input,
            )
            data["episode"] = ep
            data["wall_s"] = time.time() - t0
            mode_results.append(data)
            lag = data["detection_lag_calls"]
            print(f"  ep{ep}: success={data['success']} done_step={data['done_step']} n_calls={data['n_calls']} "
                  f"triggered={data['triggered']} onset_call={data['onset_call_idx']} trigger_call={data['trigger_call_idx']} "
                  f"lag={lag} wall={data['wall_s']:.1f}s")
        n_success = sum(r["success"] for r in mode_results)
        print(f"  {mode}: {n_success}/{args.num_trials} success")
        all_results[mode] = mode_results

    print("\n=== Summary ===")
    for mode in args.modes:
        n_success = sum(r["success"] for r in all_results[mode])
        print(f"  {mode}: {n_success}/{args.num_trials}")

    if args.results_path:
        with open(args.results_path, "w") as f:
            json.dump({"args": vars(args), "results": all_results}, f, indent=2)
        print(f"\nSaved results to {args.results_path}")


if __name__ == "__main__":
    main()
