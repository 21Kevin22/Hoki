"""
validate_vjepa_predictor.py

Cheap qualitative/quantitative checks on the 300-step smoke-test-trained
VJEPA_LatentDynamicsPredictor, per occ_vla's own repeated lesson that loss
decreasing does not imply the correction is actually useful (see H1/H2's
loss-vs-generation dissociation, and the MMaDA/dust3r/SD injection checks
throughout this project's history).

Two checks:
  1. Feature-level: for held-out (never trained on) AND in-sample (training)
     episode samples, compare occluded-token distance-to-ground-truth
     BEFORE (raw corrupted input, == untrained/zero-init behavior) vs AFTER
     (trained predictor's correction) -- cosine similarity and L2 distance.
  2. Single-frame 3-way action comparison (same methodology occ_vla used for
     MMaDA/dust3r/SD injections): baseline (clean, no occlusion) vs
     occluded+untrained vs occluded+trained, comparing which is closer to
     the baseline/GT action on one frozen frame.

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/validate_vjepa_predictor.py
"""

import os
import sys

OCC_VLA_ROOT = "/home/ubuntu/slocal1/Hoki/occ_vla"
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
sys.path.insert(0, OCC_VLA_ROOT)  # for `scripts.collect_oft_onpolicy_rollout_data` / `scripts.train_vjepa_predictor_smoke_test`
sys.path.insert(0, OFT_ROOT)  # for `experiments`, `prismatic`, `libero`
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, normalize_proprio, resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import get_model  # noqa: E402

from scripts.collect_oft_onpolicy_rollout_data import prepare_observation  # noqa: E402
from scripts.train_vjepa_predictor_smoke_test import apply_partial_patch, build_patch_token_mask, build_pixel_values  # noqa: E402

CHECKPOINT = os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa")
VJEPA_WEIGHTS = "vjepa_predictor_smoke_test.pt"
DATA_DIR = "oft_onpolicy_rollout_data"


def collect_one_fresh_episode(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, init_state):
    """Same collection logic as collect_oft_onpolicy_rollout_data.py, inline
    (avoids a subprocess) -- collects ONE episode never seen during the
    300-step smoke-test training (which only used episodes 0-4)."""
    from collections import deque
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action

    env.reset()
    obs = env.set_init_state(init_state)
    model.reset_vjepa_state()
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    agentview_frames, wrist_frames, proprio_frames = [], [], []
    t = 0
    while t < 520 + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        observation = prepare_observation(obs, resize_size)
        if len(action_queue) == 0:
            agentview_frames.append(observation["full_image"].copy())
            wrist_frames.append(observation["wrist_image"].copy())
            proprio_frames.append(observation["state"].copy())
            actions = get_action(
                cfg, model, observation, task_description, processor=processor,
                action_head=action_head, proprio_projector=proprio_projector,
                noisy_action_projector=None, use_film=cfg.use_film,
            )
            action_queue.extend(actions)
        action = action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1
        if len(proprio_frames) >= 30:  # cap -- we only need ~20-30 samples for this check
            break
    return {
        "agentview": np.stack(agentview_frames).astype(np.uint8),
        "wrist": np.stack(wrist_frames).astype(np.uint8),
        "proprio": np.stack(proprio_frames).astype(np.float32),
    }


def feature_level_check(model, processor, prompt, episode, proprio_norm_stats, label, num_samples=20):
    device, dtype = model.device, torch.bfloat16
    T = len(episode["proprio"])
    idxs = list(range(1, T))[:num_samples]

    cos_before, cos_after = [], []
    l2_before, l2_after = [], []

    for t in idxs:
        agentview_t, wrist_t = episode["agentview"][t], episode["wrist"][t]
        agentview_tm1, wrist_tm1 = episode["agentview"][t - 1], episode["wrist"][t - 1]
        proprio_t = normalize_proprio(episode["proprio"][t], proprio_norm_stats)

        wrist_t_corrupted, pixel_bounds = apply_partial_patch(wrist_t)
        occlusion_mask_np = build_patch_token_mask(pixel_bounds, camera_block_index=1, num_images=2)
        occlusion_mask = torch.from_numpy(occlusion_mask_np).to(device=device, dtype=dtype).reshape(1, -1, 1)

        with torch.no_grad():
            f_gt = model.vision_backbone(build_pixel_values(agentview_t, wrist_t, processor, prompt, device, dtype))
            past_latents = model.vision_backbone(build_pixel_values(agentview_tm1, wrist_tm1, processor, prompt, device, dtype))
            f_input = model.vision_backbone(build_pixel_values(agentview_t, wrist_t_corrupted, processor, prompt, device, dtype))
            proprio_tensor = torch.tensor(proprio_t, device=device, dtype=dtype).reshape(1, -1)
            residual = model.vjepa_predictor(f_input, past_latents, proprio_tensor)
            f_final = f_input + occlusion_mask * residual

        mask_bool = occlusion_mask_np.astype(bool)
        fg = f_gt[0, mask_bool].float()
        fi = f_input[0, mask_bool].float()
        ff = f_final[0, mask_bool].float()

        cos_before.append(torch.nn.functional.cosine_similarity(fi, fg, dim=-1).mean().item())
        cos_after.append(torch.nn.functional.cosine_similarity(ff, fg, dim=-1).mean().item())
        l2_before.append((fi - fg).norm(dim=-1).mean().item())
        l2_after.append((ff - fg).norm(dim=-1).mean().item())

    print(f"\n=== Feature-level check: {label} (n={len(idxs)} samples) ===")
    print(f"  cosine sim to GT:  before={np.mean(cos_before):.4f}  after={np.mean(cos_after):.4f}")
    print(f"  L2 dist to GT:     before={np.mean(l2_before):.4f}  after={np.mean(l2_after):.4f}")
    return {
        "cos_before": float(np.mean(cos_before)), "cos_after": float(np.mean(cos_after)),
        "l2_before": float(np.mean(l2_before)), "l2_after": float(np.mean(l2_after)),
    }


def three_way_action_check(cfg, model, processor, task_description, obs, resize_size, action_head, proprio_projector, proprio_norm_stats, prompt):
    """Single-frame: baseline (clean) vs occluded+untrained vs occluded+trained."""
    from experiments.robot.robot_utils import get_action

    observation = prepare_observation(obs, resize_size)
    wrist_corrupted, pixel_bounds = apply_partial_patch(observation["wrist_image"])
    occlusion_mask_np = build_patch_token_mask(pixel_bounds, camera_block_index=1, num_images=2)
    occlusion_mask = torch.from_numpy(occlusion_mask_np).to(device=model.device, dtype=torch.bfloat16).reshape(1, -1, 1)

    proprio = normalize_proprio(observation["state"], proprio_norm_stats)

    # A: clean baseline (no occlusion at all)
    model.reset_vjepa_state()
    obs_a = dict(observation)
    action_a = get_action(cfg, model, obs_a, task_description, processor=processor, action_head=action_head,
                           proprio_projector=proprio_projector, noisy_action_projector=None, use_film=cfg.use_film)
    action_a = np.array(action_a)

    # B: occluded, vjepa NOT engaged (occlusion_mask=None) -- matches the
    # already-established n=10 "wrist_partial" untrained result (0/10)
    model.reset_vjepa_state()
    obs_b = dict(observation, wrist_image=wrist_corrupted)
    action_b = get_action(cfg, model, obs_b, task_description, processor=processor, action_head=action_head,
                           proprio_projector=proprio_projector, noisy_action_projector=None, use_film=cfg.use_film)
    action_b = np.array(action_b)

    # C: occluded, vjepa engaged with TRAINED weights (occlusion_mask passed)
    model.reset_vjepa_state()
    obs_c = dict(observation, wrist_image=wrist_corrupted)
    action_c = get_action(cfg, model, obs_c, task_description, processor=processor, action_head=action_head,
                           proprio_projector=proprio_projector, noisy_action_projector=None, use_film=cfg.use_film,
                           occlusion_mask=occlusion_mask)
    action_c = np.array(action_c)

    dist_b = np.linalg.norm(action_a - action_b)
    dist_c = np.linalg.norm(action_a - action_c)
    cos_b = np.dot(action_a.flatten(), action_b.flatten()) / (np.linalg.norm(action_a) * np.linalg.norm(action_b) + 1e-8)
    cos_c = np.dot(action_a.flatten(), action_c.flatten()) / (np.linalg.norm(action_a) * np.linalg.norm(action_c) + 1e-8)

    print("\n=== Single-frame 3-way action comparison ===")
    print(f"  Euclidean distance to clean-baseline action:  untrained={dist_b:.4f}  trained={dist_c:.4f}")
    print(f"  Cosine similarity to clean-baseline action:    untrained={cos_b:.4f}  trained={cos_c:.4f}")
    print(f"  ({'trained is CLOSER to GT' if dist_c < dist_b else 'trained is NOT closer to GT'})")


def main():
    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10",
    )
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = 224  # OpenVLA's fixed policy input size (MODEL_IMAGE_SIZES["openvla"])
    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(8)
    task_description = task.language
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    initial_states = task_suite.get_task_init_states(8)
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # === Load in-sample (already collected, used in training) episode 0 ===
    in_sample_ep = dict(np.load(os.path.join(DATA_DIR, "episode_000.npz")))

    # === Collect ONE fresh held-out episode (init_state index 5 -- never
    # trained on; training only used episodes 0-4 / init_states 0-4) ===
    print("Collecting 1 fresh held-out episode (init_state_idx=5, not used in training)...")
    held_out_ep = collect_one_fresh_episode(
        cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_states[5]
    )
    print(f"  collected {len(held_out_ep['proprio'])} samples")

    # === Check 1: feature-level, BEFORE loading trained weights (zero-init reference) ===
    print("\n" + "=" * 70)
    print("REFERENCE (zero-init / untrained vjepa_predictor)")
    print("=" * 70)
    feature_level_check(model, processor, prompt, in_sample_ep, proprio_norm_stats, "in-sample, untrained")
    feature_level_check(model, processor, prompt, held_out_ep, proprio_norm_stats, "held-out, untrained")

    # === Load trained weights ===
    print(f"\nLoading trained weights from {VJEPA_WEIGHTS} ...")
    state_dict = torch.load(VJEPA_WEIGHTS, map_location=model.device)
    model.vjepa_predictor.load_state_dict(state_dict)
    model.vjepa_predictor.to(dtype=torch.bfloat16)

    print("\n" + "=" * 70)
    print("TRAINED (300-step smoke test)")
    print("=" * 70)
    feature_level_check(model, processor, prompt, in_sample_ep, proprio_norm_stats, "in-sample, trained")
    feature_level_check(model, processor, prompt, held_out_ep, proprio_norm_stats, "held-out, trained")

    # === Check 2: single-frame 3-way action comparison, on the held-out episode's frame ===
    env.reset()
    obs = env.set_init_state(initial_states[5])
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    # advance a bit further into the episode for a more "mid-task" frame
    for _ in range(80):
        obs, _, done, _ = env.step([0.0, 0.0, 0.0, 0, 0, 0, -1])
        if done:
            break
    three_way_action_check(cfg, model, processor, task_description, obs, resize_size, action_head, proprio_projector, proprio_norm_stats, prompt)


if __name__ == "__main__":
    main()
