"""
collect_oft_onpolicy_rollout_data.py

Collects on-policy rollout data for training VJEPA_LatentDynamicsPredictor:
real (agentview_resized, wrist_resized, proprio) triples at every VLA-query
step of real moka_pots rollouts, using the existing (unmodified,
occlusion_mask=None) OpenVLA-OFT checkpoint -- matches the deployment
distribution exactly, per the user's own stated reasoning for choosing
on-policy data over the RLDS demo dataset.

Saves one .npz per episode to --out-dir. Each .npz holds:
  agentview: (T, 224, 224, 3) uint8
  wrist:     (T, 224, 224, 3) uint8
  proprio:   (T, 8) float32  -- RAW (pre-normalization) state, matching
             obs["state"] in run_oft_camera_dropout_eval.py's prepare_observation
  success:   bool, done_step: int  -- for reference only, not used in training

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/collect_oft_onpolicy_rollout_data.py --num-episodes 5
"""

import argparse
import os
import sys
import time
from collections import deque

# Derived from __file__ (was hardcoded to the original project server's
# path, "/home/ubuntu/slocal1/Hoki/occ_vla/thirdparty/openvla-oft" -- broke
# on any other machine, e.g. a Kaggle clone under /root/oft_work/Hoki/...).
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, TaskSuite, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, resize_image_for_policy  # noqa: E402
from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model, set_seed_everywhere  # noqa: E402
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action  # noqa: E402


def prepare_observation(obs, resize_size):
    img_resized = resize_image_for_policy(get_libero_image(obs), resize_size)
    wrist_resized = resize_image_for_policy(get_libero_wrist_image(obs), resize_size)
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return {"full_image": img_resized, "wrist_image": wrist_resized, "state": state}


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode_and_collect(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_state, max_steps):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    model.reset_vjepa_state()  # keep the trajectory clean of any cross-episode state

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    agentview_frames, wrist_frames, proprio_frames = [], [], []
    t = 0
    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation = prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            # Cache this VLA-query step's real (unoccluded) observation --
            # occlusion_mask=None throughout, matching real deployment.
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
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return {
        "agentview": np.stack(agentview_frames).astype(np.uint8),
        "wrist": np.stack(wrist_frames).astype(np.uint8),
        "proprio": np.stack(proprio_frames).astype(np.float32),
        "success": success,
        "done_step": t,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)  # moka_pots
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0, help="episode index (and initial_states index) to start from -- use to append more episodes without recomputing/overwriting existing ones")
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--out-dir", default="oft_onpolicy_rollout_data")
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
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = TASK_MAX_STEPS[TaskSuite(args.task_suite)]
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    total_samples = 0
    end_episode = args.start_episode + args.num_episodes
    for ep in range(args.start_episode, end_episode):
        t0 = time.time()
        data = run_episode_and_collect(
            cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
            initial_states[ep], max_steps,
        )
        out_path = os.path.join(args.out_dir, f"episode_{ep:03d}.npz")
        np.savez_compressed(out_path, **data)
        total_samples += len(data["proprio"])
        print(
            f"ep{ep}: success={data['success']} done_step={data['done_step']} "
            f"n_samples={len(data['proprio'])} wall={time.time()-t0:.1f}s -> {out_path}"
        )

    print(f"\nTotal samples collected this run: {total_samples} across episodes {args.start_episode}-{end_episode-1}")


if __name__ == "__main__":
    main()
