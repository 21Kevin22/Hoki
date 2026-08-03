"""
collect_failure_probe_data.py

Collects (per-VLA-call LLM hidden-state activation, episode outcome) pairs
under wrist-camera partial occlusion with NO vjepa correction (`wrist_partial`
-- raw occlusion), for training a linear failure-prediction probe (occ_vla,
2026-08-03). Follows arXiv:2606.29699's finding that near-term OpenVLA
failure under visual distribution shift (occlusion) is linearly decodable
from feedforward activations -- tests whether that holds for OpenVLA-OFT
too, and whether such a probe could gate when to trust a trained
occlusion-recovery predictor (rather than the "gate" approach itself fixing
generalization -- see project memory: gating decides WHEN to correct, it
does not fix a predictor whose correction content is wrong for the task).

The hidden state is `vla.predict_action`'s own action-token final-layer LLM
hidden state (mean-pooled over the action-chunk dimension) -- already
computed internally for every call, not a new forward pass. Captured via
`get_vla_action(..., return_hidden_states=True)`.

Saves one .npz per episode to --out-dir:
  activations: (n_calls, D) float32 -- one row per VLA-query step
  success: bool, done_step: int, task_suite: str, task_id: int

Run with the openvla-oft conda env:
  python scripts/collect_failure_probe_data.py \
    --task-suite libero_10 --task-id 8 --num-episodes 15 --out-dir failure_probe_data_moka
"""

import argparse
import os
import sys
import time
from collections import deque

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = "/home/ubuntu/slocal1/Hoki/occ_vla/thirdparty/openvla-oft"
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, TaskSuite, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla_action  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, get_model, set_seed_everywhere  # noqa: E402
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action  # noqa: E402
from run_oft_camera_dropout_eval import prepare_observation  # noqa: E402


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode_and_collect(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_state, max_steps):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    activations = []
    t = 0
    n_calls = 0
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, _, _ = prepare_observation(obs, resize_size, occlude="wrist_partial", num_images=cfg.num_images_in_input)

            if len(action_queue) == 0:
                actions, hidden_state = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film,
                    occlusion_mask=None, return_hidden_states=True,
                )
                action_queue.extend(actions)
                activations.append(hidden_state)
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
        "activations": np.stack(activations).astype(np.float32),
        "success": success,
        "done_step": t,
        "n_calls": n_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=15)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--out-dir", default="failure_probe_data")
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

    n_success = 0
    end_episode = args.start_episode + args.num_episodes
    for ep in range(args.start_episode, end_episode):
        t0 = time.time()
        data = run_episode_and_collect(
            cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
            initial_states[ep], max_steps,
        )
        data["task_suite"] = args.task_suite
        data["task_id"] = args.task_id
        n_success += int(data["success"])
        out_path = os.path.join(args.out_dir, f"episode_{ep:03d}.npz")
        np.savez_compressed(out_path, **data)
        print(f"ep{ep}: success={data['success']} done_step={data['done_step']} n_calls={data['n_calls']} wall={time.time()-t0:.1f}s -> {out_path}")

    print(f"\nDone: {n_success}/{args.num_episodes} succeeded despite wrist_partial occlusion (episodes {args.start_episode}-{end_episode-1})")


if __name__ == "__main__":
    main()
