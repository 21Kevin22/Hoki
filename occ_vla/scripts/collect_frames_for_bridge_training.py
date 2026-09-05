"""occ_vla addition (2026-09-01), per user request ("それぞれのタスクごとに
学習して一般化できないですか？"): a generic, suite-agnostic real-agentview-
frame collector for training TASK/CHECKPOINT-SPECIFIC V-JEPA2 bridging
projections (train_vjepa2_bridge_projections.py), instead of reusing
vjepa2_bridge_proj_base (trained only on LIBERO-10 task1 data) across
every checkpoint/suite -- the cross-checkpoint mismatch flagged as a
real, unverified risk after Goal task7's severe (60%->15%) regression
in the full n=20 sweep.

Reuses run_libero_occluded_oracle_headroom.py's own real helper
functions (env setup, action inference) -- no reimplementation of the
policy-calling loop. Runs plain baseline rollouts (no occluder
compositing, no proactive correction) on the REAL occluded suite (same
scene the eval itself uses), saving each replan step's real agentview
frame as `{uid}_agentview.png` -- the exact glob pattern
train_vjepa2_bridge_projections.py already expects.
"""
import argparse
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from collections import deque  # noqa: E402
from PIL import Image  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_wrist_image, quat2axisangle  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, check_unnorm_key, process_action  # noqa: E402
from experiments.robot.openvla_utils import get_action_head, get_proprio_projector, get_vla, get_vla_action, get_processor  # noqa: E402

from run_libero_occluded_oracle_headroom import get_libero_env_seg, get_agentview_frames, OCCLUDED_SUITE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="10", choices=["10", "spatial", "object", "goal"])
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n-episodes", type=int, default=15)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(SCRIPTS_DIR, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    suite_occ_name = {"10": OCCLUDED_SUITE, "spatial": "libero_spatial_occluded",
                       "object": "libero_object_occluded", "goal": "libero_goal_occluded"}[args.suite]
    suite_stock_name = {"10": "libero_10", "spatial": "libero_spatial",
                         "object": "libero_object", "goal": "libero_goal"}[args.suite]

    bm = benchmark.get_benchmark_dict()
    occ_suite = bm[suite_occ_name]()
    task = occ_suite.get_task(args.task_id)
    task_description = task.language
    max_steps = TASK_MAX_STEPS[suite_stock_name]

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=suite_stock_name, seed=7,
    )
    model = get_vla(cfg)
    check_unnorm_key(cfg, model)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    env = get_libero_env_seg(task, resolution=args.resolution)
    env.seed(0)
    init_states = occ_suite.get_task_init_states(args.task_id)

    n_saved = 0
    for ep in range(min(args.n_episodes, len(init_states))):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        t = 0
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
        success = False
        while t < max_steps + 10:
            agentview_color, _ = get_agentview_frames(env, args.resolution)
            wrist_img = get_libero_wrist_image(obs).copy()
            if len(action_queue) == 0:
                observation = {
                    "full_image": agentview_color,
                    "wrist_image": wrist_img,
                    "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                }
                actions = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film,
                )
                action_queue.extend(actions)
                uid = f"task{args.task_id}_ep{ep}_t{t:05d}"
                Image.fromarray(agentview_color).save(os.path.join(args.out_dir, f"{uid}_agentview.png"))
                n_saved += 1
            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
        print(f"  ep{ep}: success={success} done_step={t} n_saved_so_far={n_saved}")
    env.close()
    print(f"\ntotal frames saved: {n_saved} -> {args.out_dir}")


if __name__ == "__main__":
    main()
