"""
run_libero_occluded_fast_scan.py

Fast baseline success-rate scan on the REAL LIBERO-Occ benchmark
(libero_10_occluded -- real deliberately-placed 3D occluder objects, e.g.
wooden_cabinet_1, per register_libero_occ_suites.py's own docstring), NOT
the plain libero_10 suite run_libero_occ_benchmark.py used earlier
(2026-08-12) -- that whole scan turned out to measure only incidental
self-occlusion on the UNMODIFIED scene, not the actual published benchmark
condition. Caught by direct user question ("見る限り同じに見えてしまいますが").

Deliberately STRIPPED DOWN vs run_libero_occ_benchmark.py for speed, per
user's explicit request ("より早く実行する技術があればそれを使いながら"):
  - NO per-step occlusion measurement (no camera_segmentations, no live
    hide-and-reveal double-render) -- this was the dominant per-step cost
    in the earlier script (2 renders/step) and is not needed for a first-
    pass success-rate-only scan. The occluder here is a REAL, ALWAYS-
    PRESENT physical object (not situational self-occlusion), so there's
    no "is it occluded THIS step" question to answer for a baseline
    success-rate read -- only "did the episode succeed."
  - Plain get_libero_env (matches the original libero_utils.py helper
    exactly, no custom OffScreenRenderEnv segmentation variant).
  - n=10 first pass (not 20) -- pilot before scaling, per this project's
    own established discipline; re-run promising/borderline tasks at
    n=20 afterward rather than paying full cost upfront on all 10.

Task order is DIFFERENT from plain libero_10 (alphabetical-by-BDDL-filename,
not the stock task_order permutation) -- confirmed directly via
get_task_names() before writing this script; e.g. "both moka pots" is
index 3 here, not 8. Don't assume index parity with the earlier scan.
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
    get_libero_env, get_libero_image, get_libero_wrist_image, get_libero_dummy_action, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, process_action, TASK_MAX_STEPS,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_vla_action, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402
import torch  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
SUITE = "libero_10_occluded"


def run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector,
                 init_state, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    while t < max_steps + cfg.num_steps_wait:
        if len(action_queue) == 0:
            observation = {
                "full_image": get_libero_image(obs).copy(),
                "wrist_image": get_libero_wrist_image(obs).copy(),
                "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            }
            with torch.inference_mode():
                actions = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=None,
                )
            action_queue.extend(actions)
        action = np.array(action_queue.popleft(), dtype=float)
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            success = True
            break

    return {"success": success, "done_step": t - cfg.num_steps_wait,
            "timeout": (not success) and (t - cfg.num_steps_wait >= max_steps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="libero_occluded_fast_scan")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)  # norm_stats key is the checkpoint's own libero_10 key, suite-independent
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    resize_size = get_image_resize_size(cfg)
    max_steps = TASK_MAX_STEPS["libero_10"]  # same 10 underlying tasks/scenes, same horizon convention

    task_suite = benchmark.get_benchmark_dict()[SUITE]()

    summary = {}
    for task_id in args.task_ids:
        task = task_suite.get_task(task_id)
        task_description = task.language
        init_states = task_suite.get_task_init_states(task_id)
        n = min(args.n_episodes, len(init_states))
        print(f"\n[fastscan] === task_id={task_id} '{task_description}' n={n} ===")

        env, _ = get_libero_env(task, cfg.model_family, resolution=resize_size)

        results = []
        for i in range(n):
            res = run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector,
                               init_states[i], max_steps)
            res["episode_idx"] = i
            results.append(res)
            print(f"[fastscan] task{task_id} ep{i} success={res['success']} done_step={res['done_step']}")
            with open(os.path.join(args.results_dir, f"task{task_id}.json"), "w") as f:
                json.dump({"task_id": task_id, "task_description": task_description, "results": results}, f, indent=2)

        succ = sum(1 for r in results if r["success"])
        summary[task_id] = {"task_description": task_description, "n": n, "success": succ, "success_rate": succ / n}
        print(f"[fastscan] task{task_id} FINAL success={succ}/{n} ({100*succ/n:.1f}%)")

    print("\n[fastscan] === ALL TASKS DONE ===")
    for tid, s in sorted(summary.items(), key=lambda kv: kv[1]["success_rate"]):
        print(f"  task{tid} '{s['task_description'][:50]}': {s['success']}/{s['n']} ({100*s['success_rate']:.1f}%)")


if __name__ == "__main__":
    main()
