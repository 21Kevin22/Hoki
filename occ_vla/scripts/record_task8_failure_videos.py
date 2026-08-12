"""
record_task8_failure_videos.py

Video capture for task8 (moka_pots) baseline failures already identified in
libero_occ_benchmark_full/task8.json (episode_idx 0,1,2 -- all timeouts,
done_step=520). Per user's explicit request (2026-08-12): before spending
compute on any proposed intervention, visually confirm WHETHER the failure
mode is visual/inference confusion synced with occlusion, or a physical/
control problem unrelated to occlusion -- run_libero_occ_benchmark.py's own
aggregate stats (occ_mean nearly identical for success/fail episodes)
couldn't resolve this, matching this project's own repeated "don't trust an
aggregate metric without looking at the actual images" discipline.

Re-runs the SAME episode_idx values (same seed=7, same init_state per index)
as the original baseline scan -- this checkpoint's L1-regression action head
is deterministic, so this reproduces the identical failure. Saves:
  - agentview MP4 (standard rollout view)
  - wrist MP4, with the current step's S_occ value burned into the frame
    (top-left text overlay) -- lets a human directly see, frame by frame,
    whether arm confusion/hesitation visually coincides with high S_occ,
    without needing to cross-reference a separate plot.
  - occlusion_trace_ep<N>.json -- raw per-step S_occ trace, for anyone who
    wants the numeric timing precisely instead of eyeballing the overlay.
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
import imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_image, get_libero_wrist_image, get_libero_dummy_action, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, process_action, TASK_MAX_STEPS,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_vla_action, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_libero_occ_benchmark import get_libero_env_seg, geom_ids_by_substring, find_segmentation_ids, get_wrist_seg  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
TASK_ID = 8
OCCLUSION_TRIGGER_THRESHOLD = 0.15


def overlay_text(img_arr, text):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 130, 18], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 0))
    return np.array(img)


def run_episode_recorded(cfg, env, task_description, model, processor, action_head, proprio_projector,
                          init_state, robot_geom_ids_, target_seg_ids, max_steps, out_dir, episode_idx):
    env.reset()
    obs = env.set_init_state(init_state)
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()
    sim = env.env.sim

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False
    occlusion_trace = []
    agent_frames, wrist_frames = [], []

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    while t < max_steps + cfg.num_steps_wait:
        target_px_now = int(np.isin(get_wrist_seg(env), target_seg_ids).sum())
        orig_alpha = sim.model.geom_rgba[robot_geom_ids_, 3].copy()
        sim.model.geom_rgba[robot_geom_ids_, 3] = 0.0
        sim.forward()
        target_px_revealed = int(np.isin(get_wrist_seg(env), target_seg_ids).sum())
        sim.model.geom_rgba[robot_geom_ids_, 3] = orig_alpha
        sim.forward()
        s_occ = float(1.0 - target_px_now / target_px_revealed) if target_px_revealed > 0 else 0.0
        occlusion_trace.append(s_occ)

        agent_img = get_libero_image(obs).copy()
        wrist_img = get_libero_wrist_image(obs).copy()
        step_num = t - cfg.num_steps_wait
        agent_frames.append(overlay_text(agent_img, f"t={step_num} s={s_occ:.2f}"))
        wrist_frames.append(overlay_text(wrist_img, f"t={step_num} s={s_occ:.2f}"))

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

    done_step = t - cfg.num_steps_wait
    agent_path = os.path.join(out_dir, f"task8_ep{episode_idx}_success={success}_agentview.mp4")
    wrist_path = os.path.join(out_dir, f"task8_ep{episode_idx}_success={success}_wrist.mp4")
    imageio.mimwrite(agent_path, agent_frames, fps=30)
    imageio.mimwrite(wrist_path, wrist_frames, fps=30)
    with open(os.path.join(out_dir, f"task8_ep{episode_idx}_occlusion_trace.json"), "w") as f:
        json.dump({"episode_idx": episode_idx, "success": success, "done_step": done_step,
                   "occlusion_trace": occlusion_trace}, f)
    print(f"[record] ep{episode_idx} success={success} done_step={done_step} "
          f"agent_video={agent_path} wrist_video={wrist_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=str, default="task8_failure_videos")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    resize_size = get_image_resize_size(cfg)
    max_steps = TASK_MAX_STEPS["libero_10"]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(TASK_ID)
    task_description = task.language
    init_states = task_suite.get_task_init_states(TASK_ID)

    env = get_libero_env_seg(task, resolution=resize_size)
    env.reset()
    env.set_init_state(init_states[0])
    for _ in range(cfg.num_steps_wait):
        env.step(get_libero_dummy_action(cfg.model_family))
    sim = env.env.sim
    robot_geom_ids_ = geom_ids_by_substring(sim, ("robot", "panda", "gripper", "mount"))
    assert robot_geom_ids_, "no robot geoms matched"

    obj_body_name, target_seg_ids = None, None
    for candidate in env.obj_of_interest:
        cand_geom_ids = geom_ids_by_substring(sim, (candidate.lower(),))
        if not cand_geom_ids:
            continue
        cand_seg_ids = find_segmentation_ids(env, sim, cand_geom_ids)
        if not cand_seg_ids:
            continue
        obj_body_name, target_seg_ids = candidate, cand_seg_ids
        break
    assert target_seg_ids is not None, f"none of {env.obj_of_interest} resolved"
    print(f"[record] target='{obj_body_name}' target_seg_ids={target_seg_ids}")

    for ep in args.episode_ids:
        run_episode_recorded(cfg, env, task_description, model, processor, action_head, proprio_projector,
                              init_states[ep], robot_geom_ids_, target_seg_ids, max_steps, args.out_dir, ep)


if __name__ == "__main__":
    main()
