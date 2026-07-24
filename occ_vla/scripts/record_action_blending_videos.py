"""Records one labeled video for a specific (task, condition, episode)
combination from the action-blending n=10 experiment
(action_blending_{moka_pots,bowl_top_drawer}_results.json), so success
and failure examples of baseline vs action_blending can be inspected
visually -- same "don't trust the metric alone, look at the video"
practice as elsewhere this session.

Re-runs the exact episode_idx (-> same init_state_idx, same SEED) with
the same per-step logic as run_action_blending_pipeline.py, so the
condition/env are reproduced; note pi0.5's own action sampling isn't
guaranteed bit-identical run to run, so the outcome (success/failure)
is checked against what's expected and reported, not assumed.

Requires only the pi0.5 RPC worker (.rpc/pi05 or override via
OCC_VLA_PI05_RPC_DIR). Run:
  python3 scripts/record_action_blending_videos.py \
      --suite libero_spatial --task-id 4 --episode 0 \
      --condition baseline --max-steps 300 --label bowl_baseline_success
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from openpi_client import image_tools
from robosuite.utils.transform_utils import quat2axisangle

_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, SCENE_BLEND_ALPHA, gated_blend_xy  # noqa: E402
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta  # noqa: E402

import os  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05

OUT_DIR = _ROOT / "action_blending_videos"


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def target_centroid_224(occ_env, obs) -> np.ndarray | None:
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    target_mask_raw = seg_dict.get(occ_env.target_body_name)
    if target_mask_raw is None:
        return None
    target_mask = target_mask_raw.squeeze(-1) != 0
    if not target_mask.any():
        return None
    h, w = target_mask.shape
    ys, xs = np.where(target_mask)
    flipped_ys = h - 1 - ys
    flipped_xs = w - 1 - xs
    scale = RESIZE_SIZE / h
    return np.array([flipped_xs.mean() * scale, flipped_ys.mean() * scale])


def label(img, text, color=(0, 255, 0)):
    img = img.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img


def write_video(frames, out_path, fps=15):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def record(suite: str, task_id: int, episode_idx: int, condition: str, max_steps: int, out_label: str) -> None:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    init_states = bench.get_task_init_states(task_id)
    instruction = bench.get_task(task_id).language

    config = LiberoOccEnvConfig(
        benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    projector = CameraProjector.from_sim(occ_env._env.sim, "agentview", resolution=RESIZE_SIZE)  # noqa: SLF001

    last_known_position = target_centroid_224(occ_env, obs)
    action_queue = collections.deque()
    frames = []
    done_step = None

    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now

        base_image_raw = obs[AGENTVIEW_KEY]
        base_image = preprocess_image(base_image_raw)
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        occluded = arm_s_occ >= GATE_THRESHOLD

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        blended_this_step = False
        if condition == "action_blending" and occluded and last_known_position is not None:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            # Current architecture (2026-07-18): gated_blend_xy, not a
            # raw fixed-alpha blend -- see its docstring in runtime.py.
            vla_xy = action[:2].astype(np.float64)
            blended = gated_blend_xy(vla_xy, pklp_delta_xy, SCENE_BLEND_ALPHA)
            action = action.copy()
            action[:2] = np.clip(blended, -1.0, 1.0)
            blended_this_step = not np.allclose(blended, vla_xy)

        text = f"[{out_label}] step {step} s_occ={arm_s_occ:.2f}" + (" OCC" if occluded else "") + (" BLEND" if blended_this_step else "")
        color = (255, 0, 255) if blended_this_step else ((0, 0, 255) if occluded else (0, 255, 0))
        frames.append(label(base_image, text, color))

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            done_step = step
            break

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{out_label}.mp4"
    write_video(frames, out_path)
    outcome = "SUCCESS" if done_step is not None else "FAILURE(timeout)"
    print(f"[{out_label}] wrote {out_path} ({len(frames)} frames, done_step={done_step}, outcome={outcome})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--condition", choices=["baseline", "action_blending"], required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    record(args.suite, args.task_id, args.episode, args.condition, args.max_steps, args.label)


if __name__ == "__main__":
    main()
