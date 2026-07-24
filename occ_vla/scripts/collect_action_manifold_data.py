"""Collects raw 7-dim pi0.5 action vectors on mug_in_microwave for the
action-manifold PCA analysis (2026-07-20): a "background manifold" from
normal baseline steps (pi0.5's own typical action distribution -- used
instead of official LIBERO human demos, which aren't downloaded in this
environment; arguably more relevant here since the question is whether
the blended action left pi0.5's OWN typical operating regime, not a
human teleoperator's), plus the specific blended/conflicting action
vectors from the ungated action_blending condition (the one already
shown to destructively cancel -- see
scripts/diagnose_blend_vector_conflict.py and CLAUDE.md) so they can be
projected onto that manifold's PCA basis afterward.

Requires only the pi0.5 RPC worker. Run:
  scripts/run_with_gpu_cleanup.sh python3 scripts/collect_action_manifold_data.py
"""

import collections
import json
import sys
from pathlib import Path

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
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, SCENE_BLEND_ALPHA  # noqa: E402
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta  # noqa: E402

import os  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
SUITE = "libero_10"
TASK_ID = 9  # mug_in_microwave
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05
N_BACKGROUND_EPISODES = 5
N_CONFLICT_EPISODES = 5

OUT_PATH = _ROOT / "action_manifold_data.json"


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


def target_centroid_224(occ_env, obs):
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


def collect_background(episode_idx: int) -> list:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(
        benchmark_suite=SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    action_queue = collections.deque()
    actions_log = []
    for step in range(MAX_STEPS):
        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])
        actions_log.append(action.tolist())
        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break
    return actions_log


def collect_conflict(episode_idx: int) -> list:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(
        benchmark_suite=SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
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
    events = []

    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        occluded = arm_s_occ >= GATE_THRESHOLD

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        if occluded and last_known_position is not None:
            vla_raw = action.copy()
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M

            vla_xy = vla_raw[:2].astype(np.float64)
            vla_norm = np.linalg.norm(vla_xy)
            pklp_norm = np.linalg.norm(pklp_delta_xy)
            cos_angle = (
                float(np.dot(vla_xy, pklp_delta_xy) / (vla_norm * pklp_norm))
                if vla_norm > 1e-8 and pklp_norm > 1e-8 else None
            )
            blended_xy = (1.0 - SCENE_BLEND_ALPHA) * vla_xy + SCENE_BLEND_ALPHA * pklp_delta_xy
            blended = action.copy()
            blended[:2] = np.clip(blended_xy, -1.0, 1.0)

            events.append({
                "step": step, "cos_angle": cos_angle,
                "vla_raw": vla_raw.tolist(), "blended": blended.tolist(),
            })
            action = blended  # ungated: always apply, matching the original harmful condition

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break
    return events


def main():
    background = []
    for ep in range(N_BACKGROUND_EPISODES):
        actions = collect_background(ep)
        background.extend(actions)
        print(f"[background ep{ep}] {len(actions)} steps collected (total={len(background)})", flush=True)

    conflict_events = []
    for ep in range(N_CONFLICT_EPISODES):
        events = collect_conflict(ep)
        conflict_events.extend(events)
        print(f"[conflict ep{ep}] {len(events)} blend events collected (total={len(conflict_events)})", flush=True)

    OUT_PATH.write_text(json.dumps({"background_actions": background, "conflict_events": conflict_events}, indent=2))
    print(f"wrote {OUT_PATH}: {len(background)} background actions, {len(conflict_events)} conflict events", flush=True)


if __name__ == "__main__":
    main()
