"""Test step-based blend-alpha decay (user request, 2026-07-23) on
moka_pots episode 7 -- the confirmed case where gate_only's continuous
correction over a genuinely long, unbroken 520-step occlusion window
prevented task completion (baseline solves the same init_state cleanly
in 375 steps; gate_only never finishes). Chosen over the two
alternatives (progress-stall detection, hard timeout+cooldown) because
it's deterministic and doesn't introduce a new indirect signal (no
"is progress happening" heuristic to itself validate -- exactly the
kind of proxy that burned moka_pots' occlusion detection earlier
today).

Decay rule: once fk_s_occ has been >= GATE_THRESHOLD for more than
DECAY_START_STEPS consecutive steps, linearly ramp blend_alpha down to
0 over the following DECAY_WINDOW steps (fully released, i.e. pure
pi0.5 action, by consecutive-occluded-step DECAY_START_STEPS +
DECAY_WINDOW). Counter resets to 0 the moment occlusion clears.

Run: python3 scripts/test_ep7_gate_decay.py
"""

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

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
BENCHMARK_SUITE = "libero_10"
TASK_ID = 8
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05
EPISODE_IDX = 7
DECAY_START_STEPS = 30
DECAY_WINDOW = 50


def preprocess_image(raw_image):
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt):
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, {"base_image": base_image, "wrist_image": wrist_image, "state": state}, {"prompt": prompt})
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
    flipped_ys, flipped_xs = h - 1 - ys, w - 1 - xs
    scale = RESIZE_SIZE / h
    return np.array([flipped_xs.mean() * scale, flipped_ys.mean() * scale])


def robot_geom_ids_for(occ_env):
    model = occ_env._env.sim.model  # noqa: SLF001
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))}
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


def target_clear_bbox_mask_224(occ_env):
    mask = np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=np.uint8)
    clear = occ_env._target_mask_clear  # noqa: SLF001
    if clear is None or not clear.any():
        return mask.astype(bool)
    h, w = clear.shape
    ys, xs = np.where(clear)
    flipped_ys, flipped_xs = h - 1 - ys, w - 1 - xs
    scale = RESIZE_SIZE / h
    x0, x1 = int(flipped_xs.min() * scale), int(flipped_xs.max() * scale)
    y0, y1 = int(flipped_ys.min() * scale), int(flipped_ys.max() * scale)
    mask[max(0, y0):min(RESIZE_SIZE, y1 + 1), max(0, x0):min(RESIZE_SIZE, x1 + 1)] = 1
    return mask.astype(bool)


def geom_radius_m(model, gid):
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]
    radius = float(size[0]) if gtype in (2, 3, 5) else float(np.median(size))
    return radius if radius > 1e-4 else 0.02


def fk_arm_mask_224(occ_env, projector, robot_geom_ids):
    sim = occ_env._env.sim  # noqa: SLF001
    model = sim.model
    mask = np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=np.uint8)
    f_px = (RESIZE_SIZE / 2) / np.tan(np.radians(projector.fovy_deg) / 2)
    for gid in robot_geom_ids:
        pos = np.array(sim.data.geom_xpos[gid], dtype=np.float64)
        depth = -(projector.cam_mat.T @ (pos - projector.cam_pos))[2]
        if depth <= 1e-3:
            continue
        radius_m = max(geom_radius_m(model, gid), 0.015)
        px, py = projector.project(pos)
        pixel_radius = max(1, int(f_px * radius_m / depth))
        if -pixel_radius <= px <= RESIZE_SIZE + pixel_radius and -pixel_radius <= py <= RESIZE_SIZE + pixel_radius:
            cv2.circle(mask, (int(px), int(py)), pixel_radius, 1, -1)
    return mask.astype(bool)


def compute_fk_s_occ(fk_mask, target_bbox_mask):
    target_area = target_bbox_mask.sum()
    return float((fk_mask & target_bbox_mask).sum()) / float(target_area) if target_area else 0.0


def decayed_alpha(consecutive_occluded_steps: int) -> float:
    if consecutive_occluded_steps <= DECAY_START_STEPS:
        return SCENE_BLEND_ALPHA
    frac_into_decay = (consecutive_occluded_steps - DECAY_START_STEPS) / DECAY_WINDOW
    return SCENE_BLEND_ALPHA * max(0.0, 1.0 - frac_into_decay)


def main():
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(BENCHMARK_SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(benchmark_suite=BENCHMARK_SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
                                 init_state_idx=EPISODE_IDX % len(init_states), seed=SEED, place_occluder=False)
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    projector = CameraProjector.from_sim(occ_env._env.sim, "agentview", resolution=RESIZE_SIZE)  # noqa: SLF001
    robot_geom_ids = robot_geom_ids_for(occ_env)
    target_bbox_mask = target_clear_bbox_mask_224(occ_env)
    last_known_position = target_centroid_224(occ_env, obs)

    action_queue = collections.deque()
    consecutive_occluded_steps = 0
    alpha_log = []
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        fk_mask = fk_arm_mask_224(occ_env, projector, robot_geom_ids)
        fk_s_occ = compute_fk_s_occ(fk_mask, target_bbox_mask)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now
        occluded = fk_s_occ >= GATE_THRESHOLD
        consecutive_occluded_steps = consecutive_occluded_steps + 1 if occluded else 0

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        alpha = decayed_alpha(consecutive_occluded_steps)
        if occluded and last_known_position is not None and alpha > 0:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            action = action.copy()
            blended = gated_blend_xy(action[:2].astype(np.float64), pklp_delta_xy, alpha)
            action[:2] = np.clip(blended, -1.0, 1.0)
        alpha_log.append(round(alpha, 3))

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break

    print(f"done_step={step if done else None}")
    print(f"final consecutive_occluded_steps={consecutive_occluded_steps}")
    print(f"alpha trace (every 20 steps): {alpha_log[::20]}")


if __name__ == "__main__":
    main()
