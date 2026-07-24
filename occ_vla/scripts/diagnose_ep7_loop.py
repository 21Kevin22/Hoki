"""Diagnostic (user request, 2026-07-23): moka_pots episode 7 (gate_only)
failed with blend_engaged_steps=520 -- the gate active for literally
every step of the episode -- identically under BOTH mask_centroid and
bbox_centroid tracking (see run_moka_pots_fk_occlusion.py results),
meaning it isn't a tracking-precision artifact. Before designing a
"safety device" to suppress this, find out whether it's a genuine
long, sustained occlusion (this init_state's task geometry keeps the
arm between camera and target for a long time) or an oscillating
on/off loop (a control-feedback artifact). Logs fk_s_occ, eef_pos, and
gate-engaged flag every step.

Run: python3 scripts/diagnose_ep7_loop.py
"""

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
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, SCENE_BLEND_ALPHA, gated_blend_xy  # noqa: E402
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta  # noqa: E402

import collections  # noqa: E402
import cv2  # noqa: E402

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
    log = []
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        fk_mask = fk_arm_mask_224(occ_env, projector, robot_geom_ids)
        fk_s_occ = compute_fk_s_occ(fk_mask, target_bbox_mask)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now
        occluded = fk_s_occ >= GATE_THRESHOLD

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        gate_fired = False
        if occluded and last_known_position is not None:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            action = action.copy()
            blended = gated_blend_xy(action[:2].astype(np.float64), pklp_delta_xy, SCENE_BLEND_ALPHA)
            action[:2] = np.clip(blended, -1.0, 1.0)
            gate_fired = True

        log.append({
            "step": step, "fk_s_occ": round(fk_s_occ, 4), "arm_s_occ": round(arm_s_occ, 4),
            "occluded": occluded, "gate_fired": gate_fired,
            "eef_pos": obs["robot0_eef_pos"].tolist(), "last_known_position": last_known_position.tolist() if last_known_position is not None else None,
        })

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break

    Path(_ROOT / "ep7_loop_diagnosis.json").write_text(json.dumps({"done_step": step if done else None, "log": log}, indent=2))
    print(f"done_step={step if done else None}, n_steps_logged={len(log)}")
    occluded_run_lengths = []
    cur = 0
    for entry in log:
        if entry["occluded"]:
            cur += 1
        else:
            if cur > 0:
                occluded_run_lengths.append(cur)
            cur = 0
    if cur > 0:
        occluded_run_lengths.append(cur)
    print(f"occluded run lengths (consecutive steps with fk_s_occ>=0.30): {occluded_run_lengths}")
    eef_positions = np.array([e["eef_pos"] for e in log])
    print(f"eef_pos range: min={eef_positions.min(axis=0)}, max={eef_positions.max(axis=0)}")
    print(f"eef_pos std: {eef_positions.std(axis=0)}")


if __name__ == "__main__":
    main()
