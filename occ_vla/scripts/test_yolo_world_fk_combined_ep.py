"""Full integration (user request, 2026-07-23): combine the two
separately-validated privilege-removal pieces into one occlusion
signal, with zero GT segmentation involved in either side:

  - ARM side: FK-projected silhouette from true 3D geom poses (sim.data
    -- the sim analogue of joint-encoder + URDF FK), see
    run_moka_pots_fk_occlusion.py's fk_arm_mask_224/geom_radius_m.
  - TARGET side: YOLO-World open-vocab detection (yolo_world_worker.py,
    validated single-episode: 94% of frames above a low 0.03 confidence
    threshold, task completed in 267/520 steps), used for BOTH:
      (a) last_known_position -- continuously updated on confident
          detections, held otherwise (already validated pattern).
      (b) the occlusion-ratio denominator -- captured ONCE right after
          settle-wait (least-occluded moment) as the target's expected
          footprint, mirroring target_clear_bbox_mask_224's role
          exactly, just from a detector instead of GT segmentation.

fk_s_occ = Area(fk_arm_mask AND initial_target_bbox) / Area(initial_target_bbox)
gates the decay-protected blend (DECAY_START_STEPS/DECAY_WINDOW,
validated on moka_pots episode 7).

The only GT-derived signal left anywhere in this script is env-internal
success detection (done flag) -- not available to the policy/gate at
all, same as a real robot's task-completion signal would come from
some other source, not from what's being tested here.

Run: python3 scripts/test_yolo_world_fk_combined_ep.py
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

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, SCENE_BLEND_ALPHA, gated_blend_xy  # noqa: E402
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
YOLO_RPC_DIR = str(_ROOT / ".rpc" / "yolo_world")
BENCHMARK_SUITE = "libero_10"
TASK_ID = 9  # mug_in_microwave
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
GATE_THRESHOLD = 0.30
EPISODE_IDX = 0
DECAY_START_STEPS = 30
DECAY_WINDOW = 50
CONF_THRESHOLD = 0.03
CLASS_NAMES = "mug|cup|microwave"


def decayed_alpha(consecutive_occluded_steps: int) -> float:
    if consecutive_occluded_steps <= DECAY_START_STEPS:
        return SCENE_BLEND_ALPHA
    frac = (consecutive_occluded_steps - DECAY_START_STEPS) / DECAY_WINDOW
    return SCENE_BLEND_ALPHA * max(0.0, 1.0 - frac)


def preprocess_image(raw_image):
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt):
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, {"base_image": base_image, "wrist_image": wrist_image, "state": state}, {"prompt": prompt})
    return resp_arrays["actions"]


def call_yolo_world(image_raw_uint8, class_names: str):
    resp_arrays, resp_fields = rpc.call(YOLO_RPC_DIR, {"image": image_raw_uint8}, {"class_names": class_names}, timeout_s=30)
    boxes, confs = resp_arrays["boxes"], resp_arrays["confs"]
    classes = resp_fields["class_names_out"].split("|") if resp_fields["class_names_out"] else []
    return boxes, confs, classes


def raw_box_to_flipped_224(box, raw_h, raw_w):
    x0, y0, x1, y1 = box
    fx0, fx1 = raw_w - 1 - x1, raw_w - 1 - x0
    fy0, fy1 = raw_h - 1 - y1, raw_h - 1 - y0
    scale = RESIZE_SIZE / raw_h
    return np.array([fx0, fy0, fx1, fy1]) * scale


def yolo_target_detect(raw_agentview, raw_h, raw_w):
    boxes, confs, classes = call_yolo_world(raw_agentview, CLASS_NAMES)
    if len(confs) == 0 or confs[0] < CONF_THRESHOLD:
        return None, (float(confs[0]) if len(confs) else 0.0)
    box_224 = raw_box_to_flipped_224(boxes[0], raw_h, raw_w)
    return box_224, float(confs[0])


BBOX_PADDING_FRAC = 1.0  # 2026-07-23: 0.35 reduced max_fk_s_occ (0.97->0.81)
# but didn't change blend_engaged_steps at all and the episode still
# failed identically -- testing whether this is simply "still not
# enough padding" or a structural issue (arm naturally spends real time
# near this fixed region during mug-into-microwave placement,
# independent of denominator size). YOLO-World's box came in tight
# relative to the object's true visible extent (~38x37px on a mug),
# which made fk_s_occ spike toward 1.0 under fairly modest arm overlap
# and drove the episode to failure -- the target-tracking piece
# (last_known_position) itself was fine (confidence stayed a stable
# 0.23-0.30 throughout), so this is a denominator-size bias, not a
# tracking failure. Pad on each side before converting to a mask.


def bbox_to_mask(box_224):
    mask = np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=np.uint8)
    x0, y0, x1, y1 = box_224
    w, h = x1 - x0, y1 - y0
    px, py = w * BBOX_PADDING_FRAC, h * BBOX_PADDING_FRAC
    x0, x1 = x0 - px, x1 + px
    y0, y1 = y0 - py, y1 + py
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0, x1 = sorted((np.clip(x0, 0, RESIZE_SIZE), np.clip(x1, 0, RESIZE_SIZE)))
    y0, y1 = sorted((np.clip(y0, 0, RESIZE_SIZE), np.clip(y1, 0, RESIZE_SIZE)))
    mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


def geom_radius_m(model, gid):
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]
    radius = float(size[0]) if gtype in (2, 3, 5) else float(np.median(size))
    return radius if radius > 1e-4 else 0.02


def robot_geom_ids_for(occ_env):
    model = occ_env._env.sim.model  # noqa: SLF001
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))}
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


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
    occ_env.capture_clear_baseline(obs)  # GT still used for env-internal S_occ LOGGING ONLY, not the gate signal below

    projector = CameraProjector.from_sim(occ_env._env.sim, "agentview", resolution=RESIZE_SIZE)  # noqa: SLF001
    robot_geom_ids = robot_geom_ids_for(occ_env)
    raw_h, raw_w = obs[AGENTVIEW_KEY].shape[:2]

    init_box_224, init_conf = yolo_target_detect(obs[AGENTVIEW_KEY], raw_h, raw_w)
    if init_box_224 is None:
        print(f"WARNING: no confident initial detection (conf={init_conf:.3f}), using a generous center-ish fallback box")
        init_box_224 = np.array([RESIZE_SIZE * 0.3, RESIZE_SIZE * 0.1, RESIZE_SIZE * 0.7, RESIZE_SIZE * 0.5])
    target_bbox_mask = bbox_to_mask(init_box_224)
    print(f"initial target bbox: {init_box_224}, conf={init_conf:.3f}", flush=True)

    last_known_position = np.array([(init_box_224[0] + init_box_224[2]) / 2, (init_box_224[1] + init_box_224[3]) / 2])

    action_queue = collections.deque()
    consecutive_occluded_steps = 0
    blend_engaged_steps = 0
    conf_log, fk_occ_log = [], []
    for step in range(MAX_STEPS):
        fk_mask = fk_arm_mask_224(occ_env, projector, robot_geom_ids)
        fk_s_occ = compute_fk_s_occ(fk_mask, target_bbox_mask)
        occluded = fk_s_occ >= GATE_THRESHOLD
        consecutive_occluded_steps = consecutive_occluded_steps + 1 if occluded else 0
        fk_occ_log.append(round(fk_s_occ, 3))

        box_now, conf = yolo_target_detect(obs[AGENTVIEW_KEY], raw_h, raw_w)
        conf_log.append(round(conf, 3))
        if box_now is not None:
            last_known_position = np.array([(box_now[0] + box_now[2]) / 2, (box_now[1] + box_now[3]) / 2])

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
            blend_engaged_steps += 1

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break

    print(f"\ndone_step={step if done else None}")
    print(f"blend_engaged_steps={blend_engaged_steps}")
    print(f"fk_s_occ trace (every 20 steps): {fk_occ_log[::20]}")
    print(f"yolo confidence trace (every 20 steps): {conf_log[::20]}")
    print(f"max_fk_s_occ={max(fk_occ_log):.3f}, mean_conf={np.mean(conf_log):.3f}")


if __name__ == "__main__":
    main()
