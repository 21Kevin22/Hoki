"""Single-episode test (user request, 2026-07-23): mug_in_microwave
gate_only, with last_known_position sourced from YOLO-World (real
open-vocab detector) instead of GT segmentation centroid. Everything
else (FK-based occlusion trigger would be a further step; this uses
the existing GT arm_s_occ trigger, matching hybrid_gate_injection's
mug_in_microwave setup, since that trigger already reliably engages
there) stays as already validated: decay-protected gated_blend_xy.

Confidence handling: YOLO-World confidence on this rendering style
tested at 0.05-0.31 for "mug" (real domain gap vs its real-photo
training data, see session log) -- NOT filtered aggressively here on
purpose, to see how the existing "hold last position, don't trust
noise" pattern behaves against this specific noise level as-is, before
any threshold tuning. CONF_THRESHOLD is deliberately low (0.03): only
reject near-zero-confidence garbage, let the position-holding logic
absorb the rest.

Run: python3 scripts/test_yolo_world_tracking_ep.py
"""

import collections
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


def yolo_target_position_224(raw_agentview_256, raw_h, raw_w):
    """raw_agentview_256 is the UNFLIPPED sim render (matches how
    preprocess_image's input is later flipped) -- detect on this, then
    convert the box center into the flipped 224 space CameraProjector
    expects, same convention as target_centroid_224 elsewhere."""
    boxes, confs, classes = call_yolo_world(raw_agentview_256, CLASS_NAMES)
    if len(confs) == 0 or confs[0] < CONF_THRESHOLD:
        return None, (float(confs[0]) if len(confs) else 0.0), (classes[0] if classes else None)
    x0, y0, x1, y1 = boxes[0]
    cx_raw, cy_raw = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    flipped_x = raw_w - 1 - cx_raw
    flipped_y = raw_h - 1 - cy_raw
    scale = RESIZE_SIZE / raw_h
    return np.array([flipped_x * scale, flipped_y * scale]), float(confs[0]), classes[0]


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
    raw_h, raw_w = obs[AGENTVIEW_KEY].shape[:2]

    last_known_position, conf, cls_name = yolo_target_position_224(obs[AGENTVIEW_KEY], raw_h, raw_w)
    print(f"initial detection: pos={last_known_position}, conf={conf:.3f}, class={cls_name}", flush=True)

    action_queue = collections.deque()
    consecutive_occluded_steps = 0
    blend_engaged_steps = 0
    yolo_calls = 0
    conf_log = []
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        occluded = arm_s_occ >= GATE_THRESHOLD
        consecutive_occluded_steps = consecutive_occluded_steps + 1 if occluded else 0

        pos_now, conf, cls_name = yolo_target_position_224(obs[AGENTVIEW_KEY], raw_h, raw_w)
        yolo_calls += 1
        conf_log.append(round(conf, 3))
        if pos_now is not None:
            last_known_position = pos_now

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
    print(f"blend_engaged_steps={blend_engaged_steps}, yolo_calls={yolo_calls}")
    print(f"confidence trace (every 20 steps): {conf_log[::20]}")
    print(f"confidence stats: min={min(conf_log):.3f}, max={max(conf_log):.3f}, mean={np.mean(conf_log):.3f}, frac_above_{CONF_THRESHOLD}={np.mean([c >= CONF_THRESHOLD for c in conf_log]):.2f}")


if __name__ == "__main__":
    main()
