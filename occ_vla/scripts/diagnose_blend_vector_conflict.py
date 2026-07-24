"""Diagnostic for the mug_in_microwave regression found in the n=10
action_blending experiment (10/10 baseline -> 4/10 action_blending,
"stuck hovering near the occlusion boundary" confirmed on video).

Hypothesis under test: SCENE_BLEND_ALPHA=0.5 linearly blends pi0.5's own
XY action with the PKLP geometric correction. If the two point in
substantially different (or opposing) directions, a linear blend of two
vectors of similar magnitude but conflicting direction can produce a
resultant with much smaller norm than either input -- i.e. the arm
"freezes" not because either signal is wrong, but because averaging them
destructively cancels. This script does not change the blending logic;
it only logs, at every step where blending actually engages,
||vla_xy||, ||pklp_xy||, the cosine angle between them, and the
resulting ||blended_xy|| (pre-clip), so the hypothesis can be checked
against real data before any fix is designed.

Requires only the pi0.5 RPC worker (.rpc/pi05 or override via
OCC_VLA_PI05_RPC_DIR). Run:
  python3 scripts/diagnose_blend_vector_conflict.py --episodes 0 1 2 6 8 9
"""

import argparse
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
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05

OUT_PATH = _ROOT / "blend_vector_conflict_diagnostic.json"


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


def run_episode(episode_idx: int, max_steps: int) -> dict:
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
    blend_events = []
    done_step = None

    for step in range(max_steps):
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
            vla_xy = action[:2].astype(np.float64)
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M

            vla_norm = float(np.linalg.norm(vla_xy))
            pklp_norm = float(np.linalg.norm(pklp_xy))
            cos_angle = (
                float(np.dot(vla_xy, pklp_xy) / (vla_norm * pklp_norm))
                if vla_norm > 1e-8 and pklp_norm > 1e-8
                else None
            )
            blended = (1.0 - SCENE_BLEND_ALPHA) * vla_xy + SCENE_BLEND_ALPHA * pklp_xy
            blended_norm = float(np.linalg.norm(blended))

            blend_events.append({
                "step": step, "s_occ": float(arm_s_occ),
                "vla_norm": vla_norm, "pklp_norm": pklp_norm,
                "cos_angle": cos_angle, "blended_norm": blended_norm,
            })

            action = action.copy()
            action[:2] = np.clip(blended, -1.0, 1.0)

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            done_step = step
            break

    return {"episode": episode_idx, "done_step": done_step, "blend_events": blend_events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=520)
    args = parser.parse_args()

    all_results = []
    for ep in args.episodes:
        result = run_episode(ep, args.max_steps)
        all_results.append(result)
        events = result["blend_events"]
        cosines = [e["cos_angle"] for e in events if e["cos_angle"] is not None]
        blended_norms = [e["blended_norm"] for e in events]
        vla_norms = [e["vla_norm"] for e in events]
        outcome = "SUCCESS" if result["done_step"] is not None else "FAILURE(timeout)"
        print(
            f"[ep{ep}] {outcome} done_step={result['done_step']} n_blend_steps={len(events)} "
            f"mean_cos={np.mean(cosines) if cosines else float('nan'):.3f} "
            f"frac_opposing(cos<0)={np.mean([c < 0 for c in cosines]) if cosines else float('nan'):.0%} "
            f"mean_vla_norm={np.mean(vla_norms) if vla_norms else float('nan'):.3f} "
            f"mean_blended_norm={np.mean(blended_norms) if blended_norms else float('nan'):.3f}",
            flush=True,
        )

    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
