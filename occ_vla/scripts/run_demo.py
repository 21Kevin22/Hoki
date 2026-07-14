"""Orchestrator: runs in the base/LIBERO+robosuite environment, drives
one occluded LIBERO episode through ControlLoop's routing logic, using
the pi0.5 and MMaDA-8B RPC workers (scripts/_workers/) for real
inference, and writes an mp4 of the rollout.

Arm occlusion (arm_s_occ) is computed from LIBERO's own segmentation
camera (SegmentationRenderEnv.get_segmentation_instances), not a
hand-labeled/synthetic mask — real per-step occlusion, same source
eval/occluder.py's S_occ search uses.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from robosuite.utils.transform_utils import quat2axisangle

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.integration.occlusion_router import OcclusionRouter, OcclusionSignals, OcclusionSource  # noqa: E402
from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402

PI05_RPC_DIR = "/tmp/occ_vla_rpc/pi05"
MMADA_RPC_DIR = "/tmp/occ_vla_rpc/mmada"
MAX_STEPS = 60
INSTRUCTION = "pick up the black bowl and place it on the plate"


def call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None, cot_anchor=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    if subgoal_image is not None:
        arrays["subgoal_image"] = subgoal_image
    fields = {"prompt": prompt}
    if cot_anchor is not None:
        fields["cot_anchor"] = cot_anchor
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, fields)
    return resp_arrays["actions"]


def call_mmada(image, arm_pixel_mask, instruction, horizon=5):
    resp_arrays, _ = rpc.call(
        MMADA_RPC_DIR,
        {"image": image, "arm_pixel_mask": arm_pixel_mask.astype(np.uint8)},
        {"instruction": instruction, "horizon": horizon},
        timeout_s=180,
    )
    return resp_arrays["image"]


def compute_arm_s_occ(env, target_body_name: str, seg_image) -> tuple[float, np.ndarray]:
    """Real per-step arm-occlusion-of-target fraction from the live
    segmentation camera, following the same
    get_segmentation_instances-based approach as eval/occluder.py."""
    seg_dict = env.get_segmentation_instances(seg_image)
    target_mask = seg_dict["target_obj"] if "target_obj" in seg_dict else seg_dict.get(target_body_name)
    if target_mask is None:
        return 0.0, np.zeros(seg_image.shape[:2], dtype=bool)
    target_mask = target_mask.squeeze(-1) != 0
    arm_mask = seg_dict["robot"].squeeze(-1) != 0
    target_area = target_mask.sum()
    if target_area == 0:
        return 0.0, arm_mask
    overlap = (target_mask & arm_mask).sum()
    return float(overlap) / float(target_area), arm_mask


def annotate(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(frame, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    config = LiberoOccEnvConfig(benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.MEDIUM)
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    print(f"occluder placed, S_occ={occ_env.last_s_occ:.3f}", flush=True)

    router = OcclusionRouter()
    plausibility_checker = PlausibilityChecker()
    target_body_name = occ_env._env.obj_of_interest[0]  # noqa: SLF001

    frames = []
    for step in range(MAX_STEPS):
        t_step = time.time()
        base_image = obs[AGENTVIEW_KEY][::-1]  # robosuite renders upside down
        wrist_image = obs["robot0_eye_in_hand_image"][::-1]
        # LIBERO's standard 8-dim state (eef pos [3] + eef orientation as
        # axis-angle [3] + gripper qpos [2]) — not the raw
        # robot0_proprio-state (39-dim, joint pos/vel/eef/gripper
        # concatenated), which is what franka's norm_stats.json is shaped
        # for and what LiberoInputs/openpi convention expects. Confirmed
        # against the live pi0.5 worker: passing the 39-dim vector raised
        # ValueError: operands could not be broadcast together with
        # shapes (39,) (8,) inside openpi's quantile-normalization step.
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)
        seg_image = obs[AGENTVIEW_SEGMENTATION_KEY]

        arm_s_occ, arm_mask = compute_arm_s_occ(occ_env._env, target_body_name, seg_image)  # noqa: SLF001
        signals = OcclusionSignals(arm_s_occ=arm_s_occ, scene_dyn_occ=False)
        source = router.route(signals)
        print(f"step {step}: arm_s_occ={arm_s_occ:.3f} source={source.name}", flush=True)

        subgoal_image = None
        cot_anchor = None
        label = f"step {step} | arm_s_occ={arm_s_occ:.2f} | {source.name}"

        if source == OcclusionSource.SELF:
            t0 = time.time()
            print("  calling MMaDA...", flush=True)
            subgoal_image = call_mmada(base_image, arm_mask, INSTRUCTION)
            print(f"  MMaDA subgoal generated in {time.time() - t0:.1f}s", flush=True)
            score = plausibility_checker.score(subgoal_image, {"original_image": base_image, "arm_pixel_mask": arm_mask})
            label += f" | plausibility={score:.2f}"
            if score < 0.5:
                subgoal_image = None
                label += " -> FALLBACK"

        print("  calling pi0.5...", flush=True)
        t0 = time.time()
        actions = call_pi05(base_image, wrist_image, state, INSTRUCTION, subgoal_image, cot_anchor)
        print(f"  pi0.5 responded in {time.time() - t0:.1f}s", flush=True)
        action = actions[0]
        action = np.concatenate([action[:6], [1.0 if action[6] > 0 else -1.0]]) if action.shape[0] >= 7 else action

        frame = annotate(base_image, label)
        frames.append(frame)

        print("  stepping sim...", flush=True)
        step_result = occ_env.step(action[:7])
        obs = step_result[0]
        done = step_result[2]
        print(f"step {step} done in {time.time() - t_step:.2f}s total", flush=True)
        if done:
            print(f"episode ended at step {step}", flush=True)
            break

    out_path = str(_ROOT / "demo_output.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote {out_path} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
