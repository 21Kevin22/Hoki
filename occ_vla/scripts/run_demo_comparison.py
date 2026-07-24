"""Baseline vs proposed-method paired comparison: runs the *same* LIBERO
task from the *same* init_state (and RNG seed) twice — once with
occlusion countermeasures disabled (baseline: routing forced to NONE)
and once with the normal ControlLoop routing (proposed: MMaDA arm-free
subgoal injection when self-occluded) — and writes a side-by-side
video plus a short report.

Observation/action preprocessing is matched to openpi's own reference
LIBERO eval script (third_party/openpi/examples/libero/main.py), since
an earlier version of this script deviated from it in several ways that
likely contributed to a 0/600-step failure in testing:
- render at 256x256 (LIBERO_ENV_RESOLUTION there), then resize_with_pad
  to 224x224 for the policy -- not render directly at a mismatched size
- flip the rendered image on *both* axes ([::-1, ::-1]), not just rows
- wait num_steps_wait=10 steps with a dummy no-op action before the
  policy starts, so dropped objects finish settling first
- pass the policy's raw action through to env.step() unmodified -- no
  gripper-dimension binarization, which the reference doesn't do either
- per-suite max_steps (libero_spatial: 220, per the reference script's
  own comment "longest training demo has 193 steps"), not a flat
  200/600 guess
- env.seed(seed) before each rollout, since openpi's script notes the
  seed affects object initial positions even under a fixed init_state

Honesty note: this is a single paired trial, not a statistically
powered evaluation (n=1 per condition, same seed). It shows whether the
mechanism does anything different in this specific rollout, not
whether it reliably helps on average. Occlusion-triggered replanning
here only changes pi0.5's inputs (a subgoal image in right_wrist_0_rgb,
or nothing); the "arm physically shifts to expose the target" (PKLP's
DynamicExposurePlanner, pklp/exposure.py) is not implemented, so any
behavioral difference comes solely from the MMaDA image injection, not
a kinematic exposure maneuver.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openpi_client import image_tools
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

# Not /tmp -- this environment has cleared /tmp mid-session before,
# silently killing workers and losing RPC state.
PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
MMADA_RPC_DIR = str(_ROOT / ".rpc" / "mmada")

BENCHMARK_SUITE = "libero_spatial"
TASK_ID = 9  # pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
MAX_STEPS = 220  # libero_spatial budget per openpi's reference eval script
NUM_STEPS_WAIT = 10  # let dropped objects settle before the policy starts
RESIZE_SIZE = 224
SEED = 7
INIT_STATE_IDX = 0
INSTRUCTION = "pick up the black bowl on the wooden cabinet and place it on the plate"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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


def compute_arm_s_occ(env, seg_image) -> tuple[float, np.ndarray]:
    seg_dict = env.get_segmentation_instances(seg_image)
    target_mask = seg_dict.get("akita_black_bowl_1")
    if target_mask is None:
        return 0.0, np.zeros(seg_image.shape[:2], dtype=bool)
    target_mask = target_mask.squeeze(-1) != 0
    arm_mask = seg_dict["robot"].squeeze(-1) != 0
    target_area = target_mask.sum()
    if target_area == 0:
        return 0.0, arm_mask
    overlap = (target_mask & arm_mask).sum()
    return float(overlap) / float(target_area), arm_mask


def state_vec(obs):
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    # matches openpi's reference: flip both axes (not just rows), then
    # resize_with_pad to the policy's expected input size
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def label(img, text, color=(0, 255, 0)):
    img = img.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def run_episode(disable_countermeasures: bool, tag: str):
    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE,
        task_id=TASK_ID,
        difficulty=Difficulty.LIGHT,
        init_state_idx=INIT_STATE_IDX,
        seed=SEED,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    print(f"[{tag}] occluder placed, S_occ={occ_env.last_s_occ:.3f}", flush=True)

    # Let dropped objects settle before the policy starts (reference script's
    # num_steps_wait) -- also matches the raw LIBERO_DUMMY_ACTION convention.
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)

    router = OcclusionRouter()
    plausibility_checker = PlausibilityChecker()

    frames = []
    done_step = None
    mmada_calls = 0
    for step in range(MAX_STEPS):
        base_image_raw = obs[AGENTVIEW_KEY]
        wrist_image_raw = obs["robot0_eye_in_hand_image"]
        base_image = preprocess_image(base_image_raw)
        wrist_image = preprocess_image(wrist_image_raw)
        state = state_vec(obs)
        seg_image = obs[AGENTVIEW_SEGMENTATION_KEY]

        arm_s_occ, arm_mask_raw = compute_arm_s_occ(occ_env._env, seg_image)  # noqa: SLF001
        signals = OcclusionSignals(arm_s_occ=arm_s_occ, scene_dyn_occ=False)
        source = OcclusionSource.NONE if disable_countermeasures else router.route(signals)

        subgoal_image = None
        label_text = f"[{tag}] step {step} | arm_s_occ={arm_s_occ:.2f} | {source.name}"

        if source == OcclusionSource.SELF:
            mmada_calls += 1
            # arm_mask_raw is at the render resolution (e.g. 256x256);
            # base_image was flipped+resized to RESIZE_SIZE via
            # preprocess_image -- resize the mask the same way (both axes
            # flipped, then matched to base_image's shape) so the two stay
            # pixel-aligned before sample_arm_free_image's own internal
            # resize to 512x512.
            arm_mask_flipped = np.ascontiguousarray(arm_mask_raw[::-1, ::-1])
            arm_mask = (
                cv2.resize(
                    arm_mask_flipped.astype(np.uint8), (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
            subgoal_image = call_mmada(base_image, arm_mask, INSTRUCTION)
            score = plausibility_checker.score(
                cv2.resize(subgoal_image, (RESIZE_SIZE, RESIZE_SIZE)), {"original_image": base_image, "arm_pixel_mask": arm_mask}
            )
            label_text += f" | plaus={score:.2f}"
            if score < 0.5:
                subgoal_image = None
                label_text += " FALLBACK"

        actions = call_pi05(base_image, wrist_image, state, INSTRUCTION, subgoal_image, None)
        # raw action, unmodified -- the reference script doesn't binarize
        # the gripper dimension either
        action = actions[0]

        color = (0, 165, 255) if disable_countermeasures else (0, 255, 0)
        frames.append(label(base_image, label_text, color))

        step_result = occ_env.step(action.tolist())
        obs = step_result[0]
        done = step_result[2]
        if done and done_step is None:
            done_step = step
            print(f"[{tag}] success at step {step}", flush=True)
            break

    print(f"[{tag}] finished: {len(frames)} steps, mmada_calls={mmada_calls}, done_step={done_step}", flush=True)
    return frames, done_step, mmada_calls


def write_video(frames, out_path, fps=10):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    single_run = os.environ.get("OCC_VLA_SINGLE_RUN") == "1"

    if single_run:
        t0 = time.time()
        frames, done_step, mmada_calls = run_episode(disable_countermeasures=False, tag="PROPOSED")
        print(f"single run took {time.time() - t0:.1f}s", flush=True)

        out_path = str(_ROOT / "demo_single_run.mp4")
        write_video(frames, out_path)
        print(f"wrote {out_path} ({len(frames)} frames)", flush=True)
        print("=== REPORT (single run) ===")
        print(f"steps={len(frames)}, success_step={done_step}, mmada_calls={mmada_calls}")
        return

    t0 = time.time()
    baseline_frames, baseline_done, _ = run_episode(disable_countermeasures=True, tag="BASELINE")
    print(f"baseline took {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    proposed_frames, proposed_done, mmada_calls = run_episode(disable_countermeasures=False, tag="PROPOSED")
    print(f"proposed took {time.time() - t0:.1f}s", flush=True)

    n = max(len(baseline_frames), len(proposed_frames))
    h, w = baseline_frames[0].shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    out_path = str(_ROOT / "demo_comparison.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (w * 2, h))
    for i in range(n):
        left = baseline_frames[i] if i < len(baseline_frames) else blank
        right = proposed_frames[i] if i < len(proposed_frames) else blank
        writer.write(cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote {out_path} ({n} paired frames)", flush=True)

    print("=== REPORT ===")
    print(f"baseline:  steps={len(baseline_frames)}, success_step={baseline_done}, mmada_calls=0")
    print(f"proposed:  steps={len(proposed_frames)}, success_step={proposed_done}, mmada_calls={mmada_calls}")
    if baseline_done is not None and proposed_done is not None:
        print(f"step difference (proposed - baseline): {proposed_done - baseline_done}")
    else:
        print("neither/one run reached success within MAX_STEPS -- no step-count efficiency claim can be made from this pair")


if __name__ == "__main__":
    main()
