"""Preview evaluation (per user request, 2026-07-17): does the tiny
(r=8, 40-step, overfit-on-1-episode) LoRA adapter from
train_arm_removal_lora_tiny.py make ANY measurable difference to pi0.5,
before investing in the much larger 2,000-pair / high-rank training run?
Cheap, GPU-light, answers the more fundamental question first: does
pi0.5 attend to a generated subgoal image at all (as opposed to
treating it as OOD noise, matching the null results already found for
the plain red-dot overlay and coordinate text)?

Runs T08 (moka pots -- same task the tiny adapter was trained on) twice,
paired by init_state:
  - baseline: no countermeasure at all
  - lora_preview: real arm_s_occ >= 0.30 triggers
    ArmFreeSubgoalGenerator.sample_arm_free_image() via the LoRA-adapted
    mmada_worker_lora.py, scored by PlausibilityChecker, injected into
    pi0.5's right_wrist_0_rgb slot if accepted (else the step behaves
    like baseline for that step).

Tracks: PlausibilityChecker accept rate, occluded-step jitter (see
run_self_occlusion_pipeline.py for the same metric), done_step, and
writes a labeled qualitative video for each condition. Requires
pi05_worker AND mmada_worker_lora.py both running.

Run: python3 scripts/run_t08_lora_preview.py
"""

import json
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
from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402
from occ_vla.world_model.arm_free_subgoal import ARM_OCC_THRESHOLD as _DEFAULT_ARM_OCC_THRESHOLD  # noqa: E402

# T08's natural arm_s_occ tops out around 0.08-0.15 (confirmed
# 2026-07-17: qualitative-video baseline run measured max 0.082 over
# 300 steps) -- well under the library default 0.30, so the first
# preview run (using 0.30) never triggered generation at all (0/3
# episodes, confirmed). Lowered to match what actually happens on this
# task, matching the same reasoning run_demo_t08.py's ADAPTIVE_ARM_THRESHOLD
# used for the same reason.
ARM_OCC_THRESHOLD = 0.05

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
MMADA_LORA_RPC_DIR = str(_ROOT / ".rpc" / "mmada_lora")

BENCHMARK_SUITE = "libero_10"
TASK_ID = 8
TARGET_BODY_NAME = "moka_pot_1"
INSTRUCTION = "put both moka pots on the stove"
MAX_STEPS = 300
NUM_STEPS_WAIT = 10
RESIZE_SIZE = 224
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
N_EPISODES = 3

OUT_DIR = _ROOT / "qualitative_videos"
RESULTS_PATH = _ROOT / "t08_lora_preview_results.json"


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    if subgoal_image is not None:
        arrays["subgoal_image"] = subgoal_image
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def call_mmada_lora(image, arm_pixel_mask, instruction):
    resp_arrays, _ = rpc.call(
        MMADA_LORA_RPC_DIR,
        {"image": image, "arm_pixel_mask": arm_pixel_mask.astype(np.uint8)},
        {"instruction": instruction, "horizon": 5},
        timeout_s=180,
    )
    return resp_arrays["image"]


def label(img, text, color=(0, 255, 0)):
    img = img.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def write_video(frames, out_path, fps=15):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def run_episode(condition: str, episode_idx: int) -> dict:
    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx, seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    plausibility_checker = PlausibilityChecker()
    import collections  # noqa: PLC0415

    action_queue = collections.deque()
    frames = []
    n_triggered, n_accepted = 0, 0
    prev_action = None
    occluded_jitter = []
    scores = []
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        base_image_raw = obs[AGENTVIEW_KEY]
        base_image = preprocess_image(base_image_raw)
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        occluded = arm_s_occ >= ARM_OCC_THRESHOLD

        subgoal_image = None
        label_text = f"[{condition}] step {step} | arm_s_occ={arm_s_occ:.2f}"
        if condition == "lora_preview" and occluded:
            n_triggered += 1
            seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
            arm_mask_raw = (seg_dict["robot"].squeeze(-1) != 0) if "robot" in seg_dict else np.zeros(base_image_raw.shape[:2], dtype=bool)
            arm_mask_flipped = np.ascontiguousarray(arm_mask_raw[::-1, ::-1])
            arm_mask = cv2.resize(arm_mask_flipped.astype(np.uint8), (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_NEAREST) > 0
            raw_generated = call_mmada_lora(base_image, arm_mask, INSTRUCTION)
            generated_224 = cv2.resize(raw_generated, (RESIZE_SIZE, RESIZE_SIZE))
            physical_context = {"original_image": base_image, "arm_pixel_mask": arm_mask}
            score = plausibility_checker.score(generated_224, physical_context)
            scores.append(score)
            accepted = not plausibility_checker.should_fallback(generated_224, physical_context)
            label_text += f" | plaus={score:.3f}"
            if accepted:
                n_accepted += 1
                subgoal_image = generated_224
                label_text += " INJECTED"
            else:
                label_text += " REJECTED"

        color = (0, 0, 255) if occluded else (0, 255, 0)
        frames.append(label(base_image, label_text, color))

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), INSTRUCTION, subgoal_image)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        if occluded and prev_action is not None:
            occluded_jitter.append(float(np.linalg.norm(action[:6] - prev_action[:6])))
        prev_action = action

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            break

    OUT_DIR.mkdir(exist_ok=True)
    write_video(frames, OUT_DIR / f"t08_{condition}_ep{episode_idx}.mp4")
    return {
        "condition": condition,
        "episode": episode_idx,
        "done_step": len(frames) - 1 if done else None,
        "n_triggered": n_triggered,
        "n_accepted": n_accepted,
        "mean_score": float(np.mean(scores)) if scores else None,
        "mean_occluded_jitter": float(np.mean(occluded_jitter)) if occluded_jitter else None,
    }


def main():
    results = []
    for condition in ("baseline", "lora_preview"):
        for episode_idx in range(N_EPISODES):
            result = run_episode(condition, episode_idx)
            results.append(result)
            print(f"[{condition} ep{episode_idx}] {result}", flush=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("=== T08 LORA PREVIEW REPORT ===")
    for condition in ("baseline", "lora_preview"):
        rows = [r for r in results if r["condition"] == condition]
        steps = [r["done_step"] for r in rows if r["done_step"] is not None]
        print(f"{condition}: {len(steps)}/{len(rows)} success, steps={steps}")
    lora_rows = [r for r in results if r["condition"] == "lora_preview"]
    total_triggered = sum(r["n_triggered"] for r in lora_rows)
    total_accepted = sum(r["n_accepted"] for r in lora_rows)
    print(f"lora_preview: {total_accepted}/{total_triggered} generations accepted by PlausibilityChecker")


if __name__ == "__main__":
    main()
