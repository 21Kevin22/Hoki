"""Best-of-N plan, step 1 (2026-07-21): before touching runtime.py,
check the more basic question -- can MMaDA-LoRA (full, r=16) ever
produce a PlausibilityChecker-passing image for a *fixed* occluded
frame, if given N independent tries? `denoise()`'s per-step sampling
is stochastic (`torch.multinomial`), so repeated calls on the same
input should genuinely differ, not just re-run identically.

Captures exactly one real occluded frame + arm mask from a
mug_in_microwave rollout (same env/seed used throughout this session),
then calls the mmada-lora RPC N times on that *same* frozen input,
scoring each with PlausibilityChecker. Reports the score distribution
and saves all N generated images plus the original for visual
inspection -- no rollout overhead, answers "does MMaDA ever get lucky"
in one shot.

Requires only the mmada-lora RPC worker (full adapter). Run:
  python3 scripts/bestofn_score_distribution.py
"""

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

import os  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
MMADA_LORA_RPC_DIR = os.environ.get("OCC_VLA_MMADA_LORA_RPC_DIR", str(_ROOT / ".rpc" / "mmada_lora_full"))

GATE_THRESHOLD = 0.30
BENCHMARK_SUITE = "libero_10"
TASK_ID = 9  # mug_in_microwave
INSTRUCTION = "put the yellow and white mug in the microwave and close it"
MAX_STEPS = 520
NUM_STEPS_WAIT = 10
RESIZE_SIZE = 224
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
N_SAMPLES = 10
EPISODE_IDX = 0

OUT_DIR = _ROOT / "bestofn_frames"


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


def call_mmada_lora(image, arm_pixel_mask, instruction):
    resp_arrays, _ = rpc.call(
        MMADA_LORA_RPC_DIR,
        {"image": image, "arm_pixel_mask": arm_pixel_mask.astype(np.uint8)},
        {"instruction": instruction, "horizon": 5},
        timeout_s=180,
    )
    return resp_arrays["image"]


def capture_first_occluded_frame():
    """Runs a real rollout with the plain pi0.5 policy until the first
    step with arm_s_occ >= GATE_THRESHOLD, then returns that frame +
    arm mask frozen (no further stepping)."""
    import collections  # noqa: PLC0415

    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=EPISODE_IDX, seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    action_queue = collections.deque()
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        base_image_raw = obs[AGENTVIEW_KEY]
        base_image = preprocess_image(base_image_raw)
        if arm_s_occ >= GATE_THRESHOLD:
            seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
            arm_mask_raw = (seg_dict["robot"].squeeze(-1) != 0) if "robot" in seg_dict else np.zeros(base_image_raw.shape[:2], dtype=bool)
            arm_mask_flipped = np.ascontiguousarray(arm_mask_raw[::-1, ::-1])
            arm_mask = cv2.resize(arm_mask_flipped.astype(np.uint8), (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_NEAREST) > 0
            print(f"captured occluded frame at step {step}, arm_s_occ={arm_s_occ:.3f}", flush=True)
            return base_image, arm_mask, step, arm_s_occ

        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), INSTRUCTION)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])
        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            raise RuntimeError(f"episode finished at step {step} before reaching arm_s_occ >= {GATE_THRESHOLD}")

    raise RuntimeError(f"never reached arm_s_occ >= {GATE_THRESHOLD} within {MAX_STEPS} steps")


def main():
    base_image, arm_mask, step, arm_s_occ = capture_first_occluded_frame()

    checker = PlausibilityChecker()
    physical_context = {"original_image": base_image, "arm_pixel_mask": arm_mask}

    samples = []
    scores = []
    for i in range(N_SAMPLES):
        raw_generated = call_mmada_lora(base_image, arm_mask, INSTRUCTION)
        generated_224 = cv2.resize(raw_generated, (RESIZE_SIZE, RESIZE_SIZE))
        score = checker.score(generated_224, physical_context)
        samples.append(generated_224)
        scores.append(score)
        print(f"[sample {i}] plausibility={score:.4f}", flush=True)

    scores_arr = np.array(scores)
    print(f"\nN={N_SAMPLES} scores: min={scores_arr.min():.4f} mean={scores_arr.mean():.4f} "
          f"max={scores_arr.max():.4f} std={scores_arr.std():.4f}")
    n_pass = int((scores_arr >= 0.5).sum())
    print(f"n_pass(>=0.5)={n_pass}/{N_SAMPLES}")
    best_idx = int(scores_arr.argmax())
    print(f"best sample: index {best_idx}, score={scores_arr[best_idx]:.4f}")

    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "original.png"), cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR))
    for i, (img, score) in enumerate(zip(samples, scores)):
        cv2.imwrite(str(OUT_DIR / f"sample_{i:02d}_score{score:.3f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # grid image: original + N samples, labeled with score
    cell = RESIZE_SIZE
    cols = 4
    rows = (N_SAMPLES + 1 + cols - 1) // cols
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    all_imgs = [("orig", base_image)] + [(f"s{i} {scores[i]:.2f}", samples[i]) for i in range(N_SAMPLES)]
    for idx, (label_text, img) in enumerate(all_imgs):
        r, c = idx // cols, idx % cols
        tile = img.copy()
        cv2.putText(tile, label_text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label_text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        grid[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = tile
    cv2.imwrite(str(OUT_DIR / "grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {OUT_DIR / 'grid.png'}", flush=True)


if __name__ == "__main__":
    main()
