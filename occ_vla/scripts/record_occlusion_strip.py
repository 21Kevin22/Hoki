"""Captures a 3-row time-series strip per task, all real data, no
generative model involved: agentview (pi0.5's actual input), wrist
camera (reference), and agentview with the arm's live segmentation
mask blacked out (visualizes exactly what the arm currently hides from
the main camera -- the same "robot" segmentation key used throughout
this session's ground-truth pair generation, not a generated image).

Runs a real baseline pi0.5 rollout (episode 0, same as the earlier
*_baseline_success recordings) and samples frames evenly across it.

Requires only the pi0.5 RPC worker. Run:
  python3 scripts/record_occlusion_strip.py \
      --suite libero_10 --task-id 8 --label moka_pots --max-steps 520
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from openpi_client import image_tools
from PIL import Image, ImageDraw, ImageFont
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

import os  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
GATE_THRESHOLD = 0.30
N_COLUMNS = 8  # number of sampled timesteps in the strip

OUT_DIR = _ROOT / "occlusion_strips"


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


def arm_mask_224(occ_env, obs) -> np.ndarray:
    """Live 'robot' segmentation mask, in the same flipped 224x224
    space preprocess_image() produces."""
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    if "robot" not in seg_dict:
        return np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=bool)
    raw_mask = seg_dict["robot"].squeeze(-1) != 0
    flipped = raw_mask[::-1, ::-1]
    resized = cv2.resize(flipped.astype(np.uint8), (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def run_and_capture(suite: str, task_id: int, label_name: str, max_steps: int) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    init_states = bench.get_task_init_states(task_id)
    instruction = bench.get_task(task_id).language

    config = LiberoOccEnvConfig(
        benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT,
        init_state_idx=0, seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    action_queue = collections.deque()
    records = []  # per-step: dict(step, s_occ, agentview, wrist, masked)
    done_step = None

    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        agentview = preprocess_image(obs[AGENTVIEW_KEY])
        wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
        mask = arm_mask_224(occ_env, obs)
        masked_view = agentview.copy()
        masked_view[mask] = 0

        records.append({"step": step, "s_occ": arm_s_occ, "agentview": agentview, "wrist": wrist, "masked": masked_view})

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(agentview, wrist, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            done_step = step
            break

    return {"records": records, "done_step": done_step, "instruction": instruction, "label": label_name}


def build_strip(result: dict, n_columns: int, out_path: Path) -> None:
    records = result["records"]
    n = len(records)
    idxs = sorted(set(int(round(i * (n - 1) / (n_columns - 1))) for i in range(n_columns)))
    cols = [records[i] for i in idxs]

    cell = RESIZE_SIZE
    pad = 4
    label_h = 22
    w = len(cols) * (cell + pad) + pad
    h = 3 * (cell + label_h) + pad
    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    row_names = ["agentview", "wrist", "arm masked"]
    row_keys = ["agentview", "wrist", "masked"]
    for r, (row_name, key) in enumerate(zip(row_names, row_keys)):
        y0 = r * (cell + label_h) + label_h
        draw.text((pad, r * (cell + label_h) + 3), row_name, fill=(255, 255, 255), font=font)
        for c, rec in enumerate(cols):
            x0 = c * (cell + pad) + pad
            img = Image.fromarray(rec[key])
            canvas.paste(img, (x0, y0))
            occluded = rec["s_occ"] >= GATE_THRESHOLD
            color = (255, 120, 0) if occluded else (0, 220, 0)
            tag = f"t={rec['step']} s={rec['s_occ']:.2f}"
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=color, width=2)
            if r == 0:
                draw.text((x0 + 2, y0 + 2), tag, fill=color, font=font)

    outcome = f"SUCCESS t={result['done_step']}" if result["done_step"] is not None else "FAILURE(timeout)"
    draw.text((pad, h - 18), f"{result['label']}: {result['instruction']} -- {outcome}", fill=(255, 255, 255), font=font)

    OUT_DIR.mkdir(exist_ok=True)
    canvas.save(out_path)
    print(f"[{result['label']}] wrote {out_path} ({len(cols)} columns, done_step={result['done_step']})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--columns", type=int, default=N_COLUMNS)
    args = parser.parse_args()

    result = run_and_capture(args.suite, args.task_id, args.label, args.max_steps)
    build_strip(result, args.columns, OUT_DIR / f"{args.label}_strip.png")


if __name__ == "__main__":
    main()
