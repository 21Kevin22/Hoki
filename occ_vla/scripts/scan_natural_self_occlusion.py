"""Before committing GPU time to a full Phase 2/4 benchmark under real
self-occlusion, check empirically whether arm_s_occ >= 0.30 (or even a
sustained >0.15) actually occurs naturally in any candidate task --
earlier sessions found T08 (moka pots) tops out around 0.15 over a
full rollout (see occ_vla/CLAUDE.md), so before assuming some other
task does better, measure it directly instead of guessing.

Real pi0.5 rollouts (natural motion) across several candidate tasks
chosen for plausible arm-camera occlusion (reaching into an enclosed
space: drawer, microwave, caddy). Tracks running max/mean arm_s_occ
per task using LiberoOccEnv.compute_arm_s_occ (place_occluder=False, no
pixel_mask -- this measures REAL, unmodified self-occlusion, nothing
artificial).

Run: python3 scripts/scan_natural_self_occlusion.py
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

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520

CANDIDATES = [
    {"suite": "libero_10", "task_id": 8, "label": "T08_moka_pots_control"},
    {"suite": "libero_10", "task_id": 3, "label": "bowl_in_drawer"},
    {"suite": "libero_10", "task_id": 9, "label": "mug_in_microwave"},
    {"suite": "libero_10", "task_id": 5, "label": "book_in_caddy"},
    {"suite": "libero_spatial", "task_id": 4, "label": "bowl_top_drawer_spatial"},
]

OUT_PATH = _ROOT / "self_occlusion_scan_results.json"
FRAMES_DIR = _ROOT / "self_occlusion_scan_frames"


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


def scan_task(suite: str, task_id: int, label: str) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    instruction = bench.get_task(task_id).language

    config = LiberoOccEnvConfig(
        benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT, init_state_idx=0, seed=SEED, place_occluder=False
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)
    print(f"[{label}] target={occ_env.target_body_name!r} instruction={instruction!r}", flush=True)

    import collections  # noqa: PLC0415

    action_queue = collections.deque()
    s_occ_trace = []
    max_s_occ = 0.0
    max_step = None
    max_frame = None
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        s_occ_trace.append(arm_s_occ)
        if arm_s_occ > max_s_occ:
            max_s_occ = arm_s_occ
            max_step = step
            max_frame = obs[AGENTVIEW_KEY].copy()

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])
        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            print(f"  [{label}] task completed at step {step}", flush=True)
            break

    if max_frame is not None:
        FRAMES_DIR.mkdir(exist_ok=True)
        from PIL import Image  # noqa: PLC0415

        Image.fromarray(max_frame).save(FRAMES_DIR / f"{label}_maxsocc_step{max_step:03d}.png")

    result = {
        "label": label,
        "instruction": instruction,
        "max_arm_s_occ": max_s_occ,
        "max_step": max_step,
        "mean_arm_s_occ": float(np.mean(s_occ_trace)),
        "frac_steps_above_0.10": float(np.mean(np.array(s_occ_trace) > 0.10)),
        "frac_steps_above_0.30": float(np.mean(np.array(s_occ_trace) > 0.30)),
        "n_steps_measured": len(s_occ_trace),
    }
    print(f"  [{label}] max_arm_s_occ={max_s_occ:.3f} at step {max_step}, mean={result['mean_arm_s_occ']:.3f}", flush=True)
    return result


def main():
    results = []
    for c in CANDIDATES:
        result = scan_task(c["suite"], c["task_id"], c["label"])
        results.append(result)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== SELF-OCCLUSION SCAN SUMMARY ===")
    for r in results:
        print(f"{r['label']:28s} max={r['max_arm_s_occ']:.3f}  mean={r['mean_arm_s_occ']:.3f}  frac>0.30={r['frac_steps_above_0.30']:.3f}")


if __name__ == "__main__":
    main()
