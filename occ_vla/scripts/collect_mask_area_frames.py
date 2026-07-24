"""Collects raw agentview + arm-segmentation-mask frame pairs across
multiple tasks and multiple points in a real rollout (arm approaching /
mid-reach / near the target), for the mask-area sweep test
(scripts/test_mask_area_sweep.py) -- see occ_vla/CLAUDE.md "Mask-area &
temperature-schedule investigation" for why a single frame (step0361)
wasn't enough to treat the 8.8%-works/19.3%-fails boundary as anything
more than a hypothesis.

Uses the real pi0.5 RPC worker (natural arm motion through several
reach phases) -- no MMaDA/RAFT needed here, just LIBERO + pi0.5.
Doesn't need the episode to succeed; MAX_STEPS is a fixed sampling
budget, not a completion target.

Run (base/system Python, pi05_worker already serving .rpc/pi05):
  python3 scripts/collect_mask_area_frames.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from openpi_client import image_tools
from PIL import Image
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

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 200
SAMPLE_STEPS = [60, 120, 180]  # rough approaching / mid-reach / near-target proxy, not contact-verified

TASKS = [
    {"suite": "libero_10", "task_id": 8, "label": "moka_pots"},
    {"suite": "libero_spatial", "task_id": 9, "label": "bowl_cabinet"},
    {"suite": "libero_10", "task_id": 4, "label": "mugs"},
]

OUT_DIR = _ROOT / "mask_area_sweep_frames"


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


def collect_task(suite: str, task_id: int, label: str) -> list[dict]:
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

    print(f"[{label}] target={occ_env.target_body_name!r} instruction={instruction!r}", flush=True)

    saved = []
    import collections  # noqa: PLC0415

    action_queue = collections.deque()
    for step in range(MAX_STEPS):
        seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
        arm_mask = (seg_dict["robot"].squeeze(-1) != 0) if "robot" in seg_dict else None

        if step in SAMPLE_STEPS and arm_mask is not None and arm_mask.sum() > 0:
            raw = obs[AGENTVIEW_KEY]
            out_stub = OUT_DIR / f"{label}_step{step:03d}"
            Image.fromarray(raw).save(f"{out_stub}_raw.png")
            Image.fromarray((arm_mask.astype(np.uint8) * 255)).save(f"{out_stub}_mask.png")
            saved.append({"label": label, "step": step, "instruction": instruction, "arm_px": int(arm_mask.sum())})
            print(f"  saved {out_stub.name} (arm_px={arm_mask.sum()})", flush=True)

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
            print(f"  [{label}] task completed early at step {step}", flush=True)
            break

    return saved


def main():
    OUT_DIR.mkdir(exist_ok=True)
    manifest = []
    for task in TASKS:
        manifest.extend(collect_task(task["suite"], task["task_id"], task["label"]))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone -- {len(manifest)} frames saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
