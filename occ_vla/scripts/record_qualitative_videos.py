"""Records one labeled qualitative video per task for the 3
self-occlusion tasks investigated this session (natural self-occlusion,
`baseline` condition -- no countermeasure, just to see what the
occlusion dynamics actually look like): T08 moka pots (control, low
natural occlusion), bowl-in-top-drawer (libero_spatial #4, highest
natural occlusion rate found in scripts/scan_natural_self_occlusion.py),
mug-in-microwave (libero_10 #9, replication task).

Each frame is labeled with step number and the real (segmentation-
measured) arm_s_occ value, so occlusion events are visible without
needing to freeze-frame externally. Requires only pi05_worker.

Run: python3 scripts/record_qualitative_videos.py
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

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 300
GATE_THRESHOLD = 0.30

TASKS = [
    {"suite": "libero_10", "task_id": 8, "label": "moka_pots"},
    {"suite": "libero_spatial", "task_id": 4, "label": "bowl_top_drawer"},
    {"suite": "libero_10", "task_id": 9, "label": "mug_in_microwave"},
]

OUT_DIR = _ROOT / "qualitative_videos"


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


def record_task(suite: str, task_id: int, label_name: str) -> None:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    instruction = bench.get_task(task_id).language
    init_states = bench.get_task_init_states(task_id)

    config = LiberoOccEnvConfig(
        benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT, init_state_idx=0, seed=SEED, place_occluder=False
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)
    print(f"[{label_name}] target={occ_env.target_body_name!r} instruction={instruction!r}", flush=True)

    import collections  # noqa: PLC0415

    action_queue = collections.deque()
    frames = []
    done_step = None
    max_s_occ = 0.0
    for step in range(MAX_STEPS):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_s_occ = max(max_s_occ, arm_s_occ)
        base_image_raw = obs[AGENTVIEW_KEY]
        base_image = preprocess_image(base_image_raw)
        occluded = arm_s_occ >= GATE_THRESHOLD
        text = f"[{label_name}] step {step} | arm_s_occ={arm_s_occ:.2f}" + (" OCCLUDED" if occluded else "")
        color = (0, 0, 255) if occluded else (0, 255, 0)
        frames.append(label(base_image, text, color))

        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])
        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            done_step = step
            break

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{label_name}_baseline.mp4"
    write_video(frames, out_path)
    print(f"[{label_name}] wrote {out_path} ({len(frames)} frames, done_step={done_step}, max_arm_s_occ={max_s_occ:.3f})", flush=True)


def main():
    for task in TASKS:
        record_task(task["suite"], task["task_id"], task["label"])


if __name__ == "__main__":
    main()
