"""Dense (every-step) agentview+wrist collection with real arm_s_occ per
step, on the task already established (scan_natural_self_occlusion.py,
CLAUDE.md) as this project's best natural self-occlusion case:
bowl_top_drawer (libero_spatial task 4, max arm_s_occ 0.438, 8.6% of
steps > 0.30).

Motivation (user request, 2026-07-22): the dust3r cross-view/novel-view/
view-selection checks so far all used data/multiview/t08_heavy's 20-step
sampling on T08 moka_pots -- which, per the earlier natural-occlusion
scan, essentially never exceeds arm_s_occ=0.107 naturally. That's why no
step in that episode had the target genuinely occluded in agentview.
Neither `data/` nor any other existing directory has dense (every-step)
paired agentview+wrist frames with real arm_s_occ recorded -- this
needs a fresh rollout, not log analysis (contra the "no GPU needed"
assumption -- a real pi0.5 rollout is required).

Saves every step's agentview+wrist pair plus arm_s_occ, so the actual
"target occluded in agentview, visible in wrist" moment can be found and
used directly (no guessing at a 20-step-aligned candidate).

Requires pi05_worker running:
  scripts/run_with_gpu_cleanup.sh python3 scripts/collect_dense_occlusion_frames.py
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

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
NUM_STEPS_WAIT = 10
SEED = 0
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = 520  # libero_10's own max_steps per the openpi reference eval table -- NOT libero_spatial's 220
RESIZE_SIZE = 224
REPLAN_STEPS = 8
SUITE, TASK_ID, LABEL = "libero_10", 9, "mug_in_microwave"
OUT_DIR = _ROOT / "dense_occlusion_frames" / f"{LABEL}_v2_with_poses"


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


def main():
    from libero.libero import benchmark  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = benchmark.get_benchmark(SUITE)()
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(
        benchmark_suite=SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=0, seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    manifest = []
    action_queue = []
    step = 0
    while step < MAX_STEPS:
        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        if action_queue:
            action = action_queue.pop(0)
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])
        obs, _, done, _ = occ_env.step(action.tolist())
        step += 1

        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        stub = f"step{step:04d}"
        Image.fromarray(obs[AGENTVIEW_KEY]).save(OUT_DIR / f"{stub}_agentview.png")
        Image.fromarray(obs["robot0_eye_in_hand_image"]).save(OUT_DIR / f"{stub}_wrist.png")

        # True camera extrinsics/intrinsics (per user's chosen "Option 2",
        # 2026-07-22: dust3r's own cross-view pose estimation broke down
        # specifically when agentview/wrist share little visual content --
        # exactly the occluded case this collection targets -- so instead
        # of estimating the relative pose from image correspondence, pull
        # the real one from the sim, same privileged-info category
        # CameraProjector.from_sim already uses for agentview).
        sim = occ_env._env.sim  # noqa: SLF001
        cam_poses = {}
        for cam_name in ("agentview", "robot0_eye_in_hand"):
            cam_id = sim.model.camera_name2id(cam_name)
            cam_poses[cam_name] = {
                "cam_pos": sim.data.cam_xpos[cam_id].tolist(),
                "cam_mat": sim.data.cam_xmat[cam_id].tolist(),  # flat 9 -- reshape(3,3) on load
                "fovy_deg": float(sim.model.cam_fovy[cam_id]),
            }

        manifest.append({"step": step, "arm_s_occ": arm_s_occ, "cam_poses": cam_poses})
        if step % 10 == 0:
            print(f"step {step}: arm_s_occ={arm_s_occ:.4f}", flush=True)

        if done:
            print(f"episode done at step {step}", flush=True)
            break

    (OUT_DIR / "manifest.json").write_text(json.dumps({"instruction": instruction, "steps": manifest}, indent=2))
    max_entry = max(manifest, key=lambda e: e["arm_s_occ"])
    n_above_030 = sum(1 for e in manifest if e["arm_s_occ"] > 0.30)
    print(f"\n{len(manifest)} steps saved to {OUT_DIR}", flush=True)
    print(f"max arm_s_occ={max_entry['arm_s_occ']:.4f} at step {max_entry['step']}", flush=True)
    print(f"steps with arm_s_occ > 0.30: {n_above_030}", flush=True)


if __name__ == "__main__":
    main()
