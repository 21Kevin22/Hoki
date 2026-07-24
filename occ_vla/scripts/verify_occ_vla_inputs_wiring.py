"""Sanity check for the new OccVlaLiberoInputs wiring (2026-07-20):
does a right_wrist_0_rgb slot actually filled with a distinctive
subgoal_image measurably change pi0.5's output, or does it get dropped
somewhere in the pipeline?

v3: like every other experiment script this session, split across the
RPC boundary -- the policy (jax/openpi) can't share a process with
robosuite/LIBERO (separate venvs), so this is the client half (system
python, has robosuite) calling a worker started with
PI05_WORKER_USE_OCC_VLA_INPUTS=1 (see pi05_worker.py).

Uses a real LIBERO agentview/wrist frame (in-distribution, like every
other experiment this session -- v1 used random-noise images and was
inconclusive, likely because a policy that's never seen noise like
that reacts chaotically regardless of subgoal_image, swamping any real
signal). Compares repeated calls with no subgoal_image key at all
(-> worker's arrays.get("subgoal_image") is None, matching every other
script's call_pi05) against repeated calls with the frame's own
base_image duplicated into the subgoal slot -- a real, in-distribution
image, not an arbitrary color.

Run (after starting the worker separately, see verify_occ_vla_inputs_wiring.sh):
  python3 scripts/verify_occ_vla_inputs_wiring.py
"""

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

import os  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05_occvla"))
RESIZE_SIZE = 224
N_CALLS = 20


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


def main():
    config = LiberoOccEnvConfig(
        benchmark_suite="libero_spatial", task_id=4, difficulty=Difficulty.LIGHT,
        init_state_idx=0, seed=7, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(10):
        obs, _, _, _ = occ_env.step([0.0] * 6 + [-1.0])

    base_image = preprocess_image(obs[AGENTVIEW_KEY])
    wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
    state = state_vec(obs)
    prompt = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"

    none_runs = np.array([call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None)[0] for _ in range(N_CALLS)])
    dup_runs = np.array([call_pi05(base_image, wrist_image, state, prompt, subgoal_image=base_image)[0] for _ in range(N_CALLS)])

    none_mean, none_std = none_runs.mean(axis=0), none_runs.std(axis=0)
    dup_mean, dup_std = dup_runs.mean(axis=0), dup_runs.std(axis=0)
    mean_shift_full = np.linalg.norm(dup_mean - none_mean)
    mean_shift_xy = np.linalg.norm(dup_mean[:2] - none_mean[:2])
    within_group_noise_full = float(np.linalg.norm(none_std) + np.linalg.norm(dup_std)) / 2
    within_group_noise_xy = float(np.linalg.norm(none_std[:2]) + np.linalg.norm(dup_std[:2])) / 2

    print(f"subgoal_image=None,        {N_CALLS} calls: mean={none_mean.round(3)}")
    print(f"                                     std={none_std.round(3)}")
    print(f"subgoal_image=dup(base),   {N_CALLS} calls: mean={dup_mean.round(3)}")
    print(f"                                     std={dup_std.round(3)}")
    print(f"[full 7-dim] shift={mean_shift_full:.4f}, within-noise={within_group_noise_full:.4f}, ratio={mean_shift_full / within_group_noise_full:.2f}")
    print(f"[xy only]    shift={mean_shift_xy:.4f}, within-noise={within_group_noise_xy:.4f}, ratio={mean_shift_xy / within_group_noise_xy if within_group_noise_xy > 0 else float('inf'):.2f}")

    from scipy import stats  # noqa: PLC0415

    dims = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    print("per-dimension Welch's t-test (None vs dup(base)):")
    n_significant = 0
    for i, name in enumerate(dims):
        t, p = stats.ttest_ind(none_runs[:, i], dup_runs[:, i], equal_var=False)
        sig = p < 0.01
        n_significant += sig
        print(f"  {name:8s}: none={none_mean[i]:+.4f}, dup={dup_mean[i]:+.4f}, t={t:+.2f}, p={p:.4f}{'  ***' if sig else ''}")

    if n_significant > 0:
        print(f"CONCLUSION: {n_significant}/7 dims differ at p<0.01 -- subgoal_image measurably changes the output. Wiring works.")
    else:
        print("CONCLUSION: no dimension differs significantly at p<0.01 -- inconclusive or inert from this test.")


if __name__ == "__main__":
    main()
