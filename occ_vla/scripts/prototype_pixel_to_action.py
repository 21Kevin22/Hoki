"""Real-sim prototype for occ_vla.pklp.pixel_to_action -- validates the
Jacobian-based pixel-delta -> world-delta conversion (Plan 3's 2D-to-Action
bridge) against an actual LIBERO/MuJoCo scene, not just the synthetic
overhead-camera unit tests.

Generalization test (2026-07-18): suite/task/target are no longer
hardcoded to moka_pots. The target object is resolved the same
task-agnostic way libero_occ_env.py already does it for S_occ
measurement -- `env.obj_of_interest[0]` + `domain.obj_body_id[name]`
(the exact pattern occluder.py._target_position uses), not a
hand-picked body name -- so running this against a different suite/task
is a genuine test of whether the pipeline generalizes, not a rewrite.

Two checks:
1. Self-consistency: project(eef_pos + world_delta) should land close to
   (current_pixel + pixel_delta) for a small, unclipped pixel_delta --
   confirms the DLS inversion is actually inverting the same projection
   formula already validated against the static moka_pot_1 body (see
   CLAUDE.md camera calibration note).
2. Direction sanity: take the task's own target object (dynamically
   resolved, see above) as a stand-in "PKLP predicted target," convert
   (target_pixel - eef_pixel) to a world delta, and check that stepping
   the eef toward it actually reduces the on-screen pixel distance to
   the target -- the thing this whole bridge is supposed to accomplish.

Run in base env (no pi0.5/mmada needed):
  python3 scripts/prototype_pixel_to_action.py \
      [--suite libero_spatial] [--task-id 4] [--init-state-idx 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.pklp.pixel_to_action import (  # noqa: E402
    CameraProjector,
    pixel_delta_to_world_delta,
    pklp_pixel_delta_to_world_delta,
    projection_jacobian,
)

SEED = 7
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
OUT_PATH = Path("/tmp/pixel_to_action_prototype.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--init-state-idx", type=int, default=0)
    args = parser.parse_args()

    config = LiberoOccEnvConfig(
        benchmark_suite=args.suite, task_id=args.task_id, difficulty=Difficulty.LIGHT,
        init_state_idx=args.init_state_idx, seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)

    env = occ_env._env  # noqa: SLF001
    sim = env.sim
    projector = CameraProjector.from_sim(sim, "agentview", resolution=config.camera_resolution)

    print(f"suite={args.suite} task_id={args.task_id} target_body_name={occ_env.target_body_name!r}")

    eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
    eef_pixel = projector.project(eef_pos)
    print(f"eef world pos: {eef_pos}, eef pixel: {eef_pixel}")

    # --- check 1a: self-consistency of the raw (unmasked) DLS inversion,
    # i.e. is the math itself correct against the real (non-overhead,
    # non-axis-aligned) camera Jacobian? Z is NOT masked here so all 3
    # world DOF are free to explain the pixel delta. ---
    small_pixel_delta = np.array([8.0, -5.0])
    jac = projection_jacobian(projector, eef_pos)
    world_delta_xyz = pixel_delta_to_world_delta(jac, small_pixel_delta, zero_z=False, max_step_m=10.0)
    reprojected_xyz = projector.project(eef_pos + world_delta_xyz)
    err_xyz = np.linalg.norm((reprojected_xyz - eef_pixel) - small_pixel_delta)
    print(f"[unmasked, all 3 DOF free] requested pixel delta: {small_pixel_delta}, "
          f"achieved: {reprojected_xyz - eef_pixel}, world_delta: {world_delta_xyz} "
          f"-> error {err_xyz:.4f}px {'OK' if err_xyz < 0.5 else 'FAIL'}")

    # --- check 1b: same, but with Z masked to zero (the deployed setting)
    # -- this is the cost of the Z-safety constraint: how much pixel error
    # remains when only X/Y world motion is allowed to explain the delta. ---
    world_delta_xy = pixel_delta_to_world_delta(jac, small_pixel_delta, zero_z=True, max_step_m=10.0)
    reprojected_xy = projector.project(eef_pos + world_delta_xy)
    err_xy = np.linalg.norm((reprojected_xy - eef_pixel) - small_pixel_delta)
    print(f"[Z-masked, deployed setting] requested pixel delta: {small_pixel_delta}, "
          f"achieved: {reprojected_xy - eef_pixel}, world_delta: {world_delta_xy} "
          f"-> residual error {err_xy:.4f}px (this is the Z-safety-constraint cost, not a bug)")

    # --- check 2: direction sanity against the task's own target object,
    # resolved the same task-agnostic way libero_occ_env.py does for S_occ
    # (env.obj_of_interest[0] -> domain.obj_body_id[name]), not a
    # hand-picked body name. ---
    domain = env.env  # ControlEnv wraps the BDDL domain env as .env, same as occluder.py
    target_id = domain.obj_body_id[occ_env.target_body_name]
    target_pos = np.array(sim.data.body_xpos[target_id], dtype=np.float64)
    target_pixel = projector.project(target_pos)

    world_delta_to_target = pklp_pixel_delta_to_world_delta(
        projector, eef_pos, eef_pixel, target_pixel, max_step_m=0.03,
    )
    stepped_eef = eef_pos + world_delta_to_target
    stepped_pixel = projector.project(stepped_eef)

    dist_before = np.linalg.norm(target_pixel - eef_pixel)
    dist_after = np.linalg.norm(target_pixel - stepped_pixel)
    print(f"target ({occ_env.target_body_name}) pixel: {target_pixel}, dist before step: {dist_before:.2f}px, "
          f"after 1 clipped step ({np.linalg.norm(world_delta_to_target)*100:.2f}cm): {dist_after:.2f}px "
          f"{'OK (moved closer)' if dist_after < dist_before else 'FAIL (moved away)'}")

    # project() matches the flipped ([::-1, ::-1]) frame, not the raw
    # MuJoCo buffer (confirmed 2026-07-18, see CLAUDE.md) -- save both
    # anyway as a per-task visual sanity check, not to re-litigate that.
    points = [(eef_pixel, (0, 255, 0)), (target_pixel, (255, 0, 0)), (stepped_pixel, (0, 128, 255))]
    for suffix, frame in [("raw", obs[AGENTVIEW_KEY].copy()), ("flipped", obs[AGENTVIEW_KEY][::-1, ::-1].copy())]:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for pt, color in points:
            x, y = pt
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=2)
        out_path = OUT_PATH.with_stem(OUT_PATH.stem + f"_{suffix}")
        img.save(out_path)
        print(f"saved annotated {suffix} frame (green=eef, red=pot/target, blue=eef-after-1-step) to {out_path}")


if __name__ == "__main__":
    main()
