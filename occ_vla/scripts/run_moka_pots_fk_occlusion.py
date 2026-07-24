"""Proposal-2 occlusion definition (user request, 2026-07-23), applied
to moka_pots (libero_10 task 8) specifically, on non-baseline
conditions only.

Today's finding: moka_pots' gate/injection mechanisms NEVER engaged
across 70 episodes (3 separate runs) under the current GT-segmentation,
target-centric arm_s_occ definition -- max_arm_s_occ rarely crosses
0.30 for this task. Open question: is that a fact about the task's
physical geometry, or an artifact of how occlusion is measured?

This tests a redefinition close to the user's "2D BBox transparency
index" proposal, isolating exactly one variable (arm mask SOURCE)
against the existing pipeline, so the comparison is clean:

  - Target side UNCHANGED: still the GT clear-baseline mask's bounding
    box (occ_env._target_mask_clear), same technique already validated
    project-wide -- this keeps "what counts as the target's expected
    region" identical to the current metric, so any difference in
    occlusion incidence is attributable to the arm side only.
  - Arm side CHANGED: instead of GT segmentation ("robot" mask from
    get_segmentation_instances), the arm's silhouette is approximated
    by projecting each robot geom's true 3D pose (sim.data.geom_xpos/
    geom_xmat -- MuJoCo's own forward-kinematics result, the same
    quantity a real robot's joint encoders + URDF FK would give) into
    the agentview pixel frame via the already-validated CameraProjector,
    drawing a filled circle of the geom's projected radius at each
    projected center. This does NOT touch MuJoCo's segmentation
    renderer at all for the arm -- only body poses/sizes, which are
    the sim analogue of real joint-encoder + URDF data, not a
    segmentation-renderer shortcut.

New metric: fk_s_occ = Area(fk_arm_mask AND target_clear_bbox) /
Area(target_clear_bbox). Both arm_s_occ (old) and fk_s_occ (new) are
logged every step for direct comparison; fk_s_occ is what actually
gates the interventions in this script.

Conditions: gate_only, injection_only, both (baseline deliberately
excluded per user request -- baseline behavior under the old metric is
already characterized and doesn't depend on which occlusion metric
gates an intervention it doesn't use).

Requires:
  - pi05_worker running WITH occ_vla inputs enabled
    (PI05_WORKER_USE_OCC_VLA_INPUTS=1)
  - sd_worker running (scripts/_workers/sd_worker.py, .venv_mmada)

Run: python3 scripts/run_moka_pots_fk_occlusion.py --episodes N
"""

import argparse
import collections
import json
import os
import sys
import time
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
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, SCENE_BLEND_ALPHA, gated_blend_xy  # noqa: E402
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta  # noqa: E402

PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
SD_RPC_DIR = os.environ.get("SD_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "sd_worker"))
BENCHMARK_SUITE = "libero_10"
TASK_ID = 8  # moka_pots
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 520
N_EPISODES = 10
GATE_THRESHOLD = 0.30
CLEAR_UPDATE_THRESHOLD = 0.05
RESULTS_PATH = _ROOT / "moka_pots_fk_occlusion_results.json"
# 2026-07-23: episode 7 (baseline solves in 375 steps; gate_only never
# finishes, gate active all 520/520 steps) showed continuous gate
# engagement over a genuinely long, unbroken occlusion window can
# actively block task completion. Fix validated on that exact episode
# (455/520 steps, success) before this broader re-check: once gate
# engagement has been continuous for DECAY_START_STEPS, linearly ramp
# blend_alpha to 0 over the following DECAY_WINDOW steps, handing
# control back to pi0.5 if the correction isn't resolving things.
DECAY_START_STEPS = 30
DECAY_WINDOW = 50

CONDITIONS = ["gate_only", "injection_only", "both"]


def decayed_alpha(consecutive_occluded_steps: int) -> float:
    if consecutive_occluded_steps <= DECAY_START_STEPS:
        return SCENE_BLEND_ALPHA
    frac_into_decay = (consecutive_occluded_steps - DECAY_START_STEPS) / DECAY_WINDOW
    return SCENE_BLEND_ALPHA * max(0.0, 1.0 - frac_into_decay)


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


def call_sd(image_raw, mask_raw, instruction):
    resp_arrays, _ = rpc.call(SD_RPC_DIR, {"image": image_raw, "mask": mask_raw.astype(np.uint8)}, {"instruction": instruction}, timeout_s=60)
    return resp_arrays["image"]


def arm_pixel_mask(occ_env, obs) -> np.ndarray:
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    if "robot" not in seg_dict:
        return np.zeros(obs[AGENTVIEW_KEY].shape[:2], dtype=bool)
    return seg_dict["robot"].squeeze(-1) != 0


def target_centroid_224(occ_env, obs, tracking_mode: str = "mask_centroid") -> np.ndarray | None:
    """tracking_mode='bbox_centroid' (2026-07-23, user request): center
    of the live segmentation mask's bounding box instead of its pixel
    mean -- a proxy for what a real detector's bbox output would give,
    to stress-test the gate's robustness to coarser target-position
    precision (already validated non-degrading on mug_in_microwave,
    which was at a 100% ceiling; moka_pots has real headroom, 8/10)."""
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    target_mask_raw = seg_dict.get(occ_env.target_body_name)
    if target_mask_raw is None:
        return None
    target_mask = target_mask_raw.squeeze(-1) != 0
    if not target_mask.any():
        return None
    h, w = target_mask.shape
    ys, xs = np.where(target_mask)
    flipped_ys = h - 1 - ys
    flipped_xs = w - 1 - xs
    scale = RESIZE_SIZE / h
    if tracking_mode == "bbox_centroid":
        cx = (flipped_xs.min() + flipped_xs.max()) / 2.0
        cy = (flipped_ys.min() + flipped_ys.max()) / 2.0
        return np.array([cx * scale, cy * scale])
    return np.array([flipped_xs.mean() * scale, flipped_ys.mean() * scale])


def robot_geom_ids_for(occ_env) -> list:
    model = occ_env._env.sim.model  # noqa: SLF001
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {
        i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))
    }
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


def target_clear_bbox_mask_224(occ_env) -> np.ndarray:
    """Bounding box of the GT clear-baseline target mask, in the same
    flipped 224x224 space CameraProjector/preprocess_image use. Kept on
    the GT-segmentation source deliberately -- only the ARM side is
    being tested here (see module docstring)."""
    mask = np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=np.uint8)
    clear = occ_env._target_mask_clear  # noqa: SLF001
    if clear is None or not clear.any():
        return mask.astype(bool)
    h, w = clear.shape
    ys, xs = np.where(clear)
    flipped_ys = h - 1 - ys
    flipped_xs = w - 1 - xs
    scale = RESIZE_SIZE / h
    x0, x1 = int(flipped_xs.min() * scale), int(flipped_xs.max() * scale)
    y0, y1 = int(flipped_ys.min() * scale), int(flipped_ys.max() * scale)
    mask[max(0, y0):min(RESIZE_SIZE, y1 + 1), max(0, x0):min(RESIZE_SIZE, x1 + 1)] = 1
    return mask.astype(bool)


def geom_radius_m(model, gid: int) -> float:
    """MuJoCo geom_size semantics differ by geom_type -- using max(size)
    across the board (first attempt, 2026-07-23) badly overestimates
    elongated BOX links (size is half-extents [x,y,z]; the long axis
    isn't the visible cross-sectional width) and caused every step of
    the first moka_pots FK run to register as occluded (max_fk_s_occ
    0.69, gate engaged all 520/520 steps, episode failed) -- a metric
    bug, not evidence of real occlusion. Fixed here: for
    SPHERE/CAPSULE/CYLINDER, size[0] genuinely is the radius; for BOX
    (and anything else/degenerate), use the median of the three extents
    as a rough cross-sectional radius instead of the max."""
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]
    # mujoco.mjtGeom: SPHERE=2, CAPSULE=3, CYLINDER=5
    if gtype in (2, 3, 5):
        radius = float(size[0])
    else:
        radius = float(np.median(size))
    if radius <= 1e-4:
        radius = 0.02  # degenerate/mesh geom fallback: typical panda-link cross-section
    return radius


def fk_arm_mask_224(occ_env, projector: CameraProjector, robot_geom_ids: list) -> np.ndarray:
    """Approximate arm silhouette from true 3D geom poses/sizes
    (sim.data.geom_xpos, sim.model.geom_size -- MuJoCo's own FK result,
    the sim analogue of joint-encoder + URDF FK on a real robot), NOT
    MuJoCo's segmentation renderer. Each geom becomes a filled circle
    at its projected center with its projected radius."""
    sim = occ_env._env.sim  # noqa: SLF001
    model = sim.model
    mask = np.zeros((RESIZE_SIZE, RESIZE_SIZE), dtype=np.uint8)
    f_px = (RESIZE_SIZE / 2) / np.tan(np.radians(projector.fovy_deg) / 2)
    for gid in robot_geom_ids:
        pos = np.array(sim.data.geom_xpos[gid], dtype=np.float64)
        depth = -(projector.cam_mat.T @ (pos - projector.cam_pos))[2]
        if depth <= 1e-3:
            continue
        radius_m = geom_radius_m(model, gid)
        radius_m = max(radius_m, 0.015)  # floor: avoid degenerate zero-size geoms (e.g. site markers)
        px, py = projector.project(pos)
        pixel_radius = max(1, int(f_px * radius_m / depth))
        if -pixel_radius <= px <= RESIZE_SIZE + pixel_radius and -pixel_radius <= py <= RESIZE_SIZE + pixel_radius:
            cv2.circle(mask, (int(px), int(py)), pixel_radius, 1, -1)
    return mask.astype(bool)


def compute_fk_s_occ(fk_mask: np.ndarray, target_bbox_mask: np.ndarray) -> float:
    target_area = target_bbox_mask.sum()
    if target_area == 0:
        return 0.0
    return float((fk_mask & target_bbox_mask).sum()) / float(target_area)


def run_episode(condition: str, episode_idx: int, max_steps: int, tracking_mode: str = "mask_centroid", use_decay: bool = False) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(BENCHMARK_SUITE)()
    init_states = bench.get_task_init_states(TASK_ID)
    instruction = bench.get_task(TASK_ID).language

    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE, task_id=TASK_ID, difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    projector = CameraProjector.from_sim(occ_env._env.sim, "agentview", resolution=RESIZE_SIZE)  # noqa: SLF001
    robot_geom_ids = robot_geom_ids_for(occ_env)
    target_bbox_mask = target_clear_bbox_mask_224(occ_env)
    last_known_position = target_centroid_224(occ_env, obs, tracking_mode)

    use_gate = condition in ("gate_only", "both")
    use_injection = condition in ("injection_only", "both")

    action_queue = collections.deque()
    blend_engaged_steps = 0
    sd_calls = 0
    max_arm_s_occ = 0.0
    max_fk_s_occ = 0.0
    consecutive_occluded_steps = 0
    max_consecutive_occluded_steps = 0
    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)  # old metric, logged for comparison only
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        fk_mask = fk_arm_mask_224(occ_env, projector, robot_geom_ids)
        fk_s_occ = compute_fk_s_occ(fk_mask, target_bbox_mask)
        max_fk_s_occ = max(max_fk_s_occ, fk_s_occ)

        centroid_now = target_centroid_224(occ_env, obs, tracking_mode)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now
        occluded = fk_s_occ >= GATE_THRESHOLD  # NEW metric gates the interventions
        consecutive_occluded_steps = consecutive_occluded_steps + 1 if occluded else 0
        max_consecutive_occluded_steps = max(max_consecutive_occluded_steps, consecutive_occluded_steps)

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])

        if action_queue:
            action = action_queue.popleft()
        else:
            subgoal_image = None
            if use_injection and occluded:
                mask_raw = arm_pixel_mask(occ_env, obs)  # SD still uses GT arm mask for the inpaint region itself
                recovered_raw = call_sd(obs[AGENTVIEW_KEY], mask_raw, instruction)
                subgoal_image = preprocess_image(recovered_raw)
                sd_calls += 1
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction, subgoal_image=subgoal_image)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        alpha = decayed_alpha(consecutive_occluded_steps) if use_decay else SCENE_BLEND_ALPHA
        if use_gate and occluded and last_known_position is not None and alpha > 0:
            eef_pos_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            eef_pixel = projector.project(eef_pos_world)
            world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos_world, eef_pixel, last_known_position)
            pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
            action = action.copy()
            blended = gated_blend_xy(action[:2].astype(np.float64), pklp_delta_xy, alpha)
            action[:2] = np.clip(blended, -1.0, 1.0)
            blend_engaged_steps += 1

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            return {"condition": condition, "tracking_mode": tracking_mode, "use_decay": use_decay, "episode": episode_idx, "done_step": step,
                    "max_arm_s_occ": max_arm_s_occ, "max_fk_s_occ": max_fk_s_occ, "max_consecutive_occluded_steps": max_consecutive_occluded_steps,
                    "sd_calls": sd_calls, "blend_engaged_steps": blend_engaged_steps}

    return {"condition": condition, "tracking_mode": tracking_mode, "use_decay": use_decay, "episode": episode_idx, "done_step": None,
            "max_arm_s_occ": max_arm_s_occ, "max_fk_s_occ": max_fk_s_occ, "max_consecutive_occluded_steps": max_consecutive_occluded_steps,
            "sd_calls": sd_calls, "blend_engaged_steps": blend_engaged_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--tracking-modes", nargs="+", default=["mask_centroid"], choices=["mask_centroid", "bbox_centroid"])
    parser.add_argument("--use-decay", action="store_true")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []

    def already_done(condition, tracking_mode, episode_idx):
        return any(
            r["condition"] == condition and r.get("tracking_mode", "mask_centroid") == tracking_mode
            and r.get("use_decay", False) == args.use_decay and r["episode"] == episode_idx
            for r in results
        )

    for tracking_mode in args.tracking_modes:
        for condition in args.conditions:
            for episode_idx in range(args.episodes):
                if already_done(condition, tracking_mode, episode_idx):
                    continue
                t0 = time.time()
                result = run_episode(condition, episode_idx, args.max_steps, tracking_mode, args.use_decay)
                result["wall_s"] = time.time() - t0
                results.append(result)
                print(f"[moka_pots_fk {tracking_mode} {condition} ep{episode_idx}] {result}", flush=True)
                args.results_path.write_text(json.dumps(results, indent=2))

    print("\n=== MOKA_POTS FK-OCCLUSION REPORT ===")
    for tracking_mode in args.tracking_modes:
        for condition in CONDITIONS:
            rows = [r for r in results if r["condition"] == condition and r.get("tracking_mode", "mask_centroid") == tracking_mode
                    and r.get("use_decay", False) == args.use_decay]
            if not rows:
                continue
            steps = [r["done_step"] for r in rows if r["done_step"] is not None]
            sd_calls_total = sum(r.get("sd_calls", 0) for r in rows)
            blend_total = sum(r.get("blend_engaged_steps", 0) for r in rows)
            n_engaged = sum(1 for r in rows if r.get("sd_calls", 0) > 0 or r.get("blend_engaged_steps", 0) > 0)
            max_fk = [round(r["max_fk_s_occ"], 3) for r in rows]
            max_old = [round(r["max_arm_s_occ"], 3) for r in rows]
            print(f"[{tracking_mode}, use_decay={args.use_decay}] {condition}: {len(steps)}/{len(rows)} success, steps={steps}")
            print(f"    engagement: {n_engaged}/{len(rows)} (sd_calls_total={sd_calls_total}, blend_total={blend_total})")
            print(f"    max_fk_s_occ per ep: {max_fk}")
            print(f"    max_arm_s_occ (old metric) per ep: {max_old}")


if __name__ == "__main__":
    main()
