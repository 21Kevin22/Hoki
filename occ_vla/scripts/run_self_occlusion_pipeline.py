"""Phase 2 (PKLP overlay) + Phase 4 (soft gating) benchmark under REAL,
naturally-occurring self-occlusion -- unlike run_t08_soft_pipeline.py,
which tested a permanent full pixel-mask blackout (arm_s_occ never
actually crossed the 0.30 gate threshold there; see occ_vla/CLAUDE.md,
"n=10" section). This uses libero_spatial task 4 ("pick up the black
bowl in the top drawer of the wooden cabinet..."), empirically confirmed
(scripts/scan_natural_self_occlusion.py) to cross arm_s_occ > 0.30
naturally on 8.6% of steps -- the best candidate of 5 tasks scanned.

No RAFT/MMaDA needed: the target's position while occluded is tracked
via ground-truth simulator segmentation (available every step in sim)
rather than optical-flow extrapolation -- simpler and more accurate
here than going through RAFT for a case where the object is static
except when the gripper itself is carrying it (see KNOWN LIMITATION
below). `last_known_position` updates whenever arm_s_occ is low
(target clearly visible) and holds otherwise.

KNOWN LIMITATION: once the bowl is actually grasped, it moves with the
gripper -- if arm_s_occ is simultaneously high at that point,
`last_known_position` goes stale (frozen at a pre-grasp location) until
next visible. Not corrected for; noted as a caveat on results, not
fixed here.

Requires only the pi0.5 RPC worker (.rpc/pi05) running.
Run: python3 scripts/run_self_occlusion_pipeline.py [--episodes N] [--max-steps N]
"""

import argparse
import collections
import json
import sys
import time
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

from occ_vla.control.occlusion_gating import apply_soft_gate, gate_scale  # noqa: E402
from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.pklp.visual_overlay import draw_kinematic_overlay  # noqa: E402

import os  # noqa: E402

# Overridable so run_parallel_conditions.py can point each condition's
# process at its own worker/RPC dir when running conditions on
# separate GPUs concurrently, instead of sharing one worker serially.
PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
BENCHMARK_SUITE = "libero_spatial"
TASK_ID = 4
INSTRUCTION_OVERRIDE = None  # use the benchmark's own instruction
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
REPLAN_STEPS = 8
MAX_STEPS = 300  # scan showed this task succeeds within ~116-250 steps
N_EPISODES = 10
GATE_THRESHOLD = 0.30  # matches control/occlusion_gating.DEFAULT_GATE_THRESHOLD
CLEAR_UPDATE_THRESHOLD = 0.05  # below this, target counted "clearly visible" -> refresh last_known_position

RESULTS_PATH = _ROOT / "self_occlusion_pipeline_results.json"

# Ablation conditions (2026-07-16, per user request): isolate which
# piece of "soft_pipeline" (gate + overlay + coordinate cot_anchor) is
# responsible for the null-to-negative result found testing it under
# real self-occlusion. Each condition below is (gate, overlay,
# cot_style); cot_style "none" sends no cot_anchor at all.
#
# - text_semantic (A): plain subtask-style restatement, no coordinates,
#   no overlay, no gating -- isolates whether coordinate-laden text
#   specifically (vs any extra text) was the problem. Hypothesis:
#   pi0.5 was fine-tuned on natural-language subtask labels, not raw
#   (x,y) pixel coordinates -- a coordinate string is plausibly just
#   unparseable noise to its language conditioning.
# - overlay_only (B): dot marker, no text, no gating -- isolates
#   whether the drawn marker itself (out-of-distribution for pi0.5's
#   frozen visual encoder, never seen in training) helps/hurts on its
#   own, separate from any text or attenuation confound.
# - spatial_text (C): overlay + a qualitative spatial-relation
#   description ("in the lower-left of the frame") instead of numeric
#   coordinates -- tests whether relational language (closer to how
#   VLA CoT is normally phrased) succeeds where raw coordinates failed.
ABLATION_CONDITIONS = {
    "baseline": {"gate": False, "overlay": False, "cot_style": "none"},
    "soft_pipeline": {"gate": True, "overlay": True, "cot_style": "coords"},
    "text_semantic": {"gate": False, "overlay": False, "cot_style": "semantic"},
    "overlay_only": {"gate": False, "overlay": True, "cot_style": "none"},
    "spatial_text": {"gate": False, "overlay": True, "cot_style": "spatial"},
}


def _spatial_phrase(position_224: np.ndarray, side: int = RESIZE_SIZE) -> str:
    """Coarse quadrant description of a 224x224-space point, e.g. 'in
    the lower-left of the frame' -- qualitative relational language
    instead of numeric pixel coordinates (Condition C)."""
    x, y = position_224
    vertical = "upper" if y < side / 2 else "lower"
    horizontal = "left" if x < side / 2 else "right"
    return f"in the {vertical}-{horizontal} of the frame"


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def call_pi05(base_image, wrist_image, state, prompt, cot_anchor=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    fields = {"prompt": prompt}
    if cot_anchor is not None:
        fields["cot_anchor"] = cot_anchor
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, fields)
    return resp_arrays["actions"]


def target_centroid_224(occ_env, obs) -> np.ndarray | None:
    """Ground-truth target centroid (from live segmentation, not RAFT),
    in the same 224x224/flipped coordinate space preprocess_image()
    produces -- i.e. what the overlay is drawn onto."""
    seg_dict = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    target_mask_raw = seg_dict.get(occ_env.target_body_name)
    if target_mask_raw is None:
        return None
    target_mask = target_mask_raw.squeeze(-1) != 0
    if not target_mask.any():
        return None
    h, w = target_mask.shape
    ys, xs = np.where(target_mask)
    # same flip preprocess_image applies, then rescale to RESIZE_SIZE
    flipped_ys = h - 1 - ys
    flipped_xs = w - 1 - xs
    scale = RESIZE_SIZE / h  # square frames, resize_with_pad degenerates to plain resize (CLAUDE.md item 9)
    return np.array([flipped_xs.mean() * scale, flipped_ys.mean() * scale])


def run_episode(condition: str, episode_idx: int, max_steps: int, suite: str = None, task_id: int = None) -> dict:
    from libero.libero import benchmark  # noqa: PLC0415

    suite = suite or BENCHMARK_SUITE
    task_id = TASK_ID if task_id is None else task_id
    bench = benchmark.get_benchmark(suite)()
    init_states = bench.get_task_init_states(task_id)
    instruction = INSTRUCTION_OVERRIDE or bench.get_task(task_id).language

    config = LiberoOccEnvConfig(
        benchmark_suite=suite,
        task_id=task_id,
        difficulty=Difficulty.LIGHT,
        init_state_idx=episode_idx % len(init_states),
        seed=SEED,
        place_occluder=False,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)
    occ_env.capture_clear_baseline(obs)

    last_known_position = target_centroid_224(occ_env, obs)
    action_queue = collections.deque()
    cfg = ABLATION_CONDITIONS[condition]
    gate_engaged_steps = 0
    max_arm_s_occ = 0.0
    # Jitter (per user request, 2026-07-16): step-to-step action delta
    # magnitude (excluding the gripper's binary open/close dim, which
    # legitimately jumps) specifically during occluded steps, to check
    # qualitatively whether the intervention smooths or roughens motion
    # right when it's supposed to help, not just whether the episode
    # finishes faster overall.
    prev_action = None
    occluded_jitter = []
    for step in range(max_steps):
        arm_s_occ = occ_env.compute_arm_s_occ(obs)
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)
        centroid_now = target_centroid_224(occ_env, obs)
        if arm_s_occ < CLEAR_UPDATE_THRESHOLD and centroid_now is not None:
            last_known_position = centroid_now

        base_image = preprocess_image(obs[AGENTVIEW_KEY])
        wrist_image = preprocess_image(obs["robot0_eye_in_hand_image"])
        cot_anchor = None
        occluded = arm_s_occ >= GATE_THRESHOLD

        if occluded:
            if cfg["gate"]:
                gate_engaged_steps += 1
                scale = gate_scale(arm_s_occ)
                if scale < 1.0:
                    base_image = apply_soft_gate(base_image, scale)

            if cfg["overlay"] and last_known_position is not None:
                base_image = draw_kinematic_overlay(base_image, last_known_position, last_known_position)

            if cfg["cot_style"] == "coords" and last_known_position is not None:
                cot_anchor = (
                    f"{instruction}. The target is currently occluded by the robot's own arm "
                    f"(S_occ={arm_s_occ:.2f}); its last known position (marked in the image) was "
                    f"approximately ({last_known_position[0]:.0f}, {last_known_position[1]:.0f})."
                )
            elif cfg["cot_style"] == "semantic":
                cot_anchor = f"Continue the task: {instruction}."
            elif cfg["cot_style"] == "spatial" and last_known_position is not None:
                cot_anchor = (
                    f"{instruction}. The target is currently occluded by the robot's own arm; "
                    f"it is {_spatial_phrase(last_known_position)}."
                )

        if action_queue:
            action = action_queue.popleft()
        else:
            actions = call_pi05(base_image, wrist_image, state_vec(obs), instruction, cot_anchor)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

        if occluded and prev_action is not None:
            occluded_jitter.append(float(np.linalg.norm(action[:6] - prev_action[:6])))
        prev_action = action

        obs, _, done, _ = occ_env.step(action.tolist())
        if done:
            return {
                "condition": condition,
                "episode": episode_idx,
                "done_step": step,
                "gate_engaged_steps": gate_engaged_steps,
                "max_arm_s_occ": max_arm_s_occ,
                "occluded_jitter_mean": float(np.mean(occluded_jitter)) if occluded_jitter else None,
                "occluded_jitter_n": len(occluded_jitter),
            }

    return {
        "condition": condition,
        "episode": episode_idx,
        "done_step": None,
        "gate_engaged_steps": gate_engaged_steps,
        "max_arm_s_occ": max_arm_s_occ,
        "occluded_jitter_mean": float(np.mean(occluded_jitter)) if occluded_jitter else None,
        "occluded_jitter_n": len(occluded_jitter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "soft_pipeline"],
        choices=list(ABLATION_CONDITIONS.keys()),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_PATH,
        help="Overridable so run_parallel_conditions.py can give each concurrent per-condition process its own file to avoid clobbering.",
    )
    parser.add_argument("--suite", type=str, default=BENCHMARK_SUITE, help="e.g. libero_10, libero_spatial")
    parser.add_argument("--task-id", type=int, default=TASK_ID)
    args = parser.parse_args()

    results = json.loads(args.results_path.read_text()) if args.results_path.exists() else []
    results = [r for r in results if not (r["condition"] in args.conditions and r["episode"] < args.episodes)]

    for condition in args.conditions:
        for episode_idx in range(args.episodes):
            t0 = time.time()
            result = run_episode(condition, episode_idx, args.max_steps, suite=args.suite, task_id=args.task_id)
            result["wall_s"] = time.time() - t0
            results.append(result)
            print(f"[{condition} ep{episode_idx}] {result}", flush=True)
            args.results_path.write_text(json.dumps(results, indent=2))

    print("=== SELF-OCCLUSION PIPELINE REPORT ===")
    for condition in ABLATION_CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        steps = [r["done_step"] for r in rows if r["done_step"] is not None]
        gate_frac = np.mean([r["gate_engaged_steps"] > 0 for r in rows]) if rows else 0.0
        jitters = [r["occluded_jitter_mean"] for r in rows if r.get("occluded_jitter_mean") is not None]
        jitter_str = f", mean_occluded_jitter={np.mean(jitters):.4f}" if jitters else ""
        print(
            f"{condition}: {len(steps)}/{len(rows)} success, steps={steps}, "
            f"episodes_with_gate_engaged={gate_frac:.0%}{jitter_str}"
        )


if __name__ == "__main__":
    main()
