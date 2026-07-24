"""T08 (Moka Pots) validation: libero_10 task_id=8,
"put both moka pots on the stove" -- chosen because the arm's reach
into the stove area keeps arm_s_occ high for extended stretches,
maximizing how often sample_arm_free_image() actually fires (unlike
the wooden-cabinet-bowl task, where it never fired once in testing).

Runs the paired BASELINE (routing forced NONE) vs PROPOSED (normal
ControlLoop routing) comparison from run_demo_comparison.py's
run_episode(), adapted to this task, and additionally logs each
MMaDA-triggered step's generated image + the action pi0.5 produced
right after seeing it, so the "does the OOD subgoal image actually
guide the policy, or confuse it" question is inspectable rather than
just inferred from the video.

Uses the same corrected preprocessing as run_demo_comparison.py
(256->224 resize_with_pad, both-axis flip, num_steps_wait settle,
env.seed) -- see that file's docstring for why each of those matters.
"""

import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from openpi_client import image_tools
from robosuite.utils.transform_utils import quat2axisangle

# LIBERO's bundled .pt init-state files predate PyTorch 2.6's
# weights_only=True default -- applied here (not just via an external
# wrapper) so this script also works as a subprocess entry point
# (run_parallel() launches `sys.executable <this file>` directly).
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.integration.occlusion_router import OcclusionRouter, OcclusionSignals, OcclusionSource  # noqa: E402
from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402

# Not /tmp -- see scripts/_workers/pi05_worker.py's RPC_DIR comment
# (this environment cleared /tmp mid-session once already, silently
# killing a worker and losing an in-progress run's output).
# Overridable so BASELINE and PROPOSED can each point at their own pi0.5
# worker (different GPU) and run as separate OS processes concurrently
# -- see run_parallel() / _run_full_comparison's "parallel" path.
PI05_RPC_DIR = os.environ.get("OCC_VLA_PI05_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
MMADA_RPC_DIR = str(_ROOT / ".rpc" / "mmada")

BENCHMARK_SUITE = "libero_10"
TASK_ID = 8  # KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
TARGET_BODY_NAME = "moka_pot_1"
INSTRUCTION = "put both moka pots on the stove"
MAX_STEPS = 520  # openpi reference: "longest training demo has 505 steps" for libero_10
NUM_STEPS_WAIT = 10
RESIZE_SIZE = 224
SEED = 7
INIT_STATE_IDX = 0
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

LOG_DIR = _ROOT / "t08_mmada_log"
FORCE_INTERVAL = 40  # inject on the real current arm mask this often when force_inject=True
# Action chunking: pi0.5 already returns an action_horizon=10 chunk per
# call (Pi0Config.action_horizon in third_party/openpi), but earlier
# versions of this script only ever used actions[0] and re-queried every
# single step -- ~1.5s/step of that was pure RPC + flow-matching
# sampling overhead, unrelated to MMaDA (which only fires occasionally).
# openpi's own reference eval script re-plans every 5 steps
# (examples/libero/main.py: replan_steps=5); 8 is used here for a bit
# more speedup at the cost of slightly more open-loop drift between
# re-queries -- direct control over this tradeoff, not a fixed choice.
REPLAN_STEPS = 8
# Deliberately lowered from PlausibilityChecker's default 0.5 -- see
# session notes: 0.5 rejected every real generation seen so far (scores
# 0.08-0.17), meaning pi0.5 had never actually been shown a real MMaDA
# image. This threshold is for validation-item-B testing (does pi0.5
# use the image sensibly or degrade?), not a recalibrated production value.
FORCED_ACCEPT_THRESHOLD = 0.1
TIMELINE_LOG_DIR = _ROOT / "t08_timeline_log"
# Route B (adaptive trigger): real arm_s_occ observed maxing out at
# 0.150-0.153 across two full T08 rollouts, well under the library
# default ARM_OCC_THRESHOLD=0.30 (arm_free_subgoal.py) -- that default
# would never fire on this task at all. Set low enough to catch genuine
# (if partial) occlusion without matching a fixed-interval schedule.
ADAPTIVE_ARM_THRESHOLD = 0.10


def call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None, cot_anchor=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    if subgoal_image is not None:
        arrays["subgoal_image"] = subgoal_image
    fields = {"prompt": prompt}
    if cot_anchor is not None:
        fields["cot_anchor"] = cot_anchor
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, fields)
    return resp_arrays["actions"]


def call_mmada(image, arm_pixel_mask, instruction, horizon=5):
    resp_arrays, _ = rpc.call(
        MMADA_RPC_DIR,
        {"image": image, "arm_pixel_mask": arm_pixel_mask.astype(np.uint8)},
        {"instruction": instruction, "horizon": horizon},
        timeout_s=180,
    )
    return resp_arrays["image"]


def compute_arm_s_occ(env, seg_image, target_mask_clear: np.ndarray) -> tuple[float, np.ndarray]:
    """Fraction of the target's CLEAR (reference, unoccluded) footprint
    that is currently covered by the arm.

    MuJoCo instance segmentation is single-layer -- each pixel holds the
    ID of whichever geometry is frontmost there, never two IDs at once.
    So `target_mask_now & arm_mask_now` (an earlier, buggy version of
    this function) is structurally ~zero regardless of real occlusion:
    a pixel where the arm is in front of the target gets the *arm's* ID,
    not the target's, so the two live masks are mutually exclusive by
    construction. The fix (matching eval/occluder.py's SoccMetric usage)
    is to capture the target's un-occluded mask once, before the episode
    starts moving the arm, and compare it against what's missing now:
    S_occ = |clear & ~now & arm| / |clear|.
    """
    seg_dict = env.get_segmentation_instances(seg_image)
    target_mask_now = seg_dict.get(TARGET_BODY_NAME)
    arm_mask = seg_dict["robot"].squeeze(-1) != 0
    clear_area = target_mask_clear.sum()
    if target_mask_now is None or clear_area == 0:
        return 0.0, arm_mask
    target_mask_now = target_mask_now.squeeze(-1) != 0
    occluded = target_mask_clear & (~target_mask_now) & arm_mask
    return float(occluded.sum()) / float(clear_area), arm_mask


def state_vec(obs):
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def label(img, text, color=(0, 255, 0)):
    img = img.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def run_episode(
    disable_countermeasures: bool,
    tag: str,
    log_mmada: bool = False,
    force_inject: bool = False,
    max_steps: int | None = None,
    adaptive_threshold: float | None = None,
):
    config = LiberoOccEnvConfig(
        benchmark_suite=BENCHMARK_SUITE,
        task_id=TASK_ID,
        difficulty=Difficulty.LIGHT,
        init_state_idx=INIT_STATE_IDX,
        seed=SEED,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    print(f"[{tag}] occluder placed, S_occ={occ_env.last_s_occ:.3f}", flush=True)

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)

    # Reference/"clear" target footprint, captured once after settling but
    # before the policy has had a chance to move the arm in front of it --
    # see compute_arm_s_occ's docstring for why this baseline is needed.
    seg_dict_clear = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    target_mask_clear_raw = seg_dict_clear.get(TARGET_BODY_NAME)
    target_mask_clear = (
        target_mask_clear_raw.squeeze(-1) != 0
        if target_mask_clear_raw is not None
        else np.zeros(obs[AGENTVIEW_SEGMENTATION_KEY].shape[:2], dtype=bool)
    )
    print(f"[{tag}] clear target footprint: {target_mask_clear.sum()} px", flush=True)

    # Route B (adaptive trigger): fire MMaDA only when the REAL,
    # ground-truth arm_s_occ (compute_arm_s_occ, MuJoCo segmentation --
    # no need for Grounded SAM here, the simulator already gives exact
    # occlusion masks) crosses a threshold, instead of the previous
    # fixed-interval force_inject regardless of actual occlusion state.
    # Threshold lowered from the library default 0.30 to ~0.10-0.15,
    # matching the real observed range in this task (max_arm_s_occ was
    # 0.150-0.153 across two full rollouts, never near 0.30) -- at 0.30
    # this task would never trigger at all.
    router = OcclusionRouter(arm_occ_threshold=adaptive_threshold) if adaptive_threshold is not None else OcclusionRouter()
    plausibility_checker = PlausibilityChecker()
    steps_to_run = max_steps if max_steps is not None else MAX_STEPS

    if log_mmada:
        LOG_DIR.mkdir(exist_ok=True)
        mmada_log = []

    frames = []
    timeline_tiles = []  # one 2x2 [raw_agentview, raw_wrist / agentview(unchanged), generated] tile per MMaDA call
    done_step = None
    mmada_calls = 0
    fallback_calls = 0
    max_arm_s_occ = 0.0
    # Action chunking (see REPLAN_STEPS docstring): only the steps where
    # this is empty pay for a pi0.5/MMaDA RPC round trip; the rest just
    # pop a pre-planned action and step the sim. arm_s_occ is still
    # computed every step (cheap, local segmentation, no RPC) so
    # max_arm_s_occ stays accurate and the video/labels reflect real
    # per-step state even between replans.
    action_queue = collections.deque()
    for step in range(steps_to_run):
        base_image_raw = obs[AGENTVIEW_KEY]
        base_image = preprocess_image(base_image_raw)
        seg_image = obs[AGENTVIEW_SEGMENTATION_KEY]
        arm_s_occ, arm_mask_raw = compute_arm_s_occ(occ_env._env, seg_image, target_mask_clear)  # noqa: SLF001
        max_arm_s_occ = max(max_arm_s_occ, arm_s_occ)

        if action_queue:
            action = action_queue.popleft()
            label_text = f"[{tag}] step {step} | arm_s_occ={arm_s_occ:.2f} | (open-loop, {len(action_queue)} queued)"
            color = (0, 165, 255) if disable_countermeasures else (0, 255, 0)
            frames.append(label(base_image, label_text, color))
            step_result = occ_env.step(action.tolist())
            obs = step_result[0]
            done = step_result[2]
            if done and done_step is None:
                done_step = step
                print(f"[{tag}] success at step {step}", flush=True)
                break
            continue

        wrist_image_raw = obs["robot0_eye_in_hand_image"]
        wrist_image = preprocess_image(wrist_image_raw)
        state = state_vec(obs)
        signals = OcclusionSignals(arm_s_occ=arm_s_occ, scene_dyn_occ=False)
        # Fixed-interval forcing (regardless of real occlusion state) is
        # gone in adaptive_threshold mode -- see comment on `router` above
        # for why: n=1 testing suggested it was disrupting an otherwise
        # converging trajectory by injecting at moments the policy didn't
        # need help. force_inject (legacy fixed-interval mode) still works
        # when adaptive_threshold is None, for backward compatibility.
        forced_this_step = force_inject and adaptive_threshold is None and step > 0 and step % FORCE_INTERVAL == 0
        if disable_countermeasures:
            source = OcclusionSource.NONE
        elif forced_this_step:
            source = OcclusionSource.SELF
        else:
            source = router.route(signals)

        subgoal_image = None
        label_text = f"[{tag}] step {step} | arm_s_occ={arm_s_occ:.2f} | {source.name}"
        if forced_this_step:
            label_text += " (FORCED)"

        if source == OcclusionSource.SELF:
            mmada_calls += 1
            # Always the REAL current arm segmentation mask now -- an
            # earlier version used a synthetic "dilated target footprint"
            # mask here for force_inject, which meant the "occluded"
            # region MMaDA was asked to regenerate was wherever the
            # target *used to be*, not wherever the arm actually *is*.
            # That made the generated result trivially similar to the
            # input (regenerating a pot where a pot already visually
            # belongs) rather than a genuine arm-removal test.
            arm_mask_flipped = np.ascontiguousarray(arm_mask_raw[::-1, ::-1])
            arm_mask = (
                cv2.resize(arm_mask_flipped.astype(np.uint8), (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_NEAREST)
                > 0
            )
            t0 = time.time()
            raw_generated_image = call_mmada(base_image, arm_mask, INSTRUCTION)
            gen_time = time.time() - t0
            score = plausibility_checker.score(
                cv2.resize(raw_generated_image, (RESIZE_SIZE, RESIZE_SIZE)), {"original_image": base_image, "arm_pixel_mask": arm_mask}
            )
            label_text += f" | plaus={score:.2f}"
            # 0.1 throughout this run (not just forced steps) -- user's
            # explicit request, since the default 0.5 rejected every real
            # generation seen so far and never let pi0.5 see one.
            accept_threshold = FORCED_ACCEPT_THRESHOLD
            rejected = score < accept_threshold
            # subgoal_image (what actually gets injected into pi0.5) is
            # nulled out on rejection, but the log always keeps
            # raw_generated_image -- what MMaDA actually produced,
            # regardless of whether PlausibilityChecker trusted it. An
            # earlier version of this logging conflated the two, so a
            # rejected call's "_generated.png" was silently a black
            # placeholder instead of MMaDA's real (if poor) output.
            subgoal_image = None if rejected else raw_generated_image
            if rejected:
                fallback_calls += 1
                label_text += " FALLBACK"
            else:
                label_text += " INJECTED"

            # Timeline tile: top row is what pi0.5 normally sees (raw
            # agentview + raw wrist); bottom row is what it sees when the
            # subgoal is injected -- agentview is untouched (occ_vla only
            # ever replaces the wrist slot, see
            # control/observation_injection.py), so the bottom-left is a
            # repeat of the same agentview frame and only bottom-right
            # (the generated image standing in for the wrist view) changes.
            gen_resized = cv2.resize(raw_generated_image, (RESIZE_SIZE, RESIZE_SIZE))
            top_left = label(base_image, f"step {step}: raw agentview", (255, 255, 255))
            top_right = label(wrist_image, "raw wrist", (255, 255, 255))
            bottom_left = label(base_image, "agentview (unchanged)", (200, 200, 0))
            bottom_right = label(
                gen_resized,
                f"generated->wrist slot (plaus={score:.2f}, {'INJECTED' if not rejected else 'FALLBACK'})",
                (0, 255, 0) if not rejected else (0, 100, 255),
            )
            tile = np.concatenate(
                [np.concatenate([top_left, top_right], axis=1), np.concatenate([bottom_left, bottom_right], axis=1)],
                axis=0,
            )
            timeline_tiles.append(tile)

            if log_mmada:
                cv2.imwrite(str(LOG_DIR / f"step{step:04d}_raw.png"), cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(
                    str(LOG_DIR / f"step{step:04d}_generated.png"),
                    cv2.cvtColor(cv2.resize(raw_generated_image, (RESIZE_SIZE, RESIZE_SIZE)), cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(LOG_DIR / f"step{step:04d}_mask.png"),
                    (arm_mask.astype(np.uint8) * 255),
                )
                mmada_log.append(
                    {
                        "step": step,
                        "forced": forced_this_step,
                        "arm_s_occ": arm_s_occ,
                        "plausibility": score,
                        "accept_threshold": accept_threshold,
                        "rejected": rejected,
                        "gen_time_s": gen_time,
                    }
                )

        if forced_this_step and source == OcclusionSource.SELF:
            # Validation item B: call pi0.5 twice on the *same* observation
            # -- once with only the raw camera feed, once with the subgoal
            # image (if accepted) in right_wrist_0_rgb -- so any behavior
            # change is attributable to the injected image, not to the
            # scene having moved between calls.
            raw_actions = call_pi05(base_image, wrist_image, state, INSTRUCTION, None, None)
            raw_action = raw_actions[0]
            if subgoal_image is not None:
                injected_actions = call_pi05(base_image, wrist_image, state, INSTRUCTION, subgoal_image, None)
                injected_action = injected_actions[0]
                action_chunk = injected_actions
            else:
                injected_action = raw_action  # rejected -> pi0.5 never saw the image, no second call needed
                action_chunk = raw_actions
            action = injected_action
            action_queue.extend(action_chunk[1:REPLAN_STEPS])

            if log_mmada:
                delta = (injected_action - raw_action).tolist()
                mmada_log[-1]["action_raw"] = raw_action.tolist()
                mmada_log[-1]["action_injected"] = injected_action.tolist()
                mmada_log[-1]["action_delta"] = delta
                print(
                    f"  [{tag}] step {step} action delta (injected-raw): "
                    f"dx={delta[0]:+.3f} dy={delta[1]:+.3f} dz={delta[2]:+.3f} "
                    f"drx={delta[3]:+.3f} dry={delta[4]:+.3f} drz={delta[5]:+.3f} dgrip={delta[6]:+.3f}",
                    flush=True,
                )
        else:
            actions = call_pi05(base_image, wrist_image, state, INSTRUCTION, subgoal_image, None)
            action = actions[0]
            action_queue.extend(actions[1:REPLAN_STEPS])

            if log_mmada and source == OcclusionSource.SELF:
                mmada_log[-1]["action_after"] = action.tolist()

        color = (0, 165, 255) if disable_countermeasures else (0, 255, 0)
        frames.append(label(base_image, label_text, color))

        step_result = occ_env.step(action.tolist())
        obs = step_result[0]
        done = step_result[2]
        if done and done_step is None:
            done_step = step
            print(f"[{tag}] success at step {step}", flush=True)
            break

    if log_mmada:
        (LOG_DIR / "mmada_log.json").write_text(json.dumps(mmada_log, indent=2))
        print(f"[{tag}] wrote {len(mmada_log)} mmada log entries to {LOG_DIR}", flush=True)

    print(
        f"[{tag}] finished: {len(frames)} steps, mmada_calls={mmada_calls}, "
        f"fallbacks={fallback_calls}, max_arm_s_occ={max_arm_s_occ:.3f}, done_step={done_step}",
        flush=True,
    )
    return frames, done_step, mmada_calls, timeline_tiles


def write_video(frames, out_path, fps=10):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def write_timeline_image(tiles, out_path):
    if not tiles:
        print(f"no MMaDA-triggered steps -- skipping {out_path}", flush=True)
        return
    stacked = np.concatenate(tiles, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_path} ({len(tiles)} tiles)", flush=True)


def run_role(role: str, adaptive_threshold: float | None) -> None:
    """Subprocess entry point for run_parallel(): runs *only* BASELINE or
    *only* PROPOSED in this process, then writes its own video + a small
    JSON result file for the parent to read back. BASELINE never touches
    MMaDA, so it only needs its own pi0.5 worker (a different GPU via
    OCC_VLA_PI05_RPC_DIR) -- no reason it can't run at the same time as
    PROPOSED's pi0.5+MMaDA pair on their own GPUs."""
    tag = role.upper()
    if role == "baseline":
        frames, done_step, mmada_calls, _ = run_episode(disable_countermeasures=True, tag=tag)
    else:
        frames, done_step, mmada_calls, timeline_tiles = run_episode(
            disable_countermeasures=False, tag=tag, log_mmada=True, adaptive_threshold=adaptive_threshold
        )
        write_timeline_image(timeline_tiles, str(_ROOT / f"demo_t08_timeline_{role}.png"))

    write_video(frames, str(_ROOT / f"demo_t08_{role}.mp4"))
    result = {"role": role, "steps": len(frames), "done_step": done_step, "mmada_calls": mmada_calls}
    (_ROOT / f"demo_t08_{role}_result.json").write_text(json.dumps(result))
    print(f"[{tag}] wrote result: {result}", flush=True)


def run_parallel(adaptive_threshold: float) -> None:
    """Launches BASELINE and PROPOSED as separate OS processes (MuJoCo/
    robosuite envs aren't safe to run twice in one process) on separate
    GPUs, waits for both, then combines their outputs. Requires a THIRD
    pi0.5 worker already running on GPU2, serving
    .rpc/pi05_baseline (see logs/README or CLAUDE.md for the launch
    command) -- this does not start that worker itself."""
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    this_script = str(Path(__file__).resolve())

    env_baseline = os.environ.copy()
    env_baseline["OCC_VLA_ROLE"] = "baseline"
    env_baseline["OCC_VLA_PI05_RPC_DIR"] = str(_ROOT / ".rpc" / "pi05_baseline")

    env_proposed = os.environ.copy()
    env_proposed["OCC_VLA_ROLE"] = "proposed"
    env_proposed["OCC_VLA_ADAPTIVE_THRESHOLD"] = str(adaptive_threshold)

    t0 = time.time()
    with open(log_dir / "t08_parallel_baseline.log", "w") as bl, open(log_dir / "t08_parallel_proposed.log", "w") as pl:
        baseline_proc = subprocess.Popen([sys.executable, this_script], env=env_baseline, stdout=bl, stderr=subprocess.STDOUT)
        proposed_proc = subprocess.Popen([sys.executable, this_script], env=env_proposed, stdout=pl, stderr=subprocess.STDOUT)
        baseline_rc = baseline_proc.wait()
        proposed_rc = proposed_proc.wait()
    print(f"parallel run took {time.time() - t0:.1f}s (baseline_rc={baseline_rc}, proposed_rc={proposed_rc})", flush=True)

    baseline_result = json.loads((_ROOT / "demo_t08_baseline_result.json").read_text())
    proposed_result = json.loads((_ROOT / "demo_t08_proposed_result.json").read_text())

    print("=== T08 PARALLEL REPORT ===")
    print(f"baseline: {baseline_result}")
    print(f"proposed: {proposed_result}")
    bd, pd = baseline_result["done_step"], proposed_result["done_step"]
    if bd is not None and pd is not None:
        print(f"step difference (proposed - baseline): {pd - bd}")
    elif pd is not None and bd is None:
        print("proposed succeeded, baseline did NOT within budget")
    elif bd is not None and pd is None:
        print("baseline succeeded, proposed did NOT within budget")
    else:
        print("neither succeeded within budget")


def _run_full_comparison(proposed_kwargs: dict, tag_suffix: str = ""):
    """Shared by OCC_VLA_FULL_COMPARISON (fixed-interval force_inject) and
    OCC_VLA_ADAPTIVE (real-arm_s_occ-threshold trigger, Route B)."""
    t0 = time.time()
    baseline_frames, baseline_done, _, _ = run_episode(disable_countermeasures=True, tag="BASELINE")
    print(f"baseline took {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    proposed_frames, proposed_done, mmada_calls, timeline_tiles = run_episode(
        disable_countermeasures=False, tag="PROPOSED", log_mmada=True, **proposed_kwargs
    )
    print(f"proposed took {time.time() - t0:.1f}s", flush=True)

    write_video(baseline_frames, str(_ROOT / f"demo_t08_baseline{tag_suffix}.mp4"))
    write_video(proposed_frames, str(_ROOT / f"demo_t08_proposed{tag_suffix}.mp4"))
    write_timeline_image(timeline_tiles, str(_ROOT / f"demo_t08_timeline{tag_suffix}.png"))

    n = max(len(baseline_frames), len(proposed_frames))
    h, w = baseline_frames[0].shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(_ROOT / f"demo_t08_comparison{tag_suffix}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w * 2, h)
    )
    for i in range(n):
        left = baseline_frames[i] if i < len(baseline_frames) else blank
        right = proposed_frames[i] if i < len(proposed_frames) else blank
        writer.write(cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote demo_t08_comparison{tag_suffix}.mp4 ({n} paired frames)", flush=True)

    print(f"=== T08 REPORT{tag_suffix} ===")
    print(f"baseline:  steps={len(baseline_frames)}, success_step={baseline_done}, mmada_calls=0")
    print(f"proposed:  steps={len(proposed_frames)}, success_step={proposed_done}, mmada_calls={mmada_calls}")
    if baseline_done is not None and proposed_done is not None:
        print(f"step difference (proposed - baseline): {proposed_done - baseline_done}")
    elif proposed_done is not None and baseline_done is None:
        print("proposed succeeded, baseline did NOT within budget")
    elif baseline_done is not None and proposed_done is None:
        print("baseline succeeded, proposed did NOT within budget")
    else:
        print("neither succeeded within budget")


def main():
    role = os.environ.get("OCC_VLA_ROLE")
    if role in ("baseline", "proposed"):
        # Subprocess entry point -- see run_parallel(). Threshold is only
        # meaningful for "proposed"; parsed here rather than passed as a
        # CLI arg since this process is launched via `sys.executable
        # <this file>` with no argv, only env vars.
        threshold = float(os.environ.get("OCC_VLA_ADAPTIVE_THRESHOLD", ADAPTIVE_ARM_THRESHOLD))
        run_role(role, threshold)
        return

    if os.environ.get("OCC_VLA_PARALLEL") == "1":
        run_parallel(ADAPTIVE_ARM_THRESHOLD)
        return

    single_run = os.environ.get("OCC_VLA_SINGLE_RUN") == "1"
    force_inject = os.environ.get("OCC_VLA_FORCE_INJECT") == "1"
    full_comparison = os.environ.get("OCC_VLA_FULL_COMPARISON") == "1"
    adaptive = os.environ.get("OCC_VLA_ADAPTIVE") == "1"

    if adaptive:
        # Route B: no fixed-interval forcing -- MMaDA fires only when the
        # real (ground-truth simulator segmentation) arm_s_occ crosses
        # ADAPTIVE_ARM_THRESHOLD. See run_episode's `router` comment for
        # why 0.30 (the library default) never fires on this task and
        # why Grounded SAM isn't needed here (the simulator already gives
        # exact occlusion masks, unlike a real-camera deployment).
        _run_full_comparison({"adaptive_threshold": ADAPTIVE_ARM_THRESHOLD}, tag_suffix="_adaptive")
        return

    if full_comparison:
        # Legacy Route (fixed-interval force_inject) -- kept for
        # comparison against Route B (adaptive=True above), which is now
        # the recommended path per session findings (fixed-interval
        # forcing likely disrupted an otherwise-converging trajectory).
        _run_full_comparison({"force_inject": True})
        return

    if force_inject:
        # Validation items B/C from the user's plan: does pi0.5 actually
        # use a real subgoal image to guide behavior, or ignore/get
        # confused by it? Shorter horizon than the full MAX_STEPS -- this
        # isn't trying to measure success rate, just observe repeated
        # real MMaDA firings.
        t0 = time.time()
        frames, done_step, mmada_calls, timeline_tiles = run_episode(
            disable_countermeasures=False, tag="FORCED", log_mmada=True, force_inject=True, max_steps=160
        )
        print(f"force-inject run took {time.time() - t0:.1f}s", flush=True)
        write_video(frames, str(_ROOT / "demo_t08_forced.mp4"))
        write_timeline_image(timeline_tiles, str(_ROOT / "demo_t08_timeline.png"))
        return

    if single_run:
        t0 = time.time()
        frames, done_step, mmada_calls, _ = run_episode(disable_countermeasures=False, tag="PROPOSED", log_mmada=True)
        print(f"single run took {time.time() - t0:.1f}s", flush=True)
        write_video(frames, str(_ROOT / "demo_t08_proposed.mp4"))
        print(f"wrote demo_t08_proposed.mp4 ({len(frames)} frames)", flush=True)
        return

    t0 = time.time()
    baseline_frames, baseline_done, _, _ = run_episode(disable_countermeasures=True, tag="BASELINE")
    print(f"baseline took {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    proposed_frames, proposed_done, mmada_calls, _ = run_episode(disable_countermeasures=False, tag="PROPOSED", log_mmada=True)
    print(f"proposed took {time.time() - t0:.1f}s", flush=True)

    write_video(baseline_frames, str(_ROOT / "demo_t08_baseline.mp4"))
    write_video(proposed_frames, str(_ROOT / "demo_t08_proposed.mp4"))

    n = max(len(baseline_frames), len(proposed_frames))
    h, w = baseline_frames[0].shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(_ROOT / "demo_t08_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (w * 2, h))
    for i in range(n):
        left = baseline_frames[i] if i < len(baseline_frames) else blank
        right = proposed_frames[i] if i < len(proposed_frames) else blank
        writer.write(cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote demo_t08_comparison.mp4 ({n} paired frames)", flush=True)

    print("=== T08 REPORT ===")
    print(f"baseline:  steps={len(baseline_frames)}, success_step={baseline_done}, mmada_calls=0")
    print(f"proposed:  steps={len(proposed_frames)}, success_step={proposed_done}, mmada_calls={mmada_calls}")


if __name__ == "__main__":
    main()
