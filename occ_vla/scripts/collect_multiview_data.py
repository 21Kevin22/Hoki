#!/usr/bin/env python
"""Collect paired (agentview, wrist) camera frames from real pi0.5
rollouts in LIBERO-Occ, as a first step toward a future VIM-style
(LIBERO-Occ/VIM, github.com/litsh/Libero-Occ) complementary-view
generator.

Why this exists: VIM's actual method is not a prompt trick on a frozen
text-to-image model -- it's a supervised, two-stage fine-tune (stage1:
visual-only generation loss, stage2: +action loss) of a UniVLA-based
world model, trained on paired multi-view captures (its
stage{1,2}_multiview_meta.pkl files, per docs/training.md on that
repo). Whatever backbone/approach occ_vla eventually uses for view
synthesis, it will need a dataset shaped like that. This script builds
occ_vla's own version of it now, decoupled from that later decision,
since LiberoOccEnv already renders both `agentview` and
`robot0_eye_in_hand` every step for free (SegmentationRenderEnv's
camera_names=[...], see eval/libero_occ_env.py).

VIM's own .pkl schema isn't publicly documented (checked docs/*.md on
github.com/litsh/Libero-Occ -- both stage files are referenced by name
only, no schema). Don't assume this output matches it byte for byte.
This writes occ_vla's own format instead: one PNG pair per saved frame
plus a global JSONL metadata record (episode/step/instruction/
arm_s_occ/proprio state/image paths) -- straightforward to reshape into
whatever schema a chosen training recipe wants later.

Requires the pi0.5 RPC worker already running (see project CLAUDE.md):
    source third_party/openpi/.venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
        python3 scripts/_workers/pi05_worker.py
pi0.5 (not a random/scripted policy) drives the arm here so collected
frames land in the same occlusion-rich states the real control loop
actually visits, not generic reachable-space noise.

Preprocessing (resize_with_pad 256->224, both-axis flip, num_steps_wait
settle, env.seed) matches openpi's reference eval pipeline exactly --
see run_demo_comparison.py's docstring for why each detail matters.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from openpi_client import image_tools
from robosuite.utils.transform_utils import quat2axisangle

# LIBERO's bundled .pt init-state files predate PyTorch 2.6's
# weights_only=True default.
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

from occ_vla.eval.libero_occ_env import (  # noqa: E402
    AGENTVIEW_KEY,
    AGENTVIEW_SEGMENTATION_KEY,
    WRIST_KEY,
    LiberoOccEnv,
    LiberoOccEnvConfig,
)
from occ_vla.eval.metrics import Difficulty  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
REPLAN_STEPS = 8
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def preprocess_image(raw_image: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(raw_image[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(flipped, RESIZE_SIZE, RESIZE_SIZE))


def state_vec(obs) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def compute_arm_s_occ(env, seg_image, target_body_name: str, target_mask_clear: np.ndarray) -> float:
    """See CLAUDE.md item 7: clear-baseline diff, not live-vs-live
    intersection (MuJoCo segmentation is single-layer)."""
    seg_dict = env.get_segmentation_instances(seg_image)
    target_mask_now = seg_dict.get(target_body_name)
    arm_mask = seg_dict["robot"].squeeze(-1) != 0
    clear_area = target_mask_clear.sum()
    if target_mask_now is None or clear_area == 0:
        return 0.0
    target_mask_now = target_mask_now.squeeze(-1) != 0
    occluded = target_mask_clear & (~target_mask_now) & arm_mask
    return float(occluded.sum()) / float(clear_area)


def compute_total_occ(env, seg_image, target_body_name: str, target_mask_clear: np.ndarray) -> float:
    """Fraction of the target's clear/baseline footprint no longer
    visible right now, regardless of *what* is blocking it (arm,
    placed scene occluder, or both combined) -- unlike arm_s_occ above,
    this isn't restricted to arm-caused occlusion, so it also captures
    the placed OccluderPlacer box's effect (eval/metrics.py's
    Difficulty/SoccMetric target). Useful because the arm's natural
    self-occlusion alone has topped out around 0.10-0.15 on T08 in past
    sessions -- MEDIUM/HEAVY difficulty scene occluders reach it higher
    and on demand, instead of waiting for it to happen naturally."""
    seg_dict = env.get_segmentation_instances(seg_image)
    target_mask_now = seg_dict.get(target_body_name)
    clear_area = target_mask_clear.sum()
    if target_mask_now is None or clear_area == 0:
        return 0.0
    target_mask_now = target_mask_now.squeeze(-1) != 0
    occluded = target_mask_clear & (~target_mask_now)
    return float(occluded.sum()) / float(clear_area)


def call_pi05(base_image, wrist_image, state, prompt):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def collect_episode(
    benchmark_suite: str,
    task_id: int,
    episode_idx: int,
    max_steps: int,
    seed: int,
    every_n_steps: int,
    frames_dir: Path,
    meta_records: list,
    action_records: list,
    difficulty: Difficulty = Difficulty.LIGHT,
    pixel_mask: bool = False,
    pixel_mask_dilate_px: int = 0,
    gate_occ_threshold: float | None = None,
) -> int:
    """pixel_mask=True: skip OccluderPlacer's physical box entirely
    (LiberoOccEnvConfig.place_occluder=False -- no collidable body in
    the scene at all) and instead blacken the target's fixed clear/
    baseline footprint directly in the rendered agentview RGB, every
    step, before it's fed to pi0.5. See libero_occ_env.py's
    place_occluder docstring: the physical box was found (2026-07-15
    session) to sit on the camera-target line, on the table, in the
    robot's own workspace, and pi0.5's gripper was repeatedly observed
    approaching/resting against the box itself rather than the real
    target -- confounding "visually occluded" with "physical obstacle
    + OOD-object attraction". This mode isolates the vision-only
    effect: the masked region is real image content removal with zero
    3D-scene footprint, so there is nothing to collide with or mistake
    for a task-relevant object."""
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(benchmark_suite)()
    instruction = bench.get_task(task_id).language
    init_states = bench.get_task_init_states(task_id)

    config = LiberoOccEnvConfig(
        benchmark_suite=benchmark_suite,
        task_id=task_id,
        difficulty=difficulty,
        init_state_idx=episode_idx % len(init_states),
        seed=seed,
        place_occluder=not pixel_mask,
    )
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    if occ_env.last_s_occ is not None:
        print(f"  episode {episode_idx}: occluder placed, S_occ={occ_env.last_s_occ:.3f}", flush=True)
    target_body_name = occ_env._env.obj_of_interest[0]  # noqa: SLF001

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)

    seg_dict_clear = occ_env._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])  # noqa: SLF001
    target_mask_clear_raw = seg_dict_clear.get(target_body_name)
    target_mask_clear = (
        target_mask_clear_raw.squeeze(-1) != 0
        if target_mask_clear_raw is not None
        else np.zeros(obs[AGENTVIEW_SEGMENTATION_KEY].shape[:2], dtype=bool)
    )

    pixel_mask_region = target_mask_clear
    if pixel_mask and pixel_mask_dilate_px > 0:
        kernel = np.ones((pixel_mask_dilate_px, pixel_mask_dilate_px), np.uint8)
        pixel_mask_region = cv2.dilate(target_mask_clear.astype(np.uint8), kernel).astype(bool)
    if pixel_mask:
        print(f"  episode {episode_idx}: pixel-space mask covering {pixel_mask_region.sum()}px (no physical body)", flush=True)

    ep_dir = frames_dir / f"{benchmark_suite}_task{task_id}" / f"ep{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    action_queue: list = []
    saved = 0
    for step in range(max_steps):
        agentview_raw = obs[AGENTVIEW_KEY]
        wrist_raw = obs[WRIST_KEY]

        arm_s_occ = compute_arm_s_occ(
            occ_env._env, obs[AGENTVIEW_SEGMENTATION_KEY], target_body_name, target_mask_clear  # noqa: SLF001
        )
        total_occ = compute_total_occ(
            occ_env._env, obs[AGENTVIEW_SEGMENTATION_KEY], target_body_name, target_mask_clear  # noqa: SLF001
        )

        if pixel_mask:
            agentview_raw = agentview_raw.copy()
            agentview_raw[pixel_mask_region] = 0
        base_image = preprocess_image(agentview_raw)
        wrist_image = preprocess_image(wrist_raw)

        # Attention-gating probe: once real occlusion (ground-truth
        # total_occ, unaffected by pixel_mask/gating themselves) crosses
        # gate_occ_threshold, zero the *entire* agentview frame pi0.5
        # sees (not just the target's own masked silhouette) -- testing
        # whether forcing full reliance on the never-masked wrist view
        # recovers the ~15-25% step-count overhead seen with only the
        # target-region pixel mask (data/multiview/t08_pixelmask), or
        # makes things worse.
        gated_this_step = gate_occ_threshold is not None and total_occ > gate_occ_threshold
        if gated_this_step:
            base_image = np.zeros_like(base_image)

        if step % every_n_steps == 0:
            agentview_path = ep_dir / f"step{step:05d}_agentview.png"
            wrist_path = ep_dir / f"step{step:05d}_wrist.png"
            from PIL import Image  # noqa: PLC0415

            Image.fromarray(base_image).save(agentview_path)
            Image.fromarray(wrist_image).save(wrist_path)
            meta_records.append(
                {
                    "suite": benchmark_suite,
                    "task_id": task_id,
                    "instruction": instruction,
                    "episode": episode_idx,
                    "step": step,
                    "arm_s_occ": arm_s_occ,
                    "total_occ": total_occ,
                    "proprio_state": state_vec(obs).tolist(),
                    "agentview_path": str(agentview_path.relative_to(frames_dir)),
                    "wrist_path": str(wrist_path.relative_to(frames_dir)),
                }
            )
            saved += 1

        if not action_queue:
            state = state_vec(obs)
            actions = call_pi05(base_image, wrist_image, state, instruction)
            action_queue = list(actions[:REPLAN_STEPS])

        action = action_queue.pop(0)
        # Always logged (unlike the image pairs above, which are
        # subsampled) -- this is the fine-grained arm_s_occ-vs-action
        # series the "does dz stall under occlusion, despite pi0.5
        # already seeing the real wrist view" baseline-dependency check
        # needs. Existing logs (t08_mmada_log/mmada_log.json) only ever
        # recorded 3 points, all clustered right before task success --
        # not enough to tell stall from normal end-of-task deceleration.
        action_records.append(
            {
                "episode": episode_idx,
                "step": step,
                "arm_s_occ": arm_s_occ,
                "total_occ": total_occ,
                "gated": gated_this_step,
                "action": action.tolist(),
            }
        )

        step_result = occ_env.step(action.tolist())
        obs = step_result[0]
        done = step_result[2]
        if done:
            print(f"  episode {episode_idx}: success at step {step}", flush=True)
            break

    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--every-n-steps", type=int, default=5, help="subsample rate for saved frames")
    parser.add_argument("--seed", type=int, default=7, help="base seed; incremented per episode")
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "data" / "multiview")
    parser.add_argument(
        "--difficulty",
        choices=["light", "medium", "heavy"],
        default="light",
        help="OccluderPlacer scene-occluder severity (target S_occ band) -- bump this to force"
        " high occlusion on demand instead of waiting for it to happen naturally via arm motion."
        " Ignored when --pixel-mask is set (no physical occluder is placed at all).",
    )
    parser.add_argument(
        "--pixel-mask",
        action="store_true",
        help="Blacken the target's clear/baseline footprint directly in the rendered agentview RGB"
        " every step, instead of placing OccluderPlacer's physical box. See collect_episode's"
        " docstring: the physical box is a real collidable body pi0.5's gripper was observed"
        " approaching/resting against, confounding vision-only occlusion with physical/OOD effects.",
    )
    parser.add_argument(
        "--pixel-mask-dilate-px",
        type=int,
        default=0,
        help="Extra margin (pixels, in the raw camera-resolution frame) to grow the pixel mask by.",
    )
    parser.add_argument(
        "--gate-occ-threshold",
        type=float,
        default=None,
        help="Attention-gating probe: once ground-truth total_occ crosses this, zero the *entire*"
        " agentview frame pi0.5 sees (not just the pixel-masked target region). None disables gating.",
    )
    args = parser.parse_args()
    difficulty = Difficulty(args.difficulty)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "metadata.jsonl"
    action_trace_path = args.out_dir / "action_trace.jsonl"
    meta_records: list = []
    action_records: list = []

    t0 = time.time()
    total_saved = 0
    for episode_idx in range(args.episodes):
        print(f"[collect] {args.suite} task {args.task_id}, episode {episode_idx}/{args.episodes}", flush=True)
        saved = collect_episode(
            benchmark_suite=args.suite,
            task_id=args.task_id,
            episode_idx=episode_idx,
            max_steps=args.max_steps,
            seed=args.seed + episode_idx,
            every_n_steps=args.every_n_steps,
            frames_dir=args.out_dir,
            meta_records=meta_records,
            action_records=action_records,
            difficulty=difficulty,
            pixel_mask=args.pixel_mask,
            pixel_mask_dilate_px=args.pixel_mask_dilate_px,
            gate_occ_threshold=args.gate_occ_threshold,
        )
        total_saved += saved
        # Flush after every episode -- a crash mid-run shouldn't lose
        # already-collected episodes' metadata.
        with open(meta_path, "w") as f:
            for record in meta_records:
                f.write(json.dumps(record) + "\n")
        with open(action_trace_path, "w") as f:
            for record in action_records:
                f.write(json.dumps(record) + "\n")

    elapsed = time.time() - t0
    print(f"[collect] done: {total_saved} frame pairs across {args.episodes} episodes in {elapsed:.0f}s", flush=True)
    print(f"[collect] metadata: {meta_path}", flush=True)
    print(f"[collect] action trace: {action_trace_path}", flush=True)


if __name__ == "__main__":
    main()
