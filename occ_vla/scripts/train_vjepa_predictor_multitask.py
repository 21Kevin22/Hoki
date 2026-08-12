"""
train_vjepa_predictor_multitask.py

Multi-task extension of train_vjepa_predictor_scaled.py: mixes on-policy
(agentview, wrist, proprio) data from MULTIPLE LIBERO tasks into one
training set for the mid-layer VJEPA predictors (vjepa_predictor_dino/
_siglip), instead of a single task's data. Goal (per user's explicit
request): test whether a predictor trained across tasks generalizes
better than the single-task (moka_pots-only) predictor did on
mug_in_microwave (0/3 zero-shot, per oft_mug_generalization_n3.json),
i.e. avoid overfitting to one task's scene/object/occlusion statistics.

Differences from train_vjepa_predictor_scaled.py (which this reuses
almost everything from -- run_vit_to_layer, build_pixel_values_batch,
apply_partial_patch, etc. are identical, just imported):
  1. --tasks accepts multiple "suite:task_id:data_dir" specs. Each task's
     episodes are loaded separately (kept in a list of per-task episode
     lists so t-1/t pairs never cross an episode OR task boundary), then
     training pairs are drawn uniformly at random ACROSS the union of all
     tasks' pairs every step (not round-robin/curriculum -- a random mix
     per batch is the simplest thing that could show cross-task transfer,
     and batches routinely contain a mix of tasks already since
     batch_size=8 > num_tasks=2).
  2. Each task gets its own env/task_description/initial_states/prompt
     (LIBERO envs are per-task) and its own periodic eval -- so training
     logs show both tasks' success rate trajectories side by side, not
     just one.
  3. build_pixel_values_batch's single shared `prompt` string is replaced
     with a per-sample prompt list, since a mixed batch now spans
     different task instructions (needed for correct multimodal
     conditioning even though pixel_values itself is expected to be
     prompt-independent -- kept correct rather than relying on that).

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/train_vjepa_predictor_multitask.py \
    --tasks "libero_10:8:oft_onpolicy_rollout_data" "libero_10:9:oft_onpolicy_rollout_data_mug" \
    --num-steps 3000 --batch-size 8 --eval-every 1000 --eval-episodes 5
"""

import argparse
import glob
import json
import os
import random
import sys
import time

OCC_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
SCRIPTS_DIR = os.path.join(OCC_VLA_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, TaskSuite, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    normalize_proprio,
    prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_image_resize_size, get_model  # noqa: E402

from run_oft_camera_dropout_eval import run_episode as eval_run_episode  # noqa: E402
from train_vjepa_predictor_scaled import (  # noqa: E402
    apply_partial_patch,
    build_patch_token_mask_256,
    run_vit_to_layer,
)

_CfgStub = type("_CfgStub", (), {"center_crop": True})


def build_pixel_values_batch_multi(agentview_imgs, wrist_imgs, prompts, processor, device, dtype):
    """Same as train_vjepa_predictor_scaled.build_pixel_values_batch but
    takes a per-sample prompt list instead of one shared prompt string,
    since a mixed-task batch spans different task instructions."""
    primary_tensors, wrist_tensors = [], []
    for av, wr, prompt in zip(agentview_imgs, wrist_imgs, prompts):
        images = prepare_images_for_vla([av, wr], _CfgStub())
        primary, wrist = images
        primary_tensors.append(processor(prompt, primary)["pixel_values"])
        wrist_tensors.append(processor(prompt, wrist)["pixel_values"])
    primary_batch = torch.cat(primary_tensors, dim=0).to(device, dtype=dtype)
    wrist_batch = torch.cat(wrist_tensors, dim=0).to(device, dtype=dtype)
    return torch.cat([primary_batch, wrist_batch], dim=1)


def load_dataset(data_dir):
    episodes = []
    for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        d = np.load(path)
        episodes.append({"agentview": d["agentview"], "wrist": d["wrist"], "proprio": d["proprio"]})
    pairs = [(ep_idx, t) for ep_idx, ep in enumerate(episodes) for t in range(1, len(ep["proprio"]))]
    return episodes, pairs


def parse_task_spec(spec):
    suite, task_id, data_dir = spec.split(":", 2)
    return suite, int(task_id), data_dir


def apply_partial_patch_jittered(img_resized, jitter_px, size_jitter_frac, rng):
    """Same fixed-geometry occlusion as apply_partial_patch (centered square,
    PARTIAL_PATCH_FRAC of h/w), but with the center offset by up to
    +/-jitter_px pixels and the size scaled by up to +/-size_jitter_frac,
    randomized per call. Training-time-only data augmentation (Solution 3,
    occ_vla 2026-08-02): with only 30 episodes of on-policy data and a
    literally-identical mask every sample, the predictor may be overfitting
    to one exact boundary shape/location rather than learning to handle
    occlusion generally -- eval-time geometry stays the fixed
    apply_partial_patch, unchanged, so this only perturbs what the model
    sees during training."""
    from train_vjepa_predictor_scaled import GRAY_FILL, PARTIAL_PATCH_FRAC

    h, w = img_resized.shape[:2]
    base_ph, base_pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
    scale = 1.0 + rng.uniform(-size_jitter_frac, size_jitter_frac)
    ph, pw = max(int(base_ph * scale), 1), max(int(base_pw * scale), 1)
    base_r0, base_c0 = (h - base_ph) // 2, (w - base_pw) // 2
    r0 = min(max(base_r0 + rng.randint(-jitter_px, jitter_px), 0), h - ph)
    c0 = min(max(base_c0 + rng.randint(-jitter_px, jitter_px), 0), w - pw)
    out = img_resized.copy()
    out[r0 : r0 + ph, c0 : c0 + pw] = GRAY_FILL
    return out, (r0, r0 + ph, c0, c0 + pw)


# ADR (Automatic Domain Randomization)-style curriculum for apply_irregular_edge_patch
# (occ_vla, 2026-08-11): both a "strong" (area 3-20%, n_freq=4, radius_jitter=0.6,
# edge_margin 0.2-0.9) and a "weakened" (area 6-18%, n_freq=2, radius_jitter=0.2,
# edge_margin 0.6-1.4) --mask-diverse run were flat at 0/10 on 2 of 3 tasks through
# the FULL 6000 steps -- the augmentation may simply be too hard from step 0 for this
# data scale (~30 episodes/task), independent of its exact final strength. Real,
# published technique (OpenAI 2019, "Solving Rubik's Cube with a Robot Hand" and
# follow-ups; general "curriculum domain randomization" in sim-to-real robotics):
# start at an EASY setting, and only widen toward the target difficulty while eval
# success stays above a threshold -- narrows back down (not just holds) if the model
# is struggling, unlike a naive fixed linear/step-based schedule. ADR_START is a near-
# -trivial version of apply_irregular_edge_patch's own parameter space (large area
# close to the original apply_partial_patch's ~35% fixed square, minimal irregularity,
# large edge_margin so the centroid lands close to the image center for typical sizes
# rather than genuinely edge-clipped) -- deliberately NOT literally apply_partial_patch
# itself, so the whole curriculum stays inside one code path (no cross-function
# blending). ADR_TARGET is supplied via the existing --mask-diverse-* CLI args (so
# both the strong and weak configs already tried remain valid curriculum endpoints).
ADR_START = {
    "area_min": 0.30, "area_max": 0.35,
    "n_freq": 1, "radius_jitter": 0.05,
    "edge_margin_min": 1.5, "edge_margin_max": 2.5,
}


def adr_curriculum_params(level, args):
    """level: 0.0 (ADR_START, easiest) .. 1.0 (the --mask-diverse-* CLI target,
    hardest). Linear interpolation per-field."""
    target = {
        "area_min": args.mask_diverse_area_min, "area_max": args.mask_diverse_area_max,
        "n_freq": args.mask_diverse_n_freq, "radius_jitter": args.mask_diverse_radius_jitter,
        "edge_margin_min": args.mask_diverse_edge_margin_min, "edge_margin_max": args.mask_diverse_edge_margin_max,
    }
    out = {}
    for k in ADR_START:
        out[k] = ADR_START[k] + (target[k] - ADR_START[k]) * level
    return out


def build_patch_token_mask_256_from_pixelmask(pixel_mask):
    """Same center-point-in-region convention as build_patch_token_mask_256
    (train_vjepa_predictor_scaled.py), generalized to an arbitrary boolean
    HxW pixel mask instead of a single rectangle -- needed for
    apply_irregular_edge_patch below, whose corrupted region isn't a
    rectangle."""
    from train_vjepa_predictor_scaled import GRID_SIDE, PATCH_PX

    mask_grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
    for i in range(GRID_SIDE):
        for j in range(GRID_SIDE):
            center_r = int(i * PATCH_PX + PATCH_PX / 2)
            center_c = int(j * PATCH_PX + PATCH_PX / 2)
            if pixel_mask[center_r, center_c]:
                mask_grid[i, j] = True
    return mask_grid.reshape(-1)


def apply_irregular_edge_patch(img_resized, rng, area_frac_range=(0.03, 0.20), n_freq=4, radius_jitter=0.6, edge_margin_range=(0.2, 0.9)):
    """Irregular, edge-anchored occlusion mask -- P2 augmentation (occ_vla
    2026-08-10), targeting the root cause found by the natural-occlusion
    (LIBERO-Occ) investigation in this project's sibling directory
    (openvla-oft/CLAUDE.md, natural_occ_report): apply_partial_patch /
    apply_partial_patch_jittered above are BOTH always a solid, CENTERED
    SQUARE (~35-39% area, only jittered a little in position/size) -- real
    LIBERO-Occ occluders are irregular blobs, much smaller (measured max
    wrist-camera area 6.56-18.23% across the 6 tasks that actually occlude
    the wrist camera), and enter the frame from a screen EDGE rather than
    sitting centered. This augmentation samples area/position/shape from
    ranges grounded in those measurements, not guessed:
      - area_frac_range default (0.03, 0.20) brackets the real measured
        6.56-18.23% severity band with a little margin on both sides.
      - the mask's centroid is anchored near one of 4 edges/4 corners
        (randomly chosen), matching real occluders' edge-entry behavior
        instead of apply_partial_patch's dead-center placement.
      - the boundary is a solid but IRREGULAR blob (polar-coordinate radius
        modulated by a few random low-frequency sinusoids around the
        centroid), not a perfect rectangle or ellipse -- real occluder
        silhouettes are irregular, per the same investigation's feature-
        level generalization check (task7, 47/256 real patches).
    Eval-time geometry is unaffected (always the fixed apply_partial_patch)
    -- this only perturbs the TRAINING distribution. Returns
    (corrupted_img, pixel_mask) -- pixel_mask is a boolean HxW array (not a
    rectangle tuple), consumed by build_patch_token_mask_256_from_pixelmask
    above, not build_patch_token_mask_256 (which expects a rectangle)."""
    from train_vjepa_predictor_scaled import GRAY_FILL

    h, w = img_resized.shape[:2]
    area_frac = rng.uniform(*area_frac_range)
    target_area = area_frac * h * w
    base_radius = np.sqrt(target_area / np.pi)

    anchor = rng.choice(["top", "bottom", "left", "right", "tl", "tr", "bl", "br"])
    edge_margin = base_radius * rng.uniform(*edge_margin_range)  # how far the centroid sits from the true edge
    if anchor == "top":
        cy, cx = edge_margin, rng.uniform(0, w)
    elif anchor == "bottom":
        cy, cx = h - edge_margin, rng.uniform(0, w)
    elif anchor == "left":
        cy, cx = rng.uniform(0, h), edge_margin
    elif anchor == "right":
        cy, cx = rng.uniform(0, h), w - edge_margin
    elif anchor == "tl":
        cy, cx = edge_margin, edge_margin
    elif anchor == "tr":
        cy, cx = edge_margin, w - edge_margin
    elif anchor == "bl":
        cy, cx = h - edge_margin, edge_margin
    else:
        cy, cx = h - edge_margin, w - edge_margin

    n_theta = 64
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radius_theta = np.ones(n_theta)
    for k in range(1, n_freq + 1):
        amp = rng.uniform(0, radius_jitter / k)
        phase = rng.uniform(0, 2 * np.pi)
        radius_theta = radius_theta + amp * np.sin(k * thetas + phase)
    radius_theta = base_radius * np.clip(radius_theta, 0.3, 2.0)

    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dy ** 2 + dx ** 2)
    theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    r_boundary = np.interp(theta, thetas, radius_theta, period=2 * np.pi)
    pixel_mask = r <= r_boundary

    out = img_resized.copy()
    out[pixel_mask] = GRAY_FILL
    return out, pixel_mask


def build_temporal_weight_table(diagnosis_json_path, task_names, n_bins=10):
    """Spatio-Temporal Adaptive Loss, time component: per-task, per-decile
    weight derived directly from diagnose_vjepa_predictor_errors.py's own
    measured mean-L1-error curve (vjepa_error_diagnosis_raw.json) -- NOT a
    hand-picked "last N%" heuristic (Step 1's design, found to target the
    wrong region for moka_pots, whose real hardest decile is 4-5/mid-
    trajectory, not the end). Returns {task_name: (n_bins,) np.ndarray},
    each row normalized to mean 1.0 so a sample's weight is
    "how much harder/easier than this task's own average" -- a task with
    uniformly flat error gets an all-1.0 (i.e. no-op) table.
    """
    with open(diagnosis_json_path) as f:
        raw = json.load(f)
    bins = np.linspace(0, 1, n_bins + 1)
    table = {}
    for task_name in task_names:
        task_rows = [r for r in raw if r["task_name"] == task_name]
        assert task_rows, f"No diagnosis samples found for task_name={task_name} in {diagnosis_json_path}"
        fracs = np.array([r["frac"] for r in task_rows])
        l1s = np.array([r["l1"] for r in task_rows])
        bin_idx = np.clip(np.digitize(fracs, bins) - 1, 0, n_bins - 1)
        decile_means = np.array([l1s[bin_idx == b].mean() if (bin_idx == b).any() else l1s.mean() for b in range(n_bins)])
        table[task_name] = decile_means / decile_means.mean()
    return table


def compute_boundary_weight_grid(mask_grid, boundary_width):
    """Spatio-Temporal Adaptive Loss, space component: erodes `mask_grid`
    (16x16 bool) inward `boundary_width` layers (8-connected), returning a
    same-shaped bool grid marking patches within that many steps of the
    mask's edge -- per Step 2's spatial heatmap finding that error
    concentrates at the occlusion boundary, not the interior, for both
    tasks identically (so this is task-agnostic, unlike the temporal
    table above)."""
    remaining = mask_grid.copy()
    boundary = np.zeros_like(mask_grid, dtype=bool)
    h, w = mask_grid.shape
    for _ in range(boundary_width):
        padded = np.pad(remaining, 1, mode="constant", constant_values=False)
        all_neighbors_remaining = np.ones_like(remaining, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                all_neighbors_remaining &= padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
        newly_boundary = remaining & ~all_neighbors_remaining
        boundary |= newly_boundary
        remaining = remaining & ~newly_boundary
    return boundary


def build_task_context(model, suite, task_id, checkpoint):
    """Builds everything a task needs for eval: env, task_description,
    prompt, initial_states, max_steps. One GenerateConfig per task since
    task_suite_name/pretrained_checkpoint are shared fields but the task
    itself varies. Calls check_unnorm_key on this task's own cfg object --
    get_vla_action reads cfg.unnorm_key directly, so every cfg instance
    passed to eval needs it set, not just the throwaway one used at
    model-load time."""
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=suite,
    )
    check_unnorm_key(cfg, model)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_id)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    initial_states = task_suite.get_task_init_states(task_id)
    max_steps = TASK_MAX_STEPS[TaskSuite(suite)]
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    return {
        "cfg": cfg, "env": env, "task_description": task_description, "prompt": prompt,
        "initial_states": initial_states, "max_steps": max_steps, "suite": suite, "task_id": task_id,
    }


def run_rollout_eval(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_states, n_episodes, max_steps):
    model.eval()
    n_success = 0
    with torch.no_grad():
        for ep in range(n_episodes):
            success, done_step, n_calls = eval_run_episode(
                cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                initial_states[ep], "wrist_partial_vjepa", max_steps,
            )
            print(f"    [eval ep{ep}] success={success} done_step={done_step} n_calls={n_calls}")
            n_success += int(success)
    return n_success, n_episodes


def eval_all_tasks(task_contexts, model, resize_size, processor, action_head, proprio_projector, n_episodes):
    results = {}
    for tc in task_contexts:
        label = f"{tc['suite']}_task{tc['task_id']}"
        print(f"  -- eval task {label} --")
        n_succ, n_tot = run_rollout_eval(
            tc["cfg"], tc["env"], tc["task_description"], model, resize_size, processor,
            action_head, proprio_projector, tc["initial_states"], n_episodes, tc["max_steps"],
        )
        print(f"  {label}: {n_succ}/{n_tot}")
        results[label] = (n_succ, n_tot)
    return results


def adr_update_level(results, level, args):
    """ADR step: widen (increase level, capped at 1.0) if mean success across all
    tasks' just-measured eval >= args.adr_threshold, else narrow back down (decrease
    by half the widen step, floored at 0.0) -- the "narrows back down if struggling"
    property that distinguishes ADR from a naive fixed schedule."""
    n_succ = sum(s for s, _ in results.values())
    n_tot = sum(t for _, t in results.values())
    mean_success = n_succ / n_tot if n_tot > 0 else 0.0
    if mean_success >= args.adr_threshold:
        new_level = min(1.0, level + args.adr_step)
    else:
        new_level = max(0.0, level - args.adr_step / 2)
    print(f"  [ADR] mean_success={mean_success:.3f} (threshold={args.adr_threshold}) curriculum_level {level:.3f} -> {new_level:.3f}")
    return new_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help='List of "suite:task_id:data_dir" specs, e.g. "libero_10:8:oft_onpolicy_rollout_data" "libero_10:9:oft_onpolicy_rollout_data_mug"',
    )
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-dynamics", type=float, default=1.0)
    parser.add_argument(
        "--precision-weight", type=float, default=1.0,
        help="Loss multiplier applied to samples in a trajectory's final --precision-phase-frac "
             "(timestep-based proxy for 'approach/insertion phase' -- no action data is collected, "
             "so this is the cheapest available proxy, not a verified phase label). 1.0 = off "
             "(uniform weighting, matches every prior run's behavior).",
    )
    parser.add_argument(
        "--precision-phase-frac", type=float, default=0.3,
        help="Fraction of each trajectory's END counted as the 'precision phase' for --precision-weight.",
    )
    parser.add_argument(
        "--temporal-weight-strength", type=float, default=0.0,
        help="Spatio-Temporal Adaptive Loss, time component. 0.0 = off. >0 scales each sample's loss by "
             "1 + strength*(normalized_measured_error[decile] - 1), using the task's own real error-by-"
             "trajectory-decile curve from --diagnosis-json (Step 2's diagnostic), NOT a hand-picked "
             "'last N%' window -- supersedes --precision-weight, which used a single guessed window shared "
             "across tasks and was found (Step 2) to miss moka_pots' true hard region (mid-trajectory, not "
             "the end).",
    )
    parser.add_argument(
        "--diagnosis-json", default="vjepa_error_diagnosis_raw.json",
        help="Path to diagnose_vjepa_predictor_errors.py's raw per-sample output, required when "
             "--temporal-weight-strength > 0.",
    )
    parser.add_argument(
        "--spatial-boundary-width", type=int, default=0,
        help="Spatio-Temporal Adaptive Loss, space component. 0 = off. >0 = boost loss weight for "
             "occlusion-mask patches within this many (8-connected) steps of the mask edge, per Step 2's "
             "spatial heatmap finding that error concentrates at the splice boundary (task-agnostic -- "
             "same pattern on both moka_pots and mug_in_microwave), not the mask interior.",
    )
    parser.add_argument(
        "--spatial-boundary-boost", type=float, default=2.0,
        help="Multiplicative weight for boundary patches when --spatial-boundary-width > 0 (interior "
             "masked patches always stay at weight 1.0).",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="vjepa_predictor_multitask.pt")
    parser.add_argument(
        "--sampling", choices=["balanced", "pooled"], default="balanced",
        help="balanced: each task drawn with equal probability per batch slot (round-robin task order, random pair within task). "
             "pooled: uniform over the flat combined pair pool (skewed toward whichever task has more pairs -- reproduces the earlier run).",
    )
    parser.add_argument(
        "--mask-jitter", action="store_true",
        help="Solution 3: randomize the occlusion mask's center/size per training sample instead of "
             "using apply_partial_patch's single fixed geometry every time (data augmentation against "
             "overfitting to one boundary shape, given only 30 episodes/task). Eval-time geometry is "
             "unaffected (always the fixed apply_partial_patch). Off by default.",
    )
    parser.add_argument("--mask-jitter-px", type=int, default=20, help="Max +/- pixel offset for --mask-jitter's center.")
    parser.add_argument("--mask-jitter-size-frac", type=float, default=0.1, help="Max +/- fractional size change for --mask-jitter.")
    parser.add_argument(
        "--mask-diverse", action="store_true",
        help="P2 (occ_vla 2026-08-10): use apply_irregular_edge_patch instead of the fixed centered "
             "square (with or without --mask-jitter) -- irregular, small, edge-anchored masks sampled "
             "from the real LIBERO-Occ severity/geometry range, directly targeting the geometric-"
             "overfitting root cause found by the natural-occlusion investigation. Mutually exclusive "
             "with --mask-jitter in effect (if both are passed, --mask-diverse takes precedence per "
             "training step). Eval-time geometry is unaffected (always the fixed apply_partial_patch).",
    )
    parser.add_argument("--mask-diverse-area-min", type=float, default=0.03)
    parser.add_argument("--mask-diverse-area-max", type=float, default=0.20)
    parser.add_argument(
        "--mask-diverse-n-freq", type=int, default=4,
        help="apply_irregular_edge_patch: number of sinusoidal harmonics perturbing the boundary radius. "
             "Lower = smoother/rounder blob, closer to an ellipse. Weakened-strength runs (occ_vla 2026-08-11, "
             "after the first --mask-diverse attempt failed 0/10 on 2 of 3 tasks) should lower this.",
    )
    parser.add_argument(
        "--mask-diverse-radius-jitter", type=float, default=0.6,
        help="apply_irregular_edge_patch: max amplitude of the boundary-radius perturbation (as a fraction of "
             "base_radius). Lower = less irregular/more circular. Same weakening rationale as --mask-diverse-n-freq.",
    )
    parser.add_argument(
        "--mask-diverse-edge-margin-min", type=float, default=0.2,
        help="apply_irregular_edge_patch: min centroid-to-edge distance (as a multiple of base_radius). "
             "Higher = less extreme edge-clipping/cropping of the occluder.",
    )
    parser.add_argument(
        "--mask-diverse-edge-margin-max", type=float, default=0.9,
        help="apply_irregular_edge_patch: max centroid-to-edge distance (as a multiple of base_radius).",
    )
    parser.add_argument(
        "--adr-curriculum", action="store_true",
        help="ADR (Automatic Domain Randomization)-style curriculum (occ_vla 2026-08-11): requires --mask-diverse. "
             "Starts at ADR_START (near-trivial: large centered-ish area, minimal irregularity) and widens toward "
             "the --mask-diverse-* CLI target only while eval success stays >= --adr-threshold, narrowing back down "
             "otherwise -- see adr_curriculum_params()'s docstring for the full rationale.",
    )
    parser.add_argument("--adr-threshold", type=float, default=0.3, help="Mean eval success rate (across all tasks) needed to widen the curriculum.")
    parser.add_argument("--adr-step", type=float, default=0.15, help="Curriculum level (0..1) change per eval checkpoint.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_specs = [parse_task_spec(s) for s in args.tasks]
    assert len(task_specs) >= 2, "multi-task training needs >=2 --tasks specs"

    # Load each task's episodes/pairs SEPARATELY -- pairs never cross a
    # task boundary, only carry a task index so the training loop knows
    # which env/prompt each sample belongs to. Keep both a flat pool
    # (pooled sampling -- reproduces the earlier moka:mug ~58:42 skewed
    # run) and a per-task list (balanced sampling -- draws each task with
    # equal probability regardless of its own pair count).
    all_episodes = []  # all_episodes[task_idx] = list of episode dicts
    pairs_by_task = []  # pairs_by_task[task_idx] = list of (ep_idx, t)
    all_pairs = []       # flat list of (task_idx, ep_idx, t), pooled sampling only
    task_names = []       # matches diagnose_vjepa_predictor_errors.py's "{suite}_task{task_id}" naming
    for task_idx, (suite, task_id, data_dir) in enumerate(task_specs):
        episodes, pairs = load_dataset(data_dir)
        assert len(pairs) > 0, f"No training pairs found in {data_dir}"
        print(f"Task {task_idx} ({suite} task_id={task_id}, {data_dir}): {len(episodes)} episodes, {len(pairs)} pairs")
        all_episodes.append(episodes)
        pairs_by_task.append(pairs)
        all_pairs.extend((task_idx, ep_idx, t) for ep_idx, t in pairs)
        task_names.append(f"{suite}_task{task_id}")
    print(f"Total mixed pool: {len(all_pairs)} (task, episode, t) training pairs across {len(task_specs)} tasks")

    temporal_weight_table = None
    if args.temporal_weight_strength > 0:
        temporal_weight_table = build_temporal_weight_table(args.diagnosis_json, task_names)
        print("Temporal weight table (Spatio-Temporal Adaptive Loss, time component), normalized_error by decile:")
        for tn in task_names:
            print(f"  {tn}: {np.array2string(temporal_weight_table[tn], precision=3)}")
    if args.sampling == "balanced":
        print(f"Sampling: BALANCED -- each task drawn with equal probability regardless of its own pair count (moka:mug would be skewed ~{100*len(pairs_by_task[0])/len(all_pairs):.0f}:{100*len(pairs_by_task[1])/len(all_pairs):.0f} under pooled sampling)")
    else:
        print("Sampling: POOLED -- uniform over the combined pair pool (skewed toward whichever task has more pairs)")

    model = get_model(GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=task_specs[0][0],
    ))
    proprio_projector = get_proprio_projector(GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=task_specs[0][0]), model.llm_dim, proprio_dim=8)
    action_head = get_action_head(GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=task_specs[0][0]), model.llm_dim)
    processor = get_processor(GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=task_specs[0][0]))
    check_unnorm_key(GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=task_specs[0][0]), model)
    resize_size = get_image_resize_size(GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=task_specs[0][0]))
    device = model.device
    dtype = torch.bfloat16
    vb = model.vision_backbone

    split_frac = vb.midlayer_split_frac
    split_layer_dino = int(len(vb.featurizer.blocks) * split_frac)
    split_layer_siglip = int(len(vb.fused_featurizer.blocks) * split_frac)
    print(f"split_frac={split_frac} -> dino block {split_layer_dino}, siglip block {split_layer_siglip}")

    print("\nBuilding per-task eval contexts (env, prompt, initial_states)...")
    task_contexts = [
        build_task_context(model, suite, task_id, args.checkpoint) for suite, task_id, _ in task_specs
    ]

    for p in model.parameters():
        p.requires_grad = False
    trainable_params = list(vb.vjepa_predictor_dino.parameters()) + list(vb.vjepa_predictor_siglip.parameters())
    for p in trainable_params:
        p.requires_grad = True
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    # All task_specs share one checkpoint, hence one unnorm_key/proprio norm stats
    # regardless of which task's data a given training sample came from.
    _tmp_cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=task_specs[0][0])
    check_unnorm_key(_tmp_cfg, model)
    proprio_norm_stats = model.norm_stats[_tmp_cfg.unnorm_key]["proprio"]

    print("\n=== Eval before training (step 0) ===")
    _pretrain_results = eval_all_tasks(task_contexts, model, resize_size, processor, action_head, proprio_projector, args.eval_episodes)

    curriculum_level = 0.0
    if args.adr_curriculum:
        curriculum_level = adr_update_level(_pretrain_results, curriculum_level, args)

    task_order = list(range(len(task_specs)))

    losses, grad_norms = [], []
    saw_nan = False
    train_t0 = time.time()

    for step in range(1, args.num_steps + 1):
        model.train()
        vb.vjepa_predictor_dino.train()
        vb.vjepa_predictor_siglip.train()

        if args.sampling == "balanced":
            # Round-robin task assignment across the batch (e.g. batch_size=8,
            # 2 tasks -> exactly 4:4 every single batch, not just in expectation),
            # then a random pair within that task's own pool -- removes the
            # ~58:42 skew pooled sampling had from moka_pots' larger pair count.
            batch_samples = []
            for i in range(args.batch_size):
                task_idx = task_order[i % len(task_order)]
                ep_idx, t = random.choice(pairs_by_task[task_idx])
                batch_samples.append((task_idx, ep_idx, t))
        else:
            batch_samples = [random.choice(all_pairs) for _ in range(args.batch_size)]
        agentview_t, wrist_t, agentview_tm1, wrist_tm1, wrist_t_corrupted, proprio_t, prompts = [], [], [], [], [], [], []
        mask_per_sample = []
        precision_weights = []
        temporal_weights = []
        for task_idx, ep_idx, t in batch_samples:
            ep = all_episodes[task_idx][ep_idx]
            tc = task_contexts[task_idx]
            agentview_t.append(ep["agentview"][t])
            wrist_t.append(ep["wrist"][t])
            agentview_tm1.append(ep["agentview"][t - 1])
            wrist_tm1.append(ep["wrist"][t - 1])
            if args.mask_diverse:
                if args.adr_curriculum:
                    p = adr_curriculum_params(curriculum_level, args)
                    corrupted, pixel_mask = apply_irregular_edge_patch(
                        ep["wrist"][t], random,
                        area_frac_range=(p["area_min"], p["area_max"]),
                        n_freq=max(1, round(p["n_freq"])),
                        radius_jitter=p["radius_jitter"],
                        edge_margin_range=(p["edge_margin_min"], p["edge_margin_max"]),
                    )
                else:
                    corrupted, pixel_mask = apply_irregular_edge_patch(
                        ep["wrist"][t], random,
                        area_frac_range=(args.mask_diverse_area_min, args.mask_diverse_area_max),
                        n_freq=args.mask_diverse_n_freq,
                        radius_jitter=args.mask_diverse_radius_jitter,
                        edge_margin_range=(args.mask_diverse_edge_margin_min, args.mask_diverse_edge_margin_max),
                    )
                sample_mask_256 = build_patch_token_mask_256_from_pixelmask(pixel_mask)
            elif args.mask_jitter:
                corrupted, pixel_bounds = apply_partial_patch_jittered(
                    ep["wrist"][t], args.mask_jitter_px, args.mask_jitter_size_frac, random
                )
                sample_mask_256 = build_patch_token_mask_256(pixel_bounds)
            else:
                corrupted, pixel_bounds = apply_partial_patch(ep["wrist"][t])
                sample_mask_256 = build_patch_token_mask_256(pixel_bounds)
            wrist_t_corrupted.append(corrupted)
            proprio_t.append(normalize_proprio(ep["proprio"][t], proprio_norm_stats))
            prompts.append(tc["prompt"])
            mask_per_sample.append(sample_mask_256)

            # Timestep-based precision-phase proxy: no action data is collected in
            # oft_onpolicy_rollout_data (only agentview/wrist/proprio), so the
            # user's alternative "action-norm" trigger isn't available without a
            # re-collection -- relative position within the episode is the cheapest
            # signal available right now. ep_len-1 matches the max valid t (pairs
            # are built over t in [1, len-1)).
            ep_len = len(ep["proprio"])
            frac_through_episode = t / max(ep_len - 1, 1)
            is_precision_phase = frac_through_episode >= (1.0 - args.precision_phase_frac)
            precision_weights.append(args.precision_weight if is_precision_phase else 1.0)

            # Spatio-Temporal Adaptive Loss, time component: continuous, task-specific,
            # derived from Step 2's real measured error-by-decile curve (see
            # build_temporal_weight_table) rather than a hand-picked window.
            if temporal_weight_table is not None:
                n_bins = len(temporal_weight_table[task_names[task_idx]])
                decile = min(int(frac_through_episode * n_bins), n_bins - 1)
                normalized_error = temporal_weight_table[task_names[task_idx]][decile]
                temporal_weights.append(1.0 + args.temporal_weight_strength * (normalized_error - 1.0))
            else:
                temporal_weights.append(1.0)

        B = args.batch_size
        if args.mask_jitter:
            # Per-sample geometry now (Solution 3) -- mask_256 is (B, 256, 1), not
            # broadcast from a single shared mask.
            mask_256_np = np.stack(mask_per_sample)  # (B, 256)
            mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(B, -1, 1)
        else:
            # apply_partial_patch uses a FIXED geometry (PARTIAL_PATCH_FRAC, centered) that
            # doesn't depend on image content, so mask_256 is identical across samples/tasks --
            # keep the per-sample list only for clarity/future-proofing, but broadcasting a single
            # mask (like the single-task script does) is equivalent and cheaper.
            mask_256_np = mask_per_sample[0]
            mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(1, -1, 1)

        # Spatio-Temporal Adaptive Loss, space component: boost patches at the
        # occlusion-mask boundary (Step 2's spatial heatmap showed error concentrates
        # there, symmetrically for both tasks -- not the mask interior). Mask geometry
        # is fixed every step (apply_partial_patch's centered square), so this is the
        # same grid every step; recomputed here (cheap, pure numpy) rather than hoisted
        # out of the loop, to keep the step loop self-contained and simple.
        if args.spatial_boundary_width > 0:
            assert not args.mask_jitter, "--spatial-boundary-width assumes one shared mask grid per step, incompatible with --mask-jitter's per-sample geometry"
            boundary_grid = compute_boundary_weight_grid(mask_256_np.reshape(16, 16), args.spatial_boundary_width)
            n_boundary_patches = int(boundary_grid.sum())
            w_space_np = np.where(boundary_grid.reshape(-1), args.spatial_boundary_boost, 1.0)
            w_space = torch.from_numpy(w_space_np).to(device=device, dtype=dtype).reshape(1, -1, 1)
        else:
            n_boundary_patches = 0
            w_space = 1.0

        with torch.no_grad():
            pv_clean_t = build_pixel_values_batch_multi(agentview_t, wrist_t, prompts, processor, device, dtype)
            _, wrist_clean_t = torch.split(pv_clean_t, [6, 6], dim=1)
            wrist_reg_clean_t, wrist_fused_clean_t = torch.split(wrist_clean_t, [3, 3], dim=1)
            f_gt_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_t, split_layer_dino)
            f_gt_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_t, split_layer_siglip)

            pv_clean_tm1 = build_pixel_values_batch_multi(agentview_tm1, wrist_tm1, prompts, processor, device, dtype)
            _, wrist_clean_tm1 = torch.split(pv_clean_tm1, [6, 6], dim=1)
            wrist_reg_clean_tm1, wrist_fused_clean_tm1 = torch.split(wrist_clean_tm1, [3, 3], dim=1)
            past_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_tm1, split_layer_dino)
            past_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_tm1, split_layer_siglip)

            pv_corrupted_t = build_pixel_values_batch_multi(agentview_t, wrist_t_corrupted, prompts, processor, device, dtype)
            _, wrist_corrupted_t = torch.split(pv_corrupted_t, [6, 6], dim=1)
            wrist_reg_corrupted_t, wrist_fused_corrupted_t = torch.split(wrist_corrupted_t, [3, 3], dim=1)
            f_input_dino = run_vit_to_layer(vb.featurizer, wrist_reg_corrupted_t, split_layer_dino)
            f_input_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_corrupted_t, split_layer_siglip)

        proprio_tensor = torch.tensor(np.stack(proprio_t), device=device, dtype=dtype)  # (B, 8)

        residual_dino = vb.vjepa_predictor_dino(f_input_dino, past_dino, proprio_tensor)
        f_final_dino = f_input_dino + mask_256 * residual_dino
        residual_siglip = vb.vjepa_predictor_siglip(f_input_siglip, past_siglip, proprio_tensor)
        f_final_siglip = f_input_siglip + mask_256 * residual_siglip

        # precision_weights defaults to all-1.0 when --precision-weight 1.0 (the default) --
        # w_time.sum()==B and w_space==1 when both new mechanisms are off, so
        # combined_weight reduces to exactly mask_256 broadcast over B and every
        # earlier run's numbers are exactly reproducible, not just approximately.
        w_time = torch.tensor(
            [p * tw for p, tw in zip(precision_weights, temporal_weights)], device=device, dtype=dtype,
        ).reshape(-1, 1, 1)
        combined_weight = mask_256 * w_time * w_space  # (B, 256, 1)

        norm_dino = combined_weight.sum() * f_final_dino.shape[-1]
        norm_siglip = combined_weight.sum() * f_final_siglip.shape[-1]
        recon_dino = (combined_weight * (f_final_dino - f_gt_dino).abs()).sum() / norm_dino
        recon_siglip = (combined_weight * (f_final_siglip - f_gt_siglip).abs()).sum() / norm_siglip
        recon_loss = recon_dino + recon_siglip

        dyn_dino = (combined_weight * ((f_final_dino - past_dino) - (f_gt_dino - past_dino)).abs()).sum() / norm_dino
        dyn_siglip = (combined_weight * ((f_final_siglip - past_siglip) - (f_gt_siglip - past_siglip)).abs()).sum() / norm_siglip
        dynamics_loss = dyn_dino + dyn_siglip

        total_loss = recon_loss + args.lambda_dynamics * dynamics_loss

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
        if not torch.isfinite(grad_norm):
            saw_nan = True
            print(f"[step {step}] NON-FINITE GRADIENT NORM: {grad_norm}")
        optimizer.step()

        losses.append(total_loss.item())
        grad_norms.append(grad_norm.item())
        if not np.isfinite(total_loss.item()):
            saw_nan = True

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - train_t0
            steps_per_sec = step / elapsed
            eta_min = (args.num_steps - step) / steps_per_sec / 60 if steps_per_sec > 0 else float("nan")
            task_counts = {}
            for task_idx, _, _ in batch_samples:
                task_counts[task_idx] = task_counts.get(task_idx, 0) + 1
            n_precision = sum(1 for pw in precision_weights if pw > 1.0)
            mean_w_time = float(np.mean(temporal_weights))
            mean_mask_size = mask_256.sum().item() / B  # per-sample mean; == exact per-sample count unless --mask-jitter
            print(
                f"[step {step:5d}/{args.num_steps}] total={total_loss.item():.6f} recon_dino={recon_dino.item():.6f} "
                f"recon_siglip={recon_siglip.item():.6f} grad_norm={grad_norm.item():.4f} "
                f"batch_task_mix={task_counts} precision_samples={n_precision}/{B} mean_w_time={mean_w_time:.3f} "
                f"boundary_patches={n_boundary_patches}/{mean_mask_size:.1f} mask_jitter={args.mask_jitter} "
                f"{steps_per_sec:.3f} steps/s, ETA {eta_min:.1f}min"
            )

        if step % args.eval_every == 0:
            print(f"\n=== Eval at step {step} ===")
            _step_results = eval_all_tasks(task_contexts, model, resize_size, processor, action_head, proprio_projector, args.eval_episodes)
            if args.adr_curriculum:
                curriculum_level = adr_update_level(_step_results, curriculum_level, args)
            ckpt_path = f"{args.save_path}.step{step}"
            torch.save({"dino": vb.vjepa_predictor_dino.state_dict(), "siglip": vb.vjepa_predictor_siglip.state_dict()}, ckpt_path)
            print(f"  saved checkpoint to {ckpt_path}\n")

    assert not saw_nan, "encountered NaN/Inf loss or gradient during training"

    first_mean = float(np.mean(losses[: min(50, len(losses))]))
    last_mean = float(np.mean(losses[-min(50, len(losses)) :]))
    print(f"\nLoss trend: first-50-steps mean={first_mean:.6f} -> last-50-steps mean={last_mean:.6f}")

    torch.save({"dino": vb.vjepa_predictor_dino.state_dict(), "siglip": vb.vjepa_predictor_siglip.state_dict()}, args.save_path)
    print(f"Saved final checkpoint to {args.save_path}")

    print(f"\n=== Final eval (n={args.final_eval_episodes}) ===")
    eval_all_tasks(task_contexts, model, resize_size, processor, action_head, proprio_projector, args.final_eval_episodes)

    del model, trainable_params, optimizer, vb
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("\nGPU memory freed (del + torch.cuda.empty_cache()). Process will now exit, releasing everything else.")


if __name__ == "__main__":
    main()
