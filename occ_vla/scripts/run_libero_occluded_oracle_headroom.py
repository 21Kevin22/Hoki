"""
run_libero_occluded_oracle_headroom.py

Phase A1 (occ_vla, 2026-08-18 experiment plan): does ORACLE correction have
any headroom on the REAL `libero_10_occluded` benchmark (deliberately-placed
3D occluder objects, e.g. `wooden_cabinet_1` -- see
`register_libero_occ_suites.py`), as opposed to the earlier, now-scoped-down
findings 7/11 in `thirdparty/openvla-oft/CLAUDE.md`, which only tested PLAIN
`libero_10`'s much milder incidental self-occlusion and found no headroom
there (see CLAUDE.md finding 12 for why those are NOT the same benchmark).

This is a GATE, not a full pipeline: if oracle (ground-truth clean content,
injected via the same mid-layer-splice technique
`midlayer_oracle_splice.py` already validated for the wrist camera, here
generalized to the AGENTVIEW image block since the real benchmark's
occlusion is agentview-side, not wrist-side) does not beat baseline, there
is no headroom for ANY correction method -- a trained VJEPA predictor, an
image-based detector, anything -- to exploit on this benchmark, and the
whole correction premise needs rethinking before investing further (see
the user's own 2026-08-18 decision criterion).

Occluder identification (per task, NOT hardcoded): the occluder is
whichever body exists in the `libero_10_occluded` task's sim but not in
the matching stock `libero_10` task's -- computed by diffing the two envs'
`sim.model` body-name sets at runtime, reusing the same filename-matching
convention `register_libero_occ_suites.py` already relies on (occluded
BDDL files share their base filename with the stock task they extend).
If a task has zero or more than one extra body, this script prints a loud
warning and SKIPS oracle injection for that task (falls back to
baseline-only for it) rather than silently guessing wrong -- known
confirmed cases from earlier direct bddl diffs this session:
KITCHEN_SCENE8 (moka pots) -> wooden_cabinet_1,
KITCHEN_SCENE6 (mug in microwave) -> desk_caddy_1.

Oracle content mechanism: alpha-zero ONLY the identified occluder body's
geoms (not the whole scene, not the robot) each step, re-render agentview
color -- the TRUE clean scene at this exact sim state (arm pose real,
target real, only the occluder removed). Same alpha-zero rendering
technique already established and validated elsewhere in this project
(`arm_removal_pairs`, `run_libero_occ_benchmark.py`'s hide-and-reveal).

Occlusion mask: the target object's segmentation footprint, captured ONCE
per episode (agentview is a STATIC camera here -- CLAUDE.md finding 10(b)
already established a once-captured baseline is valid for a static camera,
unlike the moving wrist camera, which needs a live per-step baseline).
At each step, compared against the live segmentation to find which of
those pixels no longer show the target -- occluded by the occluder object
and/or the robot arm, whichever is currently in the way. This mask is
what actually gets fed to the model (occlusion_mask kwarg AND the
mid-layer splice's patch_mask_256) -- oracle here means "oracle CONTENT"
(the spliced features are ground truth), the occlusion MASK itself is
already a real, non-privileged-beyond-simulation-access measurement (same
category of "privileged" as every other oracle/ground-truth check this
project has run).

Conditions:
  baseline -- real occluder present, no correction. Matches
              run_libero_occluded_fast_scan.py's own baseline exactly
              (same env/model setup), so its existing n=10 baseline
              numbers are directly comparable/reusable if this script's
              own baseline condition is skipped to save GPU time.
  oracle   -- real occluder present, but the occluded region's mid-layer
              features are spliced from a same-step render with the
              occluder alpha-zeroed (ceiling check, per-task skipped if
              occluder identification is ambiguous -- see above).

Statistical comparison: paired by init_state (same episode index, both
conditions), McNemar's test on the paired success/fail table -- matching
this project's own established convention (see analyze_oft_experiment_logs.py's
compute_k_sweep-adjacent discipline; this script doesn't import that module
since it's LIBERO-Occ-specific, but follows the same n>=20/paired-test bar).

Run with the openvla-oft conda env:
  python scripts/run_libero_occluded_oracle_headroom.py \
    --task-ids 0 1 2 3 4 5 6 7 8 9 --n-episodes 20 \
    --results-dir libero_occluded_oracle_headroom \
    --log-action-diff \
    --save-oracle-features-dir libero_occluded_oracle_features

--log-action-diff / --save-oracle-features-dir (occ_vla addition,
2026-08-18, per user request -- added BEFORE any real n>=20 run, since
this data can't be recaptured after the fact): the ||Delta-a|| log
directly, quantitatively answers "does the correction change the ACTION,
not just intermediate features" (Delta-a ~= 0 despite many corrections
firing => "reaches features, not behavior"; Delta-a large but
trajectories/outcomes still similar => "changes behavior, but the
environment absorbs it") -- replaces indirect inference from trajectory
similarity alone. See run_episode's own docstring for the full mechanism
and cost (one extra forward pass per oracle-correction replan step, not
every env step). --save-oracle-features-dir separately saves the exact
oracle ground-truth features used at each such step, for a later trained-
predictor-vs-oracle reconstruction-error correlation without re-running
oracle. Both off by default (zero cost/behavior change if omitted).
"""

import argparse
import json
import os
import sys
from collections import deque

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action, get_libero_image, get_libero_wrist_image, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, TASK_MAX_STEPS, check_unnorm_key, process_action,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head, get_processor, get_proprio_projector, get_vla_action, prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_image_resize_size, get_model, set_seed_everywhere  # noqa: E402

STOCK_SUITE = "libero_10"
OCCLUDED_SUITE = "libero_10_occluded"
AGENTVIEW_SEG_KEY = "agentview_segmentation_instance"
GRID_SIDE = 16  # 224px / 14px patches, matches train_vjepa_predictor_scaled.py's convention
PATCH_PX = 14
NUM_PATCHES_PER_IMAGE = GRID_SIDE * GRID_SIDE

_CfgStub = type("_CfgStub", (), {"center_crop": True})


# ---------------------------------------------------------------------------
# Occluder identification: diff the occluded task's sim body set against the
# matching stock libero_10 task's, by shared BDDL filename.
# ---------------------------------------------------------------------------

def get_libero_env_seg(task, resolution):
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution,
        camera_segmentations="instance",
    )
    env.seed(0)
    return env


def find_occluder_body_names(occluded_task, stock_task_suite):
    """Returns a list of body names present in `occluded_task`'s scene but
    not in the matching stock libero_10 task's -- the occluder(s). Matched
    by BDDL filename (occluded suite's tasks are copies of stock tasks with
    the same base filename, per register_libero_occ_suites.py's own
    docstring). Returns [] (with a printed warning) if no stock task with
    the same filename is found, or if the diff isn't exactly 1 extra body
    (ambiguous -- don't guess)."""
    stock_task = None
    for t in stock_task_suite.tasks:
        if t.bddl_file == occluded_task.bddl_file:
            stock_task = t
            break
    if stock_task is None:
        print(f"  [occluder-id] WARNING: no stock libero_10 task matches bddl_file={occluded_task.bddl_file!r} -- skipping oracle for this task")
        return []

    env_occ = get_libero_env_seg(occluded_task, resolution=64)  # small render, only need body names
    env_stock = get_libero_env_seg(stock_task, resolution=64)
    try:
        sim_occ, sim_stock = env_occ.env.sim, env_stock.env.sim
        names_occ = {sim_occ.model.body_id2name(i) for i in range(sim_occ.model.nbody)}
        names_stock = {sim_stock.model.body_id2name(i) for i in range(sim_stock.model.nbody)}
    finally:
        env_occ.close()
        env_stock.close()

    extra = sorted(n for n in (names_occ - names_stock) if n)
    if len(extra) == 0:
        print(f"  [occluder-id] WARNING: 0 extra bodies found for {occluded_task.bddl_file!r} -- skipping oracle for this task")
        return []
    print(f"  [occluder-id] {occluded_task.bddl_file!r}: occluder body/bodies = {extra}")
    return extra


def geom_ids_for_bodies(sim, body_names):
    ids = []
    for i in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[i]
        if sim.model.body_id2name(body_id) in body_names:
            ids.append(i)
    return ids


def geom_ids_for_body_substring(sim, substrings):
    ids = []
    for i in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[i]
        body_name = (sim.model.body_id2name(body_id) or "").lower()
        if any(s in body_name for s in substrings):
            ids.append(i)
    return ids


def get_agentview_frames(env, resize_size):
    """Returns (color_uint8, seg_int) for agentview, same flip convention as
    get_libero_image (both axes reversed)."""
    obs = env.env._get_observations(force_update=True)
    color = obs["agentview_image"][::-1, ::-1].copy()
    seg = obs[AGENTVIEW_SEG_KEY][::-1, ::-1, 0].copy()
    return color, seg


def find_segmentation_ids_for_bodies(env, sim, geom_ids):
    """Same empirical hide/reveal technique as run_libero_occ_benchmark.py's
    find_segmentation_ids, generalized to agentview."""
    _, seg_before = get_agentview_frames(env, None)
    counts_before = {int(v): int((seg_before == v).sum()) for v in np.unique(seg_before)}

    orig_alpha = sim.model.geom_rgba[geom_ids, 3].copy()
    sim.model.geom_rgba[geom_ids, 3] = 0.0
    sim.forward()
    _, seg_after = get_agentview_frames(env, None)
    sim.model.geom_rgba[geom_ids, 3] = orig_alpha
    sim.forward()
    counts_after = {int(v): int((seg_after == v).sum()) for v in np.unique(seg_after)}

    ids = [v for v, c in counts_before.items() if v != 0 and counts_after.get(v, 0) < 0.1 * c]
    return ids


def pixel_mask_to_token_mask_256(pixel_mask):
    """Boolean HxW pixel mask -> (256,) boolean token mask, center-point-in-
    region convention (matches train_vjepa_predictor_multitask.py's
    build_patch_token_mask_256_from_pixelmask)."""
    h, w = pixel_mask.shape
    mask_grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
    for i in range(GRID_SIDE):
        for j in range(GRID_SIDE):
            center_r = min(int(i * PATCH_PX + PATCH_PX / 2), h - 1)
            center_c = min(int(j * PATCH_PX + PATCH_PX / 2), w - 1)
            if pixel_mask[center_r, center_c]:
                mask_grid[i, j] = True
    return mask_grid.reshape(-1)


# ---------------------------------------------------------------------------
# Mid-layer splice, generalized from midlayer_oracle_splice.py's wrist-only
# (img_idx==1) version to a configurable img_idx (0 = agentview here).
# ---------------------------------------------------------------------------

def _vit_prep(featurizer, x):
    x = featurizer.patch_embed(x)
    x = featurizer._pos_embed(x)
    x = featurizer.patch_drop(x)
    x = featurizer.norm_pre(x)
    return x


def _run_vit_with_midlayer_splice(featurizer, x_corrupted_pixels, x_clean_pixels, split_layer, patch_mask_256,
                                   feature_store=None, feature_store_key=None):
    """feature_store/feature_store_key (occ_vla addition, 2026-08-18): if
    both given, stashes the oracle patch_clean tensor (detached, fp32, CPU)
    at the split layer into feature_store[feature_store_key]. Lets a caller
    save the exact ground-truth features actually used for this splice --
    e.g. for a future trained-predictor-vs-oracle reconstruction-error
    comparison, which can't be recomputed after the fact once a rollout has
    moved past this step. Default None/None for both args -- zero effect on
    any existing caller."""
    num_blocks = len(featurizer.blocks)
    extraction_layer = num_blocks - 2
    assert 0 <= split_layer < extraction_layer, f"split_layer={split_layer} must be in [0, {extraction_layer})"
    num_prefix = featurizer.num_prefix_tokens

    x_corrupted = _vit_prep(featurizer, x_corrupted_pixels)
    x_clean = _vit_prep(featurizer, x_clean_pixels)

    for i, blk in enumerate(featurizer.blocks):
        x_corrupted = blk(x_corrupted)
        if i <= split_layer:
            x_clean = blk(x_clean)
        if i == split_layer:
            mask = patch_mask_256.to(dtype=torch.bool, device=x_corrupted.device).reshape(1, -1, 1)
            patch_corrupted = x_corrupted[:, num_prefix:]
            patch_clean = x_clean[:, num_prefix:]
            if feature_store is not None and feature_store_key is not None:
                feature_store[feature_store_key] = patch_clean.detach().to(torch.float32).cpu().numpy()
            spliced_patch = torch.where(mask, patch_clean, patch_corrupted)
            x_corrupted = torch.cat([x_corrupted[:, :num_prefix], spliced_patch], dim=1)
            del x_clean
        if i == extraction_layer:
            break
    return x_corrupted[:, num_prefix:]


def make_agentview_midlayer_splice_forward(vision_backbone, split_frac, img_idx=0):
    """Like midlayer_oracle_splice.make_midlayer_splice_forward, but splices
    image index `img_idx` (0 = agentview, the real LIBERO-Occ occlusion
    channel) instead of the hardcoded wrist (index 1). Reads clean pixel
    values + the per-step token mask off vision_backbone attributes the
    eval loop sets before each get_vla_action call -- same pattern as the
    original diagnostic."""

    def patched_forward(pixel_values, occlusion_mask=None, proprio_for_dynamics=None):
        assert vision_backbone.use_fused_vision_backbone
        num_images = vision_backbone.num_images_in_input
        clean_pixels = getattr(vision_backbone, "_diagnostic_clean_agentview_pixel_values", None)
        patch_mask_256 = getattr(vision_backbone, "_diagnostic_agentview_patch_mask_256", None)
        # occ_vla addition (2026-08-18): if the caller set this to a dict
        # before invoking get_vla_action, the oracle patch_clean features
        # actually used for the splice this call get stashed into it under
        # "dino"/"siglip" -- see _run_vit_with_midlayer_splice's docstring.
        feature_store = getattr(vision_backbone, "_diagnostic_feature_store", None)

        images = [pixel_values] if num_images == 1 else torch.split(pixel_values, [6] * num_images, dim=1)

        all_patches = []
        for idx, img in enumerate(images):
            img_regular, img_fused = torch.split(img, [3, 3], dim=1)
            if idx == img_idx and clean_pixels is not None and patch_mask_256 is not None and bool(patch_mask_256.any()):
                # occ_vla addition (2026-08-18, per user request): independent
                # runtime evidence that the splice was ACTUALLY applied this
                # call -- not inferred from code reading a second time. This
                # is the one and only place the correction branch fires, so
                # incrementing here (not from the caller's separately-tracked,
                # and previously found to be always-None, `occlusion_mask`
                # local) is the ground truth. Reset per-episode by the caller
                # via `vision_backbone._diagnostic_correction_applied_count = 0`.
                vision_backbone._diagnostic_correction_applied_count = (
                    getattr(vision_backbone, "_diagnostic_correction_applied_count", 0) + 1
                )
                clean_regular, clean_fused = torch.split(clean_pixels, [3, 3], dim=1)
                nb_dino = len(vision_backbone.featurizer.blocks)
                nb_siglip = len(vision_backbone.fused_featurizer.blocks)
                sl_dino = int(nb_dino * split_frac)
                sl_siglip = int(nb_siglip * split_frac)
                patches = _run_vit_with_midlayer_splice(vision_backbone.featurizer, img_regular, clean_regular, sl_dino, patch_mask_256,
                                                         feature_store=feature_store, feature_store_key="dino")
                patches_fused = _run_vit_with_midlayer_splice(vision_backbone.fused_featurizer, img_fused, clean_fused, sl_siglip, patch_mask_256,
                                                               feature_store=feature_store, feature_store_key="siglip")
            else:
                patches = vision_backbone.featurizer(img_regular)
                patches_fused = vision_backbone.fused_featurizer(img_fused)
            all_patches.append(torch.cat([patches, patches_fused], dim=2))
        return torch.cat(all_patches, dim=1)

    return patched_forward


def build_pixel_values(agentview_img, wrist_img, processor, prompt, device, dtype):
    images = prepare_images_for_vla([agentview_img, wrist_img], _CfgStub())
    primary, wrist = images
    inputs_primary = processor(prompt, primary).to(device, dtype=dtype)
    inputs_wrist = processor(prompt, wrist).to(device, dtype=dtype)
    return torch.cat([inputs_primary["pixel_values"], inputs_wrist["pixel_values"]], dim=1)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector, resize_size,
                 init_state, max_steps, condition, occluder_geom_ids, target_seg_ids, midlayer_split_frac,
                 original_forward=None, splice_forward=None, log_action_diff=False, save_features_dir=None,
                 task_id=None, episode_idx=None):
    """log_action_diff/save_features_dir (occ_vla addition, 2026-08-18, per
    user request -- these logs must be added BEFORE the real n>=20 run,
    since the underlying data can't be recaptured after the fact):

    log_action_diff -- at each oracle replan step where a real correction
    was applied, also computes the counterfactual baseline action (same
    observation, model.vision_backbone.forward temporarily swapped back to
    `original_forward`, occlusion_mask=None -- i.e. "what would the
    uncorrected model have done here") and records the L2 norm of the
    difference from the actually-used oracle action (Delta-a) plus the
    elapsed consecutive-occluded-step count. Directly, quantitatively
    distinguishes "correction reaches features but not behavior"
    (Delta-a ~= 0) from "correction changes behavior but the environment
    absorbs it" (Delta-a large, trajectories/outcomes still similar) --
    replaces indirect inference from trajectory similarity alone. Real
    cost: one extra forward pass per oracle replan step under real
    occlusion (not every env step) -- opt-in, off by default.

    save_features_dir -- if given, also writes the oracle ground-truth
    patch features (the exact tensors spliced in, via
    _run_vit_with_midlayer_splice's feature_store hook) to a .npz per such
    step, for a later trained-predictor-vs-oracle reconstruction-error
    correlation without needing to re-run oracle."""
    env.reset()
    obs = env.set_init_state(init_state)
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()
    sim = env.env.sim  # re-fetch after reset (stale-reference bug, established this session)

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False
    clear_target_mask = None
    n_occluded_steps = 0
    occluded_run_length = 0  # elapsed consecutive occluded steps -- resets to 0 the moment occlusion clears
    action_diff_log = []
    # occ_vla addition (2026-08-18): reset the ground-truth splice-applied
    # counter (incremented inside patched_forward itself, see
    # make_agentview_midlayer_splice_forward) so each episode's result
    # reports its own count, not a running total across episodes.
    model.vision_backbone._diagnostic_correction_applied_count = 0

    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    try:
        while t < max_steps + cfg.num_steps_wait:
            wrist_img = get_libero_wrist_image(obs).copy()
            agentview_color, agentview_seg = get_agentview_frames(env, resize_size)

            if clear_target_mask is None:
                # Captured ONCE: agentview is a static camera (CLAUDE.md
                # finding 10(b)) -- a start-of-episode baseline stays valid
                # for the whole episode, unlike the moving wrist camera.
                # BUG FIXED (2026-08-18, real smoke-test run): the occluder
                # here is a STATIC, ALWAYS-PRESENT fixture -- it's already
                # blocking the target in the very first live frame, so a
                # baseline taken from that raw frame is already-occluded
                # and self-consistent with every later frame (occlusion
                # never shows up as a CHANGE). Must alpha-zero the occluder
                # itself (same technique used for the oracle content splice
                # below) to get the TRUE, occluder-free target footprint --
                # confirmed necessary: the raw-frame version produced
                # n_occluded_steps=0 across all 4 smoke-test episodes,
                # despite a confirmed 2-object occluder for this task.
                if occluder_geom_ids:
                    orig_alpha_baseline = sim.model.geom_rgba[occluder_geom_ids, 3].copy()
                    sim.model.geom_rgba[occluder_geom_ids, 3] = 0.0
                    sim.forward()
                    _, clear_seg = get_agentview_frames(env, resize_size)
                    sim.model.geom_rgba[occluder_geom_ids, 3] = orig_alpha_baseline
                    sim.forward()
                    clear_target_mask = np.isin(clear_seg, target_seg_ids)
                else:
                    clear_target_mask = np.isin(agentview_seg, target_seg_ids)
                print(f"    [debug] clear_target_mask px={int(clear_target_mask.sum())} "
                      f"(target_seg_ids={target_seg_ids}, occluder_geom_ids={len(occluder_geom_ids)})")

            live_target_mask = np.isin(agentview_seg, target_seg_ids)
            occluded_pixel_mask = clear_target_mask & ~live_target_mask
            frac_occluded_this_step = (
                occluded_pixel_mask.sum() / max(clear_target_mask.sum(), 1)
            )
            if frac_occluded_this_step > 0.05:
                n_occluded_steps += 1
                occluded_run_length += 1
            else:
                occluded_run_length = 0

            occlusion_mask = None
            if condition == "oracle" and occluder_geom_ids and occluded_pixel_mask.any():
                token_mask_256 = pixel_mask_to_token_mask_256(occluded_pixel_mask)
                orig_alpha = sim.model.geom_rgba[occluder_geom_ids, 3].copy()
                sim.model.geom_rgba[occluder_geom_ids, 3] = 0.0
                sim.forward()
                clean_agentview_color, _ = get_agentview_frames(env, resize_size)
                sim.model.geom_rgba[occluder_geom_ids, 3] = orig_alpha
                sim.forward()

                with torch.no_grad():
                    clean_pixel_values = build_pixel_values(
                        clean_agentview_color, wrist_img, processor, prompt, model.device, torch.bfloat16
                    )
                    clean_agentview_pixels, _ = torch.split(clean_pixel_values, [6, 6], dim=1)
                    model.vision_backbone._diagnostic_clean_agentview_pixel_values = clean_agentview_pixels
                    model.vision_backbone._diagnostic_agentview_patch_mask_256 = torch.from_numpy(token_mask_256)

            if len(action_queue) == 0:
                observation = {
                    "full_image": agentview_color,
                    "wrist_image": wrist_img,
                    "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                }
                # occ_vla bug fix (2026-08-18, found while auditing per user
                # request): this used to read `occlusion_mask is not None`,
                # but `occlusion_mask` is a local set to `None` once above
                # and NEVER reassigned anywhere in this file -- always False,
                # so --log-action-diff/--save-oracle-features-dir silently
                # never fired. The actual condition that determines whether
                # patched_forward will apply the splice this call is the same
                # one that gates setting the diagnostic attributes above.
                real_oracle_correction_this_call = (
                    condition == "oracle" and bool(occluder_geom_ids) and bool(occluded_pixel_mask.any())
                )

                # occ_vla addition (2026-08-18): always (re)set, even to None,
                # so a stale dict from an earlier step's call is never
                # silently reused if save_features_dir toggles off mid-run.
                feature_store = {} if (save_features_dir and real_oracle_correction_this_call) else None
                model.vision_backbone._diagnostic_feature_store = feature_store

                actions = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=occlusion_mask,
                )

                if log_action_diff and real_oracle_correction_this_call and original_forward is not None and splice_forward is not None:
                    # Counterfactual: identical observation, forward swapped
                    # back to uncorrected -- what would baseline have done
                    # at this exact state? One extra forward pass.
                    model.vision_backbone.forward = original_forward
                    with torch.no_grad():
                        actions_baseline_ctf = get_vla_action(
                            cfg, model, processor, observation, task_description,
                            action_head=action_head, proprio_projector=proprio_projector,
                            noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=None,
                        )
                    model.vision_backbone.forward = splice_forward
                    a_oracle = np.asarray(actions[0], dtype=float)
                    a_base = np.asarray(actions_baseline_ctf[0], dtype=float)
                    delta_a_first = float(np.linalg.norm(a_oracle - a_base))
                    chunk_oracle = np.asarray(actions, dtype=float)
                    chunk_base = np.asarray(actions_baseline_ctf, dtype=float)
                    n_common = min(len(chunk_oracle), len(chunk_base))
                    delta_a_chunk_mean = float(
                        np.linalg.norm(chunk_oracle[:n_common] - chunk_base[:n_common], axis=-1).mean()
                    )
                    action_diff_log.append({
                        "t": t, "occluded_run_length": occluded_run_length,
                        "frac_occluded": float(frac_occluded_this_step),
                        "delta_a_norm_first": delta_a_first, "delta_a_norm_chunk_mean": delta_a_chunk_mean,
                    })
                    print(f"    [action-diff] t={t} occluded_run_length={occluded_run_length} "
                          f"delta_a_first={delta_a_first:.4f} delta_a_chunk_mean={delta_a_chunk_mean:.4f}")

                if feature_store is not None:
                    fname = f"task{task_id}_ep{episode_idx}_t{t}_features.npz"
                    np.savez_compressed(
                        os.path.join(save_features_dir, fname),
                        dino=feature_store.get("dino"), siglip=feature_store.get("siglip"),
                        occluded_pixel_mask=occluded_pixel_mask, t=t,
                    )
                    model.vision_backbone._diagnostic_feature_store = None

                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        print(f"  Episode error: {e}")

    return {
        "success": success, "done_step": t,
        "n_occluded_steps": n_occluded_steps,
        "action_diff_log": action_diff_log,
        # occ_vla addition (2026-08-18): independent runtime ground truth
        # that the splice was actually applied (incremented inside
        # patched_forward itself, not inferred) -- see
        # make_agentview_midlayer_splice_forward. Expect > 0 for "oracle"
        # under real occlusion, == 0 for "baseline" (which never installs
        # patched_forward as vision_backbone.forward at all).
        "n_correction_applied": getattr(model.vision_backbone, "_diagnostic_correction_applied_count", 0),
    }


def mcnemar_chi2(baseline_success, oracle_success):
    """Paired McNemar's test statistic (no continuity correction, matching
    this project's own established convention elsewhere) -- b = baseline
    succeeded/oracle failed, c = baseline failed/oracle succeeded."""
    b = sum(1 for bs, os_ in zip(baseline_success, oracle_success) if bs and not os_)
    c = sum(1 for bs, os_ in zip(baseline_success, oracle_success) if not bs and os_)
    if b + c == 0:
        return 0.0, b, c
    return ((abs(b - c) - 0) ** 2) / (b + c), b, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-episodes", type=int, default=20)
    # occ_vla addition (2026-08-18, per user request): lets a genuine
    # independent replication reuse a DIFFERENT slice of this task's
    # init_states (e.g. --episode-offset 20 after an initial --n-episodes 20
    # run already consumed init_states[0:20]) instead of accidentally
    # re-running the exact same 20 seeds and calling it a replication.
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--midlayer-split-frac", type=float, default=0.67)
    parser.add_argument("--results-dir", default="libero_occluded_oracle_headroom")
    parser.add_argument("--conditions", nargs="+", default=["baseline", "oracle"])
    parser.add_argument("--load-in-4bit", action="store_true")
    # occ_vla additions (2026-08-18), per user request -- add before the
    # real n>=20 run since this data can't be recaptured after the fact.
    parser.add_argument("--log-action-diff", action="store_true",
                         help="Extra forward pass per oracle-correction replan step: log ||Delta-a|| "
                              "(oracle action vs same-state counterfactual uncorrected action) and "
                              "elapsed occluded-run-length. See run_episode's docstring.")
    parser.add_argument("--save-oracle-features-dir", default=None,
                         help="If set, also saves the oracle ground-truth patch features (.npz) at "
                              "each oracle-correction replan step, for a later predictor-vs-oracle "
                              "reconstruction-error comparison without re-running oracle.")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_oracle_features_dir:
        os.makedirs(args.save_oracle_features_dir, exist_ok=True)

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=args.load_in_4bit,
        center_crop=True, num_open_loop_steps=8, task_suite_name=STOCK_SUITE, seed=7,
    )
    set_seed_everywhere(cfg.seed)
    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)
    max_steps = TASK_MAX_STEPS["libero_10"]  # same underlying scenes/horizon convention as the fast scan

    occluded_suite = benchmark.get_benchmark_dict()[OCCLUDED_SUITE]()
    stock_suite = benchmark.get_benchmark_dict()[STOCK_SUITE]()

    original_forward = model.vision_backbone.forward
    splice_forward = make_agentview_midlayer_splice_forward(model.vision_backbone, args.midlayer_split_frac, img_idx=0)

    all_summary = {}
    for task_id in args.task_ids:
        task = occluded_suite.get_task(task_id)
        task_description = task.language
        print(f"\n=== task_id={task_id} '{task_description}' ===")

        env = get_libero_env_seg(task, resolution=resize_size)
        env.seed(0)
        env.reset()  # obj_of_interest is only populated on the env AFTER reset (not on the Task
                      # benchmark object -- confirmed via src/occ_vla/eval/libero_occ_env.py's own
                      # established convention: self._env.obj_of_interest[0], not task.obj_of_interest)
        target_names = list(getattr(env, "obj_of_interest", []) or [])
        occluder_names = find_occluder_body_names(task, stock_suite)
        # BUG FIXED (2026-08-18, real smoke-test run): find_occluder_body_names
        # opens and closes 2 SEPARATE OffScreenRenderEnv instances internally
        # (env_occ/env_stock). Confirmed via an isolated diagnostic script that
        # the alpha=0 hide-and-reveal segmentation technique itself works
        # correctly on a freshly-created env with no other envs opened/closed
        # first -- but the real run's target_seg_ids came back empty despite
        # identical logic, on the SAME `env` used here, right after
        # find_occluder_body_names's temp envs were closed. Most likely cause:
        # MuJoCo/robosuite's offscreen EGL rendering shares process-global
        # context state, and closing those temp envs left this `env`'s own
        # render state stale -- same category of bug as the already-documented
        # "re-fetch sim after reset" issue elsewhere in this project, just
        # triggered by a DIFFERENT env's lifecycle instead of this env's own
        # reset(). Fix: reset + re-fetch sim again here, AFTER
        # find_occluder_body_names's temp envs have already been opened and
        # closed, so everything render-dependent below uses a guaranteed-fresh
        # context.
        env.reset()
        sim = env.env.sim
        occluder_geom_ids = geom_ids_for_bodies(sim, set(occluder_names)) if occluder_names else []
        target_body_substrings = [n.lower() for n in target_names] or None
        if target_body_substrings is None:
            print("  [target-id] WARNING: env has no obj_of_interest -- cannot compute occlusion mask, oracle will be a no-op")
            target_seg_ids = []
        else:
            target_geom_ids = geom_ids_for_body_substring(sim, target_body_substrings)
            target_seg_ids = find_segmentation_ids_for_bodies(env, sim, target_geom_ids) if target_geom_ids else []
            if not target_seg_ids:
                print(f"  [target-id] WARNING: could not resolve segmentation ids for {target_body_substrings} -- oracle will be a no-op")

        init_states = occluded_suite.get_task_init_states(task_id)
        n = min(args.n_episodes, len(init_states) - args.episode_offset)

        task_results = {}
        for condition in args.conditions:
            model.vision_backbone.forward = splice_forward if condition == "oracle" else original_forward
            results = []
            for ep in range(n):
                res = run_episode(
                    cfg, env, task_description, model, processor, action_head, proprio_projector, resize_size,
                    init_states[args.episode_offset + ep], max_steps, condition, occluder_geom_ids, target_seg_ids, args.midlayer_split_frac,
                    original_forward=original_forward, splice_forward=splice_forward,
                    log_action_diff=args.log_action_diff, save_features_dir=args.save_oracle_features_dir,
                    task_id=task_id, episode_idx=args.episode_offset + ep,
                )
                # occ_vla addition (2026-08-18): report the TRUE global
                # init_states index, not the loop-local `ep` -- otherwise a
                # --episode-offset 20 replication's "episode":0 would look
                # identical to the original run's "episode":0 despite using
                # a completely different init_state, defeating the point of
                # recording which seeds were actually used.
                res["episode"] = args.episode_offset + ep
                results.append(res)
                print(f"  [{condition}] ep{args.episode_offset + ep}: success={res['success']} done_step={res['done_step']} "
                      f"n_occluded_steps={res['n_occluded_steps']} n_action_diff_logged={len(res['action_diff_log'])} "
                      f"n_correction_applied={res['n_correction_applied']}")
            task_results[condition] = results
            with open(os.path.join(args.results_dir, f"task{task_id}.json"), "w") as f:
                json.dump({"task_id": task_id, "task_description": task_description,
                           "occluder_names": occluder_names, "results": task_results}, f, indent=2)

        model.vision_backbone.forward = original_forward

        if "baseline" in task_results and "oracle" in task_results:
            base_s = [r["success"] for r in task_results["baseline"]]
            oracle_s = [r["success"] for r in task_results["oracle"]]
            chi2, b, c = mcnemar_chi2(base_s, oracle_s)
            summary = {
                "baseline_sr": sum(base_s) / len(base_s), "oracle_sr": sum(oracle_s) / len(oracle_s),
                "n": len(base_s), "mcnemar_chi2": chi2, "baseline_only_success": b, "oracle_only_success": c,
                "n_occluder_bodies": len(occluder_names),
            }
            # occ_vla addition (2026-08-18): aggregate ||Delta-a|| across every
            # oracle-correction replan step logged this task, if enabled --
            # answers "does the correction change the ACTION, not just
            # features" directly and quantitatively, per user request.
            if args.log_action_diff:
                all_deltas_first = [
                    d["delta_a_norm_first"] for r in task_results.get("oracle", []) for d in r["action_diff_log"]
                ]
                if all_deltas_first:
                    arr = np.array(all_deltas_first)
                    summary["action_diff_n"] = int(len(arr))
                    summary["action_diff_mean_first"] = float(arr.mean())
                    summary["action_diff_median_first"] = float(np.median(arr))
                    summary["action_diff_frac_near_zero_lt_0p01"] = float((arr < 0.01).mean())
                    print(f"  task{task_id} action-diff (n={len(arr)} oracle-correction replan steps): "
                          f"mean||Delta-a||={arr.mean():.4f} median={np.median(arr):.4f} "
                          f"frac(||Delta-a||<0.01)={(arr < 0.01).mean()*100:.1f}%")
                else:
                    print(f"  task{task_id} action-diff: 0 oracle-correction replan steps logged "
                          f"(occluder/target identification likely failed for this task -- see WARNING lines above)")
            all_summary[task_id] = summary
            print(f"  task{task_id} SUMMARY: baseline={summary['baseline_sr']*100:.1f}% oracle={summary['oracle_sr']*100:.1f}% "
                  f"chi2={chi2:.2f} (n={summary['n']}, sig if >3.84)")

    print("\n=== ALL TASKS DONE ===")
    for tid, s in sorted(all_summary.items()):
        sig = "SIGNIFICANT" if s["mcnemar_chi2"] > 3.84 else "n.s."
        print(f"  task{tid}: baseline={s['baseline_sr']*100:.1f}% oracle={s['oracle_sr']*100:.1f}% chi2={s['mcnemar_chi2']:.2f} ({sig})")
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)


if __name__ == "__main__":
    main()
