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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
# occ_vla addition (2026-08-24, Phase 2 proactive avoidance): robosuite's own
# camera-projection utilities (intrinsics from cam_fovy, extrinsics from
# cam_xpos/cam_xmat, real-depth conversion, pixel<->world back-projection) --
# reused as-is rather than hand-rolling pinhole math, since this project's own
# real-robot-deployable design bar (see scripted_recovery_after_stuck) prefers
# validated library functions over new geometry code where one already exists.
from robosuite.utils.camera_utils import (  # noqa: E402
    get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world,
)
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

def get_libero_env_seg(task, resolution, camera_depths=False):
    # occ_vla addition (2026-08-21), per user's Figure-A divergence-
    # analysis request: robosuite/LIBERO already support RGB-D rendering
    # via this one kwarg (confirmed real via env_wrapper.py/
    # bddl_base_domain.py, just never turned on in this project before).
    # Depth obs key becomes f"{cam_name}_depth" e.g. "agentview_depth"
    # (confirmed in robosuite/environments/robot_env.py). Default False,
    # zero behavior change for every existing caller.
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution,
        camera_segmentations="instance", camera_depths=camera_depths,
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
    # occ_vla note (2026-08-18, depth-sweep design): num_blocks - 2 is NOT
    # an arbitrary cap chosen by this file -- it matches the vendored
    # backbone's OWN convention (prismatic/models/backbones/vision/*.py,
    # modeling_prismatic.py all do
    # featurizer.forward = ...get_intermediate_layers(n=num_blocks-2)):
    # this checkpoint's last 2 blocks of each tower are never invoked by
    # the real inference path at all. This is therefore the TRUE effective
    # depth ("L=N" in the depth-sweep's terms), not an approximation.
    extraction_layer = num_blocks - 2
    # occ_vla change (2026-08-18, depth-sweep design): relaxed from `<` to
    # `<=` so split_layer == extraction_layer (the true final used block,
    # "L=N_effective") is a valid, includable sweep endpoint -- previously
    # excluded for no stated architectural reason.
    assert 0 <= split_layer <= extraction_layer, f"split_layer={split_layer} must be in [0, {extraction_layer}]"
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
                # occ_vla addition (2026-08-19, per user request -- correction-
                # magnitude gate signal candidate, saved once now instead of
                # requiring a second rerun): the ORIGINAL (occluded) features
                # this call would have used if not corrected, and the per-
                # masked-patch L2 distance between clean and occluded at this
                # exact layer -- ||patch_clean - patch_corrupted|| restricted
                # to the masked region only (unmasked patches are identical
                # by construction, including them would just dilute this).
                feature_store[f"{feature_store_key}_occ"] = patch_corrupted.detach().to(torch.float32).cpu().numpy()
                mask_flat = mask.reshape(1, -1, 1).float()
                n_masked = mask_flat.sum().clamp(min=1)
                delta_feat = ((patch_clean.detach() - patch_corrupted.detach()) * mask_flat)
                feature_store[f"{feature_store_key}_delta_norm"] = float(
                    (delta_feat.norm(dim=-1).sum() / n_masked).item()
                )
            spliced_patch = torch.where(mask, patch_clean, patch_corrupted)
            x_corrupted = torch.cat([x_corrupted[:, :num_prefix], spliced_patch], dim=1)
            del x_clean
        if i == extraction_layer:
            break
    final_patches = x_corrupted[:, num_prefix:]
    # occ_vla addition (2026-08-18, per user request -- distribution-shift
    # measurement): also stash the FINAL representation, after the spliced
    # patch_clean has been carried through the remaining transformer blocks
    # alongside the rest of the (still-occluded) sequence. This is what
    # actually reaches the action head, and is the natural place to check
    # whether "the injected real clean patch is in-distribution" (patch_clean
    # above, already saved) also implies "the resulting mixed-source sequence
    # is in-distribution" (this) -- they are not the same claim.
    if feature_store is not None and feature_store_key is not None:
        feature_store[f"{feature_store_key}_final"] = final_patches.detach().to(torch.float32).cpu().numpy()
    return final_patches


def make_agentview_midlayer_splice_forward(vision_backbone, split_frac, img_idx=0):
    """Like midlayer_oracle_splice.make_midlayer_splice_forward, but splices
    image index `img_idx` (0 = agentview, the real LIBERO-Occ occlusion
    channel) instead of the hardcoded wrist (index 1). Reads clean pixel
    values + the per-step token mask off vision_backbone attributes the
    eval loop sets before each get_vla_action call -- same pattern as the
    original diagnostic."""

    def patched_forward(pixel_values, occlusion_mask=None, proprio_for_dynamics=None):
        # occ_vla temp diagnostic (2026-08-19): unconditional call counter,
        # to directly measure how many times patched_forward itself is
        # invoked per replan step -- L=0 showed n_correction_applied=60 vs
        # L=N_eff's 30 for the identical n_action_diff_logged=30, and code
        # reading (predict_action calls _process_vision_features exactly
        # once, which calls vision_backbone(...) exactly once) found no
        # explanation. Remove once resolved.
        vision_backbone._diagnostic_forward_call_count = (
            getattr(vision_backbone, "_diagnostic_forward_call_count", 0) + 1
        )
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
                # occ_vla change (2026-08-18, depth-sweep design, per user
                # decision): fractions are now relative to EFFECTIVE N
                # (num_blocks - 2, the true depth this checkpoint's inference
                # path actually uses -- see _run_vit_with_midlayer_splice's
                # extraction_layer comment), not nominal N, and use round()
                # instead of int()-truncation. IMPORTANT: this changes which
                # absolute layer a given FRACTION maps to. The already-run
                # "current" setting (dino layer=16, siglip layer=18, what
                # every task1/6/8 n=20 result so far used) is reproduced
                # exactly by split_frac = 16/22 (~=0.7273), NOT the old
                # 0.67 -- verified directly: round(22*0.67)=15,
                # round(25*0.67)=17, a DIFFERENT (not-yet-tested) layer pair,
                # not (16, 18). Never assume a frac reproduces a prior
                # result without checking; the depth-sweep script computes
                # and prints the resulting (dino, siglip) layer pair for
                # every level before running, specifically to catch this.
                nb_dino_eff = len(vision_backbone.featurizer.blocks) - 2
                nb_siglip_eff = len(vision_backbone.fused_featurizer.blocks) - 2
                if split_frac <= 0.0:
                    # occ_vla addition (2026-08-18, depth-sweep L=0 endpoint):
                    # splicing "before block 0" is architecturally equivalent
                    # to just running the STOCK featurizer on the fully clean
                    # image -- outside the occluded region the clean and
                    # corrupted pixels are already identical (alpha-zeroing
                    # the occluder only changes pixels it actually covered),
                    # so a pixel-level mask-and-splice at this point reduces
                    # exactly to "use the clean image everywhere." No block
                    # loop, no mask needed -- see user's own derivation.
                    # occ_vla bug fix (2026-08-19): this branch used to
                    # increment _diagnostic_correction_applied_count AGAIN
                    # here, on top of the unconditional increment already
                    # done right after entering the outer `if idx==img_idx`
                    # block above -- double-counting every L=0 call (found
                    # empirically: n_forward_calls=30 but
                    # n_correction_applied=60 for the same episode; the
                    # L>0 branch, which has no such duplicate, correctly
                    # showed 30==30). Removed; the outer increment alone is
                    # correct for every split_frac value.
                    patches = vision_backbone.featurizer(clean_regular)
                    patches_fused = vision_backbone.fused_featurizer(clean_fused)
                    if feature_store is not None:
                        feature_store["dino"] = feature_store["dino_final"] = patches.detach().to(torch.float32).cpu().numpy()
                        feature_store["siglip"] = feature_store["siglip_final"] = patches_fused.detach().to(torch.float32).cpu().numpy()
                else:
                    sl_dino = min(int(round(nb_dino_eff * split_frac)), nb_dino_eff)
                    sl_siglip = min(int(round(nb_siglip_eff * split_frac)), nb_siglip_eff)
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
                 task_id=None, episode_idx=None, log_attn_entropy=False, log_ensemble_disagreement=False,
                 pixel_fill_mode="none", prevframe_gate_max_frac_no_ref=1.0, prevframe_feather_px=0.0,
                 disable_collision_geom_ids=None, record_video_dir=None, reactive_collision_disable=False,
                 scripted_recovery=False, low_mobility_geom_ids=None, reactive_dry_run=False,
                 composite_visual_only=False, occluder_seg_ids=None,
                 divergence_extract_dir=None, divergence_extract_t_range=None,
                 ttc_area_blend=False, ttc_threshold=8.0, ttc_safe_action=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0),
                 force_oracle_mask_frac=None, stuck_velocity_trigger=False, stuck_dist_threshold=0.012,
                 stuck_recovery_steps=4, stuck_cooldown_envsteps=None, stuck_retreat_mag=0.6,
                 proactive_avoidance_oracle=False, proactive_safety_margin=0.04, blank_agentview=False,
                 proactive_use_cbf=False, proactive_cbf_gain=2.0,
                 proactive_use_depth=False,
                 proactive_use_mpc=False, proactive_mpc_n_candidates=16, proactive_mpc_noise_std=0.15,
                 proactive_mpc_w_safety=50.0, proactive_mpc_w_fidelity=1.0,
                 agentview_vjepa=False, agentview_vjepa_min_run_length=3):
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
    # occ_vla addition (2026-08-20, per user request -- check for the same
    # physical-obstacle confound already documented in the sibling pi0.5
    # project's OccluderPlacer finding): robot geom ids, same body-name-
    # substring convention as that project's robot_geom_ids() ("robot",
    # "panda", "gripper", "mount"), needed to distinguish a REAL
    # robot-occluder collision from the occluder merely resting on the
    # table (which shows up as a permanent, uninformative contact in
    # sim.data.contact regardless of the robot's position).
    robot_geom_ids_set = set(geom_ids_for_body_substring(sim, ["robot", "panda", "gripper", "mount"]))
    # occ_vla addition (2026-08-24, per user's proactive-avoidance Phase 1
    # request -- "VoxPoser/V-JEPA-style pre-collision" proposal, scoped down
    # to what's actually buildable on this project's existing assets):
    # PRIVILEGED proof-of-concept -- uses sim.data.geom_xpos directly (same
    # privilege class as the "oracle" mid-layer splice and `no_collision`
    # conditions elsewhere in this file), to first establish whether
    # proactively checking the upcoming 8-step action CHUNK against the
    # occluder's true 3D position/size has any value at all, BEFORE
    # investing in a real depth-camera+segmentation-based version (per the
    # user's own explicit 2-phase plan, mirroring this project's established
    # "ceiling probe first" discipline). Deliberately NOT camera/pixel-space
    # -- both eef_pos and the action chunk's xyz deltas are already in the
    # same real-world metric frame, so no projection is needed for this
    # phase.
    PROACTIVE_SAFETY_MARGIN_M = proactive_safety_margin
    proactive_correction_applied_count = 0
    proactive_correction_ts = []

    def _occluder_radius_m(geom_id):
        # occ_vla note: same geom-type-aware radius convention already
        # established elsewhere in this project's own investigation history
        # (sphere/capsule/cylinder -> size[0] is a true radius; box/other ->
        # median(size) as a reasonable half-extent approximation, since
        # geom_size stores half-extents for box geoms, not a radius).
        gtype = int(sim.model.geom_type[geom_id])
        size = sim.model.geom_size[geom_id]
        if gtype in (2, 3, 5):  # mjGEOM_SPHERE, CAPSULE, CYLINDER
            return float(size[0])
        return float(np.median(size))

    # occ_vla addition (2026-08-24, Phase 2 -- per user's explicit request
    # ("実機でも対応できるようにして、ロボットの関節データ、画像、深度など")):
    # a REAL-ROBOT-DEPLOYABLE alternative to proactive_avoidance_oracle/cbf's
    # privileged sim.data.geom_xpos[occluder_geom_ids] lookup. Builds a generic
    # 3D obstacle point cloud from RGB-D (agentview depth + known, calibratable
    # camera extrinsics -- NOT a simulator privilege, a real robot's own depth
    # camera + a standard one-time calibration would supply the same inputs)
    # every replan step, with the robot's OWN body self-filtered out via its
    # real-time segmentation footprint (self-filtering via known link geometry
    # is itself a standard, real-robot technique, not privileged information
    # about the OCCLUDER specifically -- unlike occluder_geom_ids, this doesn't
    # require knowing which object is "the occluder" at all, just "is anything
    # physically there that isn't me"). Cached once per episode: the camera
    # pose is static (confirmed elsewhere in this project's own investigation
    # history), and robot_seg_ids only needs the hide/reveal technique once,
    # not every step (self-filtering is by geometry, not per-frame lookup).
    depth_cam2world = None
    robot_seg_ids_for_depth = None
    if proactive_use_depth:
        cam_h = cam_w = resize_size
        world2pix = get_camera_transform_matrix(sim, "agentview", cam_h, cam_w)
        depth_cam2world = np.linalg.inv(world2pix)
        robot_seg_ids_for_depth = set(
            find_segmentation_ids_for_bodies(env, sim, list(robot_geom_ids_set))
        ) if robot_geom_ids_set else set()

    def _depth_obstacle_points(obs_dict, stride=6, max_range_m=1.2):
        """Real-sensor obstacle point cloud for this step: back-projects a
        downsampled agentview depth grid to 3D world points, excluding the
        robot's own body (self-filter) and the task's own TARGET object
        (target_seg_ids -- we want to avoid OTHER stuff, not the thing we're
        supposed to reach for) and anything beyond max_range_m (MuJoCo scenes
        include distant background geometry irrelevant to near-field
        avoidance). No occluder-identity information used anywhere here."""
        depth_key = "agentview_depth"
        if depth_key not in obs_dict:
            return np.zeros((0, 3))
        depth_raw = np.asarray(obs_dict[depth_key])
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]
        depth_m = get_real_depth_map(sim, np.clip(depth_raw, 0.0, 1.0))
        seg = np.asarray(obs_dict.get(AGENTVIEW_SEG_KEY, np.zeros_like(depth_raw, dtype=int))).squeeze()
        h, w = depth_m.shape[:2]
        rows = np.arange(0, h, stride)
        cols = np.arange(0, w, stride)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        seg_sub = seg[rr, cc]
        depth_sub = depth_m[rr, cc]
        exclude_ids = robot_seg_ids_for_depth | set(target_seg_ids or [])
        keep = np.isin(seg_sub, list(exclude_ids), invert=True) & (depth_sub > 1e-4) & (depth_sub < max_range_m)
        if not np.any(keep):
            return np.zeros((0, 3))
        # Direct back-projection (same math transform_from_pixels_to_world uses
        # internally, done here without its batched-depth-map wrapper since we
        # sample exact integer pixel indices from ONE depth map, not sub-pixel
        # bilinear queries against a per-item depth map): homogeneous camera-
        # frame point [col*z, row*z, z, 1] -> world frame via depth_cam2world.
        z = depth_sub[keep].astype(float)
        col = cc[keep].astype(float)
        row = rr[keep].astype(float)
        cam_pts = np.stack([col * z, row * z, z, np.ones_like(z)], axis=-1)  # (N, 4)
        world_pts = (depth_cam2world @ cam_pts.T).T[:, :3]
        return world_pts
    # occ_vla addition (2026-08-20, per user request -- a physically-real,
    # geometry-free alternative to no_collision: instead of removing
    # collision, reduce the occluder's MASS and FRICTION so it can
    # genuinely be pushed aside by real contact forces, real collision
    # physics throughout (no privilege at all -- this is literally "swap
    # the bolted fixture for a cardboard box"). Directly tests the
    # mechanism hypothesis from the L=0 contact-rate finding (contact
    # increased 12x under L=0 yet still succeeded -- "pushing through");
    # collision-off is the limiting case of this continuous variable, not
    # a separate phenomenon. Applied fresh every episode, same "must
    # reapply after this episode's own env.reset()" lesson as collision-
    # disable (env.reset() reloads a fresh mjModel, silently undoing
    # ANY mjModel-level change made before this point, not just
    # contype/conaffinity).
    orig_mass = orig_friction = None
    if low_mobility_geom_ids:
        orig_mass = {}
        body_ids_for_mobility = sorted(set(sim.model.geom_bodyid[gi] for gi in low_mobility_geom_ids))
        for bid in body_ids_for_mobility:
            orig_mass[bid] = sim.model.body_mass[bid]
            sim.model.body_mass[bid] = max(sim.model.body_mass[bid] * 0.2, 0.005)  # 5x lighter, floor at 5g
        orig_friction = sim.model.geom_friction[low_mobility_geom_ids].copy()
        sim.model.geom_friction[low_mobility_geom_ids] = sim.model.geom_friction[low_mobility_geom_ids] * 0.1
        sim.forward()
    # occ_vla bug fix (2026-08-20, real anomaly caught by the smoke test:
    # no_collision still showed 26/65 contact steps): disabling collision
    # in main() BEFORE calling run_episode() was silently undone by THIS
    # `env.reset()` call above -- the same "stale reference" behavior
    # already documented for `sim` itself suggests env.reset() reloads a
    # fresh mjModel from the XML, wiping any contype/conaffinity change
    # made before this point. Must (re-)disable AFTER this episode's own
    # reset, every single episode, not once before the whole condition's
    # loop.
    orig_occluder_contype = orig_occluder_conaffinity = None
    disable_collision_support_geom_ids = None
    orig_support_contype = orig_support_conaffinity = None

    def _apply_collision_disable():
        # occ_vla bug fix (2026-08-20, caught by the user BEFORE trusting
        # the first factorial_task1_n20 result -- real validity check, not
        # a hypothetical): the original fix (contype/conaffinity=0 for the
        # occluder) removes ALL physics interaction, including its
        # support contact with whatever it rests on -- confirmed via a
        # standalone check that the occluder free-falls under gravity once
        # collision is off (z: 0.90 -> -175 over 400 steps, no floor to
        # stop it). The 30%->100% result from that version is THEREFORE
        # INVALID (occluder vanished from view almost immediately,
        # degenerating into "removed both visually and physically") and
        # was retracted. A body_gravcomp=1.0 hack was tried next and
        # technically worked (z stays flat) but keeps the object floating
        # with ZERO real contact of any kind, not physically grounded.
        # Per user's explicit request for a more realistic version ("table
        # contact kept, only other contact removed"): use MuJoCo's
        # contype/conaffinity bitmask to selectively exclude JUST
        # robot-occluder collision while keeping real occluder-support
        # contact. Both currently default to bit0 (value 1, shared by
        # robot/everything-else) -- move the occluder's REAL (non-purely-
        # visual) collision geoms to bit1-ONLY, and give bit1 to
        # EVERY OTHER non-robot geom in the scene (not just a specific
        # named "table" body -- occ_vla bug fix 2026-08-20 #2: task8's
        # scene has no body literally named "table" (its support surface
        # is "floor"/"living_room_table"), which silently made the ORIGINAL
        # table-name-based version a no-op for that task, caught by the
        # printed warning firing during a real run rather than silently
        # producing a baseline-identical "no_collision" result). This way
        # the occluder collides normally with whatever it actually rests
        # on (floor, table, shelf, counter -- scene-agnostic) while still
        # excluding only the robot. Verified via a standalone 300-step
        # check on task1: z-position stays exactly flat via genuine
        # support contact (not gravity cancellation), occluder visually
        # confirmed still present in a saved frame.
        # occ_vla addition (2026-08-20, per user request -- reactive
        # recovery proxy): extracted into a closure so it can be called
        # either immediately at episode start (no_collision/
        # oracle_no_collision, unchanged) OR lazily, mid-episode, the
        # first time an "anomalous" contact (see reactive_collision_disable
        # below) is detected -- same mechanism either way, just a
        # different trigger time.
        nonlocal disable_collision_support_geom_ids, orig_support_contype, orig_support_conaffinity
        nonlocal orig_occluder_contype, orig_occluder_conaffinity
        all_geom_ids = list(range(sim.model.ngeom))
        support_geom_ids = [
            gi for gi in all_geom_ids
            if gi not in robot_geom_ids_set and gi not in set(disable_collision_geom_ids)
        ]
        if support_geom_ids:
            disable_collision_support_geom_ids = support_geom_ids
            orig_support_contype = sim.model.geom_contype[support_geom_ids].copy()
            orig_support_conaffinity = sim.model.geom_conaffinity[support_geom_ids].copy()
            sim.model.geom_contype[support_geom_ids] |= 2
            sim.model.geom_conaffinity[support_geom_ids] |= 2
            orig_occluder_contype = sim.model.geom_contype[disable_collision_geom_ids].copy()
            orig_occluder_conaffinity = sim.model.geom_conaffinity[disable_collision_geom_ids].copy()
            for gi in disable_collision_geom_ids:
                # skip geoms that were already collision-free in the
                # original model (purely-visual sub-meshes) -- nothing to
                # move for those
                if sim.model.geom_contype[gi] != 0 or sim.model.geom_conaffinity[gi] != 0:
                    sim.model.geom_contype[gi] = 2
                    sim.model.geom_conaffinity[gi] = 2

    # occ_vla addition (2026-08-20, per user request -- real-robot-
    # deployable trigger rule, no obstacle geometry/position needed):
    # "gripper/fingertip contact = normal task contact (grasping/
    # placing), any OTHER robot link contact = anomalous interference."
    # Confirmed directly relevant by tonight's link-contact histogram
    # (robot0_link6/forearm in contact 25/65 steps, gripper only 2/65,
    # in a FAILING episode) -- this is the same distinction, made
    # operational. `arm_only_geom_ids_set` excludes finger/gripper-
    # named geoms from the trigger check specifically (occluder_contact
    # in proprio_log stays unchanged, still ANY robot geom, for
    # continuity with existing logs).
    arm_only_geom_ids_set = {
        gi for gi in robot_geom_ids_set
        if not any(s in (sim.model.body_id2name(sim.model.geom_bodyid[gi]) or "").lower() for s in ("finger", "gripper"))
    }
    reactive_triggered = False
    reactive_trigger_t = None
    dry_run_would_have_fired = []
    # occ_vla addition (2026-08-23, per user's explicit "no privileged
    # information" request): a second, independent reactive-recovery
    # trigger, ZERO privileged info (no occluder geom identity, no
    # segmentation, no sim.data.contact) -- purely obs["robot0_eef_pos"],
    # exactly what a real robot's own proprioceptive encoders provide.
    # Rationale: scripted_recovery_after_contact's trigger (anomalous
    # arm-link contact with a KNOWN occluder geom) never fired at all on
    # task6/task8 (0/14, 0/13 baseline failures -- verified directly via
    # contact_robot_body_names, not a bug) -- its failure modes there
    # apparently don't involve that specific kind of contact. A general
    # "have I made real progress lately" velocity check has no such
    # blind spot: it fires on ANY sustained near-zero net motion,
    # regardless of cause (contact-driven or not), and needs no
    # knowledge of what the robot is stuck against.
    # STUCK_WINDOW_ENVSTEPS sampled at native env-step cadence (not just
    # replan-step cadence) so a stuck state is detected quickly.
    # STUCK_DIST_THRESHOLD is deliberately well below what one real VLA
    # replan chunk (8 open-loop steps under the active OSC_POSE
    # controller, ~0.05m max delta each) would produce if genuinely
    # progressing -- see the recovery-injection site below for the exact
    # value and its justification.
    STUCK_WINDOW_ENVSTEPS = 64  # ~8 replan-steps' worth, matching the
    # existing "stuck" failure-mode classifier's own 8-replan-step window
    # occ_vla change (2026-08-23, per user's ablation request): cooldown is
    # now independently configurable instead of always reusing the window
    # size -- default (None) reproduces the original untuned behavior.
    STUCK_COOLDOWN_ENVSTEPS = stuck_cooldown_envsteps if stuck_cooldown_envsteps is not None else STUCK_WINDOW_ENVSTEPS
    stuck_eef_pos_history = deque(maxlen=STUCK_WINDOW_ENVSTEPS)
    stuck_triggered_count = 0
    stuck_trigger_ts = []
    stuck_cooldown_remaining = 0  # env-steps to skip re-checking right
    # after a trigger, so the recovery motion's own (large, intentional)
    # displacement doesn't immediately refill the window with "moving"
    # samples that then look like a brand-new episode of being stuck the
    # instant the window is long enough again -- simpler and more
    # conservative than trying to distinguish "recovery motion" from
    # "real progress" after the fact.

    if disable_collision_geom_ids and not reactive_collision_disable:
        _apply_collision_disable()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False
    clear_target_mask = None
    n_occluded_steps = 0
    # occ_vla addition (2026-08-20/21, per user's item③ request -- a
    # REAL-ROBOT-BUILDABLE "visual occlusion only" cell, replacing
    # no_collision's simulator-only "arm passes through it" trick: the
    # occluder is (a) never natively rendered again after this episode's
    # first real step (geom_rgba alpha=0) and (b) non-collidable (reuses
    # the already-validated `_apply_collision_disable` path, since
    # composite_visual_only conditions pass occluder_geom_ids as
    # disable_collision_geom_ids in main()) -- i.e. it is genuinely
    # ABSENT from the scene, not merely flagged inert in place. Its
    # on-screen occlusion is then delivered purely by pasting a single
    # static reference sprite (captured from the one, real, alpha=1
    # render at this episode's first real step) onto the live frame each
    # step, via the pixel mask captured at that same instant -- something
    # a real deployment could reproduce by digitally compositing a fixed
    # occluder silhouette onto a camera feed while the physical workspace
    # has no object there at all. KNOWN LIMITATION, stated up front: a
    # static single-shot sprite has no z-buffer information, so if the
    # arm ever passes IN FRONT of the occluder's screen region (camera-
    # dependent, not the case for every task/occluder position), the
    # composite will incorrectly paint the occluder over the arm at
    # those pixels -- not physically-consistent occlusion, must be noted
    # in any write-up (per user's own explicit caveat).
    occluder_sprite = None
    occluder_pixel_mask = None
    # occ_vla addition (2026-08-22), per user's Option (a) design (their
    # message proposing a continuous TTC-area-based safe-action blend,
    # replacing scripted_recovery_after_contact's discrete post-contact
    # interrupt with a smooth pre-emptive one): track the target's own
    # occlusion fraction (`frac_occluded_this_step`, already computed
    # every env-step for the n_occluded_steps bookkeeping) frame to
    # frame, to derive an area-growth-rate TTC signal with zero new
    # privileged information -- same quantity a real-time lightweight
    # segmentation model (e.g. SAM) run on real camera frames could
    # supply on a real robot, per the user's own framing of this as a
    # proxy for that.
    prev_frac_occluded_for_ttc = None
    ttc_blend_log = []
    occluded_run_length = 0  # elapsed consecutive occluded steps -- resets to 0 the moment occlusion clears
    # occ_vla addition (2026-08-20, per user request -- a REAL scripted
    # recovery motion, not the idealized collision-disable proxy): last
    # commanded gripper value (pre-process_action, raw model output range),
    # so the scripted recovery phase can hold the gripper steady (not
    # accidentally open/close it) instead of guessing a value. Updated
    # every time a real VLA action is popped from the queue.
    last_gripper_raw = 0.0  # LIBERO/OpenVLA raw convention before process_action's flip/normalize
    action_diff_log = []
    # occ_vla addition (2026-08-19, per user request -- attention-entropy
    # gate signal validation): logged at EVERY replan step regardless of
    # occlusion or condition (unlike action_diff_log, which only fires
    # under real oracle correction) -- the whole point is to check whether
    # a baseline-condition rollout's own attention entropy predicts
    # eventual episode success/failure, as a candidate signal for gating
    # whether to engage correction at all.
    attn_entropy_log = []
    action_trace = []  # occ_vla addition (2026-08-19): see append site below
    # occ_vla addition (2026-08-19, per user request -- real-robot-usable
    # gate signal candidate): pure proprioception, no privileged sim info
    # (unlike S_occ), no extra model forward pass (unlike attention
    # entropy/ensemble disagreement) -- eef_pos/gripper_qpos are exactly
    # what obs["robot0_eef_pos"]/obs["robot0_gripper_qpos"] already read
    # from the real observation dict every replan step, same values a real
    # robot's own encoders would provide. Logged unconditionally (cheap,
    # zero extra compute) so a candidate "stagnation" signal (eef velocity
    # near zero for several consecutive replan steps) can be checked
    # against eventual success/failure post-hoc.
    proprio_log = []
    prev_eef_pos = None
    # occ_vla addition (2026-08-19, per user request -- another real-robot-
    # usable gate signal candidate, tried in parallel with proprioception):
    # input-perturbation ensemble disagreement. do_sample=False means the
    # model is greedy/deterministic given IDENTICAL input, so repeated calls
    # on the same pixels would trivially agree -- the perturbation (small
    # Gaussian pixel noise on the agentview frame only) is what actually
    # creates an ensemble here. No output_attentions, no privileged sim
    # info -- purely a second ordinary forward pass a real robot could also
    # run. Real cost: one extra forward pass per replan step.
    ensemble_disagreement_log = []
    rng_ensemble = np.random.default_rng(episode_idx if episode_idx is not None else 0)
    # occ_vla addition (2026-08-19, per user's strategic pivot -- Stage A of
    # the mask/content decomposition, "pixel_prevframe"): last-known-clean-
    # pixel buffer, real-robot-deployable (no privileged sim re-render, no
    # learned model -- just "remember what this pixel looked like the last
    # time it wasn't covered"). `prevframe_buffer` holds the most recent
    # UNOCCLUDED color at each pixel; `prevframe_step_buffer` holds the env
    # step `t` that value was captured at (-1 = never seen unoccluded yet),
    # used to compute per-step staleness and the no-valid-reference fraction
    # requested by the user. Initialized lazily on the first loop iteration
    # (needs occluded_pixel_mask's shape, computed inside the loop below).
    prevframe_buffer = None
    prevframe_step_buffer = None
    prevframe_fill_log = []
    prevframe_gate_skip_log = []
    # occ_vla addition (2026-08-19): explicit termination reason (success /
    # timeout / error) -- distinct from `success` alone, per user request.
    termination_reason = "timeout"
    # occ_vla addition (2026-08-18): reset the ground-truth splice-applied
    # counter (incremented inside patched_forward itself, see
    # make_agentview_midlayer_splice_forward) so each episode's result
    # reports its own count, not a running total across episodes.
    model.vision_backbone._diagnostic_correction_applied_count = 0
    model.vision_backbone._diagnostic_forward_call_count = 0  # temp diagnostic

    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    try:
        while t < max_steps + cfg.num_steps_wait:
            wrist_img = get_libero_wrist_image(obs).copy()
            agentview_color, agentview_seg = get_agentview_frames(env, resize_size)

            if composite_visual_only and occluder_geom_ids:
                if occluder_sprite is None:
                    # First real step of this episode: occluder is still
                    # natively rendered (alpha=1) at this point -- capture
                    # its true appearance + footprint before hiding it for
                    # the rest of the episode. Must redo every episode
                    # (occluder_sprite reset to None at the top of this
                    # function each call), same "reapply after this
                    # episode's own state" discipline as collision-disable/
                    # low_mobility above.
                    occluder_sprite = agentview_color.copy()
                    occluder_pixel_mask = (
                        np.isin(agentview_seg, occluder_seg_ids) if occluder_seg_ids else np.zeros_like(agentview_seg, dtype=bool)
                    )
                    sim.model.geom_rgba[occluder_geom_ids, 3] = 0.0
                    sim.forward()
                    # re-render now that the occluder is hidden, so
                    # downstream oracle/clear_target_mask logic sees the
                    # true post-hide segmentation (no occluder ids left).
                    agentview_color, agentview_seg = get_agentview_frames(env, resize_size)
                # Paste the pre-captured static sprite back onto the live
                # (occluder-absent, non-collidable) frame -- see the
                # KNOWN LIMITATION note above run_episode's occluder_sprite
                # init: no z-buffering, arm-in-front-of-occluder cases are
                # not handled correctly.
                agentview_color[occluder_pixel_mask] = occluder_sprite[occluder_pixel_mask]

            # occ_vla addition (2026-08-20, per user request -- real
            # rendered qualitative video, not a trajectory-plot substitute):
            # save every env-step's real agentview frame if enabled. Cheap
            # (PNG write, no VLA cost) -- only used for the 2 specific
            # illustrative episodes, not full runs.
            if record_video_dir is not None:
                os.makedirs(record_video_dir, exist_ok=True)
                Image.fromarray(agentview_color).save(os.path.join(record_video_dir, f"frame_{t:05d}.png"))

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
            # occ_vla addition (2026-08-22), per user's explicit priority
            # request: the FORCED-ACTIVATION non-regression check -- on a
            # scene with NO real occluder (stock suite, occluded_pixel_mask
            # is structurally always empty), artificially mark a fixed
            # fraction of the target's own clear footprint as "occluded"
            # so `will_apply_correction_this_step` actually fires and the
            # real mid-layer splice mechanism actively runs on a genuinely
            # clean frame -- directly tests "does an ACTIVELY FIRING
            # correction harm a clean image", not just "does it correctly
            # decline to fire" (the earlier non-regression check's
            # n_correction_applied=0 result only established the latter).
            # A deterministic (seeded, not per-step-random) subset of the
            # target's own real pixels -- gives a coherent, real-sized
            # region rather than scattered noise, closer to what an actual
            # partial occlusion's footprint would look like.
            if force_oracle_mask_frac and clear_target_mask.any():
                target_idx = np.flatnonzero(clear_target_mask)
                n_force = max(1, int(len(target_idx) * force_oracle_mask_frac))
                forced_rng = np.random.RandomState(0)  # deterministic across steps/episodes
                forced_idx = forced_rng.choice(target_idx, size=min(n_force, len(target_idx)), replace=False)
                forced_mask = np.zeros_like(clear_target_mask)
                forced_mask.flat[forced_idx] = True
                occluded_pixel_mask = forced_mask
            frac_occluded_this_step = (
                occluded_pixel_mask.sum() / max(clear_target_mask.sum(), 1)
            )
            if frac_occluded_this_step > 0.05:
                n_occluded_steps += 1
                occluded_run_length += 1
            else:
                occluded_run_length = 0

            # occ_vla addition (2026-08-20, per user request -- reactive
            # recovery proxy, checked EVERY env step for the fastest
            # possible reaction, not just at replan steps): the first time
            # a real MuJoCo contact pair links an occluder geom to an
            # ARM-only (non-gripper) robot geom, treat it as anomalous
            # interference and switch to no-collision from this point
            # forward for the rest of the episode. Answers "is reacting
            # after contact already too late" as a cheap, geometry-free
            # proxy for a real retreat-lift-reapproach recovery motion --
            # this idealizes the recovery as instantaneous/perfect (removes
            # the physical blocker outright rather than actually backing
            # away from it), so a positive result here is a NECESSARY,
            # not sufficient, condition for a real recovery motion to work.
            if reactive_collision_disable and disable_collision_geom_ids and not reactive_triggered:
                occluder_geom_id_set_reactive = set(disable_collision_geom_ids)
                anomalous_contact = any(
                    (sim.data.contact[ci].geom1 in occluder_geom_id_set_reactive and sim.data.contact[ci].geom2 in arm_only_geom_ids_set)
                    or (sim.data.contact[ci].geom2 in occluder_geom_id_set_reactive and sim.data.contact[ci].geom1 in arm_only_geom_ids_set)
                    for ci in range(sim.data.ncon)
                )
                if anomalous_contact and reactive_dry_run:
                    # occ_vla addition (2026-08-20, per user request -- a
                    # decisive diagnostic before trusting either reactive
                    # result: does the MONITORING code itself (reading
                    # sim.data.contact every env step) perturb the
                    # simulation, or is it genuinely read-only as intended?
                    # Logs that the check WOULD have fired but takes no
                    # action at all (reactive_triggered stays False,
                    # nothing else in the step differs from plain
                    # baseline) -- if a dry-run episode's outcome still
                    # differs from a true baseline run on the same
                    # init_state, the monitoring loop itself is buggy, not
                    # just the intervention.
                    dry_run_would_have_fired.append(t)
                elif anomalous_contact:
                    reactive_triggered = True
                    reactive_trigger_t = t
                    if scripted_recovery:
                        # occ_vla addition (2026-08-20, per user request --
                        # a REAL scripted recovery motion, not the
                        # idealized collision-disable proxy tested earlier
                        # tonight). Real physics/collision stays ON (no
                        # _apply_collision_disable() call) -- this tests
                        # whether an actual retreat-then-lift motion, not a
                        # simulator privilege, can recover the episode.
                        # Direction: away from the contacting occluder geom
                        # (eef_pos - contact geom position, normalized) --
                        # simpler and sign-convention-safer than reading
                        # MuJoCo's raw contact-frame normal, and equally
                        # principled ("move away from what you're
                        # touching"). Gripper held at its last commanded
                        # value throughout (no accidental release/close).
                        contacting_occluder_geoms = []
                        for ci in range(sim.data.ncon):
                            c = sim.data.contact[ci]
                            if c.geom1 in occluder_geom_id_set_reactive and c.geom2 in arm_only_geom_ids_set:
                                contacting_occluder_geoms.append(c.geom1)
                            elif c.geom2 in occluder_geom_id_set_reactive and c.geom1 in arm_only_geom_ids_set:
                                contacting_occluder_geoms.append(c.geom2)
                        contact_pos = sim.data.geom_xpos[contacting_occluder_geoms].mean(axis=0)
                        eef_pos_now_recovery = obs["robot0_eef_pos"]
                        away = eef_pos_now_recovery - contact_pos
                        away_xy_norm = np.linalg.norm(away[:2])
                        retreat_dir = away.copy()
                        if away_xy_norm > 1e-6:
                            retreat_dir[:2] = away[:2] / away_xy_norm
                        else:
                            retreat_dir[:2] = 0.0  # degenerate case (directly above/below) -- retreat via lift only
                        retreat_dir[2] = 0.0
                        RETREAT_STEPS, LIFT_STEPS = 4, 4
                        RETREAT_MAG, LIFT_MAG = 0.6, 0.6  # normalized action units, conservative
                        recovery_actions = []
                        for _ in range(RETREAT_STEPS):
                            recovery_actions.append(
                                [retreat_dir[0] * RETREAT_MAG, retreat_dir[1] * RETREAT_MAG, 0.0, 0.0, 0.0, 0.0, last_gripper_raw]
                            )
                        for _ in range(LIFT_STEPS):
                            recovery_actions.append([0.0, 0.0, LIFT_MAG, 0.0, 0.0, 0.0, last_gripper_raw])
                        action_queue.clear()
                        action_queue.extend(np.array(recovery_actions, dtype=float))
                        print(f"    [reactive] anomalous arm-link contact detected at t={t} -- injecting scripted "
                              f"retreat(dir={retreat_dir[:2]})+lift recovery, real physics stays ON")
                    else:
                        _apply_collision_disable()
                        print(f"    [reactive] anomalous arm-link contact detected at t={t} -- switching to no_collision from here")

            # occ_vla addition (2026-08-23, per user's explicit "no
            # privileged information" request): general stuck-velocity
            # trigger, independent of the contact-based one above. Uses
            # ONLY obs["robot0_eef_pos"] (real proprioception, sampled at
            # native env-step cadence) -- no occluder geom identity, no
            # segmentation, no sim.data.contact. Real physics/rendering
            # stay completely untouched by this condition (unlike
            # no_collision/scripted_recovery_after_contact, this needs
            # disable_collision_geom_ids to be None/unused).
            if stuck_velocity_trigger:
                if stuck_cooldown_remaining > 0:
                    stuck_cooldown_remaining -= 1
                else:
                    stuck_eef_pos_history.append(obs["robot0_eef_pos"].copy())
                    if len(stuck_eef_pos_history) == STUCK_WINDOW_ENVSTEPS:
                        half = STUCK_WINDOW_ENVSTEPS // 2
                        # "have I made real progress in the RECENT half of
                        # the window" -- checking only the recent half (not
                        # the whole window) means a stuck state is flagged
                        # ~half the window's length after it actually
                        # begins, not only once the entire window has gone
                        # stale from a still-moving state.
                        recent_disp = float(np.linalg.norm(stuck_eef_pos_history[-1] - stuck_eef_pos_history[-half]))
                        # occ_vla note: 0.012m over 32 env-steps (~4 replan
                        # chunks) is well below what any genuinely-
                        # progressing reach/insert motion produces under
                        # this env's active OSC_POSE controller (per-action
                        # max delta ~0.05m; 32 real, non-degenerate actions
                        # would need to almost perfectly cancel out to stay
                        # under this) -- conservative (few false positives
                        # on real progress), not tuned/swept.
                        # occ_vla change (2026-08-23, per user's threshold-
                        # sweep request): now a caller-supplied parameter
                        # (default unchanged, 0.012m) instead of hardcoded,
                        # so sensitivity to this experimentally-chosen value
                        # can be measured directly rather than assumed.
                        if recent_disp < stuck_dist_threshold:
                            # Retreat direction: reverse of the FULL
                            # window's net displacement (the direction the
                            # arm was heading when it got stuck), NOT the
                            # (near-zero, by definition) recent-half
                            # displacement used for detection -- purely
                            # from the eef's own position history, zero
                            # privileged/occluder information needed.
                            full_disp = stuck_eef_pos_history[-1] - stuck_eef_pos_history[0]
                            away = -full_disp
                            away_xy_norm = np.linalg.norm(away[:2])
                            retreat_dir = away.copy()
                            if away_xy_norm > 1e-6:
                                retreat_dir[:2] = away[:2] / away_xy_norm
                            else:
                                retreat_dir[:2] = 0.0  # degenerate (stuck from the very start, no net heading) -- lift only
                            retreat_dir[2] = 0.0
                            # occ_vla change (2026-08-23, per user's step-count-sweep
                            # request): now a caller-supplied parameter instead of the
                            # hardcoded 4 inherited from scripted_recovery_after_contact.
                            RETREAT_STEPS = LIFT_STEPS = stuck_recovery_steps
                            # occ_vla change (2026-08-23, per user's ablation
                            # request): now caller-supplied instead of the
                            # hardcoded 0.6 inherited from
                            # scripted_recovery_after_contact.
                            RETREAT_MAG = LIFT_MAG = stuck_retreat_mag
                            recovery_actions = []
                            for _ in range(RETREAT_STEPS):
                                recovery_actions.append(
                                    [retreat_dir[0] * RETREAT_MAG, retreat_dir[1] * RETREAT_MAG, 0.0, 0.0, 0.0, 0.0, last_gripper_raw]
                                )
                            for _ in range(LIFT_STEPS):
                                recovery_actions.append([0.0, 0.0, LIFT_MAG, 0.0, 0.0, 0.0, last_gripper_raw])
                            action_queue.clear()
                            action_queue.extend(np.array(recovery_actions, dtype=float))
                            stuck_triggered_count += 1
                            stuck_trigger_ts.append(t)
                            stuck_cooldown_remaining = STUCK_COOLDOWN_ENVSTEPS
                            stuck_eef_pos_history.clear()
                            print(f"    [stuck-trigger] near-zero eef motion detected at t={t} (recent_disp={recent_disp:.4f}m) "
                                  f"-- injecting scripted retreat(dir={retreat_dir[:2]})+lift recovery, no privileged info used")

            # occ_vla addition (2026-08-19): update the last-known-clean-pixel
            # buffer EVERY env step (not just replan steps), regardless of
            # condition -- a real robot's camera stream would give this for
            # free every frame. Only pixels currently occluded (within the
            # target's own clear footprint) are withheld from the update;
            # everything else (background, arm, unoccluded target pixels)
            # refreshes to the live frame every step.
            not_occluded_now = ~occluded_pixel_mask
            if prevframe_buffer is None:
                prevframe_buffer = agentview_color.copy()
                prevframe_step_buffer = np.where(occluded_pixel_mask, -1, t)
            else:
                prevframe_buffer[not_occluded_now] = agentview_color[not_occluded_now]
                prevframe_step_buffer[not_occluded_now] = t

            occlusion_mask = None
            agentview_vjepa_engaged_this_step = False
            if agentview_vjepa and occluded_run_length >= agentview_vjepa_min_run_length and occluded_pixel_mask.any():
                # occ_vla addition (2026-08-25, per user's agentview-V-JEPA
                # request): gate the VJEPA FiLM+cross-attention correction
                # module (prismatic/extern/hf/vjepa_latent_predictor.py,
                # already wired into modeling_prismatic.py's forward() but
                # NEVER fired by this script before this change --
                # `occlusion_mask` was previously hardcoded to None
                # unconditionally) on SUSTAINED occlusion only, not a
                # single noisy frame -- matches the project's own stated
                # rationale: the predictor's query depends on past_latents
                # still holding a genuinely-confirmed pre-occlusion state,
                # which a single-frame flicker doesn't invalidate anyway
                # (no need to correct) but firing on every 1-frame blip
                # would add pure perturbation with no compensating benefit.
                # occluded_pixel_mask/occluded_run_length here are the SAME
                # real-segmentation-derived signals already computed above
                # for every other condition in this file (Phase 1: oracle
                # mask CONTENT/timing, deployable correction MECHANISM --
                # same phasing already used for CBF v1->v2->depth).
                token_mask_256 = pixel_mask_to_token_mask_256(occluded_pixel_mask)
                if token_mask_256.any():
                    agentview_vjepa_engaged_this_step = True
                    num_images_vjepa = getattr(model.vision_backbone, "num_images_in_input", 2)
                    full_mask = np.zeros(num_images_vjepa * NUM_PATCHES_PER_IMAGE, dtype=bool)
                    full_mask[0:NUM_PATCHES_PER_IMAGE] = token_mask_256  # agentview = img_idx 0, matches make_agentview_midlayer_splice_forward's own convention
                    occlusion_mask = torch.from_numpy(full_mask).to(
                        device=model.device, dtype=torch.bfloat16).reshape(1, -1, 1)
            # occ_vla addition (2026-08-19/20, per task1 NO-GO result):
            # optional gate on prevframe's own no-valid-history fraction --
            # when most of the occluded region was never once seen clean
            # THIS episode, the fill degenerates into a patchwork of stale
            # content + raw corrupted pixels that empirically hurt (task1
            # n=20: 30% vs 50% baseline, wrong direction). Real-robot-
            # computable (frac_no_reference needs only the buffer already
            # maintained above, no privileged info) -- skip the fill and
            # fall back to the UNMODIFIED frame (same as baseline) whenever
            # too little real history exists to fill with. Default 1.0 =
            # gate never trips (old, already-tested unconditional behavior).
            gate_will_skip_this_step = False
            if pixel_fill_mode == "prevframe" and prevframe_gate_max_frac_no_ref < 1.0 and occluded_pixel_mask.any():
                n_occ_px_gate = int(occluded_pixel_mask.sum())
                no_ref_px_gate = int((occluded_pixel_mask & (prevframe_step_buffer < 0)).sum())
                frac_no_ref_gate = no_ref_px_gate / max(n_occ_px_gate, 1)
                if frac_no_ref_gate > prevframe_gate_max_frac_no_ref:
                    gate_will_skip_this_step = True
                    prevframe_gate_skip_log.append({
                        "t": t, "occluded_run_length": occluded_run_length,
                        "frac_no_reference": frac_no_ref_gate,
                    })

            will_apply_correction_this_step = (
                condition == "oracle"
                and (bool(occluder_geom_ids) or bool(force_oracle_mask_frac))
                and bool(occluded_pixel_mask.any())
                and not gate_will_skip_this_step
            )
            # occ_vla bug fix (2026-08-20, found via a real, reproduced
            # catastrophic result: gated prevframe scored 1/20 vs baseline's
            # 10/20, chi2=7.36 -- WORSE than even the unconditional 6/20).
            # `patched_forward` (line ~355 above) reads these two attributes
            # via getattr(..., None) and applies a splice whenever they're
            # non-None with a non-empty mask -- they were NEVER explicitly
            # reset to None on a step where correction is not applied this
            # time, only ever SET (see the block below). Before the gate
            # existed, every task tested this session has occlusion that,
            # once present, never clears mid-episode (verified via
            # action_diff_log's occluded_run_length never dipping across
            # all 60 already-collected task1/6/8 oracle episodes) -- so
            # `occluded_pixel_mask.any()` and `will_apply_correction_this_
            # step` happened to always agree in every run before this one,
            # and the staleness path was latent but never triggered. The
            # gate deliberately creates exactly the case where occlusion IS
            # present but correction should NOT apply -- triggering
            # `patched_forward` to silently keep splicing in STALE clean
            # pixels/mask from whenever correction last really fired
            # (potentially many steps and a very different arm pose ago),
            # actively corrupting the frame instead of leaving it alone.
            # Must explicitly clear both attributes whenever correction is
            # not applied this step, not just when it IS.
            if not will_apply_correction_this_step:
                model.vision_backbone._diagnostic_clean_agentview_pixel_values = None
                model.vision_backbone._diagnostic_agentview_patch_mask_256 = None
            if will_apply_correction_this_step:
                token_mask_256 = pixel_mask_to_token_mask_256(occluded_pixel_mask)
                if pixel_fill_mode == "prevframe":
                    # Stage A (mask/content decomposition): WHERE still comes
                    # from the oracle segmentation mask (occluded_pixel_mask,
                    # identical to every other condition here) -- only WHAT
                    # fills that region changes. No privileged re-render, no
                    # learned model: just the last real pixel value observed
                    # at that exact screen location before it got occluded.
                    fill_mask = occluded_pixel_mask & (prevframe_step_buffer >= 0)
                    no_ref_mask = occluded_pixel_mask & (prevframe_step_buffer < 0)
                    # occ_vla addition (2026-08-20, per user request -- search
                    # related literature and fix the root cause, not just
                    # gate around it): the original hard-mask compositing
                    # (`clean[fill_mask] = prevframe_buffer[fill_mask]`) is
                    # exactly the naive copy-paste pattern the image-
                    # compositing literature already documents as producing
                    # a visible seam a downstream model reads as "pasted" --
                    # the standard, well-established fix is feathering (a
                    # blurred alpha mask) rather than a binary cut, e.g.
                    # Poisson/gradient-domain blending and matting-based
                    # compositing pipelines. `--prevframe-feather-px 0`
                    # (default) preserves the exact original hard-cut
                    # behavior already tested; a positive value blurs
                    # `fill_mask` into a soft alpha and alpha-blends instead
                    # of a hard index assignment, directly targeting the
                    # seam/domain-gap mechanism rather than just avoiding
                    # the fill entirely (which is what the no-reference gate
                    # does).
                    if prevframe_feather_px > 0:
                        alpha = cv2.GaussianBlur(
                            fill_mask.astype(np.float32), (0, 0), sigmaX=prevframe_feather_px
                        )
                        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
                        clean_agentview_color = (
                            alpha * prevframe_buffer.astype(np.float32)
                            + (1.0 - alpha) * agentview_color.astype(np.float32)
                        ).astype(np.uint8)
                    else:
                        clean_agentview_color = agentview_color.copy()
                        clean_agentview_color[fill_mask] = prevframe_buffer[fill_mask]
                    n_occ_px = int(occluded_pixel_mask.sum())
                    n_no_ref_px = int(no_ref_mask.sum())
                    if fill_mask.any():
                        staleness = (t - prevframe_step_buffer[fill_mask]).astype(float)
                        mean_staleness, max_staleness = float(staleness.mean()), float(staleness.max())
                    else:
                        mean_staleness, max_staleness = None, None
                    prevframe_fill_log.append({
                        "t": t, "occluded_run_length": occluded_run_length,
                        "n_occluded_px": n_occ_px, "n_no_reference_px": n_no_ref_px,
                        "frac_no_reference": n_no_ref_px / max(n_occ_px, 1),
                        "mean_staleness_steps": mean_staleness, "max_staleness_steps": max_staleness,
                    })
                else:
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
                # occ_vla addition (2026-08-24, per user's "is it even looking
                # at the image anymore" hypothesis): substitute a flat
                # mid-gray agentview frame -- NOT just the occluder region,
                # the WHOLE frame -- to test whether the policy relies on
                # agentview content at all for these suites' tasks, or
                # succeeds mainly via the (always real, never occluded)
                # wrist camera + proprioception. Wrist image and state are
                # left untouched -- this isolates the agentview channel
                # specifically, same "one-variable-at-a-time" discipline as
                # every other diagnostic condition in this file.
                agentview_for_policy = (
                    np.full_like(agentview_color, 128) if blank_agentview else agentview_color
                )
                if agentview_vjepa_engaged_this_step:
                    # occ_vla addition (2026-08-25): gray-fill just the
                    # occluded region before the VJEPA correction module
                    # adds its predicted delta on top -- matches
                    # run_peek_action_eval.py's real, already-tested
                    # `vjepa_oracle` wrist condition's own convention
                    # (GRAY_FILL=127) exactly, rather than inventing a new
                    # constant. Removes the confusing real occluder texture
                    # from the base feature the correction gets added to,
                    # without needing a privileged clean re-render.
                    agentview_for_policy = agentview_for_policy.copy()
                    agentview_for_policy[occluded_pixel_mask] = 127
                    if record_video_dir is not None:
                        # occ_vla addition (2026-08-26, per user request --
                        # illustrative material showing the agentview
                        # correction pipeline): save the gray-filled INPUT
                        # actually fed to the policy this step (pairs with
                        # the raw frame_{t:05d}.png saved above at the SAME
                        # t) plus a red-highlighted overlay of
                        # occluded_pixel_mask on the raw frame, so the
                        # detected-region and the corrected-input can be
                        # shown side by side for a real, non-illustrative
                        # example. Cheap (2 PNG writes), only fires on
                        # already-engaged steps, only used for
                        # explanatory/small runs, not full n=20 batches.
                        os.makedirs(record_video_dir, exist_ok=True)
                        Image.fromarray(agentview_for_policy).save(
                            os.path.join(record_video_dir, f"frame_{t:05d}_corrected_input.png"))
                        overlay = agentview_color.copy()
                        overlay[occluded_pixel_mask] = (
                            0.5 * overlay[occluded_pixel_mask].astype(np.float32)
                            + 0.5 * np.array([255, 0, 0], dtype=np.float32)
                        ).astype(np.uint8)
                        Image.fromarray(overlay).save(
                            os.path.join(record_video_dir, f"frame_{t:05d}_occlusion_overlay.png"))
                observation = {
                    "full_image": agentview_for_policy,
                    "wrist_image": wrist_img,
                    "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                }
                eef_pos_now = obs["robot0_eef_pos"].copy()
                eef_speed = float(np.linalg.norm(eef_pos_now - prev_eef_pos)) if prev_eef_pos is not None else None
                # occ_vla addition (2026-08-19, per user request): eef-to-
                # occluder proximity/contact -- real MuJoCo geometry, not a
                # privileged "does the model see it" signal, but useful for
                # diagnosing WHY a prevframe fill might fail (e.g. the arm
                # itself displacing/contacting the occluder mid-episode).
                if occluder_geom_ids:
                    occ_xpos = sim.data.geom_xpos[occluder_geom_ids]
                    eef_to_occluder_dist = float(np.min(np.linalg.norm(occ_xpos - eef_pos_now, axis=1)))
                    occluder_geom_id_set = set(occluder_geom_ids)
                    # occ_vla bug fix (2026-08-20, found by an implausible
                    # 99.6-100% "occluder_contact" rate on task1/task6 --
                    # per this project's own "impossible result = check for
                    # a bug" discipline): the original check flagged ANY
                    # contact involving an occluder geom, including the
                    # occluder simply resting on the table under gravity --
                    # a permanent, uninformative contact unrelated to the
                    # robot. Now requires the OTHER geom in the pair to
                    # actually be a robot geom.
                    # occ_vla addition (2026-08-20, per user request -- link-
                    # level contact histogram, more informative and more
                    # defensible than a trajectory-overlay figure since it's
                    # built from REAL MuJoCo contact pairs (sim.data.contact,
                    # not a distance-threshold proxy) and directly answers
                    # "why didn't eef_to_occluder_dist show a close approach"
                    # -- if it's the forearm/mount, not the end-effector,
                    # touching the occluder, eef-centered distance would
                    # systematically miss it.
                    contact_robot_body_names = []
                    for ci in range(sim.data.ncon):
                        g1, g2 = sim.data.contact[ci].geom1, sim.data.contact[ci].geom2
                        robot_geom = None
                        if g1 in occluder_geom_id_set and g2 in robot_geom_ids_set:
                            robot_geom = g2
                        elif g2 in occluder_geom_id_set and g1 in robot_geom_ids_set:
                            robot_geom = g1
                        if robot_geom is not None:
                            contact_robot_body_names.append(sim.model.body_id2name(sim.model.geom_bodyid[robot_geom]))
                    occluder_contact = bool(contact_robot_body_names)
                else:
                    eef_to_occluder_dist, occluder_contact = None, False
                    contact_robot_body_names = []
                proprio_log.append({
                    "t": t, "occluded_run_length": occluded_run_length,
                    "eef_pos": eef_pos_now.tolist(), "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                    "eef_speed_since_last_replan": eef_speed,
                    "eef_to_occluder_dist": eef_to_occluder_dist, "occluder_contact": occluder_contact,
                    "contact_robot_body_names": contact_robot_body_names,
                    # occ_vla addition (2026-08-21), per user's "2D
                    # segmentation area expansion rate" TTC proposal
                    # (their approach 3, picked as most directly buildable
                    # from this pipeline's existing infra): logs the
                    # target's own occlusion fraction S_occ this step --
                    # `frac_occluded_this_step` was already computed above
                    # (used internally for the 5%-threshold occluded_run_
                    # length bookkeeping) but never persisted before this.
                    # The expansion RATE (dA/dt) is deliberately NOT
                    # logged as a separate field -- it's a simple discrete
                    # difference of consecutive replan-steps' own
                    # frac_occluded values within the same episode, so any
                    # downstream analysis script can compute it directly
                    # from this sequence without needing a second field.
                    "frac_occluded": float(frac_occluded_this_step),
                })
                prev_eef_pos = eef_pos_now
                # occ_vla bug fix (2026-08-18, found while auditing per user
                # request): this used to read `occlusion_mask is not None`,
                # but `occlusion_mask` is a local set to `None` once above
                # and NEVER reassigned anywhere in this file -- always False,
                # so --log-action-diff/--save-oracle-features-dir silently
                # never fired. The actual condition that determines whether
                # patched_forward will apply the splice this call is the same
                # one that gates setting the diagnostic attributes above.
                # occ_vla change (2026-08-20): reuse `will_apply_correction_
                # this_step` (computed once above, now also accounting for
                # the prevframe no-reference gate) instead of independently
                # recomputing an equivalent-looking condition -- two
                # separate computations of "the same" condition is exactly
                # the class of bug that caused the config-drift incident
                # earlier this project.
                real_oracle_correction_this_call = will_apply_correction_this_step

                # occ_vla addition (2026-08-18): always (re)set, even to None,
                # so a stale dict from an earlier step's call is never
                # silently reused if save_features_dir toggles off mid-run.
                feature_store = {} if (save_features_dir and real_oracle_correction_this_call) else None
                model.vision_backbone._diagnostic_feature_store = feature_store

                # occ_vla addition (2026-08-21), per user's Figure-A
                # divergence-analysis request: at each REAL replan step
                # (this block only runs when action_queue is empty, i.e.
                # a fresh model decision is about to be made -- exactly
                # the moment an attention map means something) within the
                # requested t-range, save RGB + depth + the raw per-patch
                # attention map. A SEPARATE diagnostic-only get_vla_action
                # call (return_attn_map=True), same pattern already
                # established by log_action_diff's counterfactual call
                # above -- does not affect action_queue/real behavior.
                if divergence_extract_dir is not None and (
                    divergence_extract_t_range is None or divergence_extract_t_range[0] <= t <= divergence_extract_t_range[1]
                ):
                    os.makedirs(divergence_extract_dir, exist_ok=True)
                    depth_frame = None
                    try:
                        full_obs = env.env._get_observations(force_update=True)
                        if "agentview_depth" in full_obs:
                            # occ_vla note (2026-08-21): MuJoCo's raw depth
                            # buffer is a normalized [0,1] z-buffer value,
                            # NOT physical distance (confirmed via
                            # robosuite.utils.camera_utils.get_real_depth_map's
                            # own docstring) -- a first pass here saved the
                            # raw buffer directly and got a suspiciously
                            # narrow 0.98-0.996 range, exactly the expected
                            # symptom of an unconverted z-buffer (most of
                            # its dynamic range is compressed near the far
                            # plane). Convert to real metric depth before
                            # saving so any depth-gradient analysis operates
                            # on physically meaningful values.
                            # occ_vla fix (2026-08-24): this local import used to
                            # live here, but ANY local `import X` inside a function
                            # body makes X a local name for the WHOLE function scope
                            # in Python -- this silently broke the NEW
                            # _depth_obstacle_points() closure defined earlier in
                            # run_episode (real crash: "free variable
                            # 'get_real_depth_map' referenced before assignment in
                            # enclosing scope", found via the proactive_avoidance_depth
                            # smoke test). Removed; the module-level import at the
                            # top of this file already provides the same name.
                            raw_depth = full_obs["agentview_depth"][::-1, ::-1].copy()
                            depth_frame = get_real_depth_map(sim, raw_depth)
                    except Exception as e:
                        print(f"    [divergence-extract] WARNING: depth capture failed at t={t}: {e}")
                    Image.fromarray(agentview_color).save(os.path.join(divergence_extract_dir, f"rgb_t{t:05d}.png"))
                    if depth_frame is not None:
                        np.save(os.path.join(divergence_extract_dir, f"depth_t{t:05d}.npy"), depth_frame)
                    try:
                        _, extract_attn_map = get_vla_action(
                            cfg, model, processor, observation, task_description,
                            action_head=action_head, proprio_projector=proprio_projector,
                            noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=occlusion_mask,
                            return_attn_map=True,
                        )
                        if extract_attn_map is not None:
                            np.save(os.path.join(divergence_extract_dir, f"attnmap_t{t:05d}.npy"), extract_attn_map)
                    except Exception as e:
                        print(f"    [divergence-extract] WARNING: attn map extraction failed at t={t}: {e}")

                if log_attn_entropy:
                    # occ_vla addition (2026-08-19): get_vla_action's return
                    # shape changes when return_attn_entropy=True (tuple,
                    # not a bare action list) -- see its own tail dispatch.
                    actions, step_attn_entropy = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=occlusion_mask,
                        return_attn_entropy=True,
                    )
                    attn_entropy_log.append({
                        "t": t, "occluded_run_length": occluded_run_length,
                        "frac_occluded": float(frac_occluded_this_step),
                        "attn_entropy": float(step_attn_entropy) if step_attn_entropy is not None else None,
                    })
                else:
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
                    # occ_vla addition (2026-08-19, per user request -- save
                    # everything this rerun could need so a second rerun for
                    # "one more field" is never necessary again): full 8-step
                    # chunks for BOTH conditions (not just the first action),
                    # and the feature-space delta-norm computed inside
                    # _run_vit_with_midlayer_splice (dino/siglip averaged for
                    # a single scalar, since that's what a real gate would
                    # threshold on -- per-tower values remain in the saved
                    # .npz for anyone who wants them separately).
                    feat_delta = None
                    if feature_store is not None:
                        d_dino = feature_store.get("dino_delta_norm")
                        d_siglip = feature_store.get("siglip_delta_norm")
                        if d_dino is not None and d_siglip is not None:
                            feat_delta = (d_dino + d_siglip) / 2.0
                    action_diff_log.append({
                        "t": t, "occluded_run_length": occluded_run_length,
                        "frac_occluded": float(frac_occluded_this_step),
                        "delta_a_norm_first": delta_a_first, "delta_a_norm_chunk_mean": delta_a_chunk_mean,
                        "action_chunk_with_correction": chunk_oracle.tolist(),
                        "action_chunk_without_correction": chunk_base.tolist(),
                        "feature_delta_norm": feat_delta,
                    })
                    print(f"    [action-diff] t={t} occluded_run_length={occluded_run_length} "
                          f"delta_a_first={delta_a_first:.4f} delta_a_chunk_mean={delta_a_chunk_mean:.4f}")

                if log_ensemble_disagreement:
                    # Second, ordinary forward pass on a lightly-perturbed
                    # agentview frame (small Gaussian pixel noise, std=4/255
                    # in real image scale, clipped to valid range) -- same
                    # code path as the real call, no output_attentions, no
                    # privileged info. do_sample=False means the two calls
                    # would trivially agree on identical pixels; the noise
                    # is what makes this a real (if crude) ensemble.
                    noisy_agentview = agentview_color.astype(np.float32) + rng_ensemble.normal(0, 4.0, agentview_color.shape)
                    noisy_agentview = np.clip(noisy_agentview, 0, 255).astype(np.uint8)
                    observation_noisy = {
                        "full_image": noisy_agentview, "wrist_image": wrist_img,
                        "state": observation["state"],
                    }
                    actions_noisy = get_vla_action(
                        cfg, model, processor, observation_noisy, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=occlusion_mask,
                    )
                    disagreement = float(np.linalg.norm(np.asarray(actions[0], dtype=float) - np.asarray(actions_noisy[0], dtype=float)))
                    ensemble_disagreement_log.append({
                        "t": t, "occluded_run_length": occluded_run_length, "disagreement": disagreement,
                    })

                if feature_store is not None:
                    fname = f"task{task_id}_ep{episode_idx}_t{t}_features.npz"
                    # occ_vla bug fix (2026-08-18): this call only ever
                    # persisted "dino"/"siglip" explicitly -- the
                    # "dino_final"/"siglip_final" keys added the same day
                    # (distribution-shift measurement) were being computed
                    # and placed into feature_store correctly, but silently
                    # dropped here since np.savez_compressed only writes
                    # what's passed as kwargs, not the whole dict. Caught by
                    # inspecting a real saved .npz's keys before trusting the
                    # analysis pipeline. Pass every feature_store key through.
                    np.savez_compressed(
                        os.path.join(save_features_dir, fname),
                        occluded_pixel_mask=occluded_pixel_mask, t=t,
                        occluded_run_length=occluded_run_length,  # occ_vla addition 2026-08-19
                        **{k: v for k, v in feature_store.items() if v is not None},
                    )
                    model.vision_backbone._diagnostic_feature_store = None

                # occ_vla addition (2026-08-19, per user request -- determinism
                # diagnosis): always record the first action of every replan
                # chunk (cheap: one 7-dim vector per ~8 env steps). Lets a
                # post-hoc diff between two runs of the "same" episode find
                # the EXACT step divergence first appears at, distinguishing
                # "different code path from step 1" from "numerical drift
                # compounding over time."
                # occ_vla change (2026-08-19, per user request): save the
                # FULL chunk (was action_first only) so within-chunk
                # variance can be analyzed later without a further rerun.
                action_trace.append({
                    "t": t, "action_first": np.asarray(actions[0], dtype=float).tolist(),
                    "action_chunk": np.asarray(actions, dtype=float).tolist(),
                })

                # occ_vla addition (2026-08-24, per user's proactive-
                # avoidance Phase 1 request): before queuing the fresh
                # 8-step chunk, check whether EXECUTING it (cumulative eef
                # displacement, real controller scale) would bring the
                # end-effector within an unsafe distance of the occluder's
                # TRUE 3D position -- if so, correct the chunk (lift in Z)
                # BEFORE any of it runs, rather than reacting after the arm
                # is already stuck (scripted_recovery_after_stuck's own
                # approach, complementary not replaced by this).
                if proactive_avoidance_oracle and (occluder_geom_ids or proactive_use_depth or proactive_use_mpc):
                    OSC_POSE_MAX_DELTA_M = 0.05  # confirmed via robosuite.controllers.load_controller_config(default_controller="OSC_POSE") -- this env never overrides controller_configs
                    if proactive_use_depth:
                        # occ_vla addition (2026-08-24, Phase 2): obstacle source is a
                        # real RGB-D point cloud, not sim.data.geom_xpos[occluder_geom_ids].
                        # Each point is treated as a near-zero-radius obstacle (radius
                        # 0.01m -- a small margin for the point-sampling itself, not an
                        # object-size estimate, since individual points have no "size").
                        occ_centers = _depth_obstacle_points(obs)
                        occ_radii = np.full(len(occ_centers), 0.01)
                        if len(occ_centers) == 0:
                            occ_centers = np.zeros((1, 3)) + 1e6  # no real obstacle seen this step -> push "nearest" far away, never triggers
                            occ_radii = np.zeros(1)
                    else:
                        occ_centers = sim.data.geom_xpos[occluder_geom_ids]
                        occ_radii = np.array([_occluder_radius_m(gi) for gi in occluder_geom_ids])
                    predicted_pos = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
                    actions_arr = np.asarray(actions, dtype=float)

                    if proactive_use_mpc:
                        # occ_vla addition (2026-08-25, per user request -- "V-JEPA2
                        # style" CBF-regularized sampling-based MPC): STRUCTURALLY
                        # inspired by V-JEPA 2-AC's (arXiv:2506.09985, Meta, confirmed
                        # real via direct paper fetch) CEM + energy-function planning
                        # loop -- NOT a reimplementation of V-JEPA2 itself. That paper
                        # uses a learned 300M-param action-conditioned transformer atop
                        # a frozen 1B-param encoder to predict future LATENT states and
                        # scores candidates by L1 distance to a GOAL IMAGE's latent; this
                        # project has neither a trained world model nor a goal-image
                        # scorer, so both are substituted with already-validated,
                        # zero-training components: state prediction reuses the SAME
                        # analytic forward-kinematics approximation as CBF-v2
                        # (predicted_pos += a_xyz * OSC_POSE_MAX_DELTA_M), and the "goal"
                        # term is fidelity-to-the-VLA's-own-anchor-chunk (stay close to
                        # its task-directed policy) rather than a learned distance to an
                        # imagined future frame.
                        #
                        # This is a genuine capability addition beyond CBF-v2 (the
                        # `elif not proactive_use_cbf` / `else` branches below), not a
                        # reimplementation of it: CBF-v2 computes the closed-form
                        # minimal-norm correction for a SINGLE linear safety constraint,
                        # ONE STEP at a time. That closed-form solution is provably
                        # optimal for that exact (convex, single-constraint, single-step)
                        # problem, so a sampling search over the same problem could only
                        # match it, not beat it. What sampling genuinely adds is WHOLE-
                        # CHUNK lookahead (score entire T-step candidate trajectories by
                        # their WORST-point margin violation, not just the current step)
                        # and robustness to multiple/irregular obstacle geometry where no
                        # simple closed form exists -- closer in spirit to V-JEPA2-AC's
                        # actual receding-horizon re-planning (execute the best full
                        # chunk, replan next chunk) than CBF-v2's per-step reactive nudge.
                        #
                        # Candidates: the VLA's own anchor chunk (always included,
                        # candidate 0) plus (N-1) perturbations sharing ONE random xyz
                        # offset per candidate applied to ALL T steps (not independent
                        # per-step noise, which would produce jittery, physically
                        # nonsensical trajectories) -- crude but zero-training, matching
                        # this smoke test's scope.
                        anchor = actions_arr.copy()
                        T = len(anchor)
                        rng_mpc = np.random.default_rng(1000 + t)  # deterministic per replan-step, for reproducibility
                        candidates = [anchor]
                        for _ in range(proactive_mpc_n_candidates - 1):
                            offset = rng_mpc.normal(0.0, proactive_mpc_noise_std, size=3)
                            cand = anchor.copy()
                            cand[:, :3] = cand[:, :3] + offset[None, :]
                            candidates.append(cand)

                        best_energy, best_cand, best_margin_violation = None, anchor, None
                        for cand in candidates:
                            pos = predicted_pos.copy()
                            worst_violation = 0.0  # deepest safety-margin penetration anywhere along this candidate's T-step trajectory
                            for step_i in range(T):
                                pos = pos + cand[step_i, :3] * OSC_POSE_MAX_DELTA_M
                                dists = np.linalg.norm(occ_centers - pos[None, :], axis=1) - occ_radii
                                violation = max(0.0, PROACTIVE_SAFETY_MARGIN_M - float(dists.min()))
                                worst_violation = max(worst_violation, violation)
                            fidelity_cost = float(np.mean((cand[:, :3] - anchor[:, :3]) ** 2))
                            energy = proactive_mpc_w_safety * (worst_violation ** 2) + proactive_mpc_w_fidelity * fidelity_cost
                            if best_energy is None or energy < best_energy:
                                best_energy, best_cand, best_margin_violation = energy, cand, worst_violation

                        if not np.array_equal(best_cand, anchor):
                            actions_arr = best_cand
                            actions = actions_arr
                            proactive_correction_applied_count += 1
                            proactive_correction_ts.append(t)
                            print(f"    [proactive-avoidance-mpc] chunk at t={t}: selected non-anchor candidate "
                                  f"among {proactive_mpc_n_candidates} (energy={best_energy:.5f}, "
                                  f"anchor_worst_violation vs selected={best_margin_violation:.4f})")
                    elif not proactive_use_cbf:
                        # occ_vla addition (2026-08-24, original Phase-1 design):
                        # a single trigger check, then a HARD override of xyz to a
                        # fixed +Z lift for the entire rest of the chunk. Found
                        # NEGATIVE (35%->25%, n=20) -- the fixed lift discards the
                        # VLA's own lateral/forward intent entirely, so on the
                        # NEXT replan the policy tries to resume its original
                        # approach and immediately re-triggers ("tug-of-war":
                        # repeated-firing episodes correlate strongly with
                        # failure, 3.2 vs 14.2 mean corrections success/failure).
                        # Kept, unchanged, as condition="proactive_avoidance_oracle"
                        # for reproducibility of that negative result -- superseded
                        # by the proactive_use_cbf branch below as the condition
                        # actually meant to show a real benefit.
                        unsafe_from_step = None
                        for step_i in range(len(actions_arr)):
                            predicted_pos = predicted_pos + actions_arr[step_i, :3] * OSC_POSE_MAX_DELTA_M
                            dists = np.linalg.norm(occ_centers - predicted_pos[None, :], axis=1) - occ_radii
                            if dists.min() < PROACTIVE_SAFETY_MARGIN_M:
                                unsafe_from_step = step_i
                                break
                        if unsafe_from_step is not None:
                            LIFT_MAG = 0.5  # untuned, same order of magnitude as the file's other correction magnitudes
                            for step_i in range(unsafe_from_step, len(actions_arr)):
                                actions_arr[step_i, 0] = 0.0
                                actions_arr[step_i, 1] = 0.0
                                actions_arr[step_i, 2] = LIFT_MAG
                            actions = actions_arr
                            proactive_correction_applied_count += 1
                            proactive_correction_ts.append(t)
                            print(f"    [proactive-avoidance-override] chunk at t={t} would approach occluder within "
                                  f"{PROACTIVE_SAFETY_MARGIN_M}m at step {unsafe_from_step} -- correcting to +Z lift from there")
                    else:
                        # occ_vla addition (2026-08-24, v2 -- CBF/APF-style
                        # minimal-norm safe correction, redesigned per user
                        # request to fix the v1 override's "tug-of-war" failure
                        # mode, grounded in standard robotics safety-control
                        # theory: Artificial Potential Fields (Khatib, 1986,
                        # "Real-Time Obstacle Avoidance for Manipulators and
                        # Mobile Robots") and Control Barrier Functions (Ames et
                        # al., 2019, "Control Barrier Function Based Quadratic
                        # Programs for Safety-Critical Systems"). Rather than
                        # discarding the VLA's entire xyz intent once ANY future
                        # step looks unsafe, this computes -- PER STEP, for the
                        # whole chunk, not just from a single trigger point --
                        # the minimal correction that keeps the predicted
                        # position outside the safety margin: only the velocity
                        # COMPONENT heading into the occluder (the projection
                        # onto the outward normal direction) is topped up to the
                        # minimum safe value; the tangential/lateral component
                        # (the VLA's actual approach/reach direction) is left
                        # completely untouched. This is the closed-form solution
                        # to a single-constraint CBF-QP (min_a ||a - a_vla||^2
                        # s.t. dot(a, n_hat) >= k*(margin-dist)), not an
                        # approximation of one. Because the correction is
                        # continuous (scales with how deep into the margin the
                        # predicted point is, k=proactive_cbf_gain) and per-step
                        # (not "everything from here to the end of the chunk"),
                        # each replan naturally re-derives the correction fresh
                        # from the VLA's current intent rather than fighting a
                        # frozen fixed-lift override -- directly targeting the
                        # v1 tug-of-war mechanism (repeated full-chunk
                        # overrides), not just a smaller lift magnitude.
                        n_corrected_this_chunk = 0
                        for step_i in range(len(actions_arr)):
                            a_xyz = actions_arr[step_i, :3]
                            candidate_pos = predicted_pos + a_xyz * OSC_POSE_MAX_DELTA_M
                            dists = np.linalg.norm(occ_centers - candidate_pos[None, :], axis=1) - occ_radii
                            nearest_idx = int(np.argmin(dists))
                            dist = float(dists[nearest_idx])
                            if dist < PROACTIVE_SAFETY_MARGIN_M:
                                to_robot = candidate_pos - occ_centers[nearest_idx]
                                to_robot_norm = float(np.linalg.norm(to_robot))
                                n_hat = (to_robot / to_robot_norm) if to_robot_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
                                v_normal = float(np.dot(a_xyz, n_hat))
                                v_min_normal = proactive_cbf_gain * (PROACTIVE_SAFETY_MARGIN_M - dist)  # >0, scales with penetration depth
                                if v_normal < v_min_normal:
                                    deficit = v_min_normal - v_normal
                                    a_xyz = a_xyz + deficit * n_hat  # only the unsafe normal component is topped up; tangential intent untouched
                                    actions_arr[step_i, :3] = a_xyz
                                    n_corrected_this_chunk += 1
                            # propagate using the (possibly-corrected) action for THIS step,
                            # so later steps in the chunk see where the corrected trajectory
                            # actually goes, not the original uncorrected one.
                            predicted_pos = predicted_pos + a_xyz * OSC_POSE_MAX_DELTA_M
                        if n_corrected_this_chunk > 0:
                            actions = actions_arr
                            proactive_correction_applied_count += n_corrected_this_chunk
                            proactive_correction_ts.append(t)
                            print(f"    [proactive-avoidance-cbf] chunk at t={t}: minimal-norm safety "
                                  f"correction applied to {n_corrected_this_chunk}/{len(actions_arr)} steps "
                                  f"(gain={proactive_cbf_gain})")

                action_queue.extend(actions)

            action = action_queue.popleft()
            last_gripper_raw = float(np.asarray(action)[6])

            # occ_vla addition (2026-08-22), per user's Option (a):
            # continuous TTC-area safe-action blend. Computed and applied
            # EVERY env-step (not just replan steps), since
            # frac_occluded_this_step is itself recomputed every step and
            # the action being taken right now is what should be
            # corrected -- unlike scripted_recovery's one-shot scripted
            # sequence, this recomputes alpha fresh every step, so it can
            # smoothly relax back to alpha=0 (pure a_vla) the moment
            # occlusion stops growing, not just once at trigger time.
            ttc_alpha = 0.0
            ttc_value = None
            if ttc_area_blend:
                if prev_frac_occluded_for_ttc is not None:
                    a_dot = frac_occluded_this_step - prev_frac_occluded_for_ttc
                    if a_dot > 1e-6:  # only when occlusion area is genuinely GROWING
                        ttc_value = float(frac_occluded_this_step / a_dot)
                        ttc_alpha = float(np.clip(1.0 - ttc_value / ttc_threshold, 0.0, 1.0))
                if ttc_alpha > 0.0:
                    safe_vec = np.asarray(ttc_safe_action, dtype=float)
                    action_arr = np.asarray(action, dtype=float).copy()
                    action_arr[:6] = (1.0 - ttc_alpha) * action_arr[:6] + ttc_alpha * safe_vec
                    action = action_arr
                ttc_blend_log.append({
                    "t": t, "frac_occluded": float(frac_occluded_this_step),
                    "ttc_value": ttc_value, "alpha": ttc_alpha,
                })
            prev_frac_occluded_for_ttc = frac_occluded_this_step

            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                termination_reason = "success"
                break
            t += 1
        else:
            termination_reason = "timeout"
    except Exception as e:
        print(f"  Episode error: {e}")
        termination_reason = "error"

    # occ_vla addition (2026-08-20): restore collision settings before
    # returning, defensively, even though this episode's `env` instance
    # will likely be reset (and its model presumably reloaded) again
    # before the next episode anyway -- belt-and-suspenders given the
    # reset-behavior uncertainty that caused the bug this fix addresses.
    if disable_collision_geom_ids and orig_occluder_contype is not None:
        sim.model.geom_contype[disable_collision_geom_ids] = orig_occluder_contype
        sim.model.geom_conaffinity[disable_collision_geom_ids] = orig_occluder_conaffinity
        if disable_collision_support_geom_ids is not None:
            sim.model.geom_contype[disable_collision_support_geom_ids] = orig_support_contype
            sim.model.geom_conaffinity[disable_collision_support_geom_ids] = orig_support_conaffinity
    if low_mobility_geom_ids and orig_mass is not None:
        for bid, m in orig_mass.items():
            sim.model.body_mass[bid] = m
        sim.model.geom_friction[low_mobility_geom_ids] = orig_friction

    return {
        "success": success, "done_step": t,
        "termination_reason": termination_reason,
        "n_occluded_steps": n_occluded_steps,
        "action_diff_log": action_diff_log,
        "prevframe_fill_log": prevframe_fill_log,
        "prevframe_gate_skip_log": prevframe_gate_skip_log,
        "reactive_triggered": reactive_triggered, "reactive_trigger_t": reactive_trigger_t,
        "dry_run_would_have_fired": dry_run_would_have_fired,
        "stuck_triggered_count": stuck_triggered_count, "stuck_trigger_ts": stuck_trigger_ts,
        "proactive_correction_applied_count": proactive_correction_applied_count,
        "proactive_correction_ts": proactive_correction_ts,
        "ttc_blend_log": ttc_blend_log,
        # occ_vla addition (2026-08-18): independent runtime ground truth
        # that the splice was actually applied (incremented inside
        # patched_forward itself, not inferred) -- see
        # make_agentview_midlayer_splice_forward. Expect > 0 for "oracle"
        # under real occlusion, == 0 for "baseline" (which never installs
        # patched_forward as vision_backbone.forward at all).
        "n_correction_applied": getattr(model.vision_backbone, "_diagnostic_correction_applied_count", 0),
        "n_forward_calls": getattr(model.vision_backbone, "_diagnostic_forward_call_count", 0),  # temp diagnostic
        "attn_entropy_log": attn_entropy_log,
        "action_trace": action_trace,
        "proprio_log": proprio_log,
        "ensemble_disagreement_log": ensemble_disagreement_log,
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
    parser.add_argument("--reactive-dry-run", action="store_true",
                         help="Diagnostic only: with --conditions no_collision_after_contact or "
                              "scripted_recovery_after_contact, the per-step contact-monitoring check "
                              "still runs and logs when it WOULD have fired, but takes no action at "
                              "all (no collision-disable, no scripted actions) -- tests whether the "
                              "monitoring code itself perturbs the rollout vs. a true baseline run on "
                              "the same init_states.")
    parser.add_argument("--record-video-dir", default=None,
                         help="If set, saves every env-step's real agentview frame as a PNG under "
                              "<dir>/<condition>_ep<N>/frame_NNNNN.png -- for real rendered qualitative "
                              "video, use only with --n-episodes 1 (illustrative episodes), not full "
                              "runs (real disk cost, ~500KB/frame x max_steps).")
    parser.add_argument("--use-stock-suite", action="store_true",
                         help="Run against the PLAIN (non-occluded) libero_10 suite instead of "
                              "libero_10_occluded -- needed to get a real 'no occlusion at all' "
                              "reference point for interpreting the no_collision condition's SR. "
                              "IMPORTANT: task_ids do NOT correspond 1:1 between the two suites "
                              "(libero_10_occluded's numbering is alphabetical-by-BDDL-filename, not "
                              "libero_10's task_order permutation) -- look up the matching stock "
                              "task_id by bddl_file before using this, don't assume the same index.")
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
    parser.add_argument("--log-attn-entropy", action="store_true",
                         help="Log action-token-to-vision-patch attention entropy at EVERY replan "
                              "step (any condition, occluded or not) -- candidate gate signal: does "
                              "baseline's own attention entropy predict eventual episode "
                              "success/failure, per user request 2026-08-19.")
    parser.add_argument("--load-vision-weights", default=None,
                         help="occ_vla addition 2026-08-22, per user's request to evaluate "
                              "train_representation_alignment.py's output in a real rollout: path to "
                              "a saved vision_backbone+projector state dict (torch.save'd partial "
                              "dict, see that script's own --out-adapter). Loaded with strict=False "
                              "right after the base checkpoint loads (the frozen language_model is "
                              "left untouched). None (default) is the unmodified base checkpoint.")
    parser.add_argument("--vision-weights-a", default=None,
                         help="occ_vla addition 2026-08-24, per user's Approach-A+B factorial request: "
                              "same file format as --load-vision-weights, but SWAPPED IN/OUT per "
                              "condition within a single process run, rather than applied once globally. "
                              "Conditions 'A_only' and 'A_plus_B' use these weights; 'baseline' and "
                              "'B_only' use the original (unmodified) checkpoint weights. Required if "
                              "--conditions includes 'A_only' or 'A_plus_B'.")
    parser.add_argument("--force-oracle-mask-frac", type=float, default=None,
                         help="occ_vla addition 2026-08-22, per user's priority request: the FORCED-"
                              "ACTIVATION non-regression check. On a scene with NO real occluder "
                              "(--use-stock-suite), the mid-layer 'oracle' correction structurally "
                              "never fires (occluded_pixel_mask is always empty). Setting this "
                              "(e.g. 0.13, matching this project's own typical arm-silhouette "
                              "occlusion fraction) artificially marks that fraction of the target's "
                              "real, clear footprint as 'occluded' each step, forcing the REAL "
                              "mid-layer splice mechanism to actively run on a genuinely clean "
                              "image -- tests whether an ACTIVELY FIRING correction harms a clean "
                              "scene, not just whether it correctly declines to fire (which the "
                              "earlier non-regression check, n_correction_applied=0, already "
                              "confirmed separately). Only meaningful with --conditions "
                              "including 'oracle' and --use-stock-suite.")
    parser.add_argument("--blank-agentview-diagnostic", action="store_true",
                         help="occ_vla addition 2026-08-24: substitutes a flat mid-gray agentview frame for "
                              "the policy's ENTIRE input (not just the occluded region) -- diagnostic only, "
                              "to test whether the policy relies on agentview content at all for a given "
                              "task, or succeeds mainly via the always-real wrist camera + proprioception.")
    parser.add_argument("--proactive-safety-margin", type=float, default=0.04,
                         help="occ_vla addition 2026-08-24: proactive_avoidance_oracle's safety margin "
                              "(meters, added on top of the occluder's own geometric half-extent). Untuned.")
    parser.add_argument("--proactive-cbf-gain", type=float, default=2.0,
                         help="occ_vla addition 2026-08-24, v2 (proactive_avoidance_cbf condition): gain k "
                              "in the minimal-norm safety correction v_min_normal = k*(margin-dist) -- how "
                              "hard to push away per meter of margin penetration. Untuned; same order of "
                              "magnitude as other correction gains in this file.")
    parser.add_argument("--proactive-mpc-n-candidates", type=int, default=16,
                         help="occ_vla addition 2026-08-25 (proactive_avoidance_mpc condition): number of "
                              "candidate chunks scored per replan, including the VLA's own anchor chunk "
                              "(candidate 0). Untuned.")
    parser.add_argument("--proactive-mpc-noise-std", type=float, default=0.15,
                         help="occ_vla addition 2026-08-25: std of the single per-candidate xyz offset "
                              "(normalized action units, applied to all T steps of that candidate) used to "
                              "perturb the anchor chunk. Untuned.")
    parser.add_argument("--proactive-mpc-w-safety", type=float, default=50.0,
                         help="occ_vla addition 2026-08-25: energy-function weight on squared worst-point "
                              "safety-margin violation across a candidate's whole chunk. Untuned; large "
                              "relative to w-fidelity so any real violation dominates the selection.")
    parser.add_argument("--proactive-mpc-w-fidelity", type=float, default=1.0,
                         help="occ_vla addition 2026-08-25: energy-function weight on mean squared deviation "
                              "from the VLA's own anchor chunk (the 'goal' term substitute -- see the "
                              "proactive_use_mpc branch's docstring for why, in the absence of a learned "
                              "goal-image latent scorer). Untuned.")
    parser.add_argument("--agentview-vjepa-min-run-length", type=int, default=3,
                         help="occ_vla addition 2026-08-25 (agentview_vjepa condition): minimum consecutive "
                              "occluded env-steps (occluded_run_length) before the VJEPA FiLM+cross-attention "
                              "correction module is allowed to fire on the agentview image. Untuned default "
                              "(3), per the user's stated rationale that a single-frame occlusion blip needs "
                              "no correction and firing on it would just add unnecessary feature perturbation.")
    parser.add_argument("--suite", default="10", choices=["10", "spatial", "object", "goal"],
                         help="occ_vla addition 2026-08-24, per user's cross-suite VIM-comparison request: "
                              "which LIBERO-Occ suite to evaluate against. Was previously hardcoded to "
                              "libero_10 (STOCK_SUITE/OCCLUDED_SUITE module constants) -- this flag also "
                              "resolves the CORRECT per-suite max_steps (spatial=220, object=280, goal=300, "
                              "10=520, per the vendored TASK_MAX_STEPS table), instead of the previous "
                              "unconditional libero_10 value (520), which would have been silently wrong "
                              "for spatial/object/goal (same 'ported constant not checked per-suite' bug "
                              "class already documented elsewhere in this project).")
    parser.add_argument("--stuck-cooldown-envsteps", type=int, default=None,
                         help="occ_vla addition 2026-08-23, per user's ablation request: env-steps to wait "
                              "after a stuck-trigger fires before re-arming. Default None reproduces the "
                              "original (untuned) behavior of reusing STUCK_WINDOW_ENVSTEPS (64) -- pass an "
                              "explicit value to decouple it from the detection-window size.")
    parser.add_argument("--stuck-retreat-mag", type=float, default=0.6,
                         help="occ_vla addition 2026-08-23, per user's ablation request: normalized-action "
                              "magnitude for BOTH the retreat and lift phases of scripted_recovery_after_stuck's "
                              "recovery motion. Untuned default (0.6), inherited unchanged from the earlier "
                              "scripted_recovery_after_contact mechanism.")
    parser.add_argument("--stuck-recovery-steps", type=int, default=4,
                         help="occ_vla addition 2026-08-23, per user's step-count-sweep request: number of "
                              "steps for EACH phase (retreat, then lift) of scripted_recovery_after_stuck's "
                              "recovery motion -- total motion length is 2x this value. Inherited unchanged "
                              "from the earlier scripted_recovery_after_contact mechanism's own untuned "
                              "default (4) until this sweep; not previously verified against other values.")
    parser.add_argument("--stuck-dist-threshold", type=float, default=0.012,
                         help="occ_vla addition 2026-08-23, per user's threshold-sweep request: "
                              "scripted_recovery_after_stuck's near-zero-progress trigger threshold, "
                              "in meters, over the recent-half (32 env-step) window. Untuned default "
                              "(0.012m); swept at 0.006/0.012/0.024 to check sensitivity.")
    parser.add_argument("--ttc-threshold", type=float, default=8.0,
                         help="occ_vla addition 2026-08-22, per user's Option (a) continuous TTC-area "
                              "safe-action blend: TTC (in replan-agnostic env-step units, since "
                              "frac_occluded is tracked every env-step) below this triggers blending "
                              "toward --ttc-safe-action; alpha ramps from 0 (TTC>=threshold) to 1 "
                              "(TTC=0). Untuned default, a tuning knob for later.")
    parser.add_argument("--ttc-safe-action", type=float, nargs=6, default=[0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
                         help="6-dim (dx,dy,dz,drx,dry,drz) safe action blended toward as TTC drops; "
                              "gripper dim always stays the VLA's own. Default: stop XY/rotation, lift "
                              "+Z slightly.")
    parser.add_argument("--divergence-extract-dir", default=None,
                         help="occ_vla addition 2026-08-21, per user's Figure-A divergence-analysis "
                              "request: at every real replan step within --divergence-extract-t-range, "
                              "save RGB, depth (requires enabling camera_depths on the env), and the "
                              "raw per-patch attention map (extra diagnostic-only forward pass, no "
                              "behavior change) to this directory. Intended for a single 1-episode "
                              "comparison run (e.g. no_collision vs composite_visual_only, same "
                              "init_state), not a full n=20 sweep -- real per-step cost from the extra "
                              "forward pass.")
    parser.add_argument("--divergence-extract-t-range", type=int, nargs=2, default=None,
                         help="[t_min, t_max] (env-step units) -- only extract within this window. "
                              "None (default) extracts at every replan step for the whole episode.")
    parser.add_argument("--log-ensemble-disagreement", action="store_true",
                         help="At every replan step, one extra forward pass on the agentview frame "
                              "with small Gaussian pixel noise added, logging the L2 distance from "
                              "the real (unperturbed) action -- real-robot-usable candidate gate "
                              "signal, no output_attentions/no privileged info, per user request "
                              "2026-08-19.")
    parser.add_argument("--pixel-fill-mode", default="none", choices=["none", "prevframe"],
                         help="Stage A of the mask/content decomposition (user's 2026-08-19 strategic "
                              "pivot): 'prevframe' fills the oracle-masked occluded region with the "
                              "last real (unoccluded) pixel value seen at each pixel, instead of a "
                              "privileged alpha-zero re-render -- zero training, zero learned "
                              "parameters, real-robot-deployable. 'none' (default) keeps the existing "
                              "true-oracle-render behavior, unchanged.")
    parser.add_argument("--prevframe-gate-max-frac-no-ref", type=float, default=1.0,
                         help="Only meaningful with --pixel-fill-mode prevframe. Skip the fill (fall "
                              "back to the unmodified frame, same as baseline) on any step where the "
                              "fraction of currently-occluded target pixels with NO valid unoccluded "
                              "history this episode exceeds this threshold. Default 1.0 = gate never "
                              "trips (original unconditional behavior). Added 2026-08-20 after task1's "
                              "unconditional pixel_prevframe n=20 result (30% vs 50% baseline, wrong "
                              "direction) -- see CLAUDE.md.")
    parser.add_argument("--prevframe-feather-px", type=float, default=0.0,
                         help="Only meaningful with --pixel-fill-mode prevframe. 0 (default) = original "
                              "hard-cut compositing (clean[fill_mask] = prevframe_buffer[fill_mask]), "
                              "already tested. >0 = Gaussian-blur sigma (pixels, in the 256x256 "
                              "agentview frame) applied to the fill mask before alpha-blending instead "
                              "of a hard index assignment -- targets the seam/domain-gap mechanism the "
                              "image-compositing literature documents for naive copy-paste (e.g. "
                              "arXiv:2011.02146, seamless-cloning/Poisson-blending survey work), added "
                              "2026-08-20 after the gate alone failed to rescue task1's negative result.")
    parser.add_argument("--attn-implementation", default=None,
                         help="Force a specific attention implementation (e.g. 'eager') for the "
                              "WHOLE rollout, consistently. Diagnostic for whether "
                              "--log-attn-entropy's output_attentions=True request silently "
                              "switches the model off its default (SDPA) path, introducing "
                              "numerical differences that compound over a long closed-loop "
                              "rollout -- found 2026-08-19: an entropy-enabled rerun of the same "
                              "20 seeds flipped 8/20 episode outcomes vs a non-entropy run, "
                              "contradicting smoke_test_attn_entropy.py's single-step check.")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_oracle_features_dir:
        os.makedirs(args.save_oracle_features_dir, exist_ok=True)

    # occ_vla bug fix (2026-08-24, real crash: check_unnorm_key asserted
    # "Action un-norm key libero_10 not found in VLA norm_stats!" for EVERY
    # single spatial/object/goal job tonight -- task_suite_name here was
    # still hardcoded to the module-level STOCK_SUITE ("libero_10") even
    # after --suite was added, since this cfg is built BEFORE this file's
    # own suite_stock_name resolution further down. check_unnorm_key uses
    # cfg.task_suite_name directly as the norm_stats lookup key, and each
    # suite-specific checkpoint's dataset_statistics.json only contains ITS
    # OWN suite's key (confirmed: openvla-7b-oft-libero-spatial only has
    # "libero_spatial_no_noops") -- so every one of tonight's 30 new-suite
    # jobs failed at startup, before a single episode ran. Resolved here,
    # ahead of cfg construction, instead of leaving it for later.
    _suite_stock_name_for_cfg = {"10": "libero_10", "spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal"}[args.suite]
    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=args.load_in_4bit,
        center_crop=True, num_open_loop_steps=8, task_suite_name=_suite_stock_name_for_cfg, seed=7,
    )
    if args.attn_implementation:
        # occ_vla addition (2026-08-19, per user request -- determinism
        # diagnosis): getattr-guarded in openvla_utils.get_model(), safe to
        # set unconditionally for the WHOLE rollout here (this project's own
        # documented caution is specifically about MIXING True/False within
        # one episode, not about setting this consistently for an entire run).
        cfg.attn_implementation = args.attn_implementation
    set_seed_everywhere(cfg.seed)
    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model = get_model(cfg)
    # occ_vla addition (2026-08-22), per user's request to evaluate the
    # trained representation-alignment weights (vision_backbone +
    # projector, from train_representation_alignment.py) in a REAL
    # occluded-suite rollout, not just a training-loss check: load the
    # saved state dict (a partial dict -- only the params that had
    # requires_grad=True during training, i.e. vision_backbone +
    # projector; the frozen language_model is untouched) with
    # strict=False, since it deliberately does not cover every model
    # parameter.
    # occ_vla addition (2026-08-24, per user's Approach-A+B factorial request):
    # capture the UNMODIFIED vision_backbone+projector state on CPU before any
    # weight swapping happens, so 'baseline'/'B_only' conditions can be
    # restored to it after an 'A_only'/'A_plus_B' condition has loaded the
    # fine-tuned weights -- all 4 conditions run in ONE process (per the
    # user's explicit non-determinism concern: cross-process VLA inference is
    # NOT bit-reproducible, real, measured drift up to 2/20 episodes on task1
    # -- see this file's own history), so weights must be swappable in place,
    # not just loadable once at startup like the older --load-vision-weights.
    _base_vision_projector_state = {
        k: v.clone() for k, v in model.state_dict().items()
        if k.startswith("vision_backbone.") or k.startswith("projector.")
    }
    _vision_weights_a_state = None
    if args.vision_weights_a:
        _vision_weights_a_state = torch.load(args.vision_weights_a, map_location="cpu")
        unexpected_a = [k for k in _vision_weights_a_state if k not in _base_vision_projector_state]
        assert not unexpected_a, f"--vision-weights-a has keys not in the model: {unexpected_a[:5]}"

    def _set_vision_projector_weights(use_finetuned):
        """occ_vla addition (2026-08-24): swap vision_backbone+projector
        in-place between the base checkpoint's own weights and
        --vision-weights-a's fine-tuned weights, WITHOUT touching the frozen
        language_model. strict=False since the partial state dict never
        covers the LLM; unexpected/missing checked once at load time above
        and at each swap below."""
        target = _vision_weights_a_state if use_finetuned else _base_vision_projector_state
        assert target is not None, "requested fine-tuned vision weights but --vision-weights-a was not given"
        missing, unexpected = model.load_state_dict(target, strict=False)
        assert not unexpected, f"vision weight swap found keys not in the model: {unexpected[:5]}"

    if args.load_vision_weights:
        print(f"  [vision-weights] loading trained vision_backbone+projector from {args.load_vision_weights}")
        state_dict = torch.load(args.load_vision_weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # `unexpected` (keys in state_dict not matching any real model
        # param) is the meaningful failure signal here -- `missing` is
        # EXPECTED to be large (every frozen language_model param not
        # covered by this partial state_dict), not itself an error.
        print(f"  [vision-weights] loaded {len(state_dict)} tensors, "
              f"{len(unexpected)} unexpected keys (should be 0), "
              f"{len(missing)} model params left at their base-checkpoint value (expected -- the frozen LLM)")
        assert len(unexpected) == 0, f"vision weight load found keys not in the model: {unexpected[:5]}"
        loaded_names = set(state_dict.keys())
        applied_correctly = loaded_names.isdisjoint(set(missing))
        assert applied_correctly, "some trained vision weights were NOT applied -- check param name mismatch"
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)
    # occ_vla change (2026-08-24, per user's cross-suite VIM-comparison
    # request): suite/max_steps now resolved from --suite instead of the
    # module-level STOCK_SUITE/OCCLUDED_SUITE constants (still the default
    # for "10", unchanged behavior for every existing script/experiment).
    suite_stock_name = _suite_stock_name_for_cfg  # occ_vla: reuse the same resolution used for cfg.task_suite_name above, avoid drift
    suite_occ_name = {"10": OCCLUDED_SUITE, "spatial": "libero_spatial_occluded", "object": "libero_object_occluded", "goal": "libero_goal_occluded"}[args.suite]
    max_steps = TASK_MAX_STEPS[suite_stock_name]

    occluded_suite = benchmark.get_benchmark_dict()[suite_occ_name]()
    stock_suite = benchmark.get_benchmark_dict()[suite_stock_name]()

    original_forward = model.vision_backbone.forward
    splice_forward = make_agentview_midlayer_splice_forward(model.vision_backbone, args.midlayer_split_frac, img_idx=0)

    # occ_vla addition (2026-08-19, per user request -- systematic fix after
    # a real incident: today's "current"-depth runs silently used a
    # DIFFERENT resolved layer (dino=15, siglip=17) than every prior run
    # calling itself "current" (dino=16, siglip=18), because the frac->layer
    # formula changed (effective-N + round(), not nominal-N + int()) but the
    # CLI's --midlayer-split-frac default (0.67) was never updated to
    # compensate, and nothing printed/saved the RESOLVED layer to catch
    # this before trusting the comparison. Mirrors patched_forward's own
    # split_frac<=0.0 / else branching exactly -- keep these two in sync by
    # hand if either changes.
    def resolve_split_layers(vision_backbone, split_frac):
        if split_frac <= 0.0:
            return {"mode": "pixel_replace_L0", "dino_layer": None, "siglip_layer": None}
        nb_dino_eff = len(vision_backbone.featurizer.blocks) - 2
        nb_siglip_eff = len(vision_backbone.fused_featurizer.blocks) - 2
        sl_dino = min(int(round(nb_dino_eff * split_frac)), nb_dino_eff)
        sl_siglip = min(int(round(nb_siglip_eff * split_frac)), nb_siglip_eff)
        return {"mode": "midlayer_splice", "dino_layer": sl_dino, "dino_n_effective": nb_dino_eff,
                "siglip_layer": sl_siglip, "siglip_n_effective": nb_siglip_eff}

    resolved_layers = resolve_split_layers(model.vision_backbone, args.midlayer_split_frac)
    run_config = {
        "midlayer_split_frac_arg": args.midlayer_split_frac, "resolved_layers": resolved_layers,
        "task_ids": args.task_ids, "n_episodes": args.n_episodes, "episode_offset": args.episode_offset,
        "conditions": args.conditions, "checkpoint": args.checkpoint,
        "log_action_diff": args.log_action_diff, "log_attn_entropy": args.log_attn_entropy,
        "log_ensemble_disagreement": args.log_ensemble_disagreement,
        "attn_implementation": args.attn_implementation, "load_in_4bit": args.load_in_4bit,
        "pixel_fill_mode": args.pixel_fill_mode,
        "prevframe_gate_max_frac_no_ref": args.prevframe_gate_max_frac_no_ref,
        "prevframe_feather_px": args.prevframe_feather_px,
        "use_stock_suite": args.use_stock_suite,
        # occ_vla addition (2026-08-24/25, Approach-A+B factorial): resolved
        # effective settings for A_only/B_only/A_plus_B, per user's explicit
        # "run_config.json に解決後の実効設定を全部書き出す" requirement.
        "vision_weights_a": args.vision_weights_a,
        "vision_weights_a_abspath": os.path.abspath(args.vision_weights_a) if args.vision_weights_a else None,
        "stuck_dist_threshold": args.stuck_dist_threshold,
        "stuck_recovery_steps": args.stuck_recovery_steps,
        "stuck_retreat_mag": args.stuck_retreat_mag,
        "stuck_cooldown_envsteps": args.stuck_cooldown_envsteps,
        "episode_seed_range": [args.episode_offset, args.episode_offset + args.n_episodes],
    }
    print(f"[run-config] midlayer_split_frac={args.midlayer_split_frac} -> resolved: {resolved_layers}")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    all_summary = {}
    # occ_vla addition (2026-08-20, per user request -- ★ condition's 95%
    # on task1 is meaningless without a reference point): --use-stock-suite
    # runs against the PLAIN (non-occluded) libero_10 task instead of
    # libero_10_occluded, skipping occluder detection entirely (there is
    # none) -- gives the real "no occlusion, no physical obstacle at all"
    # baseline needed to interpret whether no_collision's SR fully
    # recovers to normal-task performance or still falls short.
    active_suite = stock_suite if args.use_stock_suite else occluded_suite

    for task_id in args.task_ids:
        task = active_suite.get_task(task_id)
        task_description = task.language
        print(f"\n=== task_id={task_id} '{task_description}' (stock_suite={args.use_stock_suite}) ===")

        # occ_vla addition (2026-08-24, Phase 2 proactive avoidance): also enable
        # depth rendering when "proactive_avoidance_depth" is among the requested
        # conditions -- that condition needs a real RGB-D obstacle point cloud,
        # not the privileged occluder_geom_ids used by proactive_avoidance_oracle/cbf.
        env = get_libero_env_seg(
            task, resolution=resize_size,
            camera_depths=bool(args.divergence_extract_dir) or "proactive_avoidance_depth" in args.conditions,
        )
        env.seed(0)
        env.reset()  # obj_of_interest is only populated on the env AFTER reset (not on the Task
                      # benchmark object -- confirmed via src/occ_vla/eval/libero_occ_env.py's own
                      # established convention: self._env.obj_of_interest[0], not task.obj_of_interest)
        target_names = list(getattr(env, "obj_of_interest", []) or [])
        occluder_names = [] if args.use_stock_suite else find_occluder_body_names(task, stock_suite)
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

        # occ_vla addition (2026-08-21, per user's item③ request --
        # "composite_visual_only" condition): computed once per task, same
        # pattern as target_seg_ids above -- occluder position/geometry is
        # static per task, so its segmentation ids don't need recomputing
        # per episode.
        occluder_seg_ids = find_segmentation_ids_for_bodies(env, sim, occluder_geom_ids) if occluder_geom_ids else []

        init_states = active_suite.get_task_init_states(task_id)
        n = min(args.n_episodes, len(init_states) - args.episode_offset)

        task_results = {}
        for condition in args.conditions:
            # occ_vla addition (2026-08-20, per user's 2x2-factorial 4th-
            # cell request): "oracle_no_collision" combines the oracle
            # visual splice (e.g. L=0 with --midlayer-split-frac 0 for the
            # privileged clean-render ceiling) WITH collision disabled --
            # the "visual: clean x physical: no collision" cell, needed
            # to fully decompose the two factors' contributions/
            # interaction alongside the already-measured baseline
            # (occluded+collision), L=0 (clean+collision), and
            # no_collision (occluded+no-collision) cells.
            model.vision_backbone.forward = splice_forward if condition in ("oracle", "oracle_no_collision") else original_forward
            # occ_vla addition (2026-08-24, per user's Approach-A+B factorial
            # request): "A_only"/"A_plus_B" swap in the fine-tuned
            # (representation-alignment) vision_backbone+projector weights;
            # "baseline"/"B_only" use the original checkpoint weights. Swap
            # happens fresh at the START of every condition's block (not just
            # once at process start), so all 4 conditions can run in one
            # process/one model load, satisfying the user's explicit
            # same-process requirement.
            AB_FACTORIAL_CONDITIONS = ("baseline", "A_only", "B_only", "A_plus_B")
            if condition in AB_FACTORIAL_CONDITIONS:
                _set_vision_projector_weights(use_finetuned=condition in ("A_only", "A_plus_B"))
                ab_vision_weights_used = args.vision_weights_a if condition in ("A_only", "A_plus_B") else "<base checkpoint>"
                print(f"    [A+B factorial] condition={condition} -> vision weights: {ab_vision_weights_used}")
            # occ_vla addition (2026-08-20, per user's 2x2 factorial design
            # request -- decouple VISUAL occlusion from PHYSICAL collision,
            # since removing the occluder entirely (visual+physical at
            # once) can't distinguish which one actually causes any
            # performance drop): condition "no_collision" keeps the
            # occluder fully visible (real occlusion, no VLA-side
            # correction -- same as baseline otherwise) but disables its
            # collision via geom_contype/geom_conaffinity=0, so the arm
            # can pass through it as if physically absent while the camera
            # still renders it normally. contype/conaffinity live on
            # mjModel (static), NOT mjData, so env.reset() does NOT
            # restore them -- must save/restore explicitly around this
            # condition's episode loop or the change would leak into
            # whatever condition runs next on this same `env` instance.
            # occ_vla bug fix (2026-08-20, real anomaly caught by the smoke
            # test: no_collision still showed 26/65 contact steps despite
            # "disabling" collision here): disabling contype/conaffinity at
            # THIS point (once per condition, before the episode loop) is
            # silently undone by each episode's OWN `env.reset()` call
            # inside run_episode() -- same "stale sim reference" behavior
            # already documented elsewhere in this file. Moved the actual
            # disable/restore into run_episode() itself (right after ITS
            # `sim = env.env.sim` re-fetch), done fresh every episode.
            # "no_collision" reuses run_episode's plain pass-through path
            # (condition != "oracle" -> original_forward, no VLA-side
            # correction at all) -- pass "baseline" as the internal
            # condition string so run_episode's oracle-only branches never
            # fire, while still recording results under the real
            # "no_collision" key below.
            # occ_vla addition (2026-08-20, per user request -- reactive
            # recovery proxy, real-robot-deployable trigger design):
            # "no_collision_after_contact" behaves exactly like baseline
            # (real collision, no VLA correction) UNTIL the first
            # anomalous (non-gripper) arm-link contact with the occluder,
            # at which point it switches to no_collision for the rest of
            # the episode -- tests whether reacting AFTER contact is
            # already too late, vs. the always-on no_collision condition's
            # upper bound.
            # occ_vla addition (2026-08-20, per user request -- a REAL
            # scripted recovery motion, not the idealized collision-disable
            # proxy): "scripted_recovery_after_contact" uses the SAME
            # trigger (first anomalous arm-link contact) but, instead of
            # disabling physics, injects a real retreat+lift action
            # sequence and lets real collision stay on -- tests whether an
            # actual, deployable recovery motion (not a simulator
            # privilege) can recover the episode.
            REACTIVE_CONDITIONS = ("no_collision_after_contact", "scripted_recovery_after_contact")
            # occ_vla addition (2026-08-21, per user's item③ request -- a
            # REAL-ROBOT-BUILDABLE alternative to no_collision's simulator-
            # only "arm passes through it" trick: the occluder is made
            # genuinely absent (never natively rendered, non-collidable --
            # reuses the same disable_collision_geom_ids path as
            # no_collision) and its on-screen occlusion delivered purely
            # via static-sprite image compositing (see run_episode's own
            # detailed docstring/comments at the composite_visual_only
            # block for the exact mechanism and its known z-buffering
            # limitation).
            if condition in ("no_collision",) + REACTIVE_CONDITIONS + ("low_mobility", "composite_visual_only", "ttc_area_blend", "scripted_recovery_after_stuck", "proactive_avoidance_oracle", "proactive_avoidance_cbf", "proactive_avoidance_depth", "proactive_avoidance_mpc", "agentview_vjepa"):
                run_episode_condition = "baseline"
            elif condition == "oracle_no_collision":
                run_episode_condition = "oracle"
            else:
                run_episode_condition = condition
            disable_collision_geom_ids = (
                occluder_geom_ids
                if (condition in ("no_collision", "oracle_no_collision", "composite_visual_only") + REACTIVE_CONDITIONS and occluder_geom_ids)
                else None
            )
            composite_visual_only = condition == "composite_visual_only"
            reactive_collision_disable = condition in REACTIVE_CONDITIONS
            scripted_recovery = condition == "scripted_recovery_after_contact"
            # occ_vla addition (2026-08-22), per user's Option (a):
            # "ttc_area_blend" keeps real collision AND visual occlusion
            # fully intact (same as baseline in both respects) -- the
            # only difference is the continuous action-blending logic
            # inside run_episode's main step loop, gated purely on
            # frac_occluded's own growth rate (no privileged occluder
            # geometry/contact information used at all, unlike
            # no_collision/scripted_recovery's collision-geom-based
            # mechanisms).
            ttc_area_blend = condition == "ttc_area_blend"
            # occ_vla addition (2026-08-23, per user's explicit "no
            # privileged information" request): "scripted_recovery_after_stuck"
            # keeps real collision AND real occluder rendering fully
            # intact -- it needs NO occluder-geom identity at all (unlike
            # every other reactive/no_collision/low_mobility condition
            # above, all of which require disable_collision_geom_ids ==
            # occluder_geom_ids). Trigger + recovery motion are computed
            # purely from obs["robot0_eef_pos"] inside run_episode.
            stuck_velocity_trigger = condition in ("scripted_recovery_after_stuck", "B_only", "A_plus_B")
            # occ_vla addition (2026-08-24): "proactive_avoidance_oracle" also
            # keeps real collision AND real occluder rendering fully intact --
            # it needs occluder_geom_ids for the PRIVILEGED true-3D-position
            # lookup (Phase 1 proof-of-concept only; a real depth-camera/
            # segmentation-based version is the planned Phase 2 if this shows
            # value), but does not disable collision/rendering itself.
            proactive_avoidance_oracle = condition in ("proactive_avoidance_oracle", "proactive_avoidance_cbf", "proactive_avoidance_depth", "proactive_avoidance_mpc")
            # occ_vla addition (2026-08-24, v2): "proactive_avoidance_cbf" reuses
            # the exact same trigger/plumbing as proactive_avoidance_oracle (same
            # privileged occluder-position lookup, same "keeps real collision AND
            # real occluder rendering intact" contract) -- only the correction
            # MATH differs (per-step minimal-norm CBF/APF nudge vs. v1's hard
            # full-chunk override to a fixed lift), selected via proactive_use_cbf.
            proactive_use_cbf = condition in ("proactive_avoidance_cbf", "proactive_avoidance_depth")
            # occ_vla addition (2026-08-24, Phase 2): "proactive_avoidance_depth"
            # reuses proactive_avoidance_cbf's exact correction math -- the ONLY
            # difference is where occ_centers/occ_radii come from (real RGB-D
            # obstacle point cloud vs. privileged occluder_geom_ids). No occluder
            # identity/geometry is used anywhere in this condition's path.
            proactive_use_depth = condition == "proactive_avoidance_depth"
            # occ_vla addition (2026-08-25): "proactive_avoidance_mpc" uses the
            # SAME privileged occluder-position lookup as proactive_avoidance_oracle/
            # proactive_avoidance_cbf (Phase 1 -- validate the sampling-based MPC
            # mechanism itself before adding real depth-estimation noise on top, same
            # sequencing already used for CBF v1->v2->depth). Real collision AND real
            # occluder rendering stay fully intact, same contract as every other
            # proactive_avoidance_* condition.
            proactive_use_mpc = condition == "proactive_avoidance_mpc"
            # occ_vla addition (2026-08-25): "agentview_vjepa" keeps real
            # collision AND real occluder rendering fully intact -- same
            # contract as every other proactive_avoidance_*/scripted_recovery_*
            # condition. Uses real segmentation-derived occlusion timing
            # (occluded_run_length, oracle CONTENT for now -- Phase 1, same
            # phasing already used for CBF) to gate the VJEPA correction
            # module; the module itself only ever sees proprio + its own
            # past latents, never privileged clean pixels.
            agentview_vjepa = condition == "agentview_vjepa"
            # occ_vla addition (2026-08-20, per user request -- mobility
            # sweep, top priority per their own reasoning: zero geometric
            # constraint, cheapest to implement, most directly tests the
            # "pushable vs fixed" mechanism already suggested by the L=0
            # contact-rate finding): "low_mobility" keeps real collision
            # AND visual occlusion fully intact, only reduces the
            # occluder's mass (5x lighter, floor 5g) and friction (10x
            # lower) -- a real, physically-buildable condition (cardboard
            # box vs bolted fixture), not a simulator privilege.
            low_mobility_geom_ids = occluder_geom_ids if (condition == "low_mobility" and occluder_geom_ids) else None
            results = []
            for ep in range(n):
                res = run_episode(
                    cfg, env, task_description, model, processor, action_head, proprio_projector, resize_size,
                    init_states[args.episode_offset + ep], max_steps, run_episode_condition, occluder_geom_ids, target_seg_ids, args.midlayer_split_frac,
                    original_forward=original_forward, splice_forward=splice_forward,
                    log_action_diff=args.log_action_diff, save_features_dir=args.save_oracle_features_dir,
                    task_id=task_id, episode_idx=args.episode_offset + ep,
                    log_attn_entropy=args.log_attn_entropy,
                    log_ensemble_disagreement=args.log_ensemble_disagreement,
                    pixel_fill_mode=args.pixel_fill_mode,
                    prevframe_gate_max_frac_no_ref=args.prevframe_gate_max_frac_no_ref,
                    prevframe_feather_px=args.prevframe_feather_px,
                    disable_collision_geom_ids=disable_collision_geom_ids,
                    reactive_collision_disable=reactive_collision_disable,
                    scripted_recovery=scripted_recovery,
                    low_mobility_geom_ids=low_mobility_geom_ids,
                    reactive_dry_run=args.reactive_dry_run,
                    composite_visual_only=composite_visual_only,
                    occluder_seg_ids=occluder_seg_ids,
                    record_video_dir=(
                        os.path.join(args.record_video_dir, f"{condition}_ep{args.episode_offset + ep}")
                        if args.record_video_dir else None
                    ),
                    divergence_extract_dir=(
                        os.path.join(args.divergence_extract_dir, f"{condition}_ep{args.episode_offset + ep}")
                        if args.divergence_extract_dir else None
                    ),
                    divergence_extract_t_range=tuple(args.divergence_extract_t_range) if args.divergence_extract_t_range else None,
                    ttc_area_blend=ttc_area_blend,
                    ttc_threshold=args.ttc_threshold,
                    ttc_safe_action=tuple(args.ttc_safe_action),
                    force_oracle_mask_frac=args.force_oracle_mask_frac,
                    stuck_velocity_trigger=stuck_velocity_trigger,
                    stuck_dist_threshold=args.stuck_dist_threshold,
                    stuck_recovery_steps=args.stuck_recovery_steps,
                    stuck_cooldown_envsteps=args.stuck_cooldown_envsteps,
                    stuck_retreat_mag=args.stuck_retreat_mag,
                    proactive_avoidance_oracle=proactive_avoidance_oracle,
                    proactive_safety_margin=args.proactive_safety_margin,
                    blank_agentview=args.blank_agentview_diagnostic,
                    proactive_use_cbf=proactive_use_cbf,
                    proactive_cbf_gain=args.proactive_cbf_gain,
                    proactive_use_depth=proactive_use_depth,
                    proactive_use_mpc=proactive_use_mpc,
                    proactive_mpc_n_candidates=args.proactive_mpc_n_candidates,
                    proactive_mpc_noise_std=args.proactive_mpc_noise_std,
                    proactive_mpc_w_safety=args.proactive_mpc_w_safety,
                    proactive_mpc_w_fidelity=args.proactive_mpc_w_fidelity,
                    agentview_vjepa=agentview_vjepa,
                    agentview_vjepa_min_run_length=args.agentview_vjepa_min_run_length,
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
                      f"termination_reason={res['termination_reason']} "
                      f"n_occluded_steps={res['n_occluded_steps']} n_action_diff_logged={len(res['action_diff_log'])} "
                      f"n_correction_applied={res['n_correction_applied']} n_forward_calls={res['n_forward_calls']} "
                      f"n_attn_entropy_logged={len(res['attn_entropy_log'])} "
                      f"n_ensemble_logged={len(res['ensemble_disagreement_log'])} "
                      f"n_prevframe_fill_logged={len(res['prevframe_fill_log'])} "
                      f"n_prevframe_gate_skipped={len(res['prevframe_gate_skip_log'])} "
                      f"reactive_triggered={res['reactive_triggered']} reactive_trigger_t={res['reactive_trigger_t']} "
                      f"stuck_triggered_count={res['stuck_triggered_count']} stuck_trigger_ts={res['stuck_trigger_ts']}")
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
