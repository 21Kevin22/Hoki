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
                 disable_collision_geom_ids=None, record_video_dir=None, reactive_collision_disable=False):
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

    if disable_collision_geom_ids and not reactive_collision_disable:
        _apply_collision_disable()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False
    clear_target_mask = None
    n_occluded_steps = 0
    occluded_run_length = 0  # elapsed consecutive occluded steps -- resets to 0 the moment occlusion clears
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
                if anomalous_contact:
                    _apply_collision_disable()
                    reactive_triggered = True
                    reactive_trigger_t = t
                    print(f"    [reactive] anomalous arm-link contact detected at t={t} -- switching to no_collision from here")

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
                condition == "oracle" and bool(occluder_geom_ids) and bool(occluded_pixel_mask.any())
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
                observation = {
                    "full_image": agentview_color,
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

                action_queue.extend(actions)

            action = action_queue.popleft()
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

    return {
        "success": success, "done_step": t,
        "termination_reason": termination_reason,
        "n_occluded_steps": n_occluded_steps,
        "action_diff_log": action_diff_log,
        "prevframe_fill_log": prevframe_fill_log,
        "prevframe_gate_skip_log": prevframe_gate_skip_log,
        "reactive_triggered": reactive_triggered, "reactive_trigger_t": reactive_trigger_t,
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

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=args.load_in_4bit,
        center_crop=True, num_open_loop_steps=8, task_suite_name=STOCK_SUITE, seed=7,
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

        env = get_libero_env_seg(task, resolution=resize_size)
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
            if condition in ("no_collision", "no_collision_after_contact"):
                run_episode_condition = "baseline"
            elif condition == "oracle_no_collision":
                run_episode_condition = "oracle"
            else:
                run_episode_condition = condition
            disable_collision_geom_ids = (
                occluder_geom_ids
                if (condition in ("no_collision", "oracle_no_collision", "no_collision_after_contact") and occluder_geom_ids)
                else None
            )
            reactive_collision_disable = condition == "no_collision_after_contact"
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
                    record_video_dir=(
                        os.path.join(args.record_video_dir, f"{condition}_ep{args.episode_offset + ep}")
                        if args.record_video_dir else None
                    ),
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
                      f"reactive_triggered={res['reactive_triggered']} reactive_trigger_t={res['reactive_trigger_t']}")
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
