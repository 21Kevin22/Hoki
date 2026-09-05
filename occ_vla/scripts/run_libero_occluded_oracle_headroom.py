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

def get_libero_env_seg(task, resolution, camera_depths=False, extra_camera=None):
    # occ_vla addition (2026-08-21), per user's Figure-A divergence-
    # analysis request: robosuite/LIBERO already support RGB-D rendering
    # via this one kwarg (confirmed real via env_wrapper.py/
    # bddl_base_domain.py, just never turned on in this project before).
    # Depth obs key becomes f"{cam_name}_depth" e.g. "agentview_depth"
    # (confirmed in robosuite/environments/robot_env.py). Default False,
    # zero behavior change for every existing caller.
    # occ_vla addition (2026-08-31), per user's sharp methodological
    # question about VIM's own experimental design (does an added image
    # SLOT help regardless of content, not specifically the imagined-
    # viewpoint content itself?): extra_camera, if given a real robosuite/
    # MuJoCo camera name (e.g. "frontview" -- a real, distinct, standard
    # robosuite arena camera, confirmed via env_wrapper.py's own
    # render_camera="frontview" default, NOT the same position as
    # agentview), adds it to camera_names so obs[f"{extra_camera}_image"]
    # becomes available -- a genuinely different real viewpoint with zero
    # generative/imagined content, for a clean "does ANY second view slot
    # help, or does it need to be the wrist camera specifically" ablation.
    # None (default) is the original ["agentview", "robot0_eye_in_hand"]
    # behavior, byte for byte.
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    kwargs = dict(
        bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution,
        camera_segmentations="instance", camera_depths=camera_depths,
    )
    if extra_camera:
        kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand", extra_camera]
    env = OffScreenRenderEnv(**kwargs)
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


def load_real_vjepa2(device):
    """Loads the REAL Meta V-JEPA2-AC checkpoint (facebookresearch/vjepa2,
    encoder ~1B params + AC predictor ~305M params, frozen, eval mode) --
    NOT this project's own VJEPA_LatentDynamicsPredictor (unrelated, see
    CLAUDE.md). Requires thirdparty/vjepa2 (cloned + locally patched, see
    CLAUDE.md's "Real Meta V-JEPA2-AC" entry) to already be on sys.path."""
    vjepa2_dir = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "vjepa2"))
    if vjepa2_dir not in sys.path:
        sys.path.insert(0, vjepa2_dir)
    from src.hub.backbones import vjepa2_ac_vit_giant
    encoder, predictor = vjepa2_ac_vit_giant(pretrained=True)
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)
    return encoder, predictor


def load_real_vjepa2_base(device):
    """occ_vla addition (2026-08-31, per user's verified proposal): loads
    the REAL BASE (non-action-conditioned) V-JEPA2 checkpoint
    (vjepa2_vit_giant, a SEPARATE checkpoint file from the AC one --
    different real weights, though same encoder ARCHITECTURE). This is
    V-JEPA2's own actual self-supervised pretraining objective: predict
    the representation of MASKED spatial patches from the UNMASKED
    context, within a single frame -- no temporal history, no action
    conditioning needed at all. Architecturally the right tool for a
    scene that's occluded from frame 1 (unlike the AC predictor's
    temporal rollout, which needs a real pre-occlusion observation to
    seed from -- see CLAUDE.md's "agentview_vjepa2_temporal" entries for
    why that mechanism structurally cannot help this specific case)."""
    vjepa2_dir = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "vjepa2"))
    if vjepa2_dir not in sys.path:
        sys.path.insert(0, vjepa2_dir)
    from src.hub.backbones import vjepa2_vit_giant
    encoder, predictor = vjepa2_vit_giant(pretrained=True)
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)
    return encoder, predictor


def vjepa2_amodal_complete(encoder, predictor, clip, occluded_token_mask_256, device):
    """occ_vla addition (2026-08-31): real V-JEPA2 spatial masked-patch
    completion -- masks_x (context) = UNOCCLUDED token indices, masks_y
    (target) = OCCLUDED token indices, matching the real predictor's own
    (masks_x, masks_y) index-list API (src/models/predictor.py, src/
    masks/utils.py's apply_masks). The encoder itself only ever processes
    the UNOCCLUDED tokens (masks=[masks_x] passed straight into its own
    public forward, which gathers via apply_masks BEFORE running any
    self-attention block) -- the occluded region's real pixel content
    never reaches the model's computation at all, by construction, not
    merely by convention.

    Returns a (256,1408) tensor: predicted content at occluded positions
    (real model output), ZEROS elsewhere (irrelevant -- callers only ever
    read this at the occluded positions, via the same token_mask_256
    convention used throughout this file)."""
    occluded_idx = np.flatnonzero(occluded_token_mask_256)
    unoccluded_idx = np.flatnonzero(~occluded_token_mask_256)
    if len(occluded_idx) == 0 or len(unoccluded_idx) == 0:
        return torch.zeros(256, 1408, device=device)
    masks_x = torch.from_numpy(unoccluded_idx).long().to(device).unsqueeze(0)  # (1,K_vis)
    masks_y = torch.from_numpy(occluded_idx).long().to(device).unsqueeze(0)  # (1,K_occ)
    with torch.no_grad():
        context = encoder(clip, masks=[masks_x])  # (1,K_vis,1408) -- occluded pixels never seen
        pred = predictor(context, masks_x=[masks_x], masks_y=[masks_y])  # (1,K_occ,1408)
    out = torch.zeros(256, 1408, device=device, dtype=pred.dtype)
    out[occluded_idx] = pred[0]
    return out


def vjepa2_local_adain(pred_at_occ, occluded_idx_np, z_real_full, unoccluded_idx_np, k=12):
    """occ_vla addition (2026-08-31, per user's explicit engineering
    request): a SPADE (Park et al., CVPR 2019)/AdaIN (Huang & Belongie,
    ICCV 2017)-style LOCAL statistical rescaling of the real V-JEPA2
    predictor's raw output. Diagnosed root cause: the predictor's raw
    output collapses toward the mean (std ~23% of real context std) --
    the textbook shrinkage bias of an L1/L2-regression-trained masked
    predictor under real ambiguity (V-JEPA2's own paper confirms L1
    regression loss). Rather than one GLOBAL mean/std correction (tried
    first, worked numerically but still looked like a visually distinct
    "different noise patch" from its surroundings), this rescales each
    occluded token toward the mean/std of its own k SPATIALLY NEAREST
    real unoccluded neighbor tokens (16x16 grid Euclidean distance) --
    real scenes have spatially-varying local statistics (table vs.
    cabinet vs. shadow), so a per-region reference blends far more
    seamlessly, per the same principle these two real, established
    papers are built on. Zero new training -- pure inference-time
    post-processing of the real predictor's own output."""
    coords = np.stack(np.meshgrid(np.arange(16), np.arange(16), indexing="ij"), axis=-1).reshape(-1, 2)
    occ_coords = coords[occluded_idx_np]
    unocc_coords = coords[unoccluded_idx_np]
    real_context = z_real_full[unoccluded_idx_np]
    dists = np.linalg.norm(occ_coords[:, None, :] - unocc_coords[None, :, :], axis=-1)
    nn_idx = np.argsort(dists, axis=1)[:, :k]
    src_mean = pred_at_occ.mean(dim=0)
    src_std = pred_at_occ.std(dim=0) + 1e-6
    out = pred_at_occ.clone()
    for i in range(pred_at_occ.shape[0]):
        neighbors = real_context[nn_idx[i]]
        ref_mean = neighbors.mean(dim=0)
        ref_std = neighbors.std(dim=0)
        out[i] = (pred_at_occ[i] - src_mean) / src_std * ref_std + ref_mean
    return out


def vjepa2_confidence_mask(pred_at_occ_raw, occluded_idx_np, z_real_full, unoccluded_idx_np, k=12,
                            threshold=0.0):
    """occ_vla addition (2026-09-01), per user request ("パラメータで調整
    できるテクニック... AI研究者としてリサーチして実装して"): a training-
    free, inference-time confidence gate for `agentview_vjepa2_amodal`'s
    per-token completion, grounded in the established anomaly-detection/
    image-inpainting technique of using distance-to-nearest-real-content
    as an implicit confidence/OOD score (e.g. patch-based inpainting
    confidence propagation, Criminisi et al. 2004; nearest-neighbor
    reconstruction-error anomaly scoring). Computed on the RAW predictor
    output (BEFORE vjepa2_local_adain's rescaling -- rescaling pulls
    every token toward locally-plausible statistics by construction, so
    a post-rescale confidence signal would be uninformative regardless
    of the underlying completion's real quality).

    For each occluded token, cosine-similarity against the MEAN of its
    own k spatially-nearest REAL (unoccluded) neighbor tokens is used as
    the confidence proxy: a token whose raw completion is very dissimilar
    from its real local surroundings is the signature already visually
    confirmed for this project's "blob"/regression-to-the-mean failure
    mode (CLAUDE.md's "MMaDA arm-free generation quality investigation"
    documents the same category of failure for a different generative
    model on a related task -- distance-from-real-context is the
    common, reusable diagnostic). `threshold` is a real, tunable
    parameter: -1.0 (permissive) keeps every token (byte-identical to no
    gating at all); higher values (up to 1.0) progressively restrict
    injection to only the most locally-consistent completions, falling
    back to the REAL (occluded, unmodified) content at every position
    that fails the gate -- this is a strictly more conservative fallback
    than injecting a low-confidence completion, matching this project's
    own repeatedly-validated "minimal intervention beats full
    replacement" principle (CBF v1->v2, alpha=1.0->0.3).

    Returns a boolean numpy array, shape (len(occluded_idx_np),) -- True
    = confident enough to inject, False = skip (leave real content)."""
    coords = np.stack(np.meshgrid(np.arange(16), np.arange(16), indexing="ij"), axis=-1).reshape(-1, 2)
    occ_coords = coords[occluded_idx_np]
    unocc_coords = coords[unoccluded_idx_np]
    real_context = z_real_full[unoccluded_idx_np]
    dists = np.linalg.norm(occ_coords[:, None, :] - unocc_coords[None, :, :], axis=-1)
    nn_idx = np.argsort(dists, axis=1)[:, :k]
    confident = np.zeros(len(occluded_idx_np), dtype=bool)
    sims = np.zeros(len(occluded_idx_np), dtype=np.float32)
    for i in range(pred_at_occ_raw.shape[0]):
        ref_mean = real_context[nn_idx[i]].mean(dim=0)
        sim = torch.nn.functional.cosine_similarity(
            pred_at_occ_raw[i].unsqueeze(0).float(), ref_mean.unsqueeze(0).float()
        ).item()
        sims[i] = sim
        confident[i] = sim >= threshold
    if os.environ.get("VJEPA2_CONFGATE_DEBUG"):
        print(f"    [conf-gate sim stats] min={sims.min():.3f} mean={sims.mean():.3f} "
              f"max={sims.max():.3f} std={sims.std():.3f}")
    return confident


def quat2axisangle_vjepa2(quat):
    """Same convention as openvla-oft's own proprio construction (see
    experiments/robot/robot_utils.py's quat2axisangle), duplicated here
    (not imported) to keep the V-JEPA2 addition self-contained and not
    risk touching that shared utility's behavior for every other caller."""
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    w, x, y, z = quat[3], quat[0], quat[1], quat[2]
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-8, 1 - w * w))
    axis = np.array([x, y, z]) / s if s > 1e-6 else np.array([1.0, 0.0, 0.0])
    return axis * angle


def make_agentview_vjepa2_temporal_splice_forward(vision_backbone, img_idx=0, blend_alpha=1.0):
    """blend_alpha (occ_vla addition 2026-08-31, per user's engineering
    request): 1.0 = original hard-replace behavior (byte-identical to
    every prior test this session). A value < 1.0 alpha-BLENDS the
    injected content with the model's own real (uncorrected) patch
    tokens at the occluded positions, instead of a full overwrite --
    grounded in the same "minimal intervention beats full replacement"
    principle already validated repeatedly in this project (CBF's
    minimal-norm correction, gated action blending) -- motivated by a
    real observed regression under DUAL-camera evaluation (where the
    model may already partially compensate via the real wrist camera,
    so a full synthetic overwrite of agentview's occluded region can
    remove real signal the model was already using, not just add
    missing signal)."""
    """occ_vla addition (2026-08-31, per user's explicit choice to build the
    real-V-JEPA2 temporal-recovery direction): unlike
    make_agentview_midlayer_splice_forward (which splices REAL re-rendered
    clean pixels, run through the SAME featurizer blocks up to a split
    layer), this splices content that isn't pixel-derived at all --
    a real V-JEPA2-AC latent (rolled forward from the episode's own history
    via real executed actions, see run_episode's per-step maintenance),
    projected into this checkpoint's own DINO/SigLIP token dimensions via
    NEW, UNTRAINED linear layers (vjepa2_proj_dino/_siglip -- explicitly
    disclosed as untrained; no training data or step exists for them yet).

    Design choice: overwrites the FINAL (full-depth) patch tokens, not a
    mid-network splice -- there is no principled "run V-JEPA2 content
    through DINO/SigLIP's own remaining blocks" operation (it was never
    produced by those blocks in the first place), so late/output-level
    substitution is the only architecturally coherent injection point for
    non-pixel-derived content. This is the same category as this
    project's own "L=N_effective" (late substitution) depth-sweep
    endpoint, not L=0 or a true mid-layer splice.

    Reads vision_backbone._diagnostic_vjepa2_dino_content /
    _diagnostic_vjepa2_siglip_content (each (1,256,embed_dim) or None) and
    _diagnostic_agentview_patch_mask_256 (reused from the agentview_vjepa
    condition's own convention), set by the eval loop each step this
    condition engages."""

    def patched_forward(pixel_values, occlusion_mask=None, proprio_for_dynamics=None):
        vision_backbone._diagnostic_forward_call_count = (
            getattr(vision_backbone, "_diagnostic_forward_call_count", 0) + 1
        )
        assert vision_backbone.use_fused_vision_backbone
        num_images = vision_backbone.num_images_in_input
        dino_content = getattr(vision_backbone, "_diagnostic_vjepa2_dino_content", None)
        siglip_content = getattr(vision_backbone, "_diagnostic_vjepa2_siglip_content", None)
        patch_mask_256 = getattr(vision_backbone, "_diagnostic_agentview_patch_mask_256", None)

        images = [pixel_values] if num_images == 1 else torch.split(pixel_values, [6] * num_images, dim=1)
        all_patches = []
        for idx, img in enumerate(images):
            img_regular, img_fused = torch.split(img, [3, 3], dim=1)
            patches = vision_backbone.featurizer(img_regular)
            patches_fused = vision_backbone.fused_featurizer(img_fused)
            if (idx == img_idx and dino_content is not None and siglip_content is not None
                    and patch_mask_256 is not None and bool(patch_mask_256.any())):
                vision_backbone._diagnostic_correction_applied_count = (
                    getattr(vision_backbone, "_diagnostic_correction_applied_count", 0) + 1
                )
                # occ_vla bug fix (2026-08-31, caught by a real smoke-test
                # shape-mismatch error, not by inspection): the PUBLIC
                # vision_backbone.featurizer(img)/fused_featurizer(img) calls
                # used here (unlike _run_vit_with_midlayer_splice's low-level
                # per-block loop over the RAW internal sequence) already
                # perform the checkpoint's own intermediate-layer extraction
                # AND strip prefix/register tokens internally -- `patches`/
                # `patches_fused` here are ALREADY pure (1,256,embed_dim)
                # patch-only tensors, confirmed empirically (DINO's real
                # output was 256, not 256+num_prefix_tokens; slicing
                # `[:, num_prefix:]` again wrongly removed 5 real patch
                # tokens, producing a 251-vs-256 mask mismatch). No further
                # prefix slicing needed here.
                mask = patch_mask_256.to(dtype=torch.bool, device=patches.device).reshape(1, -1, 1)
                # occ_vla addition (2026-08-31): read a PER-STEP dynamic
                # alpha if the caller set one (persistence-escalate design,
                # same closed-form schedule already validated for CBF's
                # v4-escalate persistence gate -- see run_episode's own
                # computation), falling back to the fixed closure value
                # (byte-identical to every prior test) if not set.
                # occ_vla bug fix: the attribute is explicitly set to None
                # (not deleted) when persistence-escalate is disabled, so
                # getattr's own default would never trigger -- check None
                # explicitly instead.
                step_alpha = getattr(vision_backbone, "_diagnostic_vjepa2_blend_alpha", None)
                if step_alpha is None:
                    step_alpha = blend_alpha
                blended_dino = step_alpha * dino_content.to(patches.dtype) + (1 - step_alpha) * patches
                blended_siglip = step_alpha * siglip_content.to(patches_fused.dtype) + (1 - step_alpha) * patches_fused
                patches = torch.where(mask, blended_dino, patches)
                patches_fused = torch.where(mask, blended_siglip, patches_fused)
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
                 agentview_vjepa=False, agentview_vjepa_min_run_length=3,
                 save_distillation_pairs_dir=None,
                 proactive_target_attractor_radius_m=0.0, proactive_target_attractor_decay=0.1,
                 proactive_target_attractor_max_staleness=50,
                 proactive_grasp_phase_radius_m=0.0, proactive_grasp_phase_gain_decay=1.0,
                 proactive_persistence_window=0.0, proactive_persistence_min_gain_frac=0.2,
                 proactive_persistence_mode="decay", blank_wrist=False, drop_wrist_image=False,
                 second_view_camera="robot0_eye_in_hand",
                 agentview_vjepa2_temporal=False, vjepa2_encoder=None, vjepa2_predictor=None,
                 vjepa2_proj_dino=None, vjepa2_proj_siglip=None, vjepa2_splice_forward=None,
                 agentview_vjepa2_amodal=False, vjepa2_base_encoder=None, vjepa2_base_predictor=None,
                 vjepa2_blend_alpha_ceiling=1.0, vjepa2_blend_alpha_floor=0.0,
                 vjepa2_blend_persistence_window=0, vjepa2_amodal_ema_decay=0.5,
                 vjepa2_confidence_threshold=-1.0,
                 ace_gate_enabled=False, ace_gate_scale_m=0.05, ace_gate_min_frac=0.15,
                 attn_target_excl_enabled=False, attn_target_window=5, attn_target_gap_delta=0.0,
                 object_centric_adapter_enabled=False):
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

    # occ_vla addition (2026-09-01): mutable, per-step-updated by the
    # KNOWS-style attention target-identification block in the main replan
    # loop below (assigned via plain `=` at the same nesting level as this
    # variable's declaration, so `_depth_obstacle_points`'s closure below
    # sees the CURRENT value at call time, not the value at definition
    # time -- standard Python closure-over-enclosing-scope semantics, not a
    # `nonlocal`/mutable-container workaround). None whenever
    # attn_target_excl_enabled is False or no target was confidently
    # identified this step -- zero effect on every existing condition.
    attn_identified_target_id = None

    def _depth_obstacle_points(obs_dict, stride=6, max_range_m=1.2):
        """Real-sensor obstacle point cloud for this step: back-projects a
        downsampled agentview depth grid to 3D world points, excluding the
        robot's own body (self-filter), the task's own TARGET object
        (target_seg_ids -- we want to avoid OTHER stuff, not the thing we're
        supposed to reach for), the CURRENT attention-identified target
        object if KNOWS-style exclusion is active (attn_identified_target_id
        -- addresses task9's diagnosed misfire, where CBF treats the
        destination receptacle's own geometry as an obstacle even while the
        policy is legitimately approaching it), and anything beyond
        max_range_m (MuJoCo scenes include distant background geometry
        irrelevant to near-field avoidance). No occluder-identity information
        used anywhere here."""
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
        if attn_identified_target_id is not None:
            exclude_ids = exclude_ids | {attn_identified_target_id}
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

    def _depth_target_centroid(obs_dict, stride=6, max_range_m=1.2):
        """occ_vla addition (2026-08-29, per user's 'local attractor' proposal
        for the margin-vs-target-proximity conflict diagnosed on
        libero_object task7/task1/task4 -- see run notes): mirrors
        _depth_obstacle_points's exact real-sensor back-projection, but keeps
        ONLY the TARGET's own segmentation pixels (the opposite filter),
        returning their mean 3D world position -- a zero-privileged (real
        RGB-D + segmentation, no sim.data ground truth) estimate of where the
        grasp target actually is this step. Returns None if the target isn't
        visible in this frame at all (fully occluded / out of frame) -- the
        caller must treat that as 'no attractor available this step', not
        crash or silently reuse a stale value from a different function."""
        depth_key = "agentview_depth"
        if depth_key not in obs_dict or not target_seg_ids:
            return None
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
        keep = np.isin(seg_sub, list(target_seg_ids)) & (depth_sub > 1e-4) & (depth_sub < max_range_m)
        if not np.any(keep):
            return None
        z = depth_sub[keep].astype(float)
        col = cc[keep].astype(float)
        row = rr[keep].astype(float)
        cam_pts = np.stack([col * z, row * z, z, np.ones_like(z)], axis=-1)
        world_pts = (depth_cam2world @ cam_pts.T).T[:, :3]
        return world_pts.mean(axis=0)

    # occ_vla addition (2026-09-01), per user's explicit request to ground
    # this in the real cited paper's actual method, not a from-scratch
    # invention: KNOWS (arXiv:2606.09749, "Your Model Already Knows:
    # Attention-Guided Safety Filter for Vision-Language-Action Models",
    # Park et al., UCLA -- full PDF read, no code release found) identifies
    # the object the policy is CURRENTLY approaching from a single
    # action-query x vision-key attention head, then excludes it from the
    # CBF's obstacle set (everything else stays a candidate obstacle).
    # Directly targets task9's diagnosed failure mode (CBF treating the
    # destination receptacle's own geometry as an obstacle to avoid,
    # 286.6 corrections/episode despite ~0% baseline contact).
    #
    # Faithful-subset reimplementation (full fidelity -- YOLOE fine-tuning,
    # per-object 3D ellipsoid fitting/tracking, the separating-hyperplane
    # ellipsoid CBF-QP -- is out of scope given the user's stated 3-month
    # thesis deadline, per CLAUDE.md's 2026-09-01 entry):
    #   - Object candidates: reuse this project's own REAL per-pixel
    #     segmentation (obs[AGENTVIEW_SEG_KEY], already used everywhere else
    #     in this file) instead of fitting new 3D ellipsoids -- gives the
    #     same "which object does this pixel belong to" information KNOWS'
    #     own SAM-based masks provide, without a new perception model.
    #   - Attention source: `get_vla_action(..., return_attn_map=True)`,
    #     ALREADY-EXISTING infra (2026-08-21 addition, see
    #     PrismaticForConditionalGeneration._compute_action_patch_attn_entropy
    #     in modeling_prismatic.py) returning the LAST transformer layer's
    #     action-query x vision-patch attention, mean-pooled over heads and
    #     action-chunk positions -- NOT KNOWS' own profiled (layer 12, head 3
    #     for pi0.5) single best unit; OpenVLA-OFT's own best layer/head has
    #     not been profiled this session (their Sec 3.4 procedure -- log
    #     per-(layer,head) mean attention mass on the phase-appropriate
    #     object across several real episodes -- is a real, not-yet-done
    #     next step if this coarser last-layer/all-heads version proves too
    #     noisy).
    #   - CBF integration: rather than porting KNOWS' ellipsoid-vs-ellipsoid
    #     separating-hyperplane QP, the identified target's segmentation ID
    #     is added to the EXISTING `_depth_obstacle_points` exclusion set for
    #     that step, reusing this project's already-validated per-point
    #     minimal-norm CBF correction unchanged.
    #   - Attention extraction still goes through `output_attentions=True`
    #     (not KNOWS' own hook-based, FlashAttention-kernel-untouched
    #     extraction) -- this project's OWN documented 2026-08-19 finding
    #     (SDPA->eager switch flips 8/20 episode outcomes when MIXING
    #     output_attentions=True/False calls within a rollout) is worked
    #     around the ALREADY-ESTABLISHED way (2026-08-12 CAUTION in
    #     openvla_utils.py): force `--attn-implementation eager` for the
    #     WHOLE rollout so every call is consistently eager, never mixed.
    #     This is a real, different behavior from pure-SDPA baseline (a
    #     controlled-variable comparison, not a zero-footprint one like
    #     KNOWS' hook-based design) -- any baseline this condition is
    #     compared against must ALSO run under --attn-implementation eager
    #     for the comparison to be fair; do not compare against a
    #     default-SDPA baseline number from elsewhere in this file.
    attn_target_history = deque(maxlen=attn_target_window if attn_target_window > 0 else 1)

    def _attention_target_id(attn_map_full, obs_dict):
        """Returns (target_seg_id_or_None, debug_dict). attn_map_full: raw
        (NUM_PATCHES,) from get_vla_action(return_attn_map=True); NUM_PATCHES
        = per_image_patches * num_images_in_input, agentview patches come
        FIRST (confirmed via openvla_utils.get_vla_action: `all_images =
        [obs["full_image"]]` is appended before any wrist image) -- only the
        agentview half is used here, matching KNOWS' third-person-camera
        setup."""
        if attn_map_full is None:
            return None, {}
        n_total = attn_map_full.shape[0]
        n_per_image = n_total // max(1, cfg.num_images_in_input)
        agent_map = attn_map_full[:n_per_image]
        g = int(round(np.sqrt(n_per_image)))
        if g * g != n_per_image:
            return None, {"error": f"non-square patch grid ({n_per_image} patches)"}
        agent_map_2d = agent_map.reshape(g, g)

        seg = np.asarray(obs_dict.get(AGENTVIEW_SEG_KEY, np.zeros((resize_size, resize_size), dtype=int))).squeeze()
        if seg.ndim == 3:
            seg = seg[..., 0]
        h, w = seg.shape[:2]
        # candidate objects: every real segmentation id present in-frame,
        # excluding robot/gripper -- matches KNOWS' "every manipulable
        # object is a candidate obstacle until excluded" framing, using
        # real segmentation instead of a new SAM-based detector.
        candidate_ids = sorted(set(np.unique(seg).tolist()) - set(robot_seg_ids_for_depth or []) - {0})
        if not candidate_ids:
            return None, {}

        # downsample each candidate's binary mask to the (g, g) patch grid
        # via block-mean pooling (coverage fraction per patch, matching
        # KNOWS' c_i(r,c) -- their eq. 2), accumulate mass/area into a
        # per-episode sliding window (attn_target_history, maxlen=K).
        rows_per_patch = h / g
        cols_per_patch = w / g
        step_masses, step_areas = {}, {}
        for cid in candidate_ids:
            mask = (seg == cid).astype(np.float32)
            coverage = np.zeros((g, g), dtype=np.float32)
            for r in range(g):
                r0, r1 = int(round(r * rows_per_patch)), int(round((r + 1) * rows_per_patch))
                for c in range(g):
                    c0, c1 = int(round(c * cols_per_patch)), int(round((c + 1) * cols_per_patch))
                    block = mask[r0:r1, c0:c1]
                    coverage[r, c] = block.mean() if block.size else 0.0
            step_masses[cid] = float((agent_map_2d * coverage).sum())
            step_areas[cid] = float(coverage.sum())
        attn_target_history.append((step_masses, step_areas))

        agg_mass, agg_area = {}, {}
        for masses, areas in attn_target_history:
            for cid in masses:
                agg_mass[cid] = agg_mass.get(cid, 0.0) + masses[cid]
                agg_area[cid] = agg_area.get(cid, 0.0) + areas[cid]
        densities = {cid: (agg_mass[cid] / agg_area[cid] if agg_area[cid] > 1e-8 else 0.0) for cid in agg_mass}
        if not densities:
            return None, {}
        ranked = sorted(densities.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_d = ranked[0]
        second_d = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = top_d - second_d
        debug = {"top_id": top_id, "top_density": top_d, "second_density": second_d, "gap": gap,
                  "n_candidates": len(candidate_ids)}
        if gap >= attn_target_gap_delta:
            return top_id, debug
        return None, debug

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
    # occ_vla addition (2026-08-31, agentview_vjepa2_temporal): per-episode
    # running REAL V-JEPA2-AC latent estimate + bookkeeping for the
    # step-by-step maintenance loop (see below, right after
    # occluded_pixel_mask/frac_occluded_this_step are computed each step).
    vjepa2_latent_state = None
    vjepa2_prev_frame_tensor = None
    vjepa2_last_action = None
    # occ_vla addition (2026-08-31, per user's "find and fix weaknesses"
    # request): temporal EMA buffer for agentview_vjepa2_amodal's
    # completed content -- diagnosed weakness: the completion is
    # recomputed independently every engaged replan step from the
    # slightly-shifting current frame, with no temporal consistency
    # constraint, which could itself be a distribution-deviation cost
    # (this project's own established finding: this policy is fragile to
    # ANY deviation from a live, continuously-updating, temporally
    # coherent input -- not just to missing content). None = no history
    # yet (first engaged step this episode).
    vjepa2_amodal_ema = None
    # occ_vla addition (2026-08-20, per user request -- a REAL scripted
    # recovery motion, not the idealized collision-disable proxy): last
    # commanded gripper value (pre-process_action, raw model output range),
    # so the scripted recovery phase can hold the gripper steady (not
    # accidentally open/close it) instead of guessing a value. Updated
    # every time a real VLA action is popped from the queue.
    last_gripper_raw = 0.0  # LIBERO/OpenVLA raw convention before process_action's flip/normalize
    action_diff_log = []
    distillation_manifest = []  # occ_vla addition (2026-08-27): (image, proprio, corrected-action) pairs for imitation-distillation of proactive_avoidance_depth
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
    # occ_vla addition (2026-08-29, per user's "persist target position across
    # occluded frames" proposal -- a simple state cache, NOT a learned
    # memory/retrieval module): _depth_target_centroid returns None whenever
    # the target isn't visible this exact step (occluded by the arm's own
    # pose, out of frame, etc.). Without this cache, the target-proximity
    # margin-decay fix silently disables itself during exactly the moments
    # diagnosed as most likely to matter (see CLAUDE.md run notes). Cleared
    # fresh each episode (module-level across episodes would leak state).
    last_known_target_centroid = None
    last_known_target_centroid_age = None
    # occ_vla addition (2026-08-30): consecutive-replan-chunk streak of a
    # margin violation persisting -- see the persistence-gate comment at
    # its point of use below for the full rationale.
    violation_streak = 0
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
    # occ_vla addition (2026-09-01, per user's "VLA自身のアテンション/ACEでCBFの
    # 介入をゲートする" request): reuse the already-existing, already-real-robot
    # -safe ensemble_disagreement signal (perturbed-pixel re-forward-pass L2
    # action distance -- NOT the attention-entropy signal, which is known
    # (2026-08-19 entry above) to silently force output_attentions=True and
    # flip 8/20 episode outcomes via an SDPA->eager attention-backend switch
    # -- ensemble_disagreement has no such contamination, confirmed real-
    # robot-usable, no output_attentions, no privileged info) as a candidate
    # "how confident is the base policy right now" signal to GATE (not just
    # log) the CBF's effective correction gain. Forcing it on whenever the
    # gate is enabled, rather than requiring the caller to also pass
    # --log-ensemble-disagreement separately.
    if ace_gate_enabled:
        log_ensemble_disagreement = True
    disagreement = None  # populated every replan step once log_ensemble_disagreement fires; None until then
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
            # occ_vla addition (2026-08-31): second_view_camera lets the second
            # image slot be filled by any real robosuite camera, not just the
            # wrist -- default "robot0_eye_in_hand" reproduces the original
            # get_libero_wrist_image(obs) behavior exactly (same key, same
            # flip). A non-default value (e.g. "frontview") must have been
            # requested at env-construction time too (get_libero_env_seg's
            # extra_camera) or this KeyErrors -- caller's responsibility to
            # keep the two in sync (main() does this via args.second_view_camera).
            if second_view_camera == "robot0_eye_in_hand":
                wrist_img = get_libero_wrist_image(obs).copy()
            else:
                wrist_img = obs[f"{second_view_camera}_image"][::-1, ::-1].copy()
            # occ_vla addition (2026-08-30, per user's decisive test of the
            # wrist-camera-bypass hypothesis, §3.7): blank the wrist camera
            # (gray-fill, matching the agentview_vjepa convention) to test
            # whether it is really the dominant channel letting baseline
            # bypass agentview occlusion, rather than just plausible from
            # qualitative frame inspection alone. Zero effect unless
            # explicitly enabled -- every existing condition/caller
            # untouched.
            if blank_wrist:
                wrist_img = np.full_like(wrist_img, 127)
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
                # occ_vla addition (2026-08-30, per user's "手首カメラのPOVダンプ
                # 検証" request): also save the real wrist-camera frame at the
                # SAME timestep, so agentview-occlusion vs. wrist-visibility
                # can be directly compared side by side -- diagnostic only,
                # same on/off condition (record_video_dir) as the existing
                # agentview save, zero effect unless that's already enabled.
                Image.fromarray(wrist_img).save(os.path.join(record_video_dir, f"frame_{t:05d}_wrist.png"))

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

            # occ_vla addition (2026-08-31, agentview_vjepa2_temporal):
            # maintain a running REAL V-JEPA2-AC latent every env step
            # (not just replan steps). Trusts the real observed encoding
            # whenever the target is currently unoccluded (same 0.05
            # threshold as occluded_run_length above); otherwise
            # propagates the last trusted latent forward ONE step via the
            # real AC predictor, conditioned on the REAL action actually
            # executed last step (pure state estimation, no planning/
            # hypothetical actions -- this project's own CEM+MPC script,
            # by contrast, DOES plan hypothetical actions; this is a
            # different, simpler use of the same predictor). Untrained
            # vjepa2_proj_dino/_siglip (see CLAUDE.md) mean the content
            # this ultimately injects is not expected to be meaningful --
            # this loop only tests correct wiring, not quality.
            if agentview_vjepa2_temporal:
                with torch.no_grad():
                    vjepa2_device = next(vjepa2_encoder.parameters()).device
                    frame_arr = np.asarray(Image.fromarray(agentview_color).resize((256, 256)))
                    frame_norm = (frame_arr.astype(np.float32) / 255.0 - 0.5) / 0.5
                    frame_t = torch.from_numpy(frame_norm).permute(2, 0, 1).to(vjepa2_device)
                    if vjepa2_prev_frame_tensor is None:
                        vjepa2_prev_frame_tensor = frame_t
                    vjepa2_clip = torch.stack([vjepa2_prev_frame_tensor, frame_t], dim=1).unsqueeze(0)
                    vjepa2_z_real = vjepa2_encoder(vjepa2_clip)[0]  # (256,1408)
                    vjepa2_prev_frame_tensor = frame_t
                    if frac_occluded_this_step <= 0.05 or vjepa2_latent_state is None:
                        vjepa2_latent_state = vjepa2_z_real
                    elif vjepa2_last_action is not None:
                        vjepa2_a_t = torch.from_numpy(vjepa2_last_action).float().to(vjepa2_device).view(1, 1, 7)
                        vjepa2_eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                        vjepa2_eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
                        vjepa2_axang = quat2axisangle_vjepa2(vjepa2_eef_quat).astype(np.float32)
                        vjepa2_gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
                        vjepa2_state_vec = np.concatenate(
                            [vjepa2_eef_pos, vjepa2_axang, vjepa2_gripper_qpos[:1]]
                        ).astype(np.float32)
                        vjepa2_s_t = torch.from_numpy(vjepa2_state_vec).to(vjepa2_device).view(1, 1, 7)
                        vjepa2_latent_state = vjepa2_predictor(
                            vjepa2_latent_state.unsqueeze(0), vjepa2_a_t, vjepa2_s_t
                        )[0]

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
                        # occ_vla addition (2026-08-29, per user request --
                        # "if there were no occlusion, what would this look
                        # like"): a REAL re-render of the identical sim state,
                        # occluder geoms alpha-zeroed then restored -- the
                        # same technique this file already uses for
                        # composite_visual_only/oracle content (see
                        # sim.model.geom_rgba[occluder_geom_ids, 3] = 0.0
                        # elsewhere in this function). Not a generated/
                        # hallucinated image -- genuine MuJoCo geometry with
                        # the occluder made invisible, real background behind
                        # it. Only meaningful if occluder_geom_ids identifies
                        # a real scene occluder (book/box/etc, not the arm
                        # itself, which self-occludes via robot_geom_ids --
                        # this doesn't hide the arm).
                        if occluder_geom_ids:
                            orig_alpha_illustrate = sim.model.geom_rgba[occluder_geom_ids, 3].copy()
                            sim.model.geom_rgba[occluder_geom_ids, 3] = 0.0
                            sim.forward()
                            clean_reference, _ = get_agentview_frames(env, resize_size)
                            sim.model.geom_rgba[occluder_geom_ids, 3] = orig_alpha_illustrate
                            sim.forward()
                            Image.fromarray(clean_reference).save(
                                os.path.join(record_video_dir, f"frame_{t:05d}_clean_reference.png"))

                # occ_vla addition (2026-08-31, agentview_vjepa2_temporal):
                # same sustained-occlusion gate convention as agentview_vjepa
                # (occluded_run_length >= threshold, not a single noisy
                # frame), but injects the REAL V-JEPA2 latent (maintained
                # every env step above) at the FINAL patch-token layer via
                # vjepa2_splice_forward, instead of gray-filling pixels +
                # relying on the (confirmed always-zero-output, untrained)
                # FiLM+cross-attention module agentview_vjepa uses. Sets
                # the diagnostic attributes vjepa2_splice_forward reads;
                # cleared right after the policy call so they can't leak
                # into any OTHER forward call this same step (e.g. the
                # log_action_diff counterfactual, which explicitly swaps
                # back to original_forward anyway, but this is a second,
                # independent guard).
                agentview_vjepa2_engaged_this_step = False
                if agentview_vjepa2_temporal or agentview_vjepa2_amodal:
                    # occ_vla note: unconditionally clear first, so a step
                    # that doesn't re-engage never accidentally reuses a
                    # STALE mask/content from a previous engaged step
                    # (vjepa2_splice_forward is left active for the whole
                    # episode, gated only by these attributes being
                    # non-None -- see make_agentview_vjepa2_temporal_splice_forward).
                    model.vision_backbone._diagnostic_vjepa2_dino_content = None
                    model.vision_backbone._diagnostic_vjepa2_siglip_content = None
                    model.vision_backbone._diagnostic_agentview_patch_mask_256 = None
                    model.vision_backbone._diagnostic_vjepa2_blend_alpha = None
                    # occ_vla addition (2026-08-31, per user's AI-researcher
                    # diagnosis request): persistence-ESCALATE schedule for
                    # blend_alpha -- diagnosed from real per-episode data
                    # (all 5 dual-camera failures at alpha=1.0/0.3 showed
                    # occluded_run_length pinned at its max for the WHOLE
                    # episode, unlike successes' varying/lower values) --
                    # same closed-form schedule already validated for CBF's
                    # v4-escalate persistence gate: start at a LOW floor
                    # (trust the fabricated content least right when
                    # occlusion just began, since the model may still be
                    # using real wrist-camera compensation at that point)
                    # and escalate toward the target ceiling only if
                    # occlusion genuinely PERSISTS.
                    if vjepa2_blend_persistence_window > 0:
                        f_t = min(1.0, occluded_run_length / vjepa2_blend_persistence_window)
                        dynamic_alpha = vjepa2_blend_alpha_floor + f_t * (
                            vjepa2_blend_alpha_ceiling - vjepa2_blend_alpha_floor
                        )
                        model.vision_backbone._diagnostic_vjepa2_blend_alpha = dynamic_alpha
                if (agentview_vjepa2_temporal and occluded_run_length >= agentview_vjepa_min_run_length
                        and occluded_pixel_mask.any() and vjepa2_latent_state is not None):
                    token_mask_256_v2 = pixel_mask_to_token_mask_256(occluded_pixel_mask)
                    if token_mask_256_v2.any():
                        agentview_vjepa2_engaged_this_step = True
                        with torch.no_grad():
                            dino_content = vjepa2_proj_dino(
                                vjepa2_latent_state.to(torch.bfloat16)
                            ).unsqueeze(0)
                            siglip_content = vjepa2_proj_siglip(
                                vjepa2_latent_state.to(torch.bfloat16)
                            ).unsqueeze(0)
                        model.vision_backbone._diagnostic_vjepa2_dino_content = dino_content
                        model.vision_backbone._diagnostic_vjepa2_siglip_content = siglip_content
                        model.vision_backbone._diagnostic_agentview_patch_mask_256 = torch.from_numpy(
                            token_mask_256_v2
                        ).to(model.device)
                # occ_vla addition (2026-08-31, per user's engineering
                # request): agentview_vjepa2_amodal -- real V-JEPA2 SPATIAL
                # masked-patch completion (base, non-AC predictor, no
                # temporal history needed at all -- sidesteps the
                # persistent-occlusion-since-frame-1 problem that
                # agentview_vjepa2_temporal structurally cannot solve),
                # followed by SPADE/AdaIN-style local statistical
                # rescaling (see vjepa2_local_adain's own docstring for
                # the diagnosed regression-to-the-mean root cause and why
                # local, not global, rescaling was chosen).
                if (agentview_vjepa2_amodal and occluded_run_length >= agentview_vjepa_min_run_length
                        and occluded_pixel_mask.any()):
                    token_mask_256_v3 = pixel_mask_to_token_mask_256(occluded_pixel_mask)
                    if token_mask_256_v3.any() and (~token_mask_256_v3).any():
                        agentview_vjepa2_engaged_this_step = True
                        with torch.no_grad():
                            vjepa2_device_amodal = next(vjepa2_base_encoder.parameters()).device
                            frame_arr_amodal = np.asarray(Image.fromarray(agentview_color).resize((256, 256)))
                            frame_norm_amodal = (frame_arr_amodal.astype(np.float32) / 255.0 - 0.5) / 0.5
                            frame_t_amodal = torch.from_numpy(frame_norm_amodal).permute(2, 0, 1).to(vjepa2_device_amodal)
                            clip_amodal = torch.stack([frame_t_amodal, frame_t_amodal], dim=1).unsqueeze(0)
                            z_real_full_amodal = vjepa2_base_encoder(clip_amodal)[0]
                            completed_amodal = vjepa2_amodal_complete(
                                vjepa2_base_encoder, vjepa2_base_predictor, clip_amodal,
                                token_mask_256_v3, vjepa2_device_amodal,
                            )
                            occluded_idx_amodal_full = np.flatnonzero(token_mask_256_v3)
                            unoccluded_idx_amodal = np.flatnonzero(~token_mask_256_v3)
                            pred_at_occ_amodal_full = completed_amodal[occluded_idx_amodal_full]
                            # occ_vla addition (2026-09-01): confidence-gated
                            # selective injection -- see vjepa2_confidence_mask's
                            # own docstring. Computed on the RAW (pre-AdaIN)
                            # completion, since rescaling would otherwise make
                            # every token look locally plausible regardless of
                            # its real underlying quality. threshold=-1.0
                            # (default) keeps every token, byte-identical to
                            # every prior test of this condition.
                            if vjepa2_confidence_threshold > -1.0:
                                confident_mask = vjepa2_confidence_mask(
                                    pred_at_occ_amodal_full, occluded_idx_amodal_full,
                                    z_real_full_amodal, unoccluded_idx_amodal,
                                    threshold=vjepa2_confidence_threshold,
                                )
                            else:
                                confident_mask = np.ones(len(occluded_idx_amodal_full), dtype=bool)
                            if vjepa2_confidence_threshold > -1.0:
                                print(f"    [conf-gate debug] t={t} kept={int(confident_mask.sum())}/{len(confident_mask)}")
                            occluded_idx_amodal = occluded_idx_amodal_full[confident_mask]
                            pred_at_occ_amodal = pred_at_occ_amodal_full[confident_mask]
                            token_mask_256_v3 = np.zeros_like(token_mask_256_v3)
                            token_mask_256_v3[occluded_idx_amodal] = True
                            full_amodal_latent = torch.zeros(256, 1408, device=vjepa2_device_amodal, dtype=torch.float32)
                            if len(occluded_idx_amodal) > 0:
                                rescaled_amodal = vjepa2_local_adain(
                                    pred_at_occ_amodal, occluded_idx_amodal, z_real_full_amodal, unoccluded_idx_amodal,
                                )
                                full_amodal_latent[occluded_idx_amodal] = rescaled_amodal
                            # occ_vla addition (2026-08-31): temporal EMA
                            # smoothing (decay=0.5, a reasonable untuned
                            # first value, not swept) at the currently-
                            # occluded positions, against the running
                            # per-episode buffer -- reduces step-to-step
                            # flicker in the injected content itself,
                            # independent of its underlying quality.
                            # occ_vla addition (2026-09-01): made this a real
                            # parameter (vjepa2_amodal_ema_decay), default
                            # 0.5 unchanged -- per user hypothesis that EMA
                            # staleness during a fast dynamic grasp-approach
                            # phase (Goal task7's severe regression) may be
                            # forcing a temporally-inconsistent (ghost-
                            # position) completion onto the policy; decay=0.0
                            # degenerates to the pure instantaneous value
                            # (no history mixed in at all) for this ablation.
                            ema_decay = vjepa2_amodal_ema_decay
                            if vjepa2_amodal_ema is None:
                                vjepa2_amodal_ema = full_amodal_latent.clone()
                            else:
                                smoothed = torch.zeros_like(full_amodal_latent)
                                smoothed[occluded_idx_amodal] = (
                                    ema_decay * vjepa2_amodal_ema[occluded_idx_amodal]
                                    + (1 - ema_decay) * full_amodal_latent[occluded_idx_amodal]
                                )
                                vjepa2_amodal_ema[occluded_idx_amodal] = smoothed[occluded_idx_amodal]
                                full_amodal_latent = vjepa2_amodal_ema
                            dino_content = vjepa2_proj_dino(full_amodal_latent.to(torch.bfloat16)).unsqueeze(0)
                            siglip_content = vjepa2_proj_siglip(full_amodal_latent.to(torch.bfloat16)).unsqueeze(0)
                        model.vision_backbone._diagnostic_vjepa2_dino_content = dino_content
                        model.vision_backbone._diagnostic_vjepa2_siglip_content = siglip_content
                        model.vision_backbone._diagnostic_agentview_patch_mask_256 = torch.from_numpy(
                            token_mask_256_v3
                        ).to(model.device)

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

                # occ_vla addition (2026-08-30, per user's explicit request
                # for a methodologically cleaner "agentview-only" test than
                # blank_wrist's gray-fill OOD input): drop_wrist_image
                # genuinely removes the wrist image from the model's input
                # entirely (cfg.num_images_in_input=1 -> get_vla_action's
                # own `if cfg.num_images_in_input > 1` check skips the wrist
                # image altogether; model.vision_backbone's own
                # num_images_in_input is toggled the same way, since its
                # forward() splits pixel_values by this count) rather than
                # feeding an anomalous uniform-gray frame the checkpoint was
                # never trained to expect. Both are restored immediately
                # after the call so no other condition/step in this episode
                # is affected -- this state is otherwise process-global
                # (shared model object, shared cfg object across all calls).
                if drop_wrist_image:
                    cfg.num_images_in_input = 1
                    model.vision_backbone.set_num_images_in_input(1)
                try:
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
                    elif attn_target_excl_enabled:
                        # occ_vla addition (2026-09-01), KNOWS-style
                        # (arXiv:2606.09749) attention-based target
                        # identification -- see _attention_target_id's
                        # docstring above for the full grounding/caveats.
                        # Same call as the plain branch below, just also
                        # requesting the raw per-patch attention map (no
                        # extra forward pass -- computed from the SAME
                        # output_attentions=True call).
                        actions, step_attn_map = get_vla_action(
                            cfg, model, processor, observation, task_description,
                            action_head=action_head, proprio_projector=proprio_projector,
                            noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=occlusion_mask,
                            return_attn_map=True,
                        )
                        attn_identified_target_id, attn_target_debug = _attention_target_id(step_attn_map, obs)
                        if os.environ.get("ATTN_TARGET_DEBUG"):
                            print(f"    [attn-target-debug] t={t} identified_target={attn_identified_target_id} "
                                  f"{attn_target_debug}")
                    else:
                        object_mask_t = None
                        if object_centric_adapter_enabled:
                            # occ_vla addition (2026-09-03, Month 2): build the
                            # real per-patch object-of-interest mask fresh every
                            # replan step, same grid convention/segmentation
                            # source (target_seg_ids, agentview_seg) already
                            # used to SAVE this same mask during training-data
                            # collection -- see run_libero_occluded_oracle_headroom.py's
                            # save_distillation_pairs_dir block.
                            target_pixel_mask = (
                                np.isin(agentview_seg, target_seg_ids) if target_seg_ids
                                else np.zeros_like(agentview_seg, dtype=bool)
                            )
                            agent_mask_256 = pixel_mask_to_token_mask_256(target_pixel_mask).astype(np.float32)
                            # occ_vla fix (2026-09-03, per user's VIM-style
                            # agentview-only request): when drop_wrist_image
                            # is active, cfg.num_images_in_input==1, so
                            # projected_features only has 256 real tokens --
                            # a 512-length object_mask would leave 256
                            # "phantom" key/value positions in the adapter's
                            # cross-attention with no corresponding real
                            # visual token. Match the mask length to the
                            # real token count instead of always assuming 2
                            # images.
                            if drop_wrist_image:
                                object_mask_np = agent_mask_256[:, None]
                            else:
                                wrist_mask_256 = np.zeros_like(agent_mask_256)
                                object_mask_np = np.concatenate([agent_mask_256, wrist_mask_256])[:, None]
                            object_mask_t = torch.tensor(object_mask_np, device=model.device, dtype=torch.bfloat16).unsqueeze(0)
                        actions = get_vla_action(
                            cfg, model, processor, observation, task_description,
                            action_head=action_head, proprio_projector=proprio_projector,
                            noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=occlusion_mask,
                            object_mask=object_mask_t,
                        )
                finally:
                    if drop_wrist_image:
                        cfg.num_images_in_input = 2
                        model.vision_backbone.set_num_images_in_input(2)

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
                chunk_correction_fired = False  # occ_vla addition (2026-08-27): set True in whichever branch below actually modifies `actions`, read by the distillation-pairs save hook after this block
                if os.environ.get("CBF_DEBUG"):
                    print(f"    [cbf-debug] t={t} gate_check: proactive_avoidance_oracle={proactive_avoidance_oracle} "
                          f"occluder_geom_ids={bool(occluder_geom_ids)} proactive_use_depth={proactive_use_depth} "
                          f"proactive_use_mpc={proactive_use_mpc} proactive_use_cbf={proactive_use_cbf}")
                if proactive_avoidance_oracle and (occluder_geom_ids or proactive_use_depth or proactive_use_mpc):
                    OSC_POSE_MAX_DELTA_M = 0.05  # confirmed via robosuite.controllers.load_controller_config(default_controller="OSC_POSE") -- this env never overrides controller_configs
                    if proactive_use_depth:
                        # occ_vla addition (2026-08-24, Phase 2): obstacle source is a
                        # real RGB-D point cloud, not sim.data.geom_xpos[occluder_geom_ids].
                        # Each point is treated as a near-zero-radius obstacle (radius
                        # 0.01m -- a small margin for the point-sampling itself, not an
                        # object-size estimate, since individual points have no "size").
                        occ_centers = _depth_obstacle_points(obs)
                        if os.environ.get("CBF_DEBUG"):
                            print(f"    [cbf-debug] t={t} n_depth_obstacle_pts={len(occ_centers)}")
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
                        # occ_vla addition (2026-08-29, per user's "local
                        # attractor" proposal): a real, zero-privileged
                        # (depth+segmentation, not sim ground truth) estimate
                        # of the target's own 3D position, computed ONCE per
                        # chunk -- diagnosed root cause (see run notes on
                        # libero_object task1/task4/task7): the target itself
                        # is already correctly excluded from occ_centers
                        # (_depth_obstacle_points already filters
                        # target_seg_ids out), so this is NOT a
                        # target-misclassified-as-obstacle bug -- it's that a
                        # fixed safety margin around REAL nearby obstacle
                        # geometry (e.g. a wine rack shelf) can geometrically
                        # overlap the space the gripper must occupy to reach a
                        # target sitting right next to/inside that geometry.
                        # Disabled by default (radius=0.0) -- zero effect on
                        # every existing condition/caller unless explicitly
                        # enabled.
                        # occ_vla addition (2026-08-29, per user's follow-up
                        # "persist target position across occluded frames"
                        # proposal): a plain state cache (NOT a learned
                        # memory/retrieval module -- "MemoryVLA++"-style
                        # architectures are out of scope, see CLAUDE.md run
                        # notes) so the attractor doesn't silently disable
                        # itself the instant the target drops out of view for
                        # one frame. Bounded by max_staleness (replan cycles,
                        # not env-steps) so an old estimate isn't trusted
                        # forever if the target could plausibly have moved
                        # (e.g. once actually grasped and being carried).
                        target_centroid_for_attractor = None
                        in_grasp_phase = False
                        if proactive_use_depth and proactive_target_attractor_radius_m > 0:
                            fresh_centroid = _depth_target_centroid(obs)
                            if fresh_centroid is not None:
                                last_known_target_centroid = fresh_centroid
                                last_known_target_centroid_age = 0
                            elif last_known_target_centroid is not None and last_known_target_centroid_age is not None:
                                last_known_target_centroid_age += 1
                            if (last_known_target_centroid is not None and
                                    (proactive_target_attractor_max_staleness <= 0 or
                                     last_known_target_centroid_age is None or
                                     last_known_target_centroid_age <= proactive_target_attractor_max_staleness)):
                                target_centroid_for_attractor = last_known_target_centroid
                                # occ_vla addition, per user's "task-phase
                                # dependent dynamic margin" proposal (simplified:
                                # a real distance-to-target proxy for "grasp
                                # phase" instead of a learned 3D-scene-graph
                                # phase classifier, which doesn't exist in this
                                # project): once the (real, zero-privileged)
                                # end-effector is within this radius of the
                                # target, soften the WHOLE chunk's correction
                                # gain, not just the per-point decay near the
                                # target below -- the two mechanisms compound.
                                d_eef_to_target = float(np.linalg.norm(predicted_pos - target_centroid_for_attractor))
                                in_grasp_phase = d_eef_to_target < proactive_grasp_phase_radius_m

                        effective_cbf_gain = (
                            proactive_cbf_gain * proactive_grasp_phase_gain_decay
                            if in_grasp_phase else proactive_cbf_gain
                        )

                        # occ_vla addition (2026-09-01, per user's ACE-gate
                        # request): scale the CBF gain by the base policy's
                        # OWN ensemble_disagreement this step -- the working
                        # hypothesis (task9's real diagnosed mechanism,
                        # 286.6 corrections/episode with baseline contact
                        # rate ~0%) is that CBF misfires specifically when
                        # the base policy is already confident/stable (low
                        # disagreement under pixel perturbation), i.e. it is
                        # NOT the policy that's uncertain -- the geometric
                        # detector alone is wrong. ace_scale=1.0 (full CBF
                        # trust) once disagreement reaches ace_gate_scale_m;
                        # linearly down to ace_gate_min_frac (never fully
                        # zero -- a real geometric violation should still get
                        # SOME correction even if the policy looks confident)
                        # as disagreement -> 0. ace_gate_scale_m is a raw
                        # normalized-action-space L2 distance, NOT calibrated
                        # yet -- default 0.05 is a first guess pending a real
                        # calibration pass (see scripts_figures/ or CLAUDE.md
                        # for the calibration run this default should be
                        # replaced from).
                        if ace_gate_enabled and disagreement is not None:
                            ace_frac = min(1.0, disagreement / ace_gate_scale_m) if ace_gate_scale_m > 0 else 1.0
                            ace_scale = ace_gate_min_frac + (1.0 - ace_gate_min_frac) * ace_frac
                            effective_cbf_gain = effective_cbf_gain * ace_scale
                            if os.environ.get("ACE_GATE_DEBUG"):
                                print(f"    [ace-gate-debug] t={t} disagreement={disagreement:.5f} "
                                      f"ace_frac={ace_frac:.3f} ace_scale={ace_scale:.3f} "
                                      f"effective_cbf_gain={effective_cbf_gain:.4f}")

                        # occ_vla addition (2026-08-30, per user's "continuous
                        # proximity / persistence-gated correction" proposal --
                        # implemented as a hand-crafted geometric heuristic
                        # rather than a learned discriminator, per the user's
                        # own choice of the higher-generalization option: with
                        # only ~40 task-level labeled examples available from
                        # the full sweep, a learned classifier risks
                        # overfitting to those specific tasks, whereas this
                        # needs zero training data and applies zero-shot via
                        # the same real depth+segmentation signals already
                        # used everywhere else in this file).
                        #
                        # Root cause being targeted (see CLAUDE.md's
                        # libero_object task7 run notes): the diagnosed
                        # failure signature is NOT one bad correction -- it's
                        # the SAME margin violation re-triggering across many
                        # consecutive replans without ever resolving (t=82-230
                        # in one logged episode). That recurrence is itself
                        # evidence the correction is fighting an intrinsic
                        # near-target geometry (reaching for something sitting
                        # right next to the occluder) rather than resolving a
                        # genuine one-off collision risk -- correct-perturb-
                        # re-trigger-escalate is the loop already diagnosed as
                        # the failure mechanism. So: a violation that resolves
                        # within a replan or two keeps full gain (streak==0 ->
                        # scale==1.0, byte-identical to prior behavior). A
                        # violation that keeps re-triggering for many replans
                        # in a row is a signal the correction isn't working --
                        # trust in it DECAYS the longer it persists, rather
                        # than escalating, directly breaking that loop instead
                        # of feeding it. Disabled by default (window<=0) --
                        # zero effect on every existing condition/caller
                        # unless explicitly enabled.
                        # occ_vla addition (2026-08-30, v3 result follow-up):
                        # v3 (decay-with-persistence, above) tested at n=10 on
                        # task7 and came back BYTE-IDENTICAL to v1/v2 (same
                        # 2/10, same 2 episodes) -- because every episode's
                        # FIRST correction necessarily fires at streak=0 (full,
                        # undecayed gain; a persistence gate can only detect
                        # persistence AFTER a violation has already recurred),
                        # so if the trajectory-derailing damage happens at that
                        # first correction, no later decay can undo it. Added
                        # `proactive_persistence_mode="escalate"` as the
                        # opposite polarity to test that hypothesis directly:
                        # start at the LOW floor gain unconditionally (so a
                        # first-encounter, possibly-brief-and-safe proximity is
                        # barely corrected at all) and ramp UP toward full gain
                        # only once the same violation has genuinely persisted
                        # for --proactive-persistence-window replans in a row.
                        # "decay" (default) preserves the original v3 behavior
                        # exactly -- zero change to already-reported results.
                        if proactive_persistence_window > 0:
                            frac = min(1.0, violation_streak / proactive_persistence_window)
                            if proactive_persistence_mode == "escalate":
                                persistence_scale = proactive_persistence_min_gain_frac + frac * (1.0 - proactive_persistence_min_gain_frac)
                            else:
                                persistence_scale = 1.0 - frac * (1.0 - proactive_persistence_min_gain_frac)
                        else:
                            persistence_scale = 1.0
                        effective_cbf_gain = effective_cbf_gain * persistence_scale

                        # Update the streak for the NEXT chunk based on
                        # whether THIS chunk's original (pre-correction)
                        # trajectory already violates the margin at its start
                        # position -- a lightweight, chunk-level proxy
                        # consistent with how corrections are already tracked
                        # per-chunk elsewhere in this function.
                        dists0 = np.linalg.norm(occ_centers - predicted_pos[None, :], axis=1) - occ_radii
                        chunk_would_violate = bool(np.any(dists0 < PROACTIVE_SAFETY_MARGIN_M))
                        violation_streak = violation_streak + 1 if chunk_would_violate else 0

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
                                v_min_normal = effective_cbf_gain * (PROACTIVE_SAFETY_MARGIN_M - dist)  # >0, scales with penetration depth; softened chunk-wide if in_grasp_phase
                                if target_centroid_for_attractor is not None:
                                    # Linear decay: full strength (factor=1.0) at
                                    # distance >= R from the target, decaying to
                                    # proactive_target_attractor_decay AT the
                                    # target's own position (distance=0). Only
                                    # weakens the correction near a real,
                                    # confirmed-visible target -- never near
                                    # occluder geometry the target isn't next to.
                                    d_to_target = float(np.linalg.norm(candidate_pos - target_centroid_for_attractor))
                                    frac = min(1.0, d_to_target / proactive_target_attractor_radius_m)
                                    decay_factor = proactive_target_attractor_decay + (1.0 - proactive_target_attractor_decay) * frac
                                    v_min_normal *= decay_factor
                                if v_normal < v_min_normal:
                                    deficit = v_min_normal - v_normal
                                    a_xyz = a_xyz + deficit * n_hat  # only the unsafe normal component is topped up; tangential intent untouched
                                    actions_arr[step_i, :3] = a_xyz
                                    n_corrected_this_chunk += 1
                            # propagate using the (possibly-corrected) action for THIS step,
                            # so later steps in the chunk see where the corrected trajectory
                            # actually goes, not the original uncorrected one.
                            predicted_pos = predicted_pos + a_xyz * OSC_POSE_MAX_DELTA_M
                        if os.environ.get("CBF_DEBUG"):
                            print(f"    [cbf-debug] t={t} cbf-v2 per-chunk loop done: n_corrected_this_chunk={n_corrected_this_chunk} "
                                  f"occ_centers[0]={occ_centers[0] if len(occ_centers) else None} nearest_dist0={float(np.linalg.norm(occ_centers - predicted_pos[None,:], axis=1).min() - occ_radii[0]) if len(occ_centers) else None}")
                        if n_corrected_this_chunk > 0:
                            actions = actions_arr
                            proactive_correction_applied_count += n_corrected_this_chunk
                            proactive_correction_ts.append(t)
                            chunk_correction_fired = True
                            print(f"    [proactive-avoidance-cbf] chunk at t={t}: minimal-norm safety "
                                  f"correction applied to {n_corrected_this_chunk}/{len(actions_arr)} steps "
                                  f"(gain={effective_cbf_gain}{'  [grasp-phase-softened]' if in_grasp_phase else ''}"
                                  f"{f'  [persistence-{proactive_persistence_mode}-streak={violation_streak} scale={persistence_scale:.2f}]' if proactive_persistence_window > 0 else ''})")

                # occ_vla addition (2026-08-27, per user's explicit "no
                # privileged information" requirement): save
                # (real agentview image, proprio state, FINAL corrected
                # action chunk) pairs for a later imitation-learning
                # distillation of proactive_avoidance_depth's real-RGB-D
                # correction back into the policy itself -- gated on
                # proactive_use_depth specifically (NOT plain CBF, which
                # uses privileged sim.data.geom_xpos) so the "teacher"
                # signal collected here is itself real-robot-deployable,
                # not just the final distilled policy. Saves every
                # replan step (not just corrected ones) -- when no
                # correction fires, `actions` already equals the VLA's
                # own original chunk, which is itself a valid (label ==
                # input) imitation target, not a gap in the data.
                if save_distillation_pairs_dir is not None and proactive_use_depth:
                    os.makedirs(save_distillation_pairs_dir, exist_ok=True)
                    uid = f"task{task_id}_ep{episode_idx}_t{t:05d}"
                    Image.fromarray(agentview_color).save(
                        os.path.join(save_distillation_pairs_dir, f"{uid}_agentview.png"))
                    Image.fromarray(wrist_img).save(
                        os.path.join(save_distillation_pairs_dir, f"{uid}_wrist.png"))
                    # occ_vla addition (2026-09-03, Month 2): also save the real
                    # per-patch target-object coverage mask (same
                    # pixel_mask_to_token_mask_256 grid convention already used
                    # throughout this file for occlusion_mask), for
                    # ObjectCentricZeroInitAdapter training -- built from real
                    # agentview segmentation (target_seg_ids), zero new
                    # perception dependency. Wrist slot left all-zero (target
                    # object grounding is an agentview-side concept here,
                    # matching this project's own established occlusion_mask
                    # convention of gating a single camera's 256-token block).
                    target_pixel_mask = (
                        np.isin(agentview_seg, target_seg_ids) if target_seg_ids
                        else np.zeros_like(agentview_seg, dtype=bool)
                    )
                    object_token_mask_256 = pixel_mask_to_token_mask_256(target_pixel_mask)
                    mask_path = f"{uid}_objmask.npy"
                    np.save(os.path.join(save_distillation_pairs_dir, mask_path), object_token_mask_256)
                    distillation_manifest.append({
                        "uid": uid, "task_id": task_id, "episode": episode_idx, "t": t,
                        "agentview_path": f"{uid}_agentview.png", "wrist_path": f"{uid}_wrist.png",
                        "object_mask_path": mask_path,
                        "object_mask_coverage_frac": float(object_token_mask_256.mean()),
                        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                  obs["robot0_gripper_qpos"])).tolist(),
                        "action_corrected": np.asarray(actions[0], dtype=float).tolist(),
                        "correction_applied_this_chunk": chunk_correction_fired,
                    })

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

            if agentview_vjepa2_temporal:
                # occ_vla addition (2026-08-31): capture the REAL action
                # about to be executed (pre-process_action, same raw
                # 7-dim normalized convention used throughout this
                # condition's own state/action vectors) -- consumed by
                # the NEXT loop iteration's latent-propagation step above.
                vjepa2_last_action = np.asarray(action, dtype=np.float32).copy()

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
        "distillation_manifest": distillation_manifest,
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
    parser.add_argument("--load-object-centric-adapter", default=None,
                         help="occ_vla addition 2026-09-03 (Month 2): path to a directory containing "
                              "object_centric_adapter_weights.pt, saved by "
                              "scripts/train_object_centric_adapter.py -- attaches a trained "
                              "ObjectCentricZeroInitAdapter (base model 100% untouched otherwise, "
                              "unlike --load-distillation-lora). Evaluate WITHOUT any "
                              "proactive_avoidance_* condition active, same convention as "
                              "--load-distillation-lora -- the object_mask is built fresh every "
                              "replan step from real segmentation (target_seg_ids), gated on this flag "
                              "being set, regardless of which `condition` string is running.")
    parser.add_argument("--load-distillation-lora", default=None,
                         help="occ_vla addition 2026-08-27: path to a directory containing "
                              "distillation_weights.pt, saved by scripts/train_distillation_imitation.py "
                              "(imitation distillation of proactive_avoidance_depth's zero-privileged "
                              "correction into language_model LoRA + action_head). Evaluate this WITHOUT "
                              "any proactive_avoidance_* condition active -- the point is to test whether "
                              "avoidance behavior now emerges intrinsically, not to stack it on top of the "
                              "same CBF safety layer used to generate its training data.")
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
    parser.add_argument("--proactive-target-attractor-radius-m", type=float, default=0.0,
                         help="occ_vla addition 2026-08-29, per user's 'local attractor' proposal for the "
                              "libero_object task1/task4/task7 degradation: radius (meters) around the "
                              "target's own real (depth+segmentation, zero-privileged) position within which "
                              "the CBF safety margin is progressively decayed, since a fixed margin around "
                              "real nearby obstacle geometry can otherwise overlap the space needed to reach "
                              "a target sitting close to it. 0.0 (default) disables this entirely -- zero "
                              "effect on any existing run/condition unless explicitly set.")
    parser.add_argument("--proactive-target-attractor-decay", type=float, default=0.1,
                         help="occ_vla addition 2026-08-29: minimum decay factor applied to the safety-margin "
                              "correction strength AT the target's own position (candidate_pos distance=0 from "
                              "target centroid) -- e.g. 0.1 means the correction is reduced to 10% strength "
                              "right at the target, linearly ramping back to 100% at "
                              "--proactive-target-attractor-radius-m away. Only takes effect if that radius > 0.")
    parser.add_argument("--proactive-target-attractor-max-staleness", type=int, default=50,
                         help="occ_vla addition 2026-08-29: max number of REPLAN CYCLES (not env-steps) a "
                              "cached last-known target position (from a prior step where the target WAS "
                              "visible) is trusted for the attractor decay, once the target drops out of "
                              "view. <=0 means never expire (trust forever, risky once the target might have "
                              "moved e.g. after being grasped). Only relevant if "
                              "--proactive-target-attractor-radius-m > 0.")
    parser.add_argument("--proactive-grasp-phase-radius-m", type=float, default=0.0,
                         help="occ_vla addition 2026-08-29, per user's '3DSG task-phase-dependent dynamic "
                              "margin' proposal, simplified to a distance-to-target proxy (no learned scene "
                              "graph/phase classifier exists in this project): once the end-effector is "
                              "within this radius of the (possibly cached, see max-staleness) target "
                              "position, --proactive-grasp-phase-gain-decay is applied to the CBF gain for "
                              "the WHOLE chunk, not just points near the target (compounds with the separate "
                              "--proactive-target-attractor-radius-m per-point decay). 0.0 (default) disables.")
    parser.add_argument("--proactive-grasp-phase-gain-decay", type=float, default=1.0,
                         help="occ_vla addition 2026-08-29: multiplier applied to proactive_cbf_gain for the "
                              "whole chunk once in the grasp phase (see --proactive-grasp-phase-radius-m). "
                              "1.0 (default) = no effect.")
    parser.add_argument("--attn-target-excl", action="store_true",
                         help="occ_vla addition 2026-09-01, faithful-subset reimplementation of KNOWS "
                              "(arXiv:2606.09749, 'Your Model Already Knows: Attention-Guided Safety Filter "
                              "for VLA Models', Park et al., UCLA -- no code release found): identifies the "
                              "object the policy is CURRENTLY attending to (action-query x vision-key "
                              "attention, last transformer layer, mean-pooled over heads/chunk positions -- "
                              "this project's own OpenVLA-OFT checkpoint has not been profiled for a "
                              "specific best (layer,head) the way KNOWS profiled pi0.5's layer 12/head 3) "
                              "and EXCLUDES it from the CBF's real depth-based obstacle set every replan "
                              "step -- directly targets task9's diagnosed misfire (CBF treating the "
                              "destination receptacle as an obstacle). REQUIRES --attn-implementation eager "
                              "for the whole rollout (this project's own 2026-08-19 finding: mixing "
                              "output_attentions=True/False calls within an episode silently flips 8/20 "
                              "outcomes via an SDPA->eager switch) -- any baseline compared against this "
                              "condition must ALSO run under --attn-implementation eager.")
    parser.add_argument("--attn-target-window", type=int, default=5,
                         help="occ_vla addition 2026-09-01: sliding window (replan steps) over which "
                              "per-object attention mass/area are accumulated before computing density, "
                              "matching KNOWS' own K parameter (their exact value was in an appendix not "
                              "captured by this session's fetch -- 5 is a reasonable first guess, not "
                              "calibrated against real data yet).")
    parser.add_argument("--attn-target-gap-delta", type=float, default=0.0,
                         help="occ_vla addition 2026-09-01: minimum density lead the top-ranked object "
                              "must have over the second-ranked one to be confirmed as the target (KNOWS' "
                              "own delta, gap threshold, exact value also in their appendix, not captured). "
                              "0.0 (default) = always pick the argmax, i.e. no 'not confident enough, "
                              "exclude nothing' fallback yet -- a real calibration pass against this "
                              "project's own attention-density distribution is needed before trusting a "
                              "nonzero value.")
    parser.add_argument("--ace-gate", action="store_true",
                         help="occ_vla addition 2026-09-01, per user's 'VLA自身のアテンション/ACEでCBFの介入を"
                              "ゲートする' request: scale the CBF correction gain by the base policy's OWN "
                              "ensemble_disagreement (perturbed-pixel re-forward-pass L2 action distance -- "
                              "real-robot-safe, no output_attentions, no privileged info; NOT attention "
                              "entropy, which is known to contaminate the rollout by forcing eager attention, "
                              "see the --attn-implementation entry above). Automatically forces "
                              "log_ensemble_disagreement=True. Working hypothesis: CBF misfires (task9: 286.6 "
                              "corrections/episode, ~0% baseline contact) specifically when the base policy is "
                              "already confident (low disagreement) -- trust CBF less in that regime, more "
                              "when the policy itself looks uncertain.")
    parser.add_argument("--ace-gate-scale-m", type=float, default=0.05,
                         help="occ_vla addition 2026-09-01: normalized-action-space L2 disagreement value at "
                              "which --ace-gate reaches full CBF trust (ace_scale=1.0). NOT yet calibrated "
                              "against real disagreement values -- run a calibration pass first (see CLAUDE.md).")
    parser.add_argument("--ace-gate-min-frac", type=float, default=0.15,
                         help="occ_vla addition 2026-09-01: floor multiplier for --ace-gate as disagreement -> 0 "
                              "(never fully zero -- a real geometric violation should still get some correction "
                              "even when the policy looks confident).")
    parser.add_argument("--proactive-persistence-window", type=float, default=0.0,
                         help="occ_vla addition 2026-08-30, per user's 'continuous proximity / persistence-"
                              "gated correction' proposal for the libero_object task7 degradation (implemented "
                              "as a hand-crafted geometric heuristic, chosen over a learned discriminator for "
                              "higher generalization with the ~40 task-level labeled examples available -- see "
                              "CLAUDE.md): number of consecutive REPLAN CHUNKS a margin violation must persist "
                              "across before its correction gain is decayed toward "
                              "--proactive-persistence-min-gain-frac, targeting the diagnosed correct-perturb-"
                              "re-trigger-escalate failure loop (a violation that resolves within a replan or "
                              "two keeps full gain; one that keeps re-triggering for many replans in a row is "
                              "trusted less, not more). <=0.0 (default) disables -- zero effect on any existing "
                              "run/condition unless explicitly set.")
    parser.add_argument("--proactive-persistence-min-gain-frac", type=float, default=0.2,
                         help="occ_vla addition 2026-08-30: floor multiplier applied to the CBF gain once a "
                              "margin violation has persisted for >= --proactive-persistence-window consecutive "
                              "replan chunks. Only takes effect if that window > 0.")
    parser.add_argument("--proactive-persistence-mode", default="decay", choices=["decay", "escalate"],
                         help="occ_vla addition 2026-08-30, per v3's n=10 task7 result (byte-identical to v1/v2 "
                              "-- see CLAUDE.md): 'decay' (default, original v3) starts at full gain and decays "
                              "toward the floor as a violation persists. 'escalate' is the opposite polarity, "
                              "testing the hypothesis that v3's null result is because the FIRST correction in "
                              "an episode necessarily fires at full gain (before any decay can matter) -- starts "
                              "at the floor gain and ramps UP toward full only once the violation has genuinely "
                              "persisted for --proactive-persistence-window replans. Only relevant if "
                              "--proactive-persistence-window > 0.")
    parser.add_argument("--drop-wrist-image", action="store_true",
                         help="occ_vla addition 2026-08-31, per user request: orthogonal to --conditions -- "
                              "genuinely removes the wrist image from the model's input (num_images_in_input "
                              "2->1 for the duration of each get_vla_action call, restored after) for EVERY "
                              "condition in this run, not just a dedicated 'agentview_only_true' condition. "
                              "Lets any existing condition (baseline, proactive_avoidance_depth/CBF, "
                              "agentview_vjepa, etc.) be evaluated under the true single-camera protocol "
                              "confirmed to collapse baseline to 0/90 across all 9 LIBERO-10 tasks -- see "
                              "CLAUDE.md. Default False, zero effect on every existing run unless passed.")
    parser.add_argument("--second-view-camera", default="robot0_eye_in_hand",
                         help="occ_vla addition (2026-08-31), per user's methodological question about "
                              "whether an added image SLOT helps regardless of its content (vs. needing "
                              "the wrist camera's specific near-field content): which real robosuite "
                              "camera feeds the model's SECOND image input slot. Default "
                              "'robot0_eye_in_hand' (the real wrist camera, original behavior, byte for "
                              "byte). Pass e.g. 'frontview' (a real, standard, distinct robosuite arena "
                              "camera) to test whether a content-uninformative-but-structurally-present "
                              "second view recovers any success under agentview occlusion, WITHOUT "
                              "--drop-wrist-image (that flag controls whether the slot is used at all; "
                              "this controls WHICH camera fills it when it is used).")
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
    parser.add_argument("--vjepa2-blend-alpha", type=float, default=1.0,
                         help="occ_vla addition 2026-08-31: alpha-blend fraction for "
                              "agentview_vjepa2_temporal/_amodal's injected content vs. the real "
                              "(uncorrected) patch tokens at occluded positions. 1.0 (default) = "
                              "full hard replace, byte-identical to every prior test. <1.0 blends "
                              "with real content -- a real, motivated fix for a regression observed "
                              "under dual-camera evaluation (full overwrite may remove real signal "
                              "the model was already using via the wrist camera).")
    parser.add_argument("--vjepa2-blend-alpha-floor", type=float, default=0.0,
                         help="occ_vla addition 2026-08-31: persistence-escalate floor for "
                              "--vjepa2-blend-alpha, only used when --vjepa2-blend-persistence-window > 0.")
    parser.add_argument("--vjepa2-blend-persistence-window", type=int, default=0,
                         help="occ_vla addition 2026-08-31: replan-step window over which "
                              "vjepa2_blend_alpha escalates from --vjepa2-blend-alpha-floor to "
                              "--vjepa2-blend-alpha as occluded_run_length grows -- same "
                              "closed-form schedule as CBF's own v4-escalate persistence gate. "
                              "0 (default) disables this -- --vjepa2-blend-alpha is used as a "
                              "fixed value every engaged step, byte-identical to every prior test.")
    parser.add_argument("--vjepa2-amodal-ema-decay", type=float, default=0.5,
                         help="occ_vla addition 2026-09-01: EMA decay for agentview_vjepa2_amodal's "
                              "temporal smoothing of the completed content (0.5 = original default, "
                              "unchanged). 0.0 degenerates to the pure instantaneous completion each "
                              "step (no history mixed in) -- an ablation for the hypothesis that EMA "
                              "staleness during a fast dynamic phase (e.g. Goal task7's grasp approach) "
                              "forces a temporally-inconsistent completion onto the policy.")
    parser.add_argument("--vjepa2-confidence-threshold", type=float, default=-1.0,
                         help="occ_vla addition 2026-09-01: confidence-gated selective injection for "
                              "agentview_vjepa2_amodal (see vjepa2_confidence_mask's docstring). "
                              "-1.0 (default) keeps every completed token, byte-identical to every "
                              "prior test. Higher values (up to 1.0, cosine-similarity scale) require "
                              "the raw completion to be more locally consistent with real neighboring "
                              "content before injecting it -- positions that fail the gate fall back "
                              "to the real (occluded, unmodified) content instead.")
    parser.add_argument("--vjepa2-projection-weights", default=None,
                         help="occ_vla addition 2026-08-31: directory containing trained "
                              "proj_dino.pt/proj_siglip.pt (see scripts/train_vjepa2_bridge_projections.py) "
                              "for agentview_vjepa2_temporal's bridging projections. None (default) leaves "
                              "them at their random (untrained) init.")
    parser.add_argument("--agentview-vjepa-min-run-length", type=int, default=3,
                         help="occ_vla addition 2026-08-25 (agentview_vjepa condition): minimum consecutive "
                              "occluded env-steps (occluded_run_length) before the VJEPA FiLM+cross-attention "
                              "correction module is allowed to fire on the agentview image. Untuned default "
                              "(3), per the user's stated rationale that a single-frame occlusion blip needs "
                              "no correction and firing on it would just add unnecessary feature perturbation.")
    parser.add_argument("--save-distillation-pairs-dir", default=None,
                         help="occ_vla addition 2026-08-27: directory to save (agentview image, wrist image, "
                              "proprio state, proactive_avoidance_depth-corrected action) pairs for a later "
                              "imitation-learning distillation of the zero-privileged depth-based CBF correction "
                              "back into the policy itself. Only active for the proactive_avoidance_depth "
                              "condition -- plain proactive_avoidance_cbf uses privileged occluder position and "
                              "is deliberately NOT used as the distillation teacher, per the user's explicit "
                              "no-privileged-information requirement for this thread.")
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

    if args.load_distillation_lora:
        # occ_vla addition (2026-08-27): loads the LoRA(language_model
        # attention projections) + action_head weights saved by
        # scripts/train_distillation_imitation.py -- the imitation-
        # learning distillation of proactive_avoidance_depth's
        # zero-privileged real-RGB-D CBF correction back into the
        # policy itself. Reconstructs the SAME LoraConfig used at
        # training time (rank read from the saved state dict's own
        # lora_A tensor shape, not hardcoded, so this stays correct if
        # a future training run uses a different rank) before loading.
        from peft import LoraConfig, get_peft_model
        print(f"  [distillation-lora] loading LoRA+action_head weights from {args.load_distillation_lora}")
        state_dict = torch.load(
            os.path.join(args.load_distillation_lora, "distillation_weights.pt"), map_location="cpu")
        lora_a_shapes = [v.shape for k, v in state_dict.items() if "lora_A" in k]
        assert lora_a_shapes, "no lora_A tensors found in the saved state dict -- was this really saved by train_distillation_imitation.py?"
        inferred_rank = lora_a_shapes[0][0]
        lora_config = LoraConfig(
            r=inferred_rank, lora_alpha=inferred_rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=None,
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        lm_state = {k[len("language_model."):]: v for k, v in state_dict.items() if k.startswith("language_model.")}
        ah_state = {k[len("action_head."):]: v for k, v in state_dict.items() if k.startswith("action_head.")}
        missing_lm, unexpected_lm = model.language_model.load_state_dict(lm_state, strict=False)
        missing_ah, unexpected_ah = action_head.load_state_dict(ah_state, strict=False)
        assert len(unexpected_lm) == 0, f"LoRA load found keys not in the model: {unexpected_lm[:5]}"
        assert len(unexpected_ah) == 0, f"action_head load found keys not in the model: {unexpected_ah[:5]}"
        print(f"  [distillation-lora] loaded {len(lm_state)} LoRA tensors (rank={inferred_rank}) + "
              f"{len(ah_state)} action_head tensors")

    if args.load_object_centric_adapter:
        # occ_vla addition (2026-09-03, Month 2): attach a trained
        # ObjectCentricZeroInitAdapter -- base model/action_head untouched.
        from prismatic.extern.hf.modeling_prismatic import ObjectCentricZeroInitAdapter
        print(f"  [object-centric-adapter] loading from {args.load_object_centric_adapter}")
        adapter_state = torch.load(
            os.path.join(args.load_object_centric_adapter, "object_centric_adapter_weights.pt"), map_location="cpu")
        model.object_centric_adapter = ObjectCentricZeroInitAdapter(model.llm_dim).to(model.device, dtype=torch.bfloat16)
        missing_oc, unexpected_oc = model.object_centric_adapter.load_state_dict(adapter_state, strict=True)
        print(f"  [object-centric-adapter] loaded {len(adapter_state)} tensors, "
              f"missing={len(missing_oc)} unexpected={len(unexpected_oc)} (both should be 0)")

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

    # occ_vla addition (2026-08-31, per user's explicit choice to build the
    # real-V-JEPA2 temporal-recovery direction): only load the real,
    # separate ~1.3B-param V-JEPA2-AC checkpoint (see CLAUDE.md) if a
    # condition that actually needs it was requested -- zero cost/behavior
    # change for every other run of this script.
    vjepa2_encoder = vjepa2_predictor = vjepa2_proj_dino = vjepa2_proj_siglip = None
    vjepa2_splice_forward = None
    vjepa2_base_encoder = vjepa2_base_predictor = None
    if any(c in args.conditions for c in ("agentview_vjepa2_amodal", "agentview_vjepa2_amodal_plus_depth")):
        # occ_vla addition (2026-08-31): the REAL base (non-AC) V-JEPA2
        # checkpoint, separate weights from the AC one (see
        # load_real_vjepa2_base's docstring) -- needed for spatial
        # masked-patch completion (no temporal history).
        print("[vjepa2] loading real BASE (non-AC) V-JEPA2 checkpoint (encoder + predictor)...")
        vjepa2_base_encoder, vjepa2_base_predictor = load_real_vjepa2_base(model.device)
        print("[vjepa2] base checkpoint loaded")
    if any(c in args.conditions for c in ("agentview_vjepa2_temporal", "agentview_vjepa2_amodal", "agentview_vjepa2_amodal_plus_depth")):
        if "agentview_vjepa2_temporal" in args.conditions:
            print("[vjepa2] loading real V-JEPA2-AC checkpoint (encoder + AC predictor)...")
            vjepa2_encoder, vjepa2_predictor = load_real_vjepa2(model.device)
        dino_dim = model.vision_backbone.featurizer.embed_dim
        siglip_dim = model.vision_backbone.fused_featurizer.embed_dim
        # occ_vla note: these two projections are NEW and UNTRAINED (random
        # init, seeded) -- explicitly disclosed in CLAUDE.md. They map
        # V-JEPA2's own 1408-dim latent into this checkpoint's DINO/SigLIP
        # token dimensions so the shapes are at least compatible; no claim
        # the resulting content is meaningful in those representation
        # spaces without real training data for this bridge.
        torch.manual_seed(0)
        vjepa2_proj_dino = torch.nn.Linear(1408, dino_dim).to(model.device, dtype=torch.bfloat16)
        vjepa2_proj_siglip = torch.nn.Linear(1408, siglip_dim).to(model.device, dtype=torch.bfloat16)
        vjepa2_splice_forward = make_agentview_vjepa2_temporal_splice_forward(
            model.vision_backbone, img_idx=0, blend_alpha=args.vjepa2_blend_alpha
        )
        if args.vjepa2_projection_weights:
            # occ_vla addition (2026-08-31, per user request): load
            # trained bridging projections (scripts/train_vjepa2_
            # bridge_projections.py, plain MSE regression against real
            # DINO/SigLIP targets on already-saved frames -- no rollout
            # needed) instead of leaving them at random init.
            vjepa2_proj_dino.load_state_dict(
                torch.load(os.path.join(args.vjepa2_projection_weights, "proj_dino.pt"), map_location=model.device)
            )
            vjepa2_proj_siglip.load_state_dict(
                torch.load(os.path.join(args.vjepa2_projection_weights, "proj_siglip.pt"), map_location=model.device)
            )
            print(f"[vjepa2] loaded TRAINED bridging projections from {args.vjepa2_projection_weights}")
        else:
            print(f"[vjepa2] loaded. dino_dim={dino_dim} siglip_dim={siglip_dim} "
                  f"(untrained bridging projections initialized)")

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
        # depth rendering when any requested condition needs a real RGB-D
        # obstacle point cloud (not the privileged occluder_geom_ids used by
        # proactive_avoidance_oracle/cbf alone). occ_vla bug fix (2026-09-01,
        # found while debugging why the new `stuck_recovery_plus_depth`
        # combined condition never fired a single CBF correction across a
        # whole episode): this used to check ONLY the literal string
        # "proactive_avoidance_depth" in args.conditions -- every other
        # depth-needing condition added since then
        # (agentview_vjepa_plus_depth, agentview_vjepa2_amodal_plus_depth,
        # stuck_recovery_plus_depth, proactive_avoidance_mpc when combined
        # with proactive_use_depth) silently got camera_depths=False, so
        # "agentview_depth" was never in obs_dict and
        # _depth_obstacle_points() short-circuited to zero points on EVERY
        # step, EVERY episode -- not a depth-detection failure, an env-
        # construction gap that made depth detection impossible from the
        # start. Now matches the same condition set used everywhere else in
        # this file to decide proactive_use_depth.
        _DEPTH_NEEDING_CONDITIONS = {
            "proactive_avoidance_depth", "agentview_vjepa_plus_depth",
            "agentview_vjepa2_amodal_plus_depth", "stuck_recovery_plus_depth",
            # occ_vla bug fix (2026-09-02): these two conditions (added
            # 2026-09-01) both set proactive_use_depth=True elsewhere in
            # this file but were never added here -- same missing-
            # registration pattern as the bug documented above, now
            # recurring for the ace_gate/attn_excl conditions. Confirmed
            # via real data: EVERY episode of ace_gate_task6_n10's
            # "proactive_avoidance_depth_ace_gated" condition showed
            # proactive_correction_applied_count==0 (10/10 episodes),
            # while the plain (correctly-depth-enabled)
            # "proactive_avoidance_depth" condition on the same task fires
            # in 50/50 episodes (n50_libero10_task6) -- camera_depths was
            # False the whole time, not a real gating outcome.
            "proactive_avoidance_depth_ace_gated", "proactive_avoidance_depth_attn_excl",
        }
        env = get_libero_env_seg(
            task, resolution=resize_size,
            camera_depths=bool(args.divergence_extract_dir) or any(c in _DEPTH_NEEDING_CONDITIONS for c in args.conditions),
            extra_camera=(args.second_view_camera if args.second_view_camera != "robot0_eye_in_hand" else None),
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
            model.vision_backbone.forward = (
                splice_forward if condition in ("oracle", "oracle_no_collision")
                else vjepa2_splice_forward if condition in ("agentview_vjepa2_temporal", "agentview_vjepa2_amodal", "agentview_vjepa2_amodal_plus_depth")
                else original_forward
            )
            agentview_vjepa2_temporal = condition == "agentview_vjepa2_temporal"
            agentview_vjepa2_amodal = condition in ("agentview_vjepa2_amodal", "agentview_vjepa2_amodal_plus_depth")
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
            if condition in ("no_collision",) + REACTIVE_CONDITIONS + ("low_mobility", "composite_visual_only", "ttc_area_blend", "scripted_recovery_after_stuck", "proactive_avoidance_oracle", "proactive_avoidance_cbf", "proactive_avoidance_depth", "proactive_avoidance_mpc", "agentview_vjepa", "agentview_vjepa_plus_depth", "blank_wrist", "agentview_only_true"):
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
            stuck_velocity_trigger = condition in ("scripted_recovery_after_stuck", "B_only", "A_plus_B", "stuck_recovery_plus_depth")
            # occ_vla addition (2026-08-24): "proactive_avoidance_oracle" also
            # keeps real collision AND real occluder rendering fully intact --
            # it needs occluder_geom_ids for the PRIVILEGED true-3D-position
            # lookup (Phase 1 proof-of-concept only; a real depth-camera/
            # segmentation-based version is the planned Phase 2 if this shows
            # value), but does not disable collision/rendering itself.
            proactive_avoidance_oracle = condition in ("proactive_avoidance_oracle", "proactive_avoidance_cbf", "proactive_avoidance_depth", "proactive_avoidance_mpc", "agentview_vjepa_plus_depth", "stuck_recovery_plus_depth", "proactive_avoidance_depth_ace_gated", "proactive_avoidance_depth_attn_excl")
            # occ_vla addition (2026-08-24, v2): "proactive_avoidance_cbf" reuses
            # the exact same trigger/plumbing as proactive_avoidance_oracle (same
            # privileged occluder-position lookup, same "keeps real collision AND
            # real occluder rendering intact" contract) -- only the correction
            # MATH differs (per-step minimal-norm CBF/APF nudge vs. v1's hard
            # full-chunk override to a fixed lift), selected via proactive_use_cbf.
            proactive_use_cbf = condition in ("proactive_avoidance_cbf", "proactive_avoidance_depth", "agentview_vjepa_plus_depth", "stuck_recovery_plus_depth", "proactive_avoidance_depth_ace_gated", "proactive_avoidance_depth_attn_excl")
            # occ_vla addition (2026-08-24, Phase 2): "proactive_avoidance_depth"
            # reuses proactive_avoidance_cbf's exact correction math -- the ONLY
            # difference is where occ_centers/occ_radii come from (real RGB-D
            # obstacle point cloud vs. privileged occluder_geom_ids). No occluder
            # identity/geometry is used anywhere in this condition's path.
            proactive_use_depth = condition in ("proactive_avoidance_depth", "agentview_vjepa_plus_depth", "agentview_vjepa2_amodal_plus_depth", "stuck_recovery_plus_depth", "proactive_avoidance_depth_ace_gated", "proactive_avoidance_depth_attn_excl")
            # occ_vla addition (2026-09-01), per user's "VLA自身のアテンション/
            # ACEでCBFの介入をゲートする" request: "proactive_avoidance_depth_
            # ace_gated" is byte-identical to proactive_avoidance_depth except
            # the CBF correction gain is scaled by the base policy's own
            # ensemble_disagreement each step (see run_episode's ace_gate_*
            # params). --ace-gate also allows enabling this on top of any
            # condition string via the global CLI flag, for ad hoc combination
            # with other proactive_avoidance_* variants without inventing a
            # new condition name for every combination.
            ace_gate_enabled = args.ace_gate or condition == "proactive_avoidance_depth_ace_gated"
            # occ_vla addition (2026-09-01): KNOWS-style attention-based
            # target exclusion (see run_episode's attn_target_excl_enabled
            # docstring / _attention_target_id for the full grounding).
            attn_target_excl_enabled = args.attn_target_excl or condition == "proactive_avoidance_depth_attn_excl"
            # occ_vla addition (2026-09-01), per user request ("事前回避+
            # 遮蔽耐性のあるbaselineからの差分はないですか？"): "stuck_recovery_
            # plus_depth" combines TWO independently-already-validated, zero-
            # vision-correction mechanisms -- proactive_avoidance_depth's
            # real, zero-privileged (RGB-D+segmentation) per-step minimal-
            # norm CBF correction (genuinely PREDICTIVE: computed BEFORE
            # executing the action, from the upcoming safety-margin
            # violation) and scripted_recovery_after_stuck's proprioceptive
            # stuck-detection + scripted retreat (REACTIVE: fires only once
            # already stalled). Neither alone is both predictive AND
            # occlusion-robust with confirmed significance on the same task
            # (CBF's own validated evidence is a 39-task aggregate switching
            # rule; Approach B's is a single-task McNemar-significant
            # result) -- this combination is the natural next test, not yet
            # run anywhere in this project.
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
            # occ_vla addition (2026-08-30, per user request): "agentview_
            # vjepa_plus_depth" combines VJEPA (perception-side, corrects the
            # model's INPUT vision tokens for the occluded region) with the
            # zero-privileged depth-based CBF correction (action-side,
            # corrects the model's OUTPUT action chunk) -- the two mechanisms
            # touch disjoint parts of the pipeline (vision-token patching vs.
            # post-hoc action correction) so are composable without any new
            # interaction logic; this condition just enables both flags
            # simultaneously via the same real, already-computed occlusion/
            # depth signals each condition uses independently elsewhere in
            # this file. Motivation: task9's 84%->0% CBF-alone collapse (see
            # CLAUDE.md) -- testing whether giving the model VJEPA's
            # perception-side occlusion fill ALSO reduces reliance on/
            # need for the action-side correction that caused the collapse.
            agentview_vjepa = condition in ("agentview_vjepa", "agentview_vjepa_plus_depth")
            # occ_vla addition (2026-08-30): "blank_wrist" keeps real
            # agentview occlusion identical to baseline -- ONLY the wrist
            # camera is additionally gray-filled -- to decisively test
            # whether the wrist camera is the dominant channel behind
            # baseline's high success rate under agentview occlusion (§3.7).
            blank_wrist = condition == "blank_wrist"
            # occ_vla addition (2026-08-30, per user's explicit request):
            # "agentview_only_true" is the methodologically cleaner sibling
            # of "blank_wrist" -- instead of gray-filling the wrist image
            # (a real but out-of-distribution input this checkpoint was
            # never trained to expect), it removes the wrist image from
            # the model's input ENTIRELY (num_images_in_input: 2 -> 1 for
            # the duration of each get_vla_action call, then restored),
            # matching the real LIBERO-Occ paper's likely own single-camera
            # evaluation protocol (per docs/evaluation.md's framing of
            # PERSPECTIVE_OBS_KEY as "debug/reference" only) far more
            # closely than an anomalous gray frame does. Agentview's own
            # real occlusion is left completely unchanged, same contract
            # as blank_wrist.
            drop_wrist_image = condition == "agentview_only_true" or args.drop_wrist_image
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
                    proactive_target_attractor_radius_m=args.proactive_target_attractor_radius_m,
                    proactive_target_attractor_decay=args.proactive_target_attractor_decay,
                    proactive_target_attractor_max_staleness=args.proactive_target_attractor_max_staleness,
                    proactive_grasp_phase_radius_m=args.proactive_grasp_phase_radius_m,
                    proactive_grasp_phase_gain_decay=args.proactive_grasp_phase_gain_decay,
                    proactive_persistence_window=args.proactive_persistence_window,
                    proactive_persistence_min_gain_frac=args.proactive_persistence_min_gain_frac,
                    proactive_persistence_mode=args.proactive_persistence_mode,
                    ace_gate_enabled=ace_gate_enabled,
                    ace_gate_scale_m=args.ace_gate_scale_m,
                    ace_gate_min_frac=args.ace_gate_min_frac,
                    attn_target_excl_enabled=attn_target_excl_enabled,
                    attn_target_window=args.attn_target_window,
                    attn_target_gap_delta=args.attn_target_gap_delta,
                    object_centric_adapter_enabled=bool(args.load_object_centric_adapter),
                    blank_wrist=blank_wrist,
                    drop_wrist_image=drop_wrist_image,
                    second_view_camera=args.second_view_camera,
                    proactive_use_depth=proactive_use_depth,
                    proactive_use_mpc=proactive_use_mpc,
                    proactive_mpc_n_candidates=args.proactive_mpc_n_candidates,
                    proactive_mpc_noise_std=args.proactive_mpc_noise_std,
                    proactive_mpc_w_safety=args.proactive_mpc_w_safety,
                    proactive_mpc_w_fidelity=args.proactive_mpc_w_fidelity,
                    agentview_vjepa=agentview_vjepa,
                    agentview_vjepa_min_run_length=args.agentview_vjepa_min_run_length,
                    save_distillation_pairs_dir=args.save_distillation_pairs_dir,
                    agentview_vjepa2_temporal=agentview_vjepa2_temporal,
                    vjepa2_encoder=vjepa2_encoder, vjepa2_predictor=vjepa2_predictor,
                    vjepa2_proj_dino=vjepa2_proj_dino, vjepa2_proj_siglip=vjepa2_proj_siglip,
                    vjepa2_splice_forward=vjepa2_splice_forward,
                    agentview_vjepa2_amodal=agentview_vjepa2_amodal,
                    vjepa2_base_encoder=vjepa2_base_encoder, vjepa2_base_predictor=vjepa2_base_predictor,
                    vjepa2_blend_alpha_ceiling=args.vjepa2_blend_alpha,
                    vjepa2_blend_alpha_floor=args.vjepa2_blend_alpha_floor,
                    vjepa2_blend_persistence_window=args.vjepa2_blend_persistence_window,
                    vjepa2_amodal_ema_decay=args.vjepa2_amodal_ema_decay,
                    vjepa2_confidence_threshold=args.vjepa2_confidence_threshold,
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

            # occ_vla addition (2026-08-27): flatten every episode's
            # distillation_manifest (populated only under
            # proactive_avoidance_depth) into one combined manifest.json
            # in save_distillation_pairs_dir -- rewritten after each
            # condition so a crash partway through doesn't lose earlier
            # conditions'/tasks' already-collected pairs.
            if args.save_distillation_pairs_dir:
                combined_manifest = []
                for cond_results in task_results.values():
                    for ep_res in cond_results:
                        combined_manifest.extend(ep_res.get("distillation_manifest", []))
                if combined_manifest:
                    os.makedirs(args.save_distillation_pairs_dir, exist_ok=True)
                    with open(os.path.join(args.save_distillation_pairs_dir, "manifest.json"), "w") as f:
                        json.dump(combined_manifest, f)
                    print(f"    [distillation-pairs] {len(combined_manifest)} pairs saved to {args.save_distillation_pairs_dir}/manifest.json")

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
