"""
test_natural_occlusion_generalization.py

The cheap "1-frame manual masking + cos-sim" sanity check proposed in chat
(2026-08-09): does the already-trained vjepa correction module (trained
ONLY on apply_partial_patch's fixed centered-square gray mask) generalize
to a REAL, irregularly-shaped natural/scene-induced occlusion mask from
LIBERO-Occ, once that region is painted gray to match the module's training
color distribution ("Semantic Blanking")?

Method, all on ONE real frame, no rollout/training:
  1. Load a LIBERO-Occ occluded task (real 3D occluder object placed in the
     scene -- see register_libero_occ_suites.py), step to a frame where the
     wrist camera's view is actually affected.
  2. Capture the REAL occluded wrist-camera render.
  3. Temporarily zero the occluder geom(s)' alpha and re-render the IDENTICAL
     sim state -> a ground-truth "clean" wrist render, same technique
     already validated in the sibling pi0.5 project's arm_removal_pairs
     collection (occ_vla/CLAUDE.md, "Ground-truth arm-removal pair
     generation..." -- render same state twice, alpha-zero the occluder,
     restore after). Zero generative guessing, real rendered background.
  4. Pixel-diff the two renders -> a real, irregularly-shaped occlusion
     mask (NOT apply_partial_patch's fixed centered square) -> pool to the
     16x16 patch grid the vjepa splice actually operates on.
  5. Paint the occluded pixels gray (GRAY_FILL, same constant
     train_vjepa_predictor_scaled.py's apply_partial_patch uses) in the
     REAL occluded render -- "Semantic Blanking": converts a novel occluder
     shape/color into the same gray-pixel distribution the predictor was
     actually trained on, without touching the predictor itself.
  6. Run the SAME already-trained vjepa correction pipeline
     visualize_vjepa_correction.py uses (run_vit_to_layer / residual /
     splice), but with this new mask instead of apply_partial_patch's fixed
     geometry, and report masked-region mean cos-sim(input, GT) before vs.
     after correction -- exactly the metric already used and reported in
     the published Artifact's section 04, so directly comparable.

Decision criterion (agreed in chat before running): if cos-sim improves by
roughly the same order of magnitude as the fixed-patch case (+0.20 to +0.29,
see vjepa_correction_qual_goal_task0 results), the predictor generalizes to
novel mask geometry -- Semantic Blanking is a viable path to a zero-shot
natural-occlusion-capable system. If it doesn't improve (or gets worse),
that's a real, honest Limitation for the paper: the module is overfit to
apply_partial_patch's specific fixed geometry, not gray color alone --
matching the already-documented negative finding that mask position/size
JITTERING during training was net-harmful (thirdparty/openvla-oft/CLAUDE.md
finding #2), which already hinted at exactly this risk.

Run with the openvla-oft conda env:
  python scripts/test_natural_occlusion_generalization.py \
    --checkpoint checkpoints/openvla-7b-oft-libero10-vjepa \
    --vjepa-checkpoint vjepa_predictor_multitask_3task_6000steps.pt \
    --task-suite libero_10_occluded --task-id 7 \
    --occluder-body-substrings wooden_cabinet \
    --out-prefix natural_occ_generalization_task7
(task_id=7 = LIVING_ROOM_SCENE5 two-mugs-on-plates, occluder=wooden_cabinet_1 --
picked via scan_wrist_occlusion_libero_occ.py's scan, the strongest of 6/10
libero_10_occluded tasks confirmed to actually occlude the wrist camera
-- unlike moka_pots (task_id=3), whose occluder was confirmed to affect
agentview only, 0.00% wrist coverage across 160+ scanned steps.)
"""

import argparse
import os
import sys

OCC_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
SCRIPTS_DIR = os.path.join(OCC_VLA_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import register_libero_occ_suites  # noqa: E402  (registers the occluded suites)
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_wrist_image, get_libero_image  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key, get_libero_dummy_action  # noqa: E402
from experiments.robot.openvla_utils import get_processor, normalize_proprio  # noqa: E402
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402

from train_vjepa_predictor_multitask import (  # noqa: E402
    build_pixel_values_batch_multi,
    build_patch_token_mask_256,
    run_vit_to_layer,
)
from train_vjepa_predictor_scaled import GRAY_FILL  # noqa: E402


def find_occluder_body_ids(sim, substrings):
    ids = []
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if name and any(s in name.lower() for s in substrings):
            ids.append(i)
    return ids


def render_both_cams(env, resize_size):
    # force_update=True is REQUIRED -- robosuite's _get_observations() default
    # (force_update=False) returns each Observable's cached .obs from the last
    # env.step() call, NOT a fresh render. A direct sim.model mutation (geom
    # alpha, body position) + sim.forward() alone is invisible to this cache --
    # confirmed the hard way (occ_vla 2026-08-09): an alpha=0 toggle AND a
    # body_pos translation both produced bit-identical before/after images
    # (max diff 0) until force_update=True was added, after which the SAME
    # body-position technique correctly showed real, physically-plausible
    # occlusion (18.37% wrist pixels changed, matching an independently
    # computed segmentation-based occlusion fraction of 18.23% for the same
    # task/state -- two independent measurement methods cross-validated).
    obs = env.env._get_observations(force_update=True)
    wrist = get_libero_wrist_image(obs)
    agent = get_libero_image(obs)
    from PIL import Image
    wrist_r = np.array(Image.fromarray(wrist).resize((resize_size, resize_size), Image.BILINEAR))
    agent_r = np.array(Image.fromarray(agent).resize((resize_size, resize_size), Image.BILINEAR))
    return wrist_r, agent_r


def upsample_16x16(grid, size):
    t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")
    return up.squeeze(0).squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--step", type=int, default=0, help="ADDITIONAL scripted reach-down steps after the fixed 10-step settle -- scan_wrist_occlusion_libero_occ.py found peak wrist occlusion consistently AT step 0 (right after settle, before any reach), so 0 is the right default, not a placeholder")
    parser.add_argument("--occluder-body-substrings", nargs="+", required=True,
                         help="lowercase substrings identifying the occluder's BODY (not geom) names, e.g. wooden_cabinet -- verify uniqueness first (see scan_wrist_occlusion_libero_occ.py's find_new_objects), a too-generic substring can match unrelated pre-existing scene bodies")
    parser.add_argument("--diff-threshold", type=int, default=25,
                         help="per-pixel abs RGB-diff sum threshold (0-765) for the occluder mask")
    parser.add_argument("--out-prefix", default="natural_occ_generalization")
    parser.add_argument("--masking-mode", choices=["gray_pixel", "zero_feature"], default="gray_pixel",
                         help="gray_pixel (default, original): Semantic Blanking, paint the mask region "
                              "GRAY_FILL in pixel space then encode normally -- the predictor's OWN trained "
                              "input distribution (always one fixed-shape gray patch during training). "
                              "zero_feature: encode the REAL (unmodified) occluded image, then zero the "
                              "masked patches' FEATURE vectors directly post-encoding, before the predictor "
                              "-- a cheap, no-retraining proxy for an untrained MAE-style [MASK] token (an "
                              "nn.Parameter(torch.zeros(...)) IS the zero vector before any training moves "
                              "it), to isolate whether gray-patch TEXTURE content (vs. simply being "
                              "out-of-distribution at all) drives the geometric-overfitting failure -- see "
                              "chat 2026-08-09 for the full reasoning, this does NOT test a trained mask "
                              "token, only its untrained starting point.")
    args = parser.parse_args()
    args.out_prefix = f"{args.out_prefix}_{args.masking_mode}"  # avoid clobbering the other mode's saved outputs

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=args.task_suite,
        seed=7,
    )
    # same base-suite-name fallback as run_libero_occ_baseline.py
    if cfg.task_suite_name.endswith("_occluded"):
        base_name = cfg.task_suite_name[: -len("_occluded")]
        real_name = cfg.task_suite_name
        cfg.task_suite_name = base_name

    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    device = model.device
    dtype = torch.bfloat16
    vb = model.vision_backbone
    cfg.task_suite_name = real_name if 'real_name' in dir() else args.task_suite

    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    print(f"Task: {task.language} ({args.task_suite} task_id={args.task_id})")

    env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)
    init_states = task_suite.get_task_init_states(args.task_id)
    env.reset()
    env.set_init_state(init_states[0])
    dummy = get_libero_dummy_action(cfg.model_family)
    # 10-step settle (matches scan_wrist_occlusion_libero_occ.py's convention,
    # and its own scan found max wrist-occlusion consistently AT this point,
    # step 0 of the scripted-reach phase -- before any reach motion, not
    # after) + optional additional scripted steps if --step > 0.
    for t in range(10):
        env.step(dummy)
    rng = np.random.default_rng(0)
    for t in range(args.step):
        action = np.array([0.0, 0.0, -0.3, 0.0, 0.0, 0.0, -1.0]) + rng.normal(0, 0.05, size=7)
        action[6] = -1.0
        env.step(np.clip(action, -1, 1).tolist())

    sim = env.env.sim
    occluder_body_ids = find_occluder_body_ids(sim, args.occluder_body_substrings)
    assert occluder_body_ids, f"no bodies matched {args.occluder_body_substrings} -- inspect sim.model.body_id2name(i) for all i first"
    names = [sim.model.body_id2name(i) for i in occluder_body_ids]
    print(f"Occluder bodies: {names}")

    # --- real occluded render (occluder visible, as normal) ---
    wrist_occ, agent_occ = render_both_cams(env, resize_size)

    # --- ground-truth clean render: same sim state, occluder body translated
    # far away (+5m per axis) then restored. Validated method (occ_vla
    # 2026-08-09) -- geom_rgba alpha=0 was tried FIRST and found not to work
    # at all in this robosuite/mujoco setup (see render_both_cams' docstring
    # note); body-position translation + force_update=True does work,
    # cross-validated against an independent segmentation-based occlusion
    # measurement (18.37% pixel-diff vs. 18.23% segmentation, same
    # task/state). ---
    orig_pos = {bid: sim.model.body_pos[bid].copy() for bid in occluder_body_ids}
    for bid in occluder_body_ids:
        sim.model.body_pos[bid] = orig_pos[bid] + np.array([5.0, 5.0, 5.0])
    sim.forward()
    wrist_gt, agent_gt = render_both_cams(env, resize_size)
    for bid in occluder_body_ids:
        sim.model.body_pos[bid] = orig_pos[bid]
    sim.forward()

    from PIL import Image
    Image.fromarray(wrist_occ).save(f"{args.out_prefix}_wrist_occluded_raw.png")
    Image.fromarray(wrist_gt).save(f"{args.out_prefix}_wrist_gt_raw.png")
    Image.fromarray(agent_occ).save(f"{args.out_prefix}_agentview_occluded_raw.png")

    # --- derive the natural occlusion mask from the pixel diff ---
    diff = np.abs(wrist_occ.astype(int) - wrist_gt.astype(int)).sum(axis=-1)  # (H,W)
    mask_px = diff > args.diff_threshold
    frac_pixels = mask_px.mean()
    print(f"Wrist-camera pixels changed by the occluder: {frac_pixels*100:.2f}%")

    if frac_pixels < 0.005:
        print("WARNING: <0.5% of wrist pixels affected -- this occluder barely touches the "
              "wrist camera view (consistent with the earlier finding that LIBERO-Occ's scene "
              "occluders may mainly affect agentview, not wrist). Consider a different --step "
              "or a different occluder/task before trusting the result below.")

    # pool the pixel mask to the 16x16 patch grid (same >50%-area-per-patch
    # convention used elsewhere in this project, e.g. arm_free_subgoal.py's
    # _arm_token_mask), matching apply_partial_patch's own pixel_bounds ->
    # build_patch_token_mask_256 pipeline in spirit, but from a real
    # irregular mask instead of a synthetic rectangle.
    H, W = mask_px.shape
    ph, pw = H // 16, W // 16
    mask_256_np = np.zeros(256, dtype=bool)
    for r in range(16):
        for c in range(16):
            patch = mask_px[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            if patch.mean() > 0.5:
                mask_256_np[r * 16 + c] = True
    n_occ_patches = int(mask_256_np.sum())
    print(f"Occluded patches (16x16 grid): {n_occ_patches}/256")
    assert n_occ_patches > 0, "derived mask is empty at the patch level -- try a different --step/threshold"

    # --- Semantic Blanking: paint the derived mask region gray in the REAL occluded render ---
    wrist_blanked = wrist_occ.copy()
    wrist_blanked[mask_px] = GRAY_FILL
    Image.fromarray(wrist_blanked).save(f"{args.out_prefix}_wrist_semantic_blanked.png")

    ckpt = torch.load(args.vjepa_checkpoint, map_location=device)
    vb.vjepa_predictor_dino.load_state_dict(ckpt["dino"])
    vb.vjepa_predictor_siglip.load_state_dict(ckpt["siglip"])
    model.eval()
    vb.vjepa_predictor_dino.eval()
    vb.vjepa_predictor_siglip.eval()
    split_frac = vb.midlayer_split_frac
    split_layer_dino = int(len(vb.featurizer.blocks) * split_frac)
    split_layer_siglip = int(len(vb.fused_featurizer.blocks) * split_frac)

    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    # dummy proprio (zeros) -- this single-frame probe doesn't have a real
    # robot0_proprio-state readout wired through get_libero_env's obs dict
    # inspection above; the predictor's FiLM(proprio) conditioning gets a
    # neutral input, which affects magnitude/detail of the residual but not
    # whether the mask geometry itself is handled -- acceptable for this
    # geometry-generalization check specifically.
    proprio_t = normalize_proprio(np.zeros(8, dtype=np.float32), proprio_norm_stats)

    mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(1, -1, 1)

    with torch.no_grad():
        pv_gt = build_pixel_values_batch_multi([agent_gt], [wrist_gt], [prompt], processor, device, dtype)
        _, wrist_gt_t = torch.split(pv_gt, [6, 6], dim=1)
        wrist_reg_gt, wrist_fused_gt = torch.split(wrist_gt_t, [3, 3], dim=1)
        f_gt_dino = run_vit_to_layer(vb.featurizer, wrist_reg_gt, split_layer_dino)
        f_gt_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_gt, split_layer_siglip)

        # "past" frame for the predictor's temporal input: reuse the GT frame
        # itself (t-1 unavailable for a single captured frame) -- a static
        # stand-in, same limitation noted inline, affects residual quality
        # not mask-geometry handling.
        past_dino, past_siglip = f_gt_dino, f_gt_siglip

        if args.masking_mode == "gray_pixel":
            pv_input = build_pixel_values_batch_multi([agent_occ], [wrist_blanked], [prompt], processor, device, dtype)
            _, wrist_input_t = torch.split(pv_input, [6, 6], dim=1)
            wrist_reg_input, wrist_fused_input = torch.split(wrist_input_t, [3, 3], dim=1)
            f_input_dino = run_vit_to_layer(vb.featurizer, wrist_reg_input, split_layer_dino)
            f_input_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_input, split_layer_siglip)
        else:  # zero_feature
            # encode the REAL, unmodified occluded image (real cabinet pixels,
            # not grayed) -- then overwrite the masked patch positions with
            # the zero vector directly in feature space, bypassing pixel-space
            # gray-patch encoding entirely.
            pv_input = build_pixel_values_batch_multi([agent_occ], [wrist_occ], [prompt], processor, device, dtype)
            _, wrist_input_t = torch.split(pv_input, [6, 6], dim=1)
            wrist_reg_input, wrist_fused_input = torch.split(wrist_input_t, [3, 3], dim=1)
            f_input_dino_raw = run_vit_to_layer(vb.featurizer, wrist_reg_input, split_layer_dino)
            f_input_siglip_raw = run_vit_to_layer(vb.fused_featurizer, wrist_fused_input, split_layer_siglip)
            keep = (~mask_256.bool()).to(f_input_dino_raw.dtype)
            f_input_dino = f_input_dino_raw * keep
            f_input_siglip = f_input_siglip_raw * keep

        proprio_tensor = torch.tensor(proprio_t, device=device, dtype=dtype).unsqueeze(0)

        residual_dino = vb.vjepa_predictor_dino(f_input_dino, past_dino, proprio_tensor)
        f_final_dino = f_input_dino + mask_256 * residual_dino
        residual_siglip = vb.vjepa_predictor_siglip(f_input_siglip, past_siglip, proprio_tensor)
        f_final_siglip = f_input_siglip + mask_256 * residual_siglip

        resid_norm_dino = residual_dino.float().norm(dim=-1)[0]
        resid_norm_siglip = residual_siglip.float().norm(dim=-1)[0]
        resid_norm = ((resid_norm_dino / (resid_norm_dino.max() + 1e-8))
                      + (resid_norm_siglip / (resid_norm_siglip.max() + 1e-8))) / 2

        cos_before = ((torch.nn.functional.cosine_similarity(f_input_dino, f_gt_dino, dim=-1)
                       + torch.nn.functional.cosine_similarity(f_input_siglip, f_gt_siglip, dim=-1)) / 2)[0].float()
        cos_after = ((torch.nn.functional.cosine_similarity(f_final_dino, f_gt_dino, dim=-1)
                      + torch.nn.functional.cosine_similarity(f_final_siglip, f_gt_siglip, dim=-1)) / 2)[0].float()

    resid_grid = resid_norm.cpu().numpy().reshape(16, 16)
    cos_before_grid = cos_before.cpu().numpy().reshape(16, 16)
    cos_after_grid = cos_after.cpu().numpy().reshape(16, 16)
    mask_grid = mask_256_np.reshape(16, 16)

    mean_before = float(cos_before_grid[mask_grid].mean())
    mean_after = float(cos_after_grid[mask_grid].mean())
    print(f"\n[masking_mode={args.masking_mode}] Occluded-region mean cos-sim to GT: "
          f"before={mean_before:.4f}  after={mean_after:.4f}  delta={mean_after - mean_before:+.4f}")
    print("(fixed-patch reference from vjepa_correction_qual_goal_task0: +0.202 to +0.292 across 16 frames)")
    print("(gray_pixel/Semantic Blanking reference on THIS task7 frame: delta=-0.0452, see chat 2026-08-09)")

    fig, axes = plt.subplots(1, 6, figsize=(28, 5.2))
    axes[0].imshow(wrist_gt); axes[0].set_title("GT clean wrist\n(occluder body translated +5m, same sim state)"); axes[0].axis("off")
    axes[1].imshow(wrist_occ); axes[1].set_title("Real occluded wrist\n(actual scene render)"); axes[1].axis("off")
    axes[2].imshow(upsample_16x16(mask_grid.astype(float), resize_size), cmap="gray"); axes[2].set_title(f"Derived mask\n({n_occ_patches}/256 patches, {frac_pixels*100:.1f}% px)"); axes[2].axis("off")
    panel3_title = "Semantic Blanking\n(mask region -> gray)" if args.masking_mode == "gray_pixel" else "zero_feature mode\n(masked patches -> 0 vector, post-encoding)"
    axes[3].imshow(wrist_blanked if args.masking_mode == "gray_pixel" else wrist_occ); axes[3].set_title(panel3_title); axes[3].axis("off")
    im4 = axes[4].imshow(cos_before_grid, cmap="viridis", vmin=0, vmax=1)
    axes[4].set_title(f"cos-sim BEFORE correction\n(masked mean={mean_before:.3f})"); axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)
    im5 = axes[5].imshow(cos_after_grid, cmap="viridis", vmin=0, vmax=1)
    axes[5].set_title(f"cos-sim AFTER correction\n(masked mean={mean_after:.3f})"); axes[5].axis("off")
    plt.colorbar(im5, ax=axes[5], fraction=0.046)
    fig.suptitle(f"{args.task_suite} task{args.task_id} ({task.language}), step {args.step} -- "
                 f"real natural/scene-induced occlusion, novel mask geometry vs. the fixed-patch training distribution",
                 fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{args.out_prefix}.png", dpi=150)
    print(f"Saved {args.out_prefix}.png")


if __name__ == "__main__":
    main()
