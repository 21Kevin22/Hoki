"""
visualize_vjepa_correction.py

Answers the user's question (2026-08-08): "追加したモジュールで想像した画像な
り入力を変更したものを定性的に画像や動画で示すことはできますか？" (can you
qualitatively show, via images/video, what the added module imagines or how it
changes the input?)

Honest framing, established before writing this script (see chat): the vjepa
correction module (VJEPA_LatentDynamicsPredictor) operates entirely in ViT
patch-EMBEDDING space (mid-layer DINOv2/SigLIP features, splice at
split_frac=0.67), not pixel space -- there is no decoder anywhere in this
project that turns a corrected feature back into a "generated image". Building
one would be a new, unscoped generative-decoder project (this project's own
CLAUDE.md already documents extensive, largely-failed efforts at exactly this
kind of pixel-space generation in the sibling pi0.5/MMaDA investigation --
not something to casually bolt on here).

What CAN be shown honestly, reusing diagnose_vjepa_predictor_errors.py's
already-working feature-space machinery (run_vit_to_layer / residual /
cosine-similarity, unmodified logic, just applied to one real frame instead of
aggregated over a whole dataset):
  1. The raw clean vs. occluded wrist-camera pixels (what actually changes at
     the INPUT).
  2. A per-patch residual-magnitude heatmap (||vjepa_predictor(...)||, 16x16
     grid upsampled to 224x224) overlaid on the occluded frame -- WHERE and
     HOW STRONGLY the module intervenes. This is a direct, real quantity (the
     predictor's own output norm), not a proxy.
  3. A per-patch cosine-similarity-to-ground-truth heatmap, BEFORE (occluded
     input alone) vs. AFTER (input + mask*residual, i.e. what the vision
     backbone actually sees downstream) correction -- whether the correction
     moves the occluded patches' representations toward the true clean
     content, spatially resolved. This is the closest honest analogue to
     "what does it imagine": not a picture, but a measured answer to "does
     the corrected representation look more like the true one, and where".

Run with the openvla-oft conda env:
  python scripts/visualize_vjepa_correction.py \
    --vjepa-checkpoint vjepa_predictor_goal_3task_6000steps.pt \
    --checkpoint checkpoints/openvla-7b-oft-libero-goal-vjepa \
    --task-suite libero_goal --task-id 0 \
    --data-dir goal_task0_rollout_data --episode 0 --frame 8 \
    --out-prefix vjepa_correction_qual_goal_task0

--video mode (2026-08-08, added per user follow-up request): loops the
SAME per-frame computation across every available frame of one episode
(model loaded once, not per frame -- the expensive part), saves one PNG
per frame into --video-out-dir, then shells out to ffmpeg to stitch them
into an mp4 -- so the heatmaps can be watched tracking the arm's motion
across the whole reach-and-occlude sequence, not just one static instant:
  python scripts/visualize_vjepa_correction.py \
    --vjepa-checkpoint vjepa_predictor_goal_3task_6000steps.pt \
    --checkpoint checkpoints/openvla-7b-oft-libero-goal-vjepa \
    --task-suite libero_goal --task-id 0 \
    --data-dir goal_task0_rollout_data --episode 0 \
    --video --video-out-dir vjepa_correction_qual_goal_task0_frames \
    --out-prefix vjepa_correction_qual_goal_task0
"""

import argparse
import os
import subprocess
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
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import get_processor, normalize_proprio  # noqa: E402
from experiments.robot.robot_utils import get_model  # noqa: E402

from train_vjepa_predictor_multitask import (  # noqa: E402
    apply_partial_patch,
    build_pixel_values_batch_multi,
    build_patch_token_mask_256,
    load_dataset,
    run_vit_to_layer,
)


def upsample_16x16(grid, size=224):
    """Nearest-neighbor upsample a (16,16) per-patch quantity to (size,size)
    pixel space, for overlay on the real image -- each patch is a 14x14px
    square at 224x224 (16*14=224), so this is an exact, not approximate,
    correspondence, not a smoothed guess."""
    t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")
    return up.squeeze(0).squeeze(0).numpy()


def load_model_and_predictor(checkpoint, vjepa_checkpoint, task_suite_name, task_id):
    """Loads the base model + vjepa predictor weights ONCE. Returns everything
    a per-frame call needs, so --video mode doesn't pay this cost per frame."""
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=task_suite_name,
    )
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    device = model.device
    dtype = torch.bfloat16
    vb = model.vision_backbone

    split_frac = vb.midlayer_split_frac
    split_layer_dino = int(len(vb.featurizer.blocks) * split_frac)
    split_layer_siglip = int(len(vb.fused_featurizer.blocks) * split_frac)

    ckpt = torch.load(vjepa_checkpoint, map_location=device)
    vb.vjepa_predictor_dino.load_state_dict(ckpt["dino"])
    vb.vjepa_predictor_siglip.load_state_dict(ckpt["siglip"])
    print(f"Loaded vjepa predictor weights from {vjepa_checkpoint}")
    model.eval()
    vb.vjepa_predictor_dino.eval()
    vb.vjepa_predictor_siglip.eval()

    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"

    return {
        "model": model, "processor": processor, "vb": vb, "device": device, "dtype": dtype,
        "split_layer_dino": split_layer_dino, "split_layer_siglip": split_layer_siglip,
        "proprio_norm_stats": proprio_norm_stats, "prompt": prompt,
    }


def compute_frame(ctx, ep, t):
    """Runs the vjepa correction pipeline on one (episode, frame) pair using
    an already-loaded model context (from load_model_and_predictor). Returns
    everything needed to render one panel figure."""
    processor, device, dtype = ctx["processor"], ctx["device"], ctx["dtype"]
    vb = ctx["vb"]
    split_layer_dino, split_layer_siglip = ctx["split_layer_dino"], ctx["split_layer_siglip"]
    prompt = ctx["prompt"]

    agentview_t, wrist_t = ep["agentview"][t], ep["wrist"][t]
    agentview_tm1, wrist_tm1 = ep["agentview"][t - 1], ep["wrist"][t - 1]
    wrist_t_corrupted, pixel_bounds = apply_partial_patch(wrist_t)
    proprio_t = normalize_proprio(ep["proprio"][t], ctx["proprio_norm_stats"])

    mask_256_np = build_patch_token_mask_256(pixel_bounds)  # (256,) bool, True = occluded patch
    mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(1, -1, 1)

    with torch.no_grad():
        pv_clean_t = build_pixel_values_batch_multi([agentview_t], [wrist_t], [prompt], processor, device, dtype)
        _, wrist_clean_t = torch.split(pv_clean_t, [6, 6], dim=1)
        wrist_reg_clean_t, wrist_fused_clean_t = torch.split(wrist_clean_t, [3, 3], dim=1)
        f_gt_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_t, split_layer_dino)
        f_gt_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_t, split_layer_siglip)

        pv_clean_tm1 = build_pixel_values_batch_multi([agentview_tm1], [wrist_tm1], [prompt], processor, device, dtype)
        _, wrist_clean_tm1 = torch.split(pv_clean_tm1, [6, 6], dim=1)
        wrist_reg_clean_tm1, wrist_fused_clean_tm1 = torch.split(wrist_clean_tm1, [3, 3], dim=1)
        past_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_tm1, split_layer_dino)
        past_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_tm1, split_layer_siglip)

        pv_corrupted_t = build_pixel_values_batch_multi([agentview_t], [wrist_t_corrupted], [prompt], processor, device, dtype)
        _, wrist_corrupted_t = torch.split(pv_corrupted_t, [6, 6], dim=1)
        wrist_reg_corrupted_t, wrist_fused_corrupted_t = torch.split(wrist_corrupted_t, [3, 3], dim=1)
        f_input_dino = run_vit_to_layer(vb.featurizer, wrist_reg_corrupted_t, split_layer_dino)
        f_input_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_corrupted_t, split_layer_siglip)

        proprio_tensor = torch.tensor(proprio_t, device=device, dtype=dtype).unsqueeze(0)

        residual_dino = vb.vjepa_predictor_dino(f_input_dino, past_dino, proprio_tensor)
        f_final_dino = f_input_dino + mask_256 * residual_dino
        residual_siglip = vb.vjepa_predictor_siglip(f_input_siglip, past_siglip, proprio_tensor)
        f_final_siglip = f_input_siglip + mask_256 * residual_siglip

        # residual magnitude (module's own output norm, real quantity, not a proxy)
        resid_norm_dino = residual_dino.float().norm(dim=-1)[0]  # (256,)
        resid_norm_siglip = residual_siglip.float().norm(dim=-1)[0]
        resid_norm = ((resid_norm_dino / (resid_norm_dino.max() + 1e-8))
                      + (resid_norm_siglip / (resid_norm_siglip.max() + 1e-8))) / 2

        cos_before_dino = torch.nn.functional.cosine_similarity(f_input_dino, f_gt_dino, dim=-1)[0]
        cos_before_siglip = torch.nn.functional.cosine_similarity(f_input_siglip, f_gt_siglip, dim=-1)[0]
        cos_before = ((cos_before_dino + cos_before_siglip) / 2).float()

        cos_after_dino = torch.nn.functional.cosine_similarity(f_final_dino, f_gt_dino, dim=-1)[0]
        cos_after_siglip = torch.nn.functional.cosine_similarity(f_final_siglip, f_gt_siglip, dim=-1)[0]
        cos_after = ((cos_after_dino + cos_after_siglip) / 2).float()

    resid_grid = resid_norm.cpu().numpy().reshape(16, 16)
    cos_before_grid = cos_before.cpu().numpy().reshape(16, 16)
    cos_after_grid = cos_after.cpu().numpy().reshape(16, 16)
    mask_grid = mask_256_np.reshape(16, 16)

    mean_cos_before_occ = float(cos_before_grid[mask_grid].mean())
    mean_cos_after_occ = float(cos_after_grid[mask_grid].mean())

    return {
        "wrist_t": wrist_t, "wrist_t_corrupted": wrist_t_corrupted,
        "resid_grid": resid_grid, "cos_before_grid": cos_before_grid, "cos_after_grid": cos_after_grid,
        "occluded_patches": int(mask_grid.sum()),
        "mean_cos_before_occ": mean_cos_before_occ, "mean_cos_after_occ": mean_cos_after_occ,
    }


def render_panel(d, suptitle, out_path):
    """Renders the 5-panel figure from one compute_frame() result dict."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))

    axes[0].imshow(d["wrist_t"])
    axes[0].set_title("Clean wrist frame\n(ground truth, never seen by the\nmodel once occluded)")
    axes[0].axis("off")

    axes[1].imshow(d["wrist_t_corrupted"])
    axes[1].set_title("Occluded wrist frame\n(actual model input)")
    axes[1].axis("off")

    axes[2].imshow(d["wrist_t_corrupted"])
    im2 = axes[2].imshow(upsample_16x16(d["resid_grid"]), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title("Where/how strongly the module\nintervenes (||residual||, normalized,\nper 14x14px patch)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(d["cos_before_grid"], cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(f"cos-sim(input, ground-truth)\nBEFORE correction\n(occluded-region mean={d['mean_cos_before_occ']:.3f})")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    im4 = axes[4].imshow(d["cos_after_grid"], cmap="viridis", vmin=0, vmax=1)
    axes[4].set_title(f"cos-sim(corrected, ground-truth)\nAFTER correction\n(occluded-region mean={d['mean_cos_after_occ']:.3f})")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    fig.suptitle(
        f"{suptitle} -- the vjepa module operates in ViT patch-EMBEDDING space, not pixel space, so there is no\n"
        f"\"generated image\" to show -- this instead shows WHERE it intervenes and whether that moves the "
        f"representation toward the true clean content.",
        fontsize=10.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.expanduser(
        "~/slocal/occ_vla/checkpoints/openvla-7b-oft-libero-goal-vjepa"))
    parser.add_argument("--vjepa-checkpoint", default="vjepa_predictor_goal_3task_6000steps.pt")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--data-dir", default="goal_task0_rollout_data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=8, help="single-frame mode: index t into the stored (subsampled) episode frames; needs t>=1")
    parser.add_argument("--out-prefix", default="vjepa_correction_qual")
    parser.add_argument("--video", action="store_true", help="loop over every frame (t=1..len-1) of --episode, save one PNG each, then stitch into an mp4 via ffmpeg")
    parser.add_argument("--video-out-dir", default=None, help="dir for per-frame PNGs in --video mode (default: <out-prefix>_frames)")
    parser.add_argument("--video-fps", type=int, default=2)
    args = parser.parse_args()

    episodes, _ = load_dataset(args.data_dir)
    ep = episodes[args.episode]

    ctx = load_model_and_predictor(args.checkpoint, args.vjepa_checkpoint, args.task_suite, args.task_id)

    if not args.video:
        t = args.frame
        assert t >= 1, "--frame must be >=1 (needs a t-1 frame for the predictor's `past` input)"
        print(f"Using {args.data_dir}/episode_{args.episode:03d}, frame t={t} (of {len(ep['proprio'])})")
        d = compute_frame(ctx, ep, t)
        print(f"Occluded patches: {d['occluded_patches']}/256")
        print(f"Mean cos-sim-to-GT within occluded region: before={d['mean_cos_before_occ']:.4f}  "
              f"after={d['mean_cos_after_occ']:.4f}  (delta={d['mean_cos_after_occ'] - d['mean_cos_before_occ']:+.4f})")
        out_path = f"{args.out_prefix}.png"
        render_panel(d, f"{args.task_suite} task{args.task_id}, episode {args.episode}, frame {t}", out_path)
        print(f"Saved {out_path}")
        return

    # === video mode: loop every frame of the episode, model loaded once above ===
    frames_dir = args.video_out_dir or f"{args.out_prefix}_frames"
    os.makedirs(frames_dir, exist_ok=True)
    n_frames = len(ep["proprio"])
    print(f"Rendering frames t=1..{n_frames - 1} of {args.data_dir}/episode_{args.episode:03d} into {frames_dir}/")
    for t in range(1, n_frames):
        d = compute_frame(ctx, ep, t)
        print(f"  t={t}: occluded-region cos-sim before={d['mean_cos_before_occ']:.4f} after={d['mean_cos_after_occ']:.4f} "
              f"(delta={d['mean_cos_after_occ'] - d['mean_cos_before_occ']:+.4f})")
        frame_path = os.path.join(frames_dir, f"frame_{t:03d}.png")
        render_panel(d, f"{args.task_suite} task{args.task_id}, episode {args.episode}, frame {t}/{n_frames - 1}", frame_path)

    out_mp4 = f"{args.out_prefix}.mp4"
    # -start_number matches the first rendered frame (t=1); pad_len=2 (%03d) is
    # fixed width so glob-free sequential numbering works regardless of n_frames.
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.video_fps),
        "-start_number", "1", "-i", os.path.join(frames_dir, "frame_%03d.png"),
        "-vf", "format=yuv420p", out_mp4,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved {out_mp4}")


if __name__ == "__main__":
    main()
