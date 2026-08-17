"""
train_vjepa_predictor_midlayer.py

Training smoke test for the REDESIGNED VJEPA_LatentDynamicsPredictor
architecture: two predictors (vjepa_predictor_dino, vjepa_predictor_siglip),
connected at an INTERMEDIATE ViT block (split_frac=0.67 of each backbone's
own depth) inside PrismaticVisionBackbone, not at the final extracted layer.

Supersedes train_vjepa_predictor_smoke_test.py (final-layer design) --
occ_vla's own oracle diagnostic (2026-07-31) found final-layer splicing
collapses to 0/10 task success EVEN WITH PERFECT ground-truth content
(the two token halves are never computed in the same forward pass, so
global self-attention leaves them mutually inconsistent), while splicing
at split_frac=0.67 (leaving ~2-3 remaining blocks, matching the existing
num_blocks-2 extraction convention, for self-attention to reconcile the
boundary) recovered to 8/10 -- matching baseline exactly. That result is
this training run's target ceiling.

Trains ONLY vjepa_predictor_dino/_siglip (VLA + both ViT backbones fully
frozen) on Pattern B (partial-patch) occlusion of the wrist camera, using
the SAME on-policy rollout data already collected by
collect_oft_onpolicy_rollout_data.py (raw pixel images + proprio -- no
re-collection needed, only how it's consumed at train time changes).

Loss = latent reconstruction (L1, masked to occluded tokens, summed over
both backbones) + temporal dynamics term. As before, under this script's
teacher-forcing setup (past_latents = ground-truth clean intermediate
features from t-1, not the predictors' own prior output), the dynamics
term is mathematically identical to the reconstruction term -- kept for
structural completeness and logged separately, not because it adds
independent gradient signal here.

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/train_vjepa_predictor_midlayer.py --num-steps 300
"""

import argparse
import glob
import os
import random
import sys

# Derived from __file__ (was hardcoded to the original project server's
# path, "/home/ubuntu/slocal1/Hoki/occ_vla/thirdparty/openvla-oft" -- broke
# on any other machine, e.g. a Kaggle clone under /root/oft_work/Hoki/...).
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor,
    normalize_proprio,
    prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_model  # noqa: E402

NUM_PATCHES_PER_IMAGE = 256
GRID_SIDE = 16  # 224 / 14
PATCH_PX = 14
GRAY_FILL = 127
PARTIAL_PATCH_FRAC = 0.59  # ~35% area, same geometry used throughout this investigation


def apply_partial_patch(img_resized):
    h, w = img_resized.shape[:2]
    ph, pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
    r0, c0 = (h - ph) // 2, (w - pw) // 2
    out = img_resized.copy()
    out[r0 : r0 + ph, c0 : c0 + pw] = GRAY_FILL
    return out, (r0, r0 + ph, c0, c0 + pw)


def build_patch_token_mask_256(pixel_bounds):
    """Boolean (256,) array -- single-image mask (this script only ever
    targets the wrist image's own 256-token space, matching how the
    predictors operate on ONE image's intermediate ViT tokens at a time)."""
    r0, r1, c0, c1 = pixel_bounds
    mask_grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
    for i in range(GRID_SIDE):
        for j in range(GRID_SIDE):
            center_r = i * PATCH_PX + PATCH_PX / 2
            center_c = j * PATCH_PX + PATCH_PX / 2
            if r0 <= center_r < r1 and c0 <= center_c < c1:
                mask_grid[i, j] = True
    return mask_grid.reshape(-1)


class _CfgStub:
    center_crop = True


def build_pixel_values(agentview_img, wrist_img, processor, prompt, device, dtype):
    images = prepare_images_for_vla([agentview_img, wrist_img], _CfgStub())
    primary, wrist = images
    inputs_primary = processor(prompt, primary).to(device, dtype=dtype)
    inputs_wrist = processor(prompt, wrist).to(device, dtype=dtype)
    return torch.cat([inputs_primary["pixel_values"], inputs_wrist["pixel_values"]], dim=1)


def run_vit_to_layer(featurizer, x_pixels, layer_idx):
    """Runs one ViT featurizer up to (and including) block `layer_idx`,
    returns the patch-token-only portion (B, 256, D) -- plain ground-truth
    extraction, no splicing (unlike midlayer_oracle_splice.py's diagnostic
    version, which also splices)."""
    x = featurizer.patch_embed(x_pixels)
    x = featurizer._pos_embed(x)
    x = featurizer.patch_drop(x)
    x = featurizer.norm_pre(x)
    for i, blk in enumerate(featurizer.blocks):
        x = blk(x)
        if i == layer_idx:
            return x[:, featurizer.num_prefix_tokens :]
    raise ValueError(f"layer_idx {layer_idx} out of range (only {len(featurizer.blocks)} blocks)")


def load_dataset(data_dir):
    episodes = []
    for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        d = np.load(path)
        episodes.append({"agentview": d["agentview"], "wrist": d["wrist"], "proprio": d["proprio"]})
    pairs = [(ep_idx, t) for ep_idx, ep in enumerate(episodes) for t in range(1, len(ep["proprio"]))]
    return episodes, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="oft_onpolicy_rollout_data")
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-dynamics", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="vjepa_predictor_midlayer.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episodes, pairs = load_dataset(args.data_dir)
    assert len(pairs) > 0, f"No training pairs found in {args.data_dir}"
    print(f"Loaded {len(episodes)} episodes, {len(pairs)} (t-1, t) training pairs")

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=args.task_suite,
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
    print(f"split_frac={split_frac} -> dino block {split_layer_dino}/{len(vb.featurizer.blocks)}, "
          f"siglip block {split_layer_siglip}/{len(vb.fused_featurizer.blocks)}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task = benchmark_dict[args.task_suite]().get_task(args.task_id)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"

    # === Freeze everything except the two mid-layer predictors ===
    for p in model.parameters():
        p.requires_grad = False
    trainable_params = list(vb.vjepa_predictor_dino.parameters()) + list(vb.vjepa_predictor_siglip.parameters())
    for p in trainable_params:
        p.requires_grad = True
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]

    losses, grad_norms = [], []
    saw_nan = False

    for step in range(args.num_steps):
        ep_idx, t = random.choice(pairs)
        ep = episodes[ep_idx]

        agentview_t, wrist_t = ep["agentview"][t], ep["wrist"][t]
        agentview_tm1, wrist_tm1 = ep["agentview"][t - 1], ep["wrist"][t - 1]
        proprio_t = normalize_proprio(ep["proprio"][t], proprio_norm_stats)

        wrist_t_corrupted, pixel_bounds = apply_partial_patch(wrist_t)
        mask_256_np = build_patch_token_mask_256(pixel_bounds)
        mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(1, -1, 1)

        with torch.no_grad():
            pv_clean_t = build_pixel_values(agentview_t, wrist_t, processor, prompt, device, dtype)
            _, wrist_clean_t = torch.split(pv_clean_t, [6, 6], dim=1)
            wrist_reg_clean_t, wrist_fused_clean_t = torch.split(wrist_clean_t, [3, 3], dim=1)
            f_gt_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_t, split_layer_dino)
            f_gt_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_t, split_layer_siglip)

            pv_clean_tm1 = build_pixel_values(agentview_tm1, wrist_tm1, processor, prompt, device, dtype)
            _, wrist_clean_tm1 = torch.split(pv_clean_tm1, [6, 6], dim=1)
            wrist_reg_clean_tm1, wrist_fused_clean_tm1 = torch.split(wrist_clean_tm1, [3, 3], dim=1)
            past_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_tm1, split_layer_dino)
            past_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_tm1, split_layer_siglip)

            pv_corrupted_t = build_pixel_values(agentview_t, wrist_t_corrupted, processor, prompt, device, dtype)
            _, wrist_corrupted_t = torch.split(pv_corrupted_t, [6, 6], dim=1)
            wrist_reg_corrupted_t, wrist_fused_corrupted_t = torch.split(wrist_corrupted_t, [3, 3], dim=1)
            f_input_dino = run_vit_to_layer(vb.featurizer, wrist_reg_corrupted_t, split_layer_dino)
            f_input_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_corrupted_t, split_layer_siglip)

        proprio_tensor = torch.tensor(proprio_t, device=device, dtype=dtype).reshape(1, -1)

        residual_dino = vb.vjepa_predictor_dino(f_input_dino, past_dino, proprio_tensor)
        f_final_dino = f_input_dino + mask_256 * residual_dino
        residual_siglip = vb.vjepa_predictor_siglip(f_input_siglip, past_siglip, proprio_tensor)
        f_final_siglip = f_input_siglip + mask_256 * residual_siglip

        n_occ = mask_256.sum()
        recon_dino = (mask_256 * (f_final_dino - f_gt_dino).abs()).sum() / (n_occ * f_final_dino.shape[-1])
        recon_siglip = (mask_256 * (f_final_siglip - f_gt_siglip).abs()).sum() / (n_occ * f_final_siglip.shape[-1])
        recon_loss = recon_dino + recon_siglip

        # See module docstring: degenerate with recon_loss under teacher forcing (past == ground truth).
        dyn_dino = (mask_256 * ((f_final_dino - past_dino) - (f_gt_dino - past_dino)).abs()).sum() / (n_occ * f_final_dino.shape[-1])
        dyn_siglip = (mask_256 * ((f_final_siglip - past_siglip) - (f_gt_siglip - past_siglip)).abs()).sum() / (n_occ * f_final_siglip.shape[-1])
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

        if step % args.log_every == 0 or step == args.num_steps - 1:
            print(
                f"[step {step:4d}] total={total_loss.item():.6f} recon_dino={recon_dino.item():.6f} "
                f"recon_siglip={recon_siglip.item():.6f} grad_norm={grad_norm.item():.4f} "
                f"n_occluded_tokens={int(n_occ.item())}"
            )

    assert not saw_nan, "encountered NaN/Inf loss or gradient -- smoke test FAILED"
    assert all(g > 0 for g in grad_norms), "found a zero gradient norm step -- backprop may not be wired correctly"

    first10_mean = float(np.mean(losses[:10]))
    last10_mean = float(np.mean(losses[-10:]))
    print(f"\nLoss trend: first-10-steps mean={first10_mean:.6f} -> last-10-steps mean={last10_mean:.6f}")
    assert last10_mean < first10_mean, f"loss did not decrease overall ({first10_mean:.6f} -> {last10_mean:.6f})"

    torch.save(
        {"dino": vb.vjepa_predictor_dino.state_dict(), "siglip": vb.vjepa_predictor_siglip.state_dict()},
        args.save_path,
    )
    print(f"\nSaved {{'dino', 'siglip'}} state dicts to {args.save_path}")
    print("SMOKE TEST PASSED: gradients flow, no NaN/Inf, loss trends down.")


if __name__ == "__main__":
    main()
