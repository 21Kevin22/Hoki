"""
train_vjepa_predictor_smoke_test.py

Small-scale (200-300 step) training smoke test for VJEPA_LatentDynamicsPredictor,
per occ_vla's own "verify the wiring before scaling up" convention. Trains ONLY
`model.vjepa_predictor` (VLA backbone frozen) on Pattern B (partial-patch)
occlusion of the wrist camera, using on-policy rollout data collected by
collect_oft_onpolicy_rollout_data.py.

Loss = latent reconstruction (L1, masked to occluded tokens) + temporal
dynamics term. NOTE: under this script's teacher-forcing setup (`past_latents`
= ground-truth clean features from t-1, not the model's own prior prediction),
the dynamics term is mathematically identical to the reconstruction term --
see the review discussion before this script was written. Implemented anyway
for structural completeness and logged separately, but do not expect it to
move independently of recon_loss until a free-running (autoregressive
past_latents) training variant is built.

Goal of THIS script: verify gradients flow correctly into vjepa_predictor
(and only vjepa_predictor), no NaN/Inf, loss trends down -- NOT to produce a
deployable adapter. Real training (more data, more steps, held-out eval)
is a later step.

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/train_vjepa_predictor_smoke_test.py --num-steps 300
"""

import argparse
import glob
import os
import random
import sys

OFT_ROOT = "/home/ubuntu/slocal1/Hoki/occ_vla/thirdparty/openvla-oft"
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
PARTIAL_PATCH_FRAC = 0.59  # ~35% area, same geometry as run_oft_camera_dropout_eval.py's Check B


def apply_partial_patch(img_resized):
    h, w = img_resized.shape[:2]
    ph, pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
    r0, c0 = (h - ph) // 2, (w - pw) // 2
    out = img_resized.copy()
    out[r0 : r0 + ph, c0 : c0 + pw] = GRAY_FILL
    return out, (r0, r0 + ph, c0, c0 + pw)


def build_patch_token_mask(pixel_bounds, camera_block_index, num_images):
    """Boolean (256*num_images,) mask: True for tokens whose 14x14 pixel-patch
    CENTER falls inside the occluded pixel region, restricted to the given
    camera's 256-token block (0=agentview, 1=wrist, matching image order)."""
    r0, r1, c0, c1 = pixel_bounds
    mask_grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
    for i in range(GRID_SIDE):
        for j in range(GRID_SIDE):
            center_r = i * PATCH_PX + PATCH_PX / 2
            center_c = j * PATCH_PX + PATCH_PX / 2
            if r0 <= center_r < r1 and c0 <= center_c < c1:
                mask_grid[i, j] = True
    per_image_mask = mask_grid.reshape(-1)  # (256,)
    full_mask = np.zeros(NUM_PATCHES_PER_IMAGE * num_images, dtype=bool)
    full_mask[camera_block_index * NUM_PATCHES_PER_IMAGE : (camera_block_index + 1) * NUM_PATCHES_PER_IMAGE] = per_image_mask
    return full_mask


def build_pixel_values(agentview_img, wrist_img, processor, prompt, device, dtype):
    images = prepare_images_for_vla([agentview_img, wrist_img], cfg_stub)
    primary, wrist = images
    inputs_primary = processor(prompt, primary).to(device, dtype=dtype)
    inputs_wrist = processor(prompt, wrist).to(device, dtype=dtype)
    return torch.cat([inputs_primary["pixel_values"], inputs_wrist["pixel_values"]], dim=1)


class _CfgStub:
    """prepare_images_for_vla only reads cfg.center_crop -- avoid importing the full GenerateConfig for this helper."""

    center_crop = True


cfg_stub = _CfgStub()


def load_dataset(data_dir):
    episodes = []
    for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        d = np.load(path)
        episodes.append({"agentview": d["agentview"], "wrist": d["wrist"], "proprio": d["proprio"]})
    pairs = []  # (episode_idx, t) for t >= 1
    for ep_idx, ep in enumerate(episodes):
        for t in range(1, len(ep["proprio"])):
            pairs.append((ep_idx, t))
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
    parser.add_argument("--save-path", default="vjepa_predictor_smoke_test.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episodes, pairs = load_dataset(args.data_dir)
    assert len(pairs) > 0, f"No training pairs found in {args.data_dir} -- run collect_oft_onpolicy_rollout_data.py first"
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

    benchmark_dict = benchmark.get_benchmark_dict()
    task = benchmark_dict[args.task_suite]().get_task(args.task_id)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"

    # === Freeze everything except vjepa_predictor ===
    for p in model.parameters():
        p.requires_grad = False
    for p in model.vjepa_predictor.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.vjepa_predictor.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    optimizer = torch.optim.AdamW(model.vjepa_predictor.parameters(), lr=args.lr)

    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]

    losses, recon_losses, dynamics_losses, grad_norms = [], [], [], []
    saw_nan = False

    for step in range(args.num_steps):
        ep_idx, t = random.choice(pairs)
        ep = episodes[ep_idx]

        agentview_t, wrist_t = ep["agentview"][t], ep["wrist"][t]
        agentview_tm1, wrist_tm1 = ep["agentview"][t - 1], ep["wrist"][t - 1]
        proprio_t = normalize_proprio(ep["proprio"][t], proprio_norm_stats)

        wrist_t_corrupted, pixel_bounds = apply_partial_patch(wrist_t)
        occlusion_mask_np = build_patch_token_mask(pixel_bounds, camera_block_index=1, num_images=2)
        occlusion_mask = torch.from_numpy(occlusion_mask_np).to(device=device, dtype=dtype).reshape(1, -1, 1)

        with torch.no_grad():
            pixel_values_clean_t = build_pixel_values(agentview_t, wrist_t, processor, prompt, device, dtype)
            f_gt = model.vision_backbone(pixel_values_clean_t)  # (1, 512, 2176), frozen backbone -- no grad needed

            pixel_values_tm1 = build_pixel_values(agentview_tm1, wrist_tm1, processor, prompt, device, dtype)
            past_latents = model.vision_backbone(pixel_values_tm1)  # ground-truth clean past (on-policy trajectory)

            pixel_values_corrupted_t = build_pixel_values(agentview_t, wrist_t_corrupted, processor, prompt, device, dtype)
            f_input = model.vision_backbone(pixel_values_corrupted_t)  # frozen backbone -- no grad needed

        proprio_tensor = torch.tensor(proprio_t, device=device, dtype=dtype).reshape(1, -1)

        residual = model.vjepa_predictor(f_input, past_latents, proprio_tensor)  # (1, 512, 2176), requires_grad
        f_final = f_input + occlusion_mask * residual

        n_occluded = occlusion_mask.sum()
        recon_loss = (occlusion_mask * (f_final - f_gt).abs()).sum() / (n_occluded * f_final.shape[-1])
        # NOTE (see module docstring): with past_latents == ground truth, this
        # reduces algebraically to recon_loss. Kept separate for logging /
        # structural fidelity to the originally requested loss formula.
        pred_delta = f_final - past_latents
        true_delta = f_gt - past_latents
        dynamics_loss = (occlusion_mask * (pred_delta - true_delta).abs()).sum() / (n_occluded * f_final.shape[-1])

        total_loss = recon_loss + args.lambda_dynamics * dynamics_loss

        optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.vjepa_predictor.parameters(), max_norm=10.0)
        if not torch.isfinite(grad_norm):
            saw_nan = True
            print(f"[step {step}] NON-FINITE GRADIENT NORM: {grad_norm}")
        optimizer.step()

        losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        dynamics_losses.append(dynamics_loss.item())
        grad_norms.append(grad_norm.item())

        if not np.isfinite(total_loss.item()):
            saw_nan = True

        if step % args.log_every == 0 or step == args.num_steps - 1:
            print(
                f"[step {step:4d}] total={total_loss.item():.6f} recon={recon_loss.item():.6f} "
                f"dynamics={dynamics_loss.item():.6f} grad_norm={grad_norm.item():.4f} "
                f"n_occluded_tokens={int(n_occluded.item())}"
            )

    # === Smoke-test assertions ===
    assert not saw_nan, "encountered NaN/Inf loss or gradient during training -- smoke test FAILED"
    assert all(g > 0 for g in grad_norms), "found a zero gradient norm step -- backprop into vjepa_predictor may not be wired correctly"

    first10_mean = float(np.mean(losses[:10]))
    last10_mean = float(np.mean(losses[-10:]))
    print(f"\nLoss trend: first-10-steps mean={first10_mean:.6f} -> last-10-steps mean={last10_mean:.6f}")
    assert last10_mean < first10_mean, (
        f"loss did not decrease overall ({first10_mean:.6f} -> {last10_mean:.6f}) -- "
        "wiring may be correct but the smoke test doesn't show learning happening"
    )

    torch.save(model.vjepa_predictor.state_dict(), args.save_path)
    print(f"\nSaved vjepa_predictor state dict to {args.save_path}")
    print("SMOKE TEST PASSED: gradients flow, no NaN/Inf, loss trends down.")


if __name__ == "__main__":
    main()
