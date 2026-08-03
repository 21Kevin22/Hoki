"""
train_vjepa_predictor_scaled.py

Full-scale (not smoke-test) training run for the mid-layer VJEPA predictors
(vjepa_predictor_dino/_siglip, connected at split_frac=0.67 inside
PrismaticVisionBackbone). Adds three things the 300-step smoke test
(train_vjepa_predictor_midlayer.py) didn't need:
  1. Batched training (default batch_size=8) -- run_vit_to_layer() and
     VJEPA_LatentDynamicsPredictor.forward() already generalize to batch
     dim > 1 for free; only the data-loading side needed changing.
  2. Periodic IN-PROCESS rollout evaluation (every --eval-every steps, n =
     --eval-episodes real episodes on moka_pots under wrist partial
     occlusion) using the SAME already-loaded model -- no ~10s+ reload
     overhead per checkpoint, and lets you watch task success rate rise (or
     not) directly against the already-established ceilings: 0/10 untrained,
     8/10 oracle (split_frac=0.67).
  3. Explicit GPU/cache cleanup at the end (del + torch.cuda.empty_cache()),
     on top of the natural cleanup from the process exiting.

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/train_vjepa_predictor_scaled.py --num-steps 3000 --batch-size 8 \
    --eval-every 1000 --eval-episodes 5
"""

import argparse
import glob
import os
import random
import sys
import time

OCC_VLA_ROOT = "/home/ubuntu/slocal1/Hoki/occ_vla"
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

NUM_PATCHES_PER_IMAGE = 256
GRID_SIDE = 16
PATCH_PX = 14
GRAY_FILL = 127
PARTIAL_PATCH_FRAC = 0.59


def apply_partial_patch(img_resized):
    h, w = img_resized.shape[:2]
    ph, pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
    r0, c0 = (h - ph) // 2, (w - pw) // 2
    out = img_resized.copy()
    out[r0 : r0 + ph, c0 : c0 + pw] = GRAY_FILL
    return out, (r0, r0 + ph, c0, c0 + pw)


def build_patch_token_mask_256(pixel_bounds):
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


def build_pixel_values_batch(agentview_imgs, wrist_imgs, processor, prompt, device, dtype):
    """agentview_imgs/wrist_imgs: lists of B [H,W,3] uint8 arrays. Returns
    (B, 12, 224, 224) -- per-sample processor calls (HF processors here
    don't batch arbitrary PIL images cleanly), stacked via torch.cat."""
    primary_tensors, wrist_tensors = [], []
    for av, wr in zip(agentview_imgs, wrist_imgs):
        images = prepare_images_for_vla([av, wr], _CfgStub())
        primary, wrist = images
        primary_tensors.append(processor(prompt, primary)["pixel_values"])
        wrist_tensors.append(processor(prompt, wrist)["pixel_values"])
    primary_batch = torch.cat(primary_tensors, dim=0).to(device, dtype=dtype)
    wrist_batch = torch.cat(wrist_tensors, dim=0).to(device, dtype=dtype)
    return torch.cat([primary_batch, wrist_batch], dim=1)


def run_vit_to_layer(featurizer, x_pixels, layer_idx):
    x = featurizer.patch_embed(x_pixels)
    x = featurizer._pos_embed(x)
    x = featurizer.patch_drop(x)
    x = featurizer.norm_pre(x)
    for i, blk in enumerate(featurizer.blocks):
        x = blk(x)
        if i == layer_idx:
            return x[:, featurizer.num_prefix_tokens :]
    raise ValueError(f"layer_idx {layer_idx} out of range")


def load_dataset(data_dir):
    episodes = []
    for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        d = np.load(path)
        episodes.append({"agentview": d["agentview"], "wrist": d["wrist"], "proprio": d["proprio"]})
    pairs = [(ep_idx, t) for ep_idx, ep in enumerate(episodes) for t in range(1, len(ep["proprio"]))]
    return episodes, pairs


def run_rollout_eval(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_states, n_episodes, max_steps):
    """In-process eval reusing run_oft_camera_dropout_eval.py's run_episode,
    against the ALREADY-LOADED model (whatever its current predictor weights
    are) -- no reload. Returns (n_success, n_episodes)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="oft_onpolicy_rollout_data")
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-dynamics", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="vjepa_predictor_scaled.pt")
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
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)
    device = model.device
    dtype = torch.bfloat16
    vb = model.vision_backbone

    split_frac = vb.midlayer_split_frac
    split_layer_dino = int(len(vb.featurizer.blocks) * split_frac)
    split_layer_siglip = int(len(vb.fused_featurizer.blocks) * split_frac)
    print(f"split_frac={split_frac} -> dino block {split_layer_dino}, siglip block {split_layer_siglip}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = TASK_MAX_STEPS[TaskSuite(args.task_suite)]
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

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

    print(f"\n=== Eval before training (step 0, untrained/whatever checkpoint state) ===")
    n_succ, n_tot = run_rollout_eval(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_states, args.eval_episodes, max_steps)
    print(f"  step 0 eval: {n_succ}/{n_tot}")

    losses, grad_norms = [], []
    saw_nan = False
    train_t0 = time.time()

    for step in range(1, args.num_steps + 1):
        model.train()
        vb.vjepa_predictor_dino.train()
        vb.vjepa_predictor_siglip.train()

        batch_pairs = [random.choice(pairs) for _ in range(args.batch_size)]
        agentview_t, wrist_t, agentview_tm1, wrist_tm1, wrist_t_corrupted, proprio_t = [], [], [], [], [], []
        pixel_bounds = None
        for ep_idx, t in batch_pairs:
            ep = episodes[ep_idx]
            agentview_t.append(ep["agentview"][t])
            wrist_t.append(ep["wrist"][t])
            agentview_tm1.append(ep["agentview"][t - 1])
            wrist_tm1.append(ep["wrist"][t - 1])
            corrupted, pixel_bounds = apply_partial_patch(ep["wrist"][t])
            wrist_t_corrupted.append(corrupted)
            proprio_t.append(normalize_proprio(ep["proprio"][t], proprio_norm_stats))

        mask_256_np = build_patch_token_mask_256(pixel_bounds)  # same fixed geometry every step -- broadcasts over batch
        mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(1, -1, 1)
        B = args.batch_size

        with torch.no_grad():
            pv_clean_t = build_pixel_values_batch(agentview_t, wrist_t, processor, prompt, device, dtype)
            _, wrist_clean_t = torch.split(pv_clean_t, [6, 6], dim=1)
            wrist_reg_clean_t, wrist_fused_clean_t = torch.split(wrist_clean_t, [3, 3], dim=1)
            f_gt_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_t, split_layer_dino)
            f_gt_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_t, split_layer_siglip)

            pv_clean_tm1 = build_pixel_values_batch(agentview_tm1, wrist_tm1, processor, prompt, device, dtype)
            _, wrist_clean_tm1 = torch.split(pv_clean_tm1, [6, 6], dim=1)
            wrist_reg_clean_tm1, wrist_fused_clean_tm1 = torch.split(wrist_clean_tm1, [3, 3], dim=1)
            past_dino = run_vit_to_layer(vb.featurizer, wrist_reg_clean_tm1, split_layer_dino)
            past_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_clean_tm1, split_layer_siglip)

            pv_corrupted_t = build_pixel_values_batch(agentview_t, wrist_t_corrupted, processor, prompt, device, dtype)
            _, wrist_corrupted_t = torch.split(pv_corrupted_t, [6, 6], dim=1)
            wrist_reg_corrupted_t, wrist_fused_corrupted_t = torch.split(wrist_corrupted_t, [3, 3], dim=1)
            f_input_dino = run_vit_to_layer(vb.featurizer, wrist_reg_corrupted_t, split_layer_dino)
            f_input_siglip = run_vit_to_layer(vb.fused_featurizer, wrist_fused_corrupted_t, split_layer_siglip)

        proprio_tensor = torch.tensor(np.stack(proprio_t), device=device, dtype=dtype)  # (B, 8)

        residual_dino = vb.vjepa_predictor_dino(f_input_dino, past_dino, proprio_tensor)
        f_final_dino = f_input_dino + mask_256 * residual_dino
        residual_siglip = vb.vjepa_predictor_siglip(f_input_siglip, past_siglip, proprio_tensor)
        f_final_siglip = f_input_siglip + mask_256 * residual_siglip

        n_occ = mask_256.sum()
        norm_dino = n_occ * f_final_dino.shape[-1] * B
        norm_siglip = n_occ * f_final_siglip.shape[-1] * B
        recon_dino = (mask_256 * (f_final_dino - f_gt_dino).abs()).sum() / norm_dino
        recon_siglip = (mask_256 * (f_final_siglip - f_gt_siglip).abs()).sum() / norm_siglip
        recon_loss = recon_dino + recon_siglip

        dyn_dino = (mask_256 * ((f_final_dino - past_dino) - (f_gt_dino - past_dino)).abs()).sum() / norm_dino
        dyn_siglip = (mask_256 * ((f_final_siglip - past_siglip) - (f_gt_siglip - past_siglip)).abs()).sum() / norm_siglip
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
            print(
                f"[step {step:5d}/{args.num_steps}] total={total_loss.item():.6f} recon_dino={recon_dino.item():.6f} "
                f"recon_siglip={recon_siglip.item():.6f} grad_norm={grad_norm.item():.4f} "
                f"{steps_per_sec:.3f} steps/s, ETA {eta_min:.1f}min"
            )

        if step % args.eval_every == 0:
            print(f"\n=== Eval at step {step} ===")
            n_succ, n_tot = run_rollout_eval(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_states, args.eval_episodes, max_steps)
            print(f"  step {step} eval: {n_succ}/{n_tot}\n")
            ckpt_path = f"{args.save_path}.step{step}"
            torch.save({"dino": vb.vjepa_predictor_dino.state_dict(), "siglip": vb.vjepa_predictor_siglip.state_dict()}, ckpt_path)
            print(f"  saved checkpoint to {ckpt_path}")

    assert not saw_nan, "encountered NaN/Inf loss or gradient during training"

    first_mean = float(np.mean(losses[: min(50, len(losses))]))
    last_mean = float(np.mean(losses[-min(50, len(losses)) :]))
    print(f"\nLoss trend: first-50-steps mean={first_mean:.6f} -> last-50-steps mean={last_mean:.6f}")

    torch.save({"dino": vb.vjepa_predictor_dino.state_dict(), "siglip": vb.vjepa_predictor_siglip.state_dict()}, args.save_path)
    print(f"Saved final checkpoint to {args.save_path}")

    print(f"\n=== Final eval (n={args.final_eval_episodes}) ===")
    n_succ, n_tot = run_rollout_eval(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector, initial_states, args.final_eval_episodes, max_steps)
    print(f"  FINAL: {n_succ}/{n_tot}")

    # === Explicit GPU/cache cleanup ===
    del model, trainable_params, optimizer, vb
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("\nGPU memory freed (del + torch.cuda.empty_cache()). Process will now exit, releasing everything else.")


if __name__ == "__main__":
    main()
