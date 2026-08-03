"""
diagnose_vjepa_predictor_errors.py

Step 2 diagnostic (offline, no LIBERO rollout needed): for an already-
trained vjepa_predictor_dino/_siglip checkpoint, compute per-sample
prediction error (masked-patch L1 distance + cosine similarity vs.
ground truth) across every (task, episode, t) pair in the on-policy
rollout data, then:
  1. Bin by relative trajectory position (decile of t/episode_length)
     to see whether error concentrates late in the trajectory
     ("precision phase") -- and whether this differs between tasks,
     per the user's hypothesis motivating Step 1's precision-weighted
     loss result (mug_in_microwave 10%->20%, moka_pots 40%->20%).
  2. Aggregate per-patch-position (16x16 grid) mean L1 error to check
     for a spatial/geometric blind spot within the wrist camera's
     occluded region.

Uses the SAME occlusion mask geometry (apply_partial_patch, a fixed
centered square -- every sample in this dataset uses this identical
mask, so the spatial heatmap reflects error-by-position within that
one fixed footprint, not a sweep over different occlusion shapes) and
the same encoder/predictor forward path as training -- imports
directly from train_vjepa_predictor_multitask.py rather than
reimplementing, so this diagnostic is guaranteed to match what the
training loss actually optimizes.

Deliberately uses the balanced-sampling, non-precision-weighted
checkpoint (vjepa_predictor_multitask_balanced_6000steps.pt) by
default -- this shows the "natural" error distribution across the
trajectory BEFORE any timestep-based reweighting was applied, which is
the cleanest signal for deciding how (or whether) to reweight further.

Note on data: reuses the same on-policy rollout episodes the predictor
was trained on (no held-out split exists for this data) -- fine for
this purpose (understanding error SHAPE, not measuring held-out
quality), but keep in mind results reflect training-distribution fit.

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python -u \
    scripts/diagnose_vjepa_predictor_errors.py \
    --vjepa-checkpoint vjepa_predictor_multitask_balanced_6000steps.pt
"""

import argparse
import json
import os
import sys

OCC_VLA_ROOT = "/home/ubuntu/slocal1/Hoki/occ_vla"
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
    parse_task_spec,
    run_vit_to_layer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+",
        default=["libero_10:8:oft_onpolicy_rollout_data", "libero_10:9:oft_onpolicy_rollout_data_mug"],
    )
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-prefix", default="vjepa_error_diagnosis")
    args = parser.parse_args()

    task_specs = [parse_task_spec(s) for s in args.tasks]

    all_episodes, pairs_by_task, task_names = [], [], []
    for task_idx, (suite, task_id, data_dir) in enumerate(task_specs):
        episodes, pairs = load_dataset(data_dir)
        all_episodes.append(episodes)
        pairs_by_task.append(pairs)
        task_names.append(f"{suite}_task{task_id}")
        print(f"Task {task_idx} ({task_names[task_idx]}, {data_dir}): {len(episodes)} episodes, {len(pairs)} pairs")

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=task_specs[0][0],
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
    print(f"split_frac={split_frac} -> dino block {split_layer_dino}, siglip block {split_layer_siglip}")

    ckpt = torch.load(args.vjepa_checkpoint, map_location=device)
    vb.vjepa_predictor_dino.load_state_dict(ckpt["dino"])
    vb.vjepa_predictor_siglip.load_state_dict(ckpt["siglip"])
    print(f"Loaded vjepa predictor weights from {args.vjepa_checkpoint}")
    model.eval()
    vb.vjepa_predictor_dino.eval()
    vb.vjepa_predictor_siglip.eval()

    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]

    prompts_by_task = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite, task_id, _ in task_specs:
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        prompts_by_task.append(f"In: What action should the robot take to {task.language.lower()}?\nOut:")

    results = []
    with torch.no_grad():
        for task_idx in range(len(task_specs)):
            pairs = pairs_by_task[task_idx]
            episodes = all_episodes[task_idx]
            prompt = prompts_by_task[task_idx]
            for batch_start in range(0, len(pairs), args.batch_size):
                batch_pairs = pairs[batch_start: batch_start + args.batch_size]
                B = len(batch_pairs)
                agentview_t, wrist_t, agentview_tm1, wrist_tm1, wrist_t_corrupted, proprio_t = [], [], [], [], [], []
                fracs = []
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
                    ep_len = len(ep["proprio"])
                    fracs.append(t / max(ep_len - 1, 1))

                mask_256_np = build_patch_token_mask_256(pixel_bounds)
                mask_256 = torch.from_numpy(mask_256_np).to(device=device, dtype=dtype).reshape(1, -1, 1)
                prompts = [prompt] * B

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

                proprio_tensor = torch.tensor(np.stack(proprio_t), device=device, dtype=dtype)

                residual_dino = vb.vjepa_predictor_dino(f_input_dino, past_dino, proprio_tensor)
                f_final_dino = f_input_dino + mask_256 * residual_dino
                residual_siglip = vb.vjepa_predictor_siglip(f_input_siglip, past_siglip, proprio_tensor)
                f_final_siglip = f_input_siglip + mask_256 * residual_siglip

                l1_dino_patch = (f_final_dino - f_gt_dino).abs().mean(dim=-1)  # (B, 256)
                l1_siglip_patch = (f_final_siglip - f_gt_siglip).abs().mean(dim=-1)
                l1_patch = (l1_dino_patch + l1_siglip_patch) / 2

                cos_dino = torch.nn.functional.cosine_similarity(f_final_dino, f_gt_dino, dim=-1)
                cos_siglip = torch.nn.functional.cosine_similarity(f_final_siglip, f_gt_siglip, dim=-1)
                cos_patch = (cos_dino + cos_siglip) / 2

                l1_patch_np = l1_patch.float().cpu().numpy()
                cos_patch_np = cos_patch.float().cpu().numpy()
                for i in range(B):
                    results.append({
                        "task_idx": task_idx, "task_name": task_names[task_idx],
                        "frac": fracs[i],
                        "l1": float(l1_patch_np[i][mask_256_np].mean()),
                        "cos": float(cos_patch_np[i][mask_256_np].mean()),
                        "l1_patch": l1_patch_np[i].tolist(),
                    })
                print(f"  task{task_idx} ({task_names[task_idx]}) batch {batch_start}-{batch_start + B}/{len(pairs)}: {len(results)} total samples so far")

    with open(f"{args.out_prefix}_raw.json", "w") as f:
        json.dump(results, f)
    print(f"\nSaved raw per-sample results to {args.out_prefix}_raw.json ({len(results)} samples)")

    # === Aggregate: error by trajectory-progress decile, per task ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    colors = ["tab:blue", "tab:orange"]
    for task_idx, task_name in enumerate(task_names):
        task_results = [r for r in results if r["task_idx"] == task_idx]
        fracs = np.array([r["frac"] for r in task_results])
        l1s = np.array([r["l1"] for r in task_results])
        coss = np.array([r["cos"] for r in task_results])
        bin_idx = np.clip(np.digitize(fracs, bins) - 1, 0, 9)

        print(f"\n=== {task_name}: masked-patch error by trajectory decile ===")
        mean_l1_by_bin, mean_cos_by_bin = [], []
        for b in range(10):
            sel = bin_idx == b
            n = int(sel.sum())
            if n == 0:
                mean_l1_by_bin.append(np.nan)
                mean_cos_by_bin.append(np.nan)
                continue
            m_l1, m_cos = float(l1s[sel].mean()), float(coss[sel].mean())
            mean_l1_by_bin.append(m_l1)
            mean_cos_by_bin.append(m_cos)
            print(f"  decile {b} ({bins[b]:.1f}-{bins[b+1]:.1f}): n={n:4d}  mean_l1={m_l1:.5f}  mean_cos={m_cos:.5f}")

        axes[0].plot(bin_centers, mean_l1_by_bin, marker="o", label=task_name, color=colors[task_idx % len(colors)])
        axes[1].plot(bin_centers, mean_cos_by_bin, marker="o", label=task_name, color=colors[task_idx % len(colors)])

        # spatial heatmap, this task only
        patch_arr = np.array([r["l1_patch"] for r in task_results])
        mean_patch = patch_arr.mean(axis=0).reshape(16, 16)
        np.save(f"{args.out_prefix}_{task_name}_spatial_l1.npy", mean_patch)

    axes[0].set_xlabel("fraction through episode (0=start, 1=end)")
    axes[0].set_ylabel("mean masked-patch L1 error")
    axes[0].set_title("Reconstruction error vs. trajectory progress")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("fraction through episode (0=start, 1=end)")
    axes[1].set_ylabel("mean masked-patch cosine similarity")
    axes[1].set_title("Cosine similarity vs. trajectory progress")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_temporal_curves.png", dpi=150)
    print(f"\nSaved temporal error curves to {args.out_prefix}_temporal_curves.png")

    # === Spatial heatmaps, side by side ===
    fig, axes = plt.subplots(1, len(task_names), figsize=(6 * len(task_names), 5))
    if len(task_names) == 1:
        axes = [axes]
    vmax = max(
        np.load(f"{args.out_prefix}_{tn}_spatial_l1.npy").max() for tn in task_names
    )
    for ax, task_name in zip(axes, task_names):
        mean_patch = np.load(f"{args.out_prefix}_{task_name}_spatial_l1.npy")
        im = ax.imshow(mean_patch, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{task_name}\nmean L1 error by patch position (16x16 grid)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_spatial_heatmaps.png", dpi=150)
    print(f"Saved spatial heatmaps to {args.out_prefix}_spatial_heatmaps.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
