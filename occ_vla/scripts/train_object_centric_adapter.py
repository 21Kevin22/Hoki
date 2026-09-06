"""occ_vla addition (2026-09-03), Month 2 of the occlusion/collision-robustness
thesis plan: trains ONLY the newly-wired `ObjectCentricZeroInitAdapter`
(see modeling_prismatic.py's class docstring + CLAUDE.md's "Month 2 design"
entry for the full rationale) -- base VLA weights (vision_backbone,
projector, language_model, action_head) stay 100% frozen throughout. This is
architecturally distinct from Month 1's `train_distillation_imitation.py`
(LoRA on language_model attention + action_head, all pre-existing weights),
which is exactly the mechanism this adapter design is meant to avoid the
catastrophic-forgetting failure mode of.

Data: reuses `run_libero_occluded_oracle_headroom.py --save-distillation-
pairs-dir`'s same real (agentview, wrist, proprio, action_corrected) pairs,
PLUS the new `object_mask_path` field (real per-patch target-object
coverage, from real segmentation -- see that script's 2026-09-03 addition).

Loss: same L1-regression-in-normalized-action-space objective and the same
action_head.predict_action monkey-patch trick already established and
documented in train_distillation_imitation.py -- reused verbatim, not
reinvented.
"""
import argparse
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_vla, get_processor, get_action_head, get_proprio_projector,
)
from prismatic.extern.hf.modeling_prismatic import ObjectCentricZeroInitAdapter  # noqa: E402
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType, ACTION_DIM  # noqa: E402

ACTION_DIM_FOR_RESHAPE = ACTION_DIM


def normalize_action(action_np, action_norm_stats, device, dtype):
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        high, low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        high, low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    mask = action_norm_stats.get("mask", np.ones_like(low, dtype=bool))
    normed = np.where(mask, 2 * (action_np - low) / (high - low + 1e-8) - 1, action_np)
    return torch.tensor(normed, device=device, dtype=dtype)


def load_sample(entry, data_dir, processor, cfg, device, ablation_mask_mode="real"):
    agentview = Image.open(os.path.join(data_dir, entry["agentview_path"])).convert("RGB")
    wrist = Image.open(os.path.join(data_dir, entry["wrist_path"])).convert("RGB")
    prompt = "In: What action should the robot take?\nOut:"
    inputs = processor(prompt, agentview).to(device, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, wrist).to(device, dtype=torch.bfloat16)
    pixel_values = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

    # object_mask: real per-patch target-object coverage, agentview slot only
    # (256 tokens), wrist slot zero-filled -- matches how object_mask_path
    # was saved (agentview-side only, see run_libero_occluded_oracle_headroom.py's
    # 2026-09-03 addition).
    agent_mask_256 = np.load(os.path.join(data_dir, entry["object_mask_path"])).astype(np.float32)  # (256,)
    # occ_vla addition (2026-09-06, per user request): ablation to isolate
    # WHETHER the real per-patch object-location signal is what drives any
    # measured effect, vs. the adapter simply being "extra trainable
    # capacity fit to this distillation data regardless of what conditions
    # it". "zero" removes the object-location information entirely (the
    # adapter still trains, still sees the same actions/images, just gets a
    # constant all-zero key/value input); "random" replaces the real mask
    # with independent U(0,1) noise of the same shape (same information
    # RATE as the real signal, but no correlation with true object
    # location). Default "real" is byte-for-byte the original, unablated
    # behavior.
    if ablation_mask_mode == "zero":
        agent_mask_256 = np.zeros_like(agent_mask_256)
    elif ablation_mask_mode == "random":
        agent_mask_256 = np.random.uniform(0.0, 1.0, size=agent_mask_256.shape).astype(np.float32)
    elif ablation_mask_mode != "real":
        raise ValueError(f"unknown ablation_mask_mode: {ablation_mask_mode}")
    wrist_mask_256 = np.zeros_like(agent_mask_256)
    object_mask = np.concatenate([agent_mask_256, wrist_mask_256])[:, None]  # (512, 1)
    object_mask_t = torch.tensor(object_mask, device=device, dtype=torch.bfloat16).unsqueeze(0)  # (1, 512, 1)

    return inputs["input_ids"], inputs["attention_mask"], pixel_values, object_mask_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    ap.add_argument("--data-dir", default="month2_pairs_task2_n15")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=0,
                     help="occ_vla addition (2026-09-03), per user's diagnosis of the step150/300 "
                          "collapse: linear LR warmup from 0 to --lr over this many steps. 0 (default) "
                          "disables -- matches the original (collapsing) run's behavior unless set.")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                     help="occ_vla addition (2026-09-03): torch.nn.utils.clip_grad_norm_ max_norm. "
                          "0 (default) disables -- matches the original run's behavior unless set.")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--out-adapter", default="object_centric_adapter_task2")
    ap.add_argument("--val-holdout-episodes", type=int, default=1)
    ap.add_argument("--causal-sensitivity-weight", type=float, default=0.0,
                     help="occ_vla addition (2026-09-06), fix for the 'causal confusion' finding "
                          "(de Haan/Jayaraman/Levine, NeurIPS 2019, arXiv:1905.11979 -- see CLAUDE.md's "
                          "'Month 2 CRITICAL FINDING' entry): 0 (default) disables, fully backward-"
                          "compatible. When > 0, on every training step where "
                          "correction_applied_this_chunk=True, runs a SECOND forward pass with the SAME "
                          "sample's object_mask rolled to a random shift (same statistics, wrong "
                          "location) and adds a hinge loss max(0, margin - ||f(real)-f(shifted)||), "
                          "directly penalizing the network for producing near-identical outputs under a "
                          "real-vs-relocated mask -- the exact failure mode test_causal_mask_sensitivity.py "
                          "measured (shifted producing ||diff||=0.0 on the unfixed adapter).")
    ap.add_argument("--causal-sensitivity-margin", type=float, default=0.05,
                     help="Hinge margin (normalized-action-space L2) for --causal-sensitivity-weight.")
    ap.add_argument("--ablation-mask-mode", default="real", choices=["real", "zero", "random"],
                     help="occ_vla addition (2026-09-06): 'real' (default) is unablated. 'zero' trains "
                          "the adapter with the object-location signal removed entirely (constant "
                          "all-zero K/V) -- isolates whether the object-centric mask specifically "
                          "matters, vs. the adapter just being extra trainable capacity fit to this "
                          "distillation data. 'random' keeps the same information RATE but removes "
                          "correlation with true object location.")
    args = ap.parse_args()
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(SCRIPTS_DIR, args.data_dir)
    if not os.path.isabs(args.out_adapter):
        args.out_adapter = os.path.join(SCRIPTS_DIR, args.out_adapter)

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    print(f"loaded manifest: {len(manifest)} pairs")
    episodes = sorted(set(e["episode"] for e in manifest))
    val_episodes = set(episodes[-args.val_holdout_episodes:]) if args.val_holdout_episodes else set()
    train_manifest = [e for e in manifest if e["episode"] not in val_episodes]
    val_manifest = [e for e in manifest if e["episode"] in val_episodes]
    print(f"train pairs: {len(train_manifest)} (episodes {sorted(set(e['episode'] for e in train_manifest))}), "
          f"held-out pairs: {len(val_manifest)} (episodes {sorted(val_episodes)})")
    import random
    random.Random(0).shuffle(train_manifest)

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    device = vla.device
    unnorm_key = next(iter(vla.norm_stats.keys())) if len(vla.norm_stats) == 1 else "libero_10_no_noops"
    cfg.unnorm_key = unnorm_key
    action_norm_stats = vla.get_action_stats(unnorm_key)

    vla.language_model.gradient_checkpointing_enable()

    # The core Month-2 design point: EVERYTHING stays frozen except the new
    # adapter -- no LoRA, no action_head fine-tuning, no vision_backbone/
    # projector unfreezing. This is a much smaller optimization than Month
    # 1's LoRA sweep, by construction.
    for p in vla.parameters():
        p.requires_grad = False
    for p in action_head.parameters():
        p.requires_grad = False

    vla.object_centric_adapter = ObjectCentricZeroInitAdapter(vla.llm_dim, n_heads=args.n_heads).to(device, dtype=torch.bfloat16)
    trainable_params = list(vla.object_centric_adapter.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in vla.parameters()) + sum(p.numel() for p in action_head.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.4f}%) "
          f"[ObjectCentricZeroInitAdapter ONLY -- base model + action_head 100% frozen]")

    _orig_action_head_predict = action_head.predict_action.__func__

    def _stashing_predict_action(self_, actions_hidden_states):
        raw = _orig_action_head_predict(self_, actions_hidden_states)
        action_head._diagnostic_last_raw_output = raw
        return raw

    import types
    action_head.predict_action = types.MethodType(_stashing_predict_action, action_head)

    # occ_vla fix (2026-09-03), per user's real diagnosis: torch.optim.AdamW's
    # default weight_decay=0.01 was left unset in the ORIGINAL run that
    # collapsed (step150/300 -> 0/10) -- decoupled weight decay pulls EVERY
    # trainable param, including the zero-initialized layer, back toward
    # zero every step, fighting the gradient signal trying to move it away
    # from zero. Excluding all-bias params AND the zero-init layer's own
    # weight/bias from decay is the user's own proposed fix -- everything
    # else (geo_cnn's conv/linear layers) keeps normal decay.
    #
    # occ_vla fix (2026-09-06), ControlVLA-faithful redesign: the zero-init
    # layer moved from `out_proj` to `kv_proj` (see ObjectCentricZeroInitAdapter's
    # class docstring) -- updated the name match accordingly.
    decay_params, no_decay_params = [], []
    for name, p in vla.object_centric_adapter.named_parameters():
        if "kv_proj" in name or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)
    print(f"  [optimizer] {len(decay_params)} params with weight_decay=0.01, "
          f"{len(no_decay_params)} params (kv_proj + all biases) with weight_decay=0.0")

    def lr_at_step(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr

    def roll_object_mask(object_mask_t, shift):
        """occ_vla addition (2026-09-06): rolls the AGENTVIEW half (first 256
        patches) of an already-built (1, 512, 1) object_mask tensor by
        `shift` positions -- identical sum/coverage statistics, different
        spatial layout. Wrist half (always zero) is untouched. Same
        operation test_causal_mask_sensitivity.py used to demonstrate the
        unfixed adapter's output was invariant to this."""
        agent = object_mask_t[:, :256, :]
        wrist = object_mask_t[:, 256:, :]
        agent_shifted = torch.roll(agent, shifts=shift, dims=1)
        return torch.cat([agent_shifted, wrist], dim=1)

    def forward_raw_action(input_ids, attention_mask, pixel_values, proprio, object_mask):
        action_head._diagnostic_last_raw_output = None
        _ = vla.predict_action(
            input_ids=input_ids, unnorm_key=unnorm_key, proprio=proprio,
            proprio_projector=proprio_projector, action_head=action_head,
            use_film=cfg.use_film, pixel_values=pixel_values, attention_mask=attention_mask,
            object_mask=object_mask,
        )
        pred_raw = action_head._diagnostic_last_raw_output
        if pred_raw is None:
            raise RuntimeError("stashing hook did not fire -- action_head.predict_action monkey-patch is broken")
        return pred_raw.reshape(-1, ACTION_DIM_FOR_RESHAPE)

    def run_step(entry, train: bool):
        input_ids, attention_mask, pixel_values, object_mask = load_sample(
            entry, args.data_dir, processor, cfg, device, ablation_mask_mode=args.ablation_mask_mode)
        proprio = np.array(entry["state"], dtype=np.float32)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            pred_normalized = forward_raw_action(input_ids, attention_mask, pixel_values, proprio, object_mask)
            label_raw = np.array(entry["action_corrected"], dtype=np.float32)
            label_norm = normalize_action(label_raw, action_norm_stats, device, pred_normalized.dtype)
            action_loss = torch.nn.functional.l1_loss(pred_normalized[0].float(), label_norm.float())

            # occ_vla addition (2026-09-06): causal-sensitivity regularizer --
            # see --causal-sensitivity-weight's own help text and CLAUDE.md's
            # "Month 2 CRITICAL FINDING" entry for the full motivation. Only
            # applied on steps where the object's location should plausibly
            # matter (a real CBF correction fired), and only during training
            # (never on the held-out loss, which stays a pure measure of
            # action-prediction quality for comparability with prior runs).
            sensitivity_loss = torch.zeros((), device=device)
            if train and args.causal_sensitivity_weight > 0 and entry.get("correction_applied_this_chunk"):
                shift = int(np.random.randint(32, 224))
                shifted_mask = roll_object_mask(object_mask, shift)
                pred_shifted = forward_raw_action(input_ids, attention_mask, pixel_values, proprio, shifted_mask)
                diff = torch.norm(pred_normalized[0].float() - pred_shifted[0].float())
                sensitivity_loss = torch.clamp(args.causal_sensitivity_margin - diff, min=0.0)

            loss = action_loss + args.causal_sensitivity_weight * sensitivity_loss
        return loss, action_loss, sensitivity_loss

    def save_checkpoint(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        state_to_save = {name: p.detach().cpu() for name, p in vla.object_centric_adapter.named_parameters()}
        torch.save(state_to_save, os.path.join(out_dir, "object_centric_adapter_weights.pt"))

    train_losses, val_losses = [], []
    for step in range(args.n_steps):
        cur_lr = lr_at_step(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        entry = train_manifest[step % len(train_manifest)]
        loss, action_loss, sensitivity_loss = run_step(entry, train=True)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
        optimizer.step()
        train_losses.append(float(loss.item()))

        val_loss_str = ""
        if val_manifest and (step % 5 == 0 or step == args.n_steps - 1):
            val_entry = val_manifest[(step // 5) % len(val_manifest)]
            with torch.no_grad():
                vloss, _, _ = run_step(val_entry, train=False)
            val_losses.append((step, float(vloss.item())))
            val_loss_str = f" held_out_loss={vloss.item():.6f} (uid={val_entry['uid']})"
        grad_norm_str = f" grad_norm={grad_norm:.4f}" if grad_norm is not None else ""
        # occ_vla fix (2026-09-06): the zero-init layer moved from `out_proj`
        # to `kv_proj` in the ControlVLA-faithful redesign -- log that instead.
        kv_proj_norm = vla.object_centric_adapter.kv_proj.weight.norm().item()
        sens_str = f" sensitivity_loss={sensitivity_loss.item():.5f}" if args.causal_sensitivity_weight > 0 else ""
        print(f"step {step}: lr={cur_lr:.2e} train_loss={loss.item():.6f} action_loss={action_loss.item():.6f}{sens_str} "
              f"(uid={entry['uid']}, "
              f"corrected={entry['correction_applied_this_chunk']}, "
              f"obj_coverage={entry['object_mask_coverage_frac']:.3f}){grad_norm_str} "
              f"kv_proj_norm={kv_proj_norm:.4f}{val_loss_str}")

        if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
            ckpt_dir = os.path.join(args.out_adapter, f"step{step + 1}")
            save_checkpoint(ckpt_dir)
            print(f"  [checkpoint] saved to {ckpt_dir}/")

    print(f"\ntrain loss trend: first={train_losses[0]:.6f} last={train_losses[-1]:.6f}")
    if val_losses:
        print(f"held-out loss trend: first={val_losses[0][1]:.6f} last={val_losses[-1][1]:.6f}")

    save_checkpoint(args.out_adapter)
    with open(os.path.join(args.out_adapter, "loss_log.json"), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses, "n_trainable_params": n_trainable}, f, indent=2)
    print(f"saved to {args.out_adapter}/")


if __name__ == "__main__":
    main()
