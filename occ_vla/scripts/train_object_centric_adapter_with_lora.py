"""occ_vla addition (2026-09-06), Month 2 priority ③ (per the researched
priority order 1データ量→2既存位置分散→3狭いLoRA, "その優先順位でやってみて"):
combines `ObjectCentricZeroInitAdapter` (unchanged from
train_object_centric_adapter.py -- see that file and modeling_prismatic.py's
class docstring) with a NARROW LoRA restricted to the first
`--lora-layers` transformer layers of the (frozen) language_model,
immediately downstream of the adapter's own injection point.

Rationale (see CLAUDE.md's "Priority (3)" discussion): the adapter alone
can inject an arbitrarily well-designed, genuinely position-encoded
signal into the post-projector vision token stream, but the frozen 7B
LLM was never trained to specifically read a per-object positional
signal from this injection point -- there is no guarantee it has a
learned pathway to actually USE that signal, and repeated experiments
(pos_embed / ControlVLA-redesign / doubled-data, all with the adapter
ALONE) show the same "position-blind, in-distribution-vs-out-of-
distribution content-blind" pattern regardless of how the injected
signal is constructed. A tiny amount of LoRA on the FIRST 1-2 layers
(not all 32, unlike Month 1's train_distillation_imitation.py, which
LoRA'd every layer's q/k/v/o and catastrophically forgot the base
grasping skill on higher-correction-density tasks) gives the frozen
backbone a genuine, if narrow, learned pathway to start using the
injected content, while leaving the vast majority of the network
(layers 2-31, the action head, the vision backbone/projector) fully
frozen -- a much smaller, more targeted risk than Month 1's approach.

Everything else (data format, load_sample, the causal-sensitivity
regularizer, the zero/random ablation modes, the L1-regression loss,
the action_head.predict_action monkey-patch for gradient access) is
reused VERBATIM from train_object_centric_adapter.py, not reinvented.
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

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_vla, get_processor, get_action_head, get_proprio_projector,
)
from prismatic.extern.hf.modeling_prismatic import ObjectCentricZeroInitAdapter  # noqa: E402
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType, ACTION_DIM  # noqa: E402
from train_object_centric_adapter import normalize_action, load_sample  # noqa: E402

ACTION_DIM_FOR_RESHAPE = ACTION_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    ap.add_argument("--data-dir", default="month2_pairs_task2_n15")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=60)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--out-adapter", default="object_centric_adapter_task2_lora")
    ap.add_argument("--val-holdout-episodes", type=int, default=1)
    ap.add_argument("--causal-sensitivity-weight", type=float, default=0.0)
    ap.add_argument("--causal-sensitivity-margin", type=float, default=0.05)
    ap.add_argument("--ablation-mask-mode", default="real", choices=["real", "zero", "random"])
    # occ_vla addition (2026-09-06, priority (3)): narrow LoRA on the first
    # N transformer layers immediately downstream of the adapter's
    # injection point (the post-projector vision token stream).
    ap.add_argument("--lora-layers", default="0,1",
                     help="Comma-separated 0-indexed layer indices of vla.language_model.model.layers "
                          "to apply LoRA to (out of 32 total). Default '0,1' -- the first 2 layers only, "
                          "immediately downstream of the ObjectCentricZeroInitAdapter's injection point. "
                          "Everything else (layers 2-31, action_head, vision_backbone, projector) stays "
                          "100% frozen, unlike Month 1's train_distillation_imitation.py which LoRA'd "
                          "every layer.")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    args = ap.parse_args()
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(SCRIPTS_DIR, args.data_dir)
    if not os.path.isabs(args.out_adapter):
        args.out_adapter = os.path.join(SCRIPTS_DIR, args.out_adapter)
    lora_layer_idxs = [int(x) for x in args.lora_layers.split(",") if x.strip() != ""]

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

    for p in vla.parameters():
        p.requires_grad = False
    for p in action_head.parameters():
        p.requires_grad = False

    # occ_vla addition (2026-09-06, priority (3)): narrow LoRA, applied
    # BEFORE freezing the adapter's own params (order doesn't matter here
    # since get_peft_model only touches vla.language_model, but kept
    # explicit). Verified via a standalone check (see CLAUDE.md) that
    # layers_to_transform correctly restricts LoRA to exactly the
    # requested layer indices, not all 32.
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=lora_layer_idxs,
        layers_pattern="layers",
        task_type=None,  # not a standard HF task head; used only for parameter injection
    )
    vla.language_model = get_peft_model(vla.language_model, lora_config)
    n_lora_params = 0
    for name, p in vla.language_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            n_lora_params += p.numel()
        else:
            p.requires_grad = False
    print(f"  [lora] layers {lora_layer_idxs} (of 32), rank={args.lora_rank}, "
          f"{n_lora_params:,} trainable LoRA params")

    vla.object_centric_adapter = ObjectCentricZeroInitAdapter(vla.llm_dim).to(device, dtype=torch.bfloat16)
    lora_params = [p for name, p in vla.language_model.named_parameters() if "lora_" in name]
    adapter_params = list(vla.object_centric_adapter.parameters())
    trainable_params = adapter_params + lora_params
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in vla.parameters()) + sum(p.numel() for p in action_head.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.4f}%) "
          f"[ObjectCentricZeroInitAdapter + LoRA(layers {lora_layer_idxs}) -- "
          f"layers 2-31, action_head, vision_backbone, projector 100% frozen]")

    _orig_action_head_predict = action_head.predict_action.__func__

    def _stashing_predict_action(self_, actions_hidden_states):
        raw = _orig_action_head_predict(self_, actions_hidden_states)
        action_head._diagnostic_last_raw_output = raw
        return raw

    import types
    action_head.predict_action = types.MethodType(_stashing_predict_action, action_head)

    # Same weight-decay-exclusion recipe as train_object_centric_adapter.py's
    # v2-fixed run (kv_proj + all biases excluded from decay -- the
    # zero-init layer specifically), applied to the adapter's own params;
    # LoRA params get plain, standard weight_decay=0.01 (they are NOT
    # zero-initialized -- PEFT's own default LoRA init is lora_A~kaiming,
    # lora_B=zeros, so the composed update starts at zero exactly like any
    # standard LoRA, but this project's specific "decay fights zero-init"
    # failure mode was diagnosed for out_proj/kv_proj specifically, not
    # for lora_B in general -- no evidence yet that LoRA's own B=0 init
    # has the same fragility, so it is not special-cased here).
    decay_params, no_decay_params = [], []
    for name, p in vla.object_centric_adapter.named_parameters():
        if "kv_proj" in name or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    decay_params += lora_params
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
        adapter_state = {name: p.detach().cpu() for name, p in vla.object_centric_adapter.named_parameters()}
        torch.save(adapter_state, os.path.join(out_dir, "object_centric_adapter_weights.pt"))
        lora_state = {name: p.detach().cpu() for name, p in vla.language_model.named_parameters() if "lora_" in name}
        torch.save(lora_state, os.path.join(out_dir, "lora_weights.pt"))

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
        kv_proj_norm = vla.object_centric_adapter.kv_proj.weight.norm().item()
        lora_norm = torch.stack([p.detach().float().norm() for p in lora_params]).norm().item()
        sens_str = f" sensitivity_loss={sensitivity_loss.item():.5f}" if args.causal_sensitivity_weight > 0 else ""
        print(f"step {step}: lr={cur_lr:.2e} train_loss={loss.item():.6f} action_loss={action_loss.item():.6f}{sens_str} "
              f"(uid={entry['uid']}, "
              f"corrected={entry['correction_applied_this_chunk']}, "
              f"obj_coverage={entry['object_mask_coverage_frac']:.3f}){grad_norm_str} "
              f"kv_proj_norm={kv_proj_norm:.4f} lora_norm={lora_norm:.4f}{val_loss_str}")

        if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
            ckpt_dir = os.path.join(args.out_adapter, f"step{step + 1}")
            save_checkpoint(ckpt_dir)
            print(f"  [checkpoint] saved to {ckpt_dir}/")

    print(f"\ntrain loss trend: first={train_losses[0]:.6f} last={train_losses[-1]:.6f}")
    if val_losses:
        print(f"held-out loss trend: first={val_losses[0][1]:.6f} last={val_losses[-1][1]:.6f}")

    save_checkpoint(args.out_adapter)
    with open(os.path.join(args.out_adapter, "loss_log.json"), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses, "n_trainable_params": n_trainable,
                    "lora_layers": lora_layer_idxs, "lora_rank": args.lora_rank}, f, indent=2)
    print(f"saved to {args.out_adapter}/")


if __name__ == "__main__":
    main()
