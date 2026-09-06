"""occ_vla addition (2026-08-27), per user's explicit "no privileged
information" requirement: distills proactive_avoidance_depth's real-RGB-D
(zero-privileged) CBF correction back into the policy itself via
imitation-learning fine-tuning, rather than relying on a CBF safety layer
running at inference time forever.

Data: (agentview image, wrist image, proprio state, action_corrected) pairs
from `run_libero_occluded_oracle_headroom.py --conditions
proactive_avoidance_depth --save-distillation-pairs-dir ...` -- the
action label is whatever proactive_avoidance_depth's REAL RGB-D+segmentation
correction actually executed (== the VLA's own original action on chunks
where no correction fired, == the corrected action on chunks where it did).

Loss: L1 regression in NORMALIZED action space (matching how this
checkpoint was actually trained -- use_l1_regression=True, BOUNDS_Q99
normalization), NOT a feature-matching loss like
train_representation_alignment.py. This needs a real trick to get a
gradient at all: `OpenVLAForActionPrediction.predict_action()` calls
`_unnormalize_actions()`, which does `np.where(...)` on the model's
raw output -- silently converting it to a detached numpy array (confirmed
by reading modeling_prismatic.py directly, not assumed). We monkey-patch
a bound `_unnormalize_actions` on the live model instance that stashes the
pre-conversion tensor (still attached to the autograd graph) onto
`vla._diagnostic_last_normalized_actions` before calling the original --
the same "_diagnostic_*" stashing convention this codebase already uses
elsewhere (e.g. `_diagnostic_correction_applied_count`), not a new pattern.

Trainable params: vision_backbone + projector + action_head (Approach A's
own representation-alignment script only touched vision_backbone+projector,
since its loss was purely a feature-matching objective with no action
supervision at all -- an action-imitation loss needs the action_head itself
to also be trainable, or the label signal has nowhere to flow into that
would actually change the OUTPUT action). language_model stays fully frozen,
matching Approach A's own "LLM side untouched" scope.
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
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType, ACTION_DIM  # noqa: E402

ACTION_DIM_FOR_RESHAPE = ACTION_DIM


def normalize_action(action_np, action_norm_stats, device, dtype):
    """Mirrors _unnormalize_actions's own math, inverted -- maps a raw
    (unnormalized) 7-dim action into the same [-1, 1]-ish BOUNDS_Q99 space
    the model's raw output lives in, so the L1 loss compares like with like."""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        high, low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        high, low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    mask = action_norm_stats.get("mask", np.ones_like(low, dtype=bool))
    normed = np.where(mask, 2 * (action_np - low) / (high - low + 1e-8) - 1, action_np)
    return torch.tensor(normed, device=device, dtype=dtype)


def load_sample(entry, data_dir, processor, cfg, device):
    agentview = Image.open(os.path.join(data_dir, entry["agentview_path"])).convert("RGB")
    wrist = Image.open(os.path.join(data_dir, entry["wrist_path"])).convert("RGB")
    prompt = "In: What action should the robot take?\nOut:"
    inputs = processor(prompt, agentview).to(device, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, wrist).to(device, dtype=torch.bfloat16)
    pixel_values = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
    return inputs["input_ids"], inputs["attention_mask"], pixel_values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    ap.add_argument("--data-dir", default="distillation_pairs_task1_n30")
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--checkpoint-every", type=int, default=50,
                     help="occ_vla addition 2026-08-27: save an intermediate checkpoint every N steps "
                          "(into out_adapter/stepN/), so a 'golden checkpoint' search via real rollouts "
                          "at several points along training is actually possible. 0 disables.")
    ap.add_argument("--lora-rank", type=int, default=32,
                     help="occ_vla addition 2026-08-27, v3: LoRA rank on language_model's attention "
                          "projections (q/k/v/o_proj), matching the real, verified LIBERO.md OFT recipe's "
                          "own --lora_rank 32 default, not an arbitrary choice.")
    ap.add_argument("--out-adapter", default="distillation_imitation_task1")
    ap.add_argument("--val-holdout-episodes", type=int, default=1,
                     help="hold out the LAST N episode indices from training, matching this project's "
                          "own established train/val convention (e.g. train_arm_removal_lora_full.py's "
                          "split_train_val) -- report train-vs-held-out loss separately, not just train loss.")
    ap.add_argument("--interleave-sampling", action="store_true",
                     help="occ_vla addition (2026-08-27), per user's 'symmetric/interleave sampling' "
                          "proposal, adapted to this script's real per-sample-SGD design (there is no "
                          "minibatch here to split 4:4 within -- see module note above run_step). When set, "
                          "EVEN training steps draw from the 'clean'/uncorrected pool and ODD steps draw "
                          "from the 'corrected'/avoidance pool, each cycling through its own independently-"
                          "shuffled order -- guaranteeing an exact 1:1 clean:corrected ratio across training "
                          "regardless of the pools' true relative sizes (91.8%/8.2% in this dataset), instead "
                          "of leaving that ratio to chance under plain uniform shuffling. Default off "
                          "(matches the v3 run that already produced step450) so this is an explicit, "
                          "deliberate A/B against that baseline, not a silent behavior change.")
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
    # occ_vla bug fix (2026-08-27): the manifest is recorded in
    # episode/timestep order (real chronological rollout order), NOT
    # shuffled. The training loop below reads `train_manifest[step %
    # len(train_manifest)]` in that same fixed order every epoch --
    # confirmed by direct inspection that steps 0-8 of episode 0 happen
    # to be a long consecutive run of correction_applied_this_chunk=True
    # (CBF firing continuously during that specific approach), so the
    # first ~50 steps of training saw a real, un-representative
    # over-concentration of "avoid" labels relative to the dataset's
    # true 8.2% overall rate -- a likely contributor to the early
    # catastrophic-looking rollout failures, independent of whether the
    # dataset itself is balanced (it is: 91.8% of pairs are the VLA's
    # own ordinary, uncorrected action). Fixed with a one-time shuffle
    # (fixed seed for reproducibility) instead of collecting new "clean"
    # data -- the clean/ordinary-action data already exists in this
    # same manifest, it just wasn't being sampled representatively.
    rng = np.random.default_rng(0)
    train_manifest = list(train_manifest)
    rng.shuffle(train_manifest)
    n_corrected_train = sum(1 for e in train_manifest if e["correction_applied_this_chunk"])
    print(f"train pairs: {len(train_manifest)} (episodes {sorted(episodes[:-args.val_holdout_episodes] if args.val_holdout_episodes else episodes)}), "
          f"held-out pairs: {len(val_manifest)} (episodes {sorted(val_episodes)}), "
          f"train correction rate: {n_corrected_train}/{len(train_manifest)} ({100*n_corrected_train/len(train_manifest):.1f}%) -- SHUFFLED")

    clean_pool = [e for e in train_manifest if not e["correction_applied_this_chunk"]]
    corrected_pool = [e for e in train_manifest if e["correction_applied_this_chunk"]]
    if args.interleave_sampling:
        rng2 = np.random.default_rng(1)
        rng2.shuffle(clean_pool)
        rng3 = np.random.default_rng(2)
        rng3.shuffle(corrected_pool)
        print(f"[interleave-sampling] ENABLED: clean_pool={len(clean_pool)}, corrected_pool={len(corrected_pool)} "
              f"-- even steps draw from clean_pool, odd steps draw from corrected_pool (each cycling its own "
              f"shuffled order), guaranteeing exact 1:1 alternation instead of the dataset's natural "
              f"{100*n_corrected_train/len(train_manifest):.1f}% corrected ratio.")

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
    action_norm_stats = vla.get_action_stats(unnorm_key)

    # occ_vla design choice v3 (2026-08-27, per user's own revised
    # proposal after v2's action_head-only run showed unstable/rising
    # train_loss): action_head-only (v2) avoided the OOM but likely
    # under-fits -- the backbone's own attention (QKV) never adapts to
    # this task at all, matching the real LIBERO.md-documented OFT
    # recipe's own use of `--lora_rank 32` on the backbone (confirmed
    # real, not assumed, by reading LIBERO.md directly). v3 restores
    # backbone adaptation via 16-bit LoRA (rank=32, matching that same
    # documented recipe) on language_model's attention projections,
    # PLUS gradient checkpointing (confirmed real and supported --
    # `PrismaticPreTrainedModel.supports_gradient_checkpointing = True`,
    # standard HF `language_model.gradient_checkpointing_enable()` --
    # verified in modeling_prismatic.py before assuming it, not guessed)
    # instead of QLoRA, per the user's own correctly-reasoned objection
    # that 4-bit quantization risks corrupting continuous-action
    # precision in a way pure LoRA (base weights stay bf16) does not.
    # vision_backbone/projector stay frozen (unlike train_representation_
    # alignment.py) -- this run isolates "does backbone attention
    # adaptation help" from "does vision-encoder adaptation help",
    # rather than changing both at once.
    from peft import LoraConfig, get_peft_model

    vla.language_model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=None,  # not a standard HF task head; we only use this for parameter injection, not generate()
    )
    vla.language_model = get_peft_model(vla.language_model, lora_config)

    for p in vla.parameters():
        p.requires_grad = False
    trainable_params = []
    for name, p in vla.language_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            trainable_params.append(p)
    for p in action_head.parameters():
        p.requires_grad = True
        trainable_params.append(p)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in vla.parameters()) + sum(p.numel() for p in action_head.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%) "
          f"[language_model LoRA(r={args.lora_rank}) + action_head; vision_backbone/projector frozen]")

    # occ_vla trick (see module docstring): the FIRST attempt hooked
    # `vla._unnormalize_actions`, but that was already too late -- reading
    # `_regression_or_discrete_prediction` directly (modeling_prismatic.py:1245)
    # showed the L1-regression path does
    # `action_head.predict_action(...).float().cpu().detach().numpy()`
    # itself, so the tensor is already detached before _unnormalize_actions
    # ever sees it. Hook `action_head.predict_action` instead -- the actual
    # boundary where the gradient-carrying tensor still exists.
    _orig_action_head_predict = action_head.predict_action.__func__

    def _stashing_predict_action(self_, actions_hidden_states):
        raw = _orig_action_head_predict(self_, actions_hidden_states)
        action_head._diagnostic_last_raw_output = raw  # still attached to the autograd graph
        return raw

    import types
    action_head.predict_action = types.MethodType(_stashing_predict_action, action_head)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    def run_step(entry, train: bool):
        input_ids, attention_mask, pixel_values = load_sample(entry, args.data_dir, processor, cfg, device)
        proprio = np.array(entry["state"], dtype=np.float32)
        action_head._diagnostic_last_raw_output = None
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            _ = vla.predict_action(
                input_ids=input_ids, unnorm_key=unnorm_key, proprio=proprio,
                proprio_projector=proprio_projector, action_head=action_head,
                use_film=cfg.use_film, pixel_values=pixel_values, attention_mask=attention_mask,
            )
            pred_raw = action_head._diagnostic_last_raw_output  # still has grad (train=True) or is a plain no_grad tensor (train=False)
            if pred_raw is None:
                raise RuntimeError("stashing hook did not fire -- action_head.predict_action monkey-patch is broken")
            pred_normalized = pred_raw.reshape(-1, ACTION_DIM_FOR_RESHAPE)  # (NUM_ACTIONS_CHUNK, ACTION_DIM), matches _regression_or_discrete_prediction's own reshape
            label_raw = np.array(entry["action_corrected"], dtype=np.float32)  # (ACTION_DIM,) -- only the FIRST action of the chunk was logged
            label_norm = normalize_action(label_raw, action_norm_stats, device, pred_normalized.dtype)
            # Only the first action of the predicted chunk has a real label
            # (the collection script only logged action_corrected =
            # actions[0] per replan step, matching how it's actually
            # EXECUTED before the next replan -- the rest of the chunk is
            # this project's own already-documented "future timesteps this
            # step's geometry doesn't describe" caveat from the Plan 3
            # action-blending work, reused here rather than re-litigated).
            loss = torch.nn.functional.l1_loss(pred_normalized[0].float(), label_norm.float())
        return loss

    def save_checkpoint(out_dir):
        # occ_vla addition (2026-08-27, per user's "golden checkpoint"
        # request): factored out so it can be called PERIODICALLY, not
        # just once at the very end -- the whole point of comparing
        # step50/100/200/etc. via real rollouts requires each of those
        # checkpoints to actually exist on disk, which the original
        # single-save-at-completion design never provided.
        os.makedirs(out_dir, exist_ok=True)
        state_to_save = {name: p.detach().cpu() for name, p in vla.named_parameters() if p.requires_grad}
        state_to_save.update({f"action_head.{name}": p.detach().cpu() for name, p in action_head.named_parameters()})
        torch.save(state_to_save, os.path.join(out_dir, "distillation_weights.pt"))

    train_losses, val_losses = [], []
    for step in range(args.n_steps):
        if args.interleave_sampling:
            if step % 2 == 0:
                entry = clean_pool[(step // 2) % len(clean_pool)]
            else:
                entry = corrected_pool[(step // 2) % len(corrected_pool)]
        else:
            entry = train_manifest[step % len(train_manifest)]
        loss = run_step(entry, train=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        val_loss_str = ""
        if val_manifest and (step % 5 == 0 or step == args.n_steps - 1):
            val_entry = val_manifest[(step // 5) % len(val_manifest)]
            with torch.no_grad():
                vloss = run_step(val_entry, train=False)
            val_losses.append((step, float(vloss.item())))
            val_loss_str = f" held_out_loss={vloss.item():.6f} (uid={val_entry['uid']})"
        print(f"step {step}: train_loss={loss.item():.6f} (uid={entry['uid']}, corrected={entry['correction_applied_this_chunk']}){val_loss_str}")

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
