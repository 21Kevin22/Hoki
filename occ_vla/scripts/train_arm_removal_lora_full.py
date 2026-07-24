"""Full-scale LoRA training run (per user request, 2026-07-17): trains
on the full arm_removal_pairs_policy/ dataset (real pi0.5-rollout-driven
arm poses across 5 tasks, WMPO-style distribution alignment) instead of
train_arm_removal_lora_tiny.py's deliberate 15-sample/1-episode overfit
smoke test.

Same techniques that resolved the tiny run's VRAM ceiling (2026-07-17,
confirmed reproducible): 4-bit (nf4) quantization + MaskedFFOut
saliency-aware backward (final vocab projection only computed at
masked/arm-region token positions). rank=128 confirmed to fit in a
quick standalone check (~6.8GB for base+LoRA setup, comfortable
headroom on a 24GB GPU).

Unlike the tiny smoke test, this HOLDS OUT the last episode of each
task as validation (never trained on) so acceptance/quality on unseen
episodes can be checked honestly, not just memorization on the training
set. Periodically runs the real iterative denoise() on a few held-out
samples and scores them with PlausibilityChecker, logging progress so
training can be stopped early by inspection rather than running blind
for a fixed step count.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/train_arm_removal_lora_full.py [--steps 1500] [--rank 128]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_MMADA_ROOT = _ROOT / "third_party" / "mmada"
sys.path.insert(0, str(_MMADA_ROOT))

from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402
from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator, TOKEN_GRID_SIDE  # noqa: E402
from occ_vla.world_model.mmada import MASK_TOKEN_ID, MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import CODEBOOK_SIZE, MagvitV2Tokenizer, SUBGOAL_IMAGE_SIDE, SUBGOAL_NUM_TOKENS  # noqa: E402

PAIRS_DIR = _ROOT / "arm_removal_pairs_policy"
LR = 1e-4
DEFAULT_STEPS = 1500
DEFAULT_RANK = 64  # stepped down from 128 (2026-07-17): 128 OOM'd even with 8-bit Adam, at a normal attention
                   # layer forward pass, before backward -- confirmed via isolated test that 64 fits (~21.5GB/24GB)
LORA_ALPHA_MULT = 2  # alpha = rank * 2, matching the 128/256 ratio suggested
EVAL_EVERY = 100
MAX_ACTIVE_POSITIONS = 150  # see load_sample's docstring -- bounds per-step memory regardless of mask size
ADAPTER_OUT = _ROOT / "arm_removal_lora_full_adapter"
EVAL_DIR = _ROOT / "arm_removal_pairs_policy" / "eval_progress"

gen = ArmFreeSubgoalGenerator(world_model=None)
STUB_RE = re.compile(r"^(.*)_ep(\d+)_s\d+$")


class MaskedFFOut(torch.nn.Module):
    """Same technique as train_arm_removal_lora_tiny.py's MaskedFFOut --
    see that file for the full rationale. Defaults to full-sequence
    (inference-safe); training loop sets active_positions explicitly."""

    def __init__(self, orig_ff_out: torch.nn.Module):
        super().__init__()
        self.orig = orig_ff_out
        self.active_positions: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_positions is None:
            return self.orig(x)
        return self.orig(x[:, self.active_positions, :])


def load_sample(stub: str, tokenizer: MagvitV2Tokenizer, image_token_offset: int):
    armvis = np.array(Image.open(PAIRS_DIR / f"{stub}_armvis.png").convert("RGB"))
    armfree = np.array(Image.open(PAIRS_DIR / f"{stub}_armfree.png").convert("RGB"))
    armmask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png")) > 127

    armvis_512 = cv2.resize(armvis, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    armfree_512 = cv2.resize(armfree, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(armmask.astype(np.uint8), (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_NEAREST) > 0

    token_mask = gen._arm_token_mask(mask_512)  # noqa: SLF001
    vis_ids = tokenizer.encode(armvis_512)
    free_ids = tokenizer.encode(armfree_512)

    masked_input_ids = np.where(token_mask, MASK_TOKEN_ID, vis_ids + image_token_offset)
    masked_positions = np.where(token_mask)[0]
    # Real pi0.5-rollout poses vary the arm's on-screen footprint far
    # more than the tiny smoke test's 15 curated samples did (measured
    # 2026-07-17: 75-220 active tokens across this dataset, median 129).
    # Repeated OOMs during full-scale training (even after capping the
    # *upper* bound) were traced to the PyTorch caching allocator
    # fragmenting across steps when MaskedFFOut's sliced tensor shape
    # varies sample to sample -- forcing every sample to EXACTLY
    # MAX_ACTIVE_POSITIONS (subsample without replacement if over,
    # resample with replacement if under) keeps that shape constant
    # every step, letting the allocator reuse the same blocks instead
    # of fragmenting. Resampling with replacement means a few masked
    # positions get double-weighted in the loss for small-mask samples
    # -- a minor training-signal quirk, not a correctness bug.
    rng = np.random.default_rng(hash(stub) % (2**32))
    if len(masked_positions) >= MAX_ACTIVE_POSITIONS:
        keep = rng.choice(len(masked_positions), size=MAX_ACTIVE_POSITIONS, replace=False)
    else:
        keep = rng.choice(len(masked_positions), size=MAX_ACTIVE_POSITIONS, replace=True)
    masked_positions = masked_positions[np.sort(keep)]
    active_labels = free_ids[masked_positions]
    return (
        torch.from_numpy(masked_input_ids).long().unsqueeze(0),
        torch.from_numpy(masked_positions).long(),
        torch.from_numpy(active_labels).long(),
    )


def split_train_val(manifest: list[dict]) -> tuple[list[dict], list[dict]]:
    """Held-out validation = the last episode collected per task label --
    never seen during training, so acceptance there tests generalization,
    not memorization."""
    max_ep = {}
    for e in manifest:
        m = STUB_RE.match(e["stub"])
        if m:
            label, ep = m.group(1), int(m.group(2))
            max_ep[label] = max(max_ep.get(label, -1), ep)

    train, val = [], []
    for e in manifest:
        m = STUB_RE.match(e["stub"])
        ep = int(m.group(2)) if m else -1
        label = m.group(1) if m else e["label"]
        (val if ep == max_ep.get(label, -1) else train).append(e)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    args = parser.parse_args()

    device = "cuda"

    manifest = json.loads((PAIRS_DIR / "manifest.json").read_text())
    train_manifest, val_manifest = split_train_val(manifest)
    print(f"dataset: {len(train_manifest)} train, {len(val_manifest)} held-out val", flush=True)

    from peft import prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from training.prompting_utils import UniversalPrompting  # noqa: PLC0415
    from models import MMadaModelLM  # noqa: PLC0415

    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    text_tokenizer = AutoTokenizer.from_pretrained(world_model.checkpoint_path, padding_side="left")
    world_model._uni_prompting = UniversalPrompting(  # noqa: SLF001
        text_tokenizer, max_text_len=128, max_seq_len=SUBGOAL_NUM_TOKENS + 8
    )
    text_vocab = world_model.image_token_offset
    num_vq_tokens = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE

    # Cache tokenized data to disk (2026-07-17): tokenizing all ~2000
    # samples takes ~20-25 min, and OOM-debugging the training loop
    # below needed several restarts -- re-paying that cost each retry
    # was the dominant time sink. Cache key includes MAX_ACTIVE_POSITIONS
    # since that affects the cached content.
    cache_path = PAIRS_DIR / f"tokenized_cache_max{MAX_ACTIVE_POSITIONS}.pt"
    tokenizer.load()  # needed either way -- for prep() below, or for later eval/decode calls
    if cache_path.exists():
        print(f"loading cached tokenized data from {cache_path}", flush=True)
        cached = torch.load(cache_path, weights_only=False)
        train_data, val_data = cached["train"], cached["val"]
    else:
        def prep(entries):
            prepped = []
            for e in entries:
                masked_ids, masked_positions, active_labels = load_sample(e["stub"], tokenizer, text_vocab)
                prepped.append((e["stub"], e["instruction"], masked_ids, masked_positions, active_labels))
            return prepped

        train_data = prep(train_manifest)
        val_data = prep(val_manifest)
        torch.save({"train": train_data, "val": val_data}, cache_path)
        print(f"tokenized {len(train_data)} train + {len(val_data)} val samples, cached to {cache_path}", flush=True)

    tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base_model = MMadaModelLM.from_pretrained(
        world_model.checkpoint_path, quantization_config=bnb_config, device_map={"": device}
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    world_model._model = base_model  # noqa: SLF001

    ff_out_wrapper = MaskedFFOut(base_model.model.transformer.ff_out)
    base_model.model.transformer.ff_out = ff_out_wrapper

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * LORA_ALPHA_MULT,
        target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    def build_prompt_batch(stub, instruction, masked_ids):
        prompt_batch = world_model.build_prompt(instruction, masked_ids.to(device))
        seq_len = prompt_batch.input_ids.shape[1]
        image_start = seq_len - num_vq_tokens - 1
        return prompt_batch.input_ids.cpu(), prompt_batch.attention_mask.cpu(), image_start

    train_batches = []
    for stub, instruction, masked_ids, masked_positions, active_labels in train_data:
        input_ids, attention_mask, image_start = build_prompt_batch(stub, instruction, masked_ids)
        global_positions = masked_positions + image_start
        train_batches.append((input_ids, attention_mask, global_positions, active_labels))

    checker = PlausibilityChecker()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    def run_validation(step_label):
        model.eval()
        ff_out_wrapper.active_positions = None
        tokenizer._model = tokenizer._model.to(device)  # noqa: SLF001
        world_model._model = model  # noqa: SLF001
        scores = []
        for stub, instruction, masked_ids, _, _ in val_data[:3]:
            eval_batch = world_model.build_prompt(instruction, masked_ids.to(device))
            with torch.no_grad():
                resolved_ids = world_model.denoise(eval_batch, timesteps=18)
            image_out = tokenizer.decode(resolved_ids[0].cpu().numpy())
            armvis = np.array(Image.open(PAIRS_DIR / f"{stub}_armvis.png").convert("RGB").resize((SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE)))
            armmask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png").resize((SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE))) > 127
            score = checker.score(image_out, {"original_image": armvis, "arm_pixel_mask": armmask})
            scores.append(score)
            Image.fromarray(image_out).save(EVAL_DIR / f"{step_label}_{stub}.png")
        tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
        torch.cuda.empty_cache()
        model.train()
        print(f"[val @ {step_label}] scores={[round(s, 4) for s in scores]} mean={np.mean(scores):.4f}", flush=True)

    # 8-bit Adam (bitsandbytes) instead of plain AdamW: r=128 has ~16x
    # more trainable params than the tiny run's r=8 (134M vs 8.4M) --
    # confirmed OOM during an actual training step (2026-07-17, model
    # alone fit at ~6.8GB but that check never created an optimizer or
    # ran forward/backward). Adam's fp32 momentum+variance states are
    # the standard QLoRA-paired fix for exactly this, cutting optimizer
    # memory roughly 4x for a modest, well-established quality tradeoff.
    import bitsandbytes as bnb  # noqa: PLC0415

    optimizer = bnb.optim.Adam8bit([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train_batches))
    for step in range(args.steps):
        idx = order[step % len(order)]
        if step % len(order) == 0 and step > 0:
            order = rng.permutation(len(train_batches))
        input_ids, attention_mask, global_positions, active_labels = train_batches[idx]
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        global_positions, active_labels = global_positions.to(device), active_labels.to(device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)

        ff_out_wrapper.active_positions = global_positions
        logits = model(input_ids, attention_bias=attention_bias).logits
        logits = logits[:, :, text_vocab : text_vocab + CODEBOOK_SIZE]
        loss = F.cross_entropy(logits.reshape(-1, CODEBOOK_SIZE), active_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        n_active = global_positions.numel()
        del logits, loss, input_ids, attention_mask, attention_bias, global_positions, active_labels

        # Temporary per-step memory diagnostic (2026-07-17): 5 consecutive
        # OOMs at the same ~step-1-to-19 window even after fixing shape
        # variance -- log every step until the actual growth pattern is
        # visible, instead of guessing at more fixes blind.
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"step {step}: loss={loss_value:.4f} n_active={n_active} allocated={allocated:.3f}GB reserved={reserved:.3f}GB", flush=True)
        if step > 0 and step % EVAL_EVERY == 0:
            run_validation(f"step{step}")
            ADAPTER_OUT.mkdir(exist_ok=True)
            model.save_pretrained(str(ADAPTER_OUT))

    run_validation("final")
    ADAPTER_OUT.mkdir(exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    print(f"saved final LoRA adapter to {ADAPTER_OUT}", flush=True)


if __name__ == "__main__":
    main()
