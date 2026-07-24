"""Smoke test ONLY (per train_arm_removal_lora_tiny.py's precedent): does
upweighting the masked cross-entropy loss at the TRUE object sub-region
(env.obj_of_interest segmentation, not the discredited
_gripper_end_bbox_token_mask heuristic) train stably -- no NaN, no
gradient blow-up, no VRAM regression vs. the already-proven r=16
pipeline -- before committing to a full-population data collection +
production run?

Not meant to produce a useful/general adapter: trains on the same 9
frames from grounded_holdout_frames/ used for the grounded-collage probe
(2026-07-21) -- these are technically episode-19 ("held-out" by
train_arm_removal_lora_full.py's split) and are the only frames on disk
so far with a REAL object mask attached, precisely because this is a
throwaway plumbing check, not an eval. A real production run needs a
freshly collected, properly-split training population (not implemented
here).

Motivation recap (CLAUDE.md, "Grounding-corrected collage test..."):
scored on the object's true footprint (not the full arm mask), MMaDA-LoRA
loses to classical cv2.inpaint on ground-truth accuracy, 9/9 frames --
the model isn't attending to the ~150 masked-token training signal at
the object sub-region, which is typically <5 of the ~150 sampled
positions (object footprint measured at 0.03%-0.5% of the frame). This
tests whether raising that handful of positions' loss weight (relative
weight lambda_obj) changes what the model prioritizes, without any new
architecture.

Runs lambda_obj in {1.0 (control), 10.0, 50.0}, ~20 real steps each
(matching the project's own "verify several consecutive steps, not one"
lesson from the r=64/r=128 instability investigation), fresh model load
per condition to avoid cross-condition contamination, logging per-step
total/object-only/background-only loss components and GPU memory.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/train_object_weighted_loss_smoke.py
"""

import json
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

from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator, TOKEN_GRID_SIDE  # noqa: E402
from occ_vla.world_model.mmada import MASK_TOKEN_ID, MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import CODEBOOK_SIZE, MagvitV2Tokenizer, SUBGOAL_IMAGE_SIDE  # noqa: E402

FRAMES_DIR = _ROOT / "grounded_holdout_frames"
LR = 1e-4
STEPS_PER_CONDITION = 20
LORA_RANK = 16  # the only rank confirmed stable in real multi-step training (2026-07-17) -- do not raise
LORA_ALPHA = 32
LAMBDA_CONDITIONS = [1.0, 10.0, 50.0]
OBJECT_MASK_DILATE_PX = 4  # in native 256px space -- ensures thin/small object footprints still tag >=1 token
RESULTS_PATH = _ROOT / "texture_ceiling_probe" / "object_weighted_loss_smoke_results.json"

gen = ArmFreeSubgoalGenerator(world_model=None)


class MaskedFFOut(torch.nn.Module):
    """Same technique as train_arm_removal_lora_tiny.py -- see that file."""

    def __init__(self, orig_ff_out: torch.nn.Module):
        super().__init__()
        self.orig = orig_ff_out
        self.active_positions: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_positions is None:
            return self.orig(x)
        return self.orig(x[:, self.active_positions, :])


def lenient_token_mask(pixel_mask: np.ndarray) -> np.ndarray:
    """Like ArmFreeSubgoalGenerator._arm_token_mask but ANY overlap
    (not >50%) counts a token as covered -- the true object footprint
    (0.03%-0.5% of the frame) is often smaller than one token's
    receptive field, so a majority-vote threshold would silently zero
    out the object mask for most samples."""
    h, w = pixel_mask.shape
    ph, pw = h // TOKEN_GRID_SIDE, w // TOKEN_GRID_SIDE
    pooled = pixel_mask[: ph * TOKEN_GRID_SIDE, : pw * TOKEN_GRID_SIDE].reshape(TOKEN_GRID_SIDE, ph, TOKEN_GRID_SIDE, pw)
    return pooled.any(axis=(1, 3)).reshape(-1)


def load_sample(stub: str, instruction: str, tokenizer: MagvitV2Tokenizer, image_token_offset: int):
    armvis = np.array(Image.open(FRAMES_DIR / f"{stub}_armvis.png").convert("RGB"))
    armfree = np.array(Image.open(FRAMES_DIR / f"{stub}_armfree.png").convert("RGB"))
    armmask = np.array(Image.open(FRAMES_DIR / f"{stub}_armmask.png")) > 127
    objmask = np.array(Image.open(FRAMES_DIR / f"{stub}_objmask.png")) > 127

    kernel = np.ones((OBJECT_MASK_DILATE_PX, OBJECT_MASK_DILATE_PX), np.uint8)
    objmask_dilated = cv2.dilate(objmask.astype(np.uint8), kernel).astype(bool)

    armvis_512 = cv2.resize(armvis, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    armfree_512 = cv2.resize(armfree, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(armmask.astype(np.uint8), (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_NEAREST) > 0
    objmask_512 = cv2.resize(objmask_dilated.astype(np.uint8), (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_NEAREST) > 0

    token_mask = gen._arm_token_mask(mask_512)  # noqa: SLF001
    object_token_mask = lenient_token_mask(objmask_512) & token_mask  # never flag a token outside the arm mask itself
    vis_ids = tokenizer.encode(armvis_512)
    free_ids = tokenizer.encode(armfree_512)

    masked_input_ids = np.where(token_mask, MASK_TOKEN_ID, vis_ids + image_token_offset)
    masked_positions = np.where(token_mask)[0]
    active_labels = free_ids[masked_positions]
    is_object = object_token_mask[masked_positions]
    return (
        torch.from_numpy(masked_input_ids).long().unsqueeze(0),
        torch.from_numpy(masked_positions).long(),
        torch.from_numpy(active_labels).long(),
        torch.from_numpy(is_object).bool(),
    )


def run_condition(lambda_obj: float, samples: list[dict], device: str):
    from peft import prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from training.prompting_utils import UniversalPrompting  # noqa: PLC0415
    from models import MMadaModelLM  # noqa: PLC0415
    from occ_vla.world_model.tokenizer import SUBGOAL_NUM_TOKENS  # noqa: PLC0415

    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    text_tokenizer = AutoTokenizer.from_pretrained(world_model.checkpoint_path, padding_side="left")
    world_model._uni_prompting = UniversalPrompting(  # noqa: SLF001
        text_tokenizer, max_text_len=128, max_seq_len=SUBGOAL_NUM_TOKENS + 8
    )
    text_vocab = world_model.image_token_offset
    num_vq_tokens = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base_model = MMadaModelLM.from_pretrained(world_model.checkpoint_path, quantization_config=bnb_config, device_map={"": device})
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    world_model._model = base_model  # noqa: SLF001

    ff_out_wrapper = MaskedFFOut(base_model.model.transformer.ff_out)
    base_model.model.transformer.ff_out = ff_out_wrapper

    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=["q_proj", "k_proj", "v_proj", "attn_out"], lora_dropout=0.0, bias="none")
    model = get_peft_model(base_model, lora_config)

    batch = []
    n_object_tokens_per_sample = []
    for e in samples:
        masked_ids, masked_positions, active_labels, is_object = load_sample(e["stub"], e["instruction"], tokenizer, text_vocab)
        prompt_batch = world_model.build_prompt(e["instruction"], masked_ids.to(device))
        seq_len = prompt_batch.input_ids.shape[1]
        image_start = seq_len - num_vq_tokens - 1
        global_positions = masked_positions + image_start
        weights = is_object.float() * (lambda_obj - 1.0) + 1.0
        batch.append((prompt_batch.input_ids.cpu(), prompt_batch.attention_mask.cpu(), global_positions, active_labels, is_object, weights))
        n_object_tokens_per_sample.append(int(is_object.sum()))

    tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
    torch.cuda.empty_cache()
    print(f"  object tokens/sample (of ~{sum(len(b[2]) for b in batch) // len(batch)} masked): {n_object_tokens_per_sample}", flush=True)

    import bitsandbytes as bnb  # noqa: PLC0415

    optimizer = bnb.optim.Adam8bit([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    rng = np.random.default_rng(0)
    step_logs = []
    for step in range(STEPS_PER_CONDITION):
        idx = rng.integers(len(batch))
        input_ids, attention_mask, global_positions, active_labels, is_object, weights = batch[idx]
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        global_positions, active_labels = global_positions.to(device), active_labels.to(device)
        is_object, weights = is_object.to(device), weights.to(device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)

        ff_out_wrapper.active_positions = global_positions
        logits = model(input_ids, attention_bias=attention_bias).logits
        logits = logits[:, :, text_vocab : text_vocab + CODEBOOK_SIZE]
        per_pos_loss = F.cross_entropy(logits.reshape(-1, CODEBOOK_SIZE), active_labels.reshape(-1), reduction="none")
        loss = (per_pos_loss * weights).sum() / weights.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            obj_loss = per_pos_loss[is_object].mean().item() if is_object.any() else float("nan")
            bg_loss = per_pos_loss[~is_object].mean().item() if (~is_object).any() else float("nan")
        reserved = torch.cuda.memory_reserved() / 1e9
        is_finite = bool(torch.isfinite(loss))
        step_logs.append({"step": step, "loss": loss.item(), "obj_loss": obj_loss, "bg_loss": bg_loss, "reserved_gb": reserved, "finite": is_finite})
        print(f"  [lambda={lambda_obj}] step {step}: loss={loss.item():.4f} obj_loss={obj_loss:.4f} bg_loss={bg_loss:.4f} reserved={reserved:.2f}GB finite={is_finite}", flush=True)
        del logits, loss, per_pos_loss, input_ids, attention_mask, attention_bias

    del model, base_model, optimizer, world_model, tokenizer
    torch.cuda.empty_cache()
    return step_logs


def main():
    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text())
    device = "cuda"
    print(f"{len(manifest)} smoke-test samples (episode 19, throwaway -- see module docstring)", flush=True)

    all_results = {}
    for lam in LAMBDA_CONDITIONS:
        print(f"\n=== lambda_obj={lam} ===", flush=True)
        logs = run_condition(lam, manifest, device)
        all_results[str(lam)] = logs
        RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    print("\n=== summary ===")
    for lam, logs in all_results.items():
        finite_all = all(row["finite"] for row in logs)
        final_obj = logs[-1]["obj_loss"]
        final_bg = logs[-1]["bg_loss"]
        max_reserved = max(row["reserved_gb"] for row in logs)
        print(f"lambda={lam}: all_finite={finite_all}  final obj_loss={final_obj:.4f} bg_loss={final_bg:.4f}  max_reserved={max_reserved:.2f}GB")
    print(f"\nfull logs: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
