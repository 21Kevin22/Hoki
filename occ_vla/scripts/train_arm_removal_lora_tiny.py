"""Smoke test ONLY: overfit a tiny LoRA adapter on 1 episode's worth of
ground-truth arm-removal pairs (~15 samples, from
collect_arm_removal_pairs.py) to verify the training pipeline
(tokenize -> mask -> LoRA forward -> masked cross-entropy -> backward)
is technically wired end to end. NOT meant to produce a useful/general
adapter -- deliberately overfits a handful of samples, per user's
explicit "lowest risk first" staging request. A real training run
(more data, held-out eval, tuned rank/lr/steps) is a separate, larger
follow-up if this pipeline check passes.

Unlike arm_free_subgoal.py's zero-shot path (which had to approximate
the arm region with a bbox heuristic because it never sees ground
truth), this uses the REAL arm segmentation mask directly -- no bbox
approximation needed now that supervised pairs exist.

Uses peft (r=8 here, deliberately smaller than the r=128 suggested for
a real run -- this is a plumbing check, not a quality run; rank barely
matters for "does the loss go down on 15 memorized samples").
target_modules=[q_proj,k_proj,v_proj,attn_out] confirmed via
model.named_modules() against the actual loaded checkpoint (MMaDA uses
separate q/k/v/attn_out Linears, not a fused qkv here).

4-bit (nf4) quantization + saliency-aware backward (MaskedFFOut below):
the final vocab projection (hidden_dim -> 134,656) is only run on the
masked (arm-region) token positions during training, not the full
~1150-token sequence -- this is what actually closed a reproducible
~230MB VRAM shortfall at the backward step (2026-07-17 session), since
the bottleneck was the SIZE of that layer's forward/backward tensors
(proportional to seq_len x vocab_size), not raw compute. Defaults to
full-sequence behavior (opt-in only inside the training loop), so
world_model.denoise()'s inference path is unaffected.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/train_arm_removal_lora_tiny.py
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

PAIRS_DIR = _ROOT / "arm_removal_pairs"
EPISODE_PREFIX = "moka_pots_ep00_"  # single episode used for the overfit smoke test
LR = 1e-4
STEPS = 40
LORA_RANK = 8
LORA_ALPHA = 16
ADAPTER_OUT = _ROOT / "arm_removal_lora_tiny_adapter"
EVAL_OUT = _ROOT / "arm_removal_pairs" / "lora_tiny_eval.png"

gen = ArmFreeSubgoalGenerator(world_model=None)


class MaskedFFOut(torch.nn.Module):
    """Wraps the model's final vocab-projection layer (hidden_dim ->
    134,656) to optionally restrict computation to a subset of sequence
    positions. Avoids ever materializing the full (seq_len x vocab)
    logits/gradient tensor, which is what OOM'd the last ~230MB of
    backward memory (2026-07-17, confirmed reproducible). Defaults to
    computing the full sequence (`active_positions=None`), so
    `world_model.denoise()`'s inference path is completely unaffected
    unless `active_positions` is explicitly set -- only the training
    loop below ever sets it, and resets it to None before eval."""

    def __init__(self, orig_ff_out: torch.nn.Module):
        super().__init__()
        self.orig = orig_ff_out
        self.active_positions: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_positions is None:
            return self.orig(x)
        return self.orig(x[:, self.active_positions, :])  # (batch, n_active, vocab) -- not padded back to full seq_len


def load_sample(stub: str, tokenizer: MagvitV2Tokenizer, image_token_offset: int):
    """Returns CPU tensors -- caller moves to device just-in-time inside
    the training loop, so all samples' tensors aren't resident on GPU
    simultaneously. `active_labels` is the ground-truth codebook id at
    just the masked (arm-region) positions, aligned to
    `masked_positions` (indices into the image-token span) -- the small,
    sequence-position-sliced counterpart of the old full-width
    -100-padded `labels` tensor."""
    armvis = np.array(Image.open(PAIRS_DIR / f"{stub}_armvis.png").convert("RGB"))
    armfree = np.array(Image.open(PAIRS_DIR / f"{stub}_armfree.png").convert("RGB"))
    armmask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png")) > 127

    armvis_512 = cv2.resize(armvis, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    armfree_512 = cv2.resize(armfree, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(armmask.astype(np.uint8), (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_NEAREST) > 0

    token_mask = gen._arm_token_mask(mask_512)  # noqa: SLF001 -- real mask now, no bbox approximation needed
    vis_ids = tokenizer.encode(armvis_512)
    free_ids = tokenizer.encode(armfree_512)

    masked_input_ids = np.where(token_mask, MASK_TOKEN_ID, vis_ids + image_token_offset)
    masked_positions = np.where(token_mask)[0]  # indices into the 1024-wide image-token span
    active_labels = free_ids[masked_positions]  # raw codebook ids (0..CODEBOOK_SIZE-1)
    return (
        torch.from_numpy(masked_input_ids).long().unsqueeze(0),
        torch.from_numpy(masked_positions).long(),
        torch.from_numpy(active_labels).long(),
    )


def main():
    device = "cuda"
    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()

    # 8-bit base weights instead of world_model.load()'s bf16 path: a
    # bf16 forward+backward OOM'd a 24GB GPU on the very first sample
    # (confirmed by direct measurement 2026-07-17: ~15.6GB just for
    # bf16 base weights, exceeding the ~8GB left over once activations
    # for backprop through all 32 frozen layers are added -- this is a
    # real capacity constraint, not a bug). 8-bit cuts base weights to
    # ~9GB, leaving enough headroom for LoRA's backward pass.
    from peft import prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import BitsAndBytesConfig  # noqa: PLC0415
    from training.prompting_utils import UniversalPrompting  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from occ_vla.world_model.tokenizer import SUBGOAL_NUM_TOKENS  # noqa: PLC0415

    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    text_tokenizer = AutoTokenizer.from_pretrained(world_model.checkpoint_path, padding_side="left")
    world_model._uni_prompting = UniversalPrompting(  # noqa: SLF001
        text_tokenizer, max_text_len=128, max_seq_len=SUBGOAL_NUM_TOKENS + 8
    )
    # 4-bit (nf4) instead of 8-bit: 8-bit alone (bnb's LLM.int8() path)
    # still OOM'd on the first forward+backward (confirmed 2026-07-17) --
    # its outlier-handling path keeps some fp16 workspace that partly
    # offsets the memory saving. nf4 has a leaner compute path.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
    )
    from models import MMadaModelLM  # noqa: PLC0415

    base_model = MMadaModelLM.from_pretrained(
        world_model.checkpoint_path, quantization_config=bnb_config, device_map={"": device}
    )
    # MMadaModelLM doesn't support gradient checkpointing (vendored
    # class never implements it) -- skip prepare_model_for_kbit_training's
    # default auto-enable of it and rely on 8-bit quantization alone for
    # memory headroom.
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    world_model._model = base_model  # noqa: SLF001

    # Saliency-aware backward (per user request, 2026-07-17): only run
    # the huge final vocab projection on the masked (arm-region) token
    # positions during training, instead of the full ~1150-token
    # sequence -- this is what actually closes the ~230MB gap (the full
    # ff_out's forward+backward tensor, not raw FLOPs, was the
    # bottleneck). ff_out_wrapper.active_positions stays None (full
    # sequence) except inside the training loop below, so the eval
    # denoise() call further down is unaffected.
    ff_out_wrapper = MaskedFFOut(base_model.model.transformer.ff_out)
    base_model.model.transformer.ff_out = ff_out_wrapper

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    manifest = json.loads((PAIRS_DIR / "manifest.json").read_text())
    stubs = [e["stub"] for e in manifest if e["stub"].startswith(EPISODE_PREFIX)]
    print(f"training on {len(stubs)} samples from {EPISODE_PREFIX}*", flush=True)

    prompt = "put both moka pots on the stove"
    text_vocab = world_model.image_token_offset
    num_vq_tokens = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE

    batch = []
    for stub in stubs:
        masked_ids, masked_positions, active_labels = load_sample(stub, tokenizer, text_vocab)
        prompt_batch = world_model.build_prompt(prompt, masked_ids.to(device))
        seq_len = prompt_batch.input_ids.shape[1]
        image_start = seq_len - num_vq_tokens - 1  # matches the -(num_vq_tokens+1):-1 convention used elsewhere
        global_positions = masked_positions + image_start
        # keep on CPU; moved to device just-in-time in the training loop below
        batch.append((prompt_batch.input_ids.cpu(), prompt_batch.attention_mask.cpu(), global_positions, active_labels))

    # Free the MAGVIT-v2 tokenizer's GPU memory now that all samples are
    # encoded -- it's only needed again for the final eval decode() below,
    # and the 8B LLM's forward+backward is tight enough on a 24GB GPU
    # that this is worth reclaiming (confirmed OOM'ing right at the
    # final ff_out projection otherwise, 2026-07-17).
    tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for step in range(STEPS):
        total_loss = 0.0
        for input_ids, attention_mask, global_positions, active_labels in batch:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            global_positions, active_labels = global_positions.to(device), active_labels.to(device)
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)

            ff_out_wrapper.active_positions = global_positions  # only these positions get projected to vocab space
            logits = model(input_ids, attention_bias=attention_bias).logits  # (1, n_active, full_vocab)
            logits = logits[:, :, text_vocab : text_vocab + CODEBOOK_SIZE]
            loss = F.cross_entropy(logits.reshape(-1, CODEBOOK_SIZE), active_labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"step {step}: mean_loss={total_loss / len(batch):.4f}", flush=True)

    ADAPTER_OUT.mkdir(exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    print(f"saved LoRA adapter to {ADAPTER_OUT}", flush=True)

    # eval: run the real iterative denoise() (not a single forward) on one
    # training sample's masked input, using the now-adapted model, and
    # decode -- checks the pipeline round-trips end to end, not just that
    # the training loss went down.
    model.eval()
    ff_out_wrapper.active_positions = None  # back to full-sequence behavior for the real denoise() eval below
    del optimizer
    torch.cuda.empty_cache()
    tokenizer._model = tokenizer._model.to(device)  # noqa: SLF001 -- bring back for encode/decode below

    eval_stub = stubs[0]
    masked_ids, _, _ = load_sample(eval_stub, tokenizer, text_vocab)
    eval_batch = world_model.build_prompt(prompt, masked_ids.to(device))
    world_model._model = model  # noqa: SLF001 -- swap in the LoRA-wrapped model for inference
    with torch.no_grad():
        resolved_ids = world_model.denoise(eval_batch, timesteps=18)
    image_out = tokenizer.decode(resolved_ids[0].cpu().numpy())
    Image.fromarray(image_out).save(EVAL_OUT)
    print(f"saved eval decode ({eval_stub}) to {EVAL_OUT}", flush=True)


if __name__ == "__main__":
    main()
