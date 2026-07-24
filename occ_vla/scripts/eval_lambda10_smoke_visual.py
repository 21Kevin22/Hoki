"""Visual sanity check requested by user before scaling to full-population
training: does the lambda_obj=10 object-weighted-loss smoke-adapter
(train_object_weighted_loss_smoke.py) actually turn the object region
from a "washed-out blob" into a recognizable shape, or does the lower
obj_loss number not translate to a visible improvement?

Deliberately an overfit check (same 9 frames used for training, 20
steps) -- if it can't even show visible structure on memorized samples,
a larger training run is unlikely to either. Not a generalization claim.

Saves per-frame contact sheets (armvis | mask overlay | cv2.inpaint |
lambda10-adapter output | pre-existing arm_removal_lora_full_adapter
(lambda=1 equivalent) output | ground truth) plus a zoomed crop on just
the object bounding box, matching generate_grounded_collage.py's style.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/eval_lambda10_smoke_visual.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
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
STEPS = 20
LAMBDA_OBJ = 10.0
LORA_RANK = 16
LORA_ALPHA = 32
OBJECT_MASK_DILATE_PX = 4
BASELINE_ADAPTER = _ROOT / "arm_removal_lora_full_adapter"  # lambda=1-equivalent, already trained (2026-07-17)
SMOKE_ADAPTER_OUT = _ROOT / "object_weighted_lora_smoke_lambda10_adapter"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "lambda10_visual_check"

gen = ArmFreeSubgoalGenerator(world_model=None)


class MaskedFFOut(torch.nn.Module):
    def __init__(self, orig_ff_out):
        super().__init__()
        self.orig = orig_ff_out
        self.active_positions = None

    def forward(self, x):
        if self.active_positions is None:
            return self.orig(x)
        return self.orig(x[:, self.active_positions, :])


def lenient_token_mask(pixel_mask):
    h, w = pixel_mask.shape
    ph, pw = h // TOKEN_GRID_SIDE, w // TOKEN_GRID_SIDE
    pooled = pixel_mask[: ph * TOKEN_GRID_SIDE, : pw * TOKEN_GRID_SIDE].reshape(TOKEN_GRID_SIDE, ph, TOKEN_GRID_SIDE, pw)
    return pooled.any(axis=(1, 3)).reshape(-1)


def load_sample(stub, tokenizer, image_token_offset):
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
    object_token_mask = lenient_token_mask(objmask_512) & token_mask
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
        armvis_512, armfree_512, mask_512, objmask_512,
    )


def build_base_model(device, tokenizer=None):
    from peft import prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from training.prompting_utils import UniversalPrompting  # noqa: PLC0415
    from models import MMadaModelLM  # noqa: PLC0415
    from occ_vla.world_model.tokenizer import SUBGOAL_NUM_TOKENS  # noqa: PLC0415

    # reuse a caller-supplied tokenizer instead of loading a second
    # MAGVIT-v2 instance onto the same GPU -- two base MMaDA-8B (4-bit)
    # copies is already the point of this script (lambda10 vs baseline
    # adapter, loaded/freed sequentially), a second tokenizer alongside
    # is pure unnecessary VRAM risk.
    if tokenizer is None:
        tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
        tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    text_tokenizer = AutoTokenizer.from_pretrained(world_model.checkpoint_path, padding_side="left")
    world_model._uni_prompting = UniversalPrompting(text_tokenizer, max_text_len=128, max_seq_len=SUBGOAL_NUM_TOKENS + 8)  # noqa: SLF001
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base_model = MMadaModelLM.from_pretrained(world_model.checkpoint_path, quantization_config=bnb_config, device_map={"": device})
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    world_model._model = base_model  # noqa: SLF001
    return tokenizer, world_model, base_model


def train_lambda10(device):
    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text())
    tokenizer, world_model, base_model = build_base_model(device)
    text_vocab = world_model.image_token_offset
    num_vq_tokens = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE

    ff_out_wrapper = MaskedFFOut(base_model.model.transformer.ff_out)
    base_model.model.transformer.ff_out = ff_out_wrapper
    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=["q_proj", "k_proj", "v_proj", "attn_out"], lora_dropout=0.0, bias="none")
    model = get_peft_model(base_model, lora_config)

    batch = []
    for e in manifest:
        masked_ids, masked_positions, active_labels, is_object, *_ = load_sample(e["stub"], tokenizer, text_vocab)
        prompt_batch = world_model.build_prompt(e["instruction"], masked_ids.to(device))
        seq_len = prompt_batch.input_ids.shape[1]
        image_start = seq_len - num_vq_tokens - 1
        global_positions = masked_positions + image_start
        weights = is_object.float() * (LAMBDA_OBJ - 1.0) + 1.0
        batch.append((prompt_batch.input_ids.cpu(), prompt_batch.attention_mask.cpu(), global_positions, active_labels, weights))

    tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
    torch.cuda.empty_cache()

    import bitsandbytes as bnb  # noqa: PLC0415

    optimizer = bnb.optim.Adam8bit([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        idx = rng.integers(len(batch))
        input_ids, attention_mask, global_positions, active_labels, weights = batch[idx]
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        global_positions, active_labels, weights = global_positions.to(device), active_labels.to(device), weights.to(device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        ff_out_wrapper.active_positions = global_positions
        logits = model(input_ids, attention_bias=attention_bias).logits
        logits = logits[:, :, text_vocab : text_vocab + CODEBOOK_SIZE]
        per_pos_loss = F.cross_entropy(logits.reshape(-1, CODEBOOK_SIZE), active_labels.reshape(-1), reduction="none")
        loss = (per_pos_loss * weights).sum() / weights.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss={loss.item():.4f}", flush=True)
        del logits, loss, per_pos_loss

    SMOKE_ADAPTER_OUT.mkdir(exist_ok=True)
    model.save_pretrained(str(SMOKE_ADAPTER_OUT))
    print(f"saved lambda=10 smoke adapter to {SMOKE_ADAPTER_OUT}", flush=True)

    ff_out_wrapper.active_positions = None
    del optimizer
    torch.cuda.empty_cache()
    tokenizer._model = tokenizer._model.to(device)  # noqa: SLF001
    return tokenizer, world_model, model  # model is the lambda10-adapted PeftModel, still loaded


def main():
    device = "cuda"
    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text())
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    tokenizer, world_model, lambda10_model = train_lambda10(device)
    text_vocab = world_model.image_token_offset

    def generate(model_to_use, masked_ids, instruction):
        world_model._model = model_to_use  # noqa: SLF001
        batch = world_model.build_prompt(instruction, masked_ids.to(device))
        with torch.no_grad():
            resolved_ids = world_model.denoise(batch, timesteps=18)
        return tokenizer.decode(resolved_ids[0].cpu().numpy())

    lambda10_model.eval()
    print("generating with lambda=10 smoke adapter...", flush=True)
    lambda10_outputs = {}
    for e in manifest:
        masked_ids, _, _, _, armvis_512, armfree_512, mask_512, objmask_512 = load_sample(e["stub"], tokenizer, text_vocab)
        lambda10_outputs[e["stub"]] = generate(lambda10_model, masked_ids, e["instruction"])

    del lambda10_model
    torch.cuda.empty_cache()

    # swap in the pre-existing lambda=1-equivalent adapter for a same-footing comparison
    print("loading baseline (lambda=1-equivalent) arm_removal_lora_full_adapter...", flush=True)
    _, world_model2, base_model2 = build_base_model(device, tokenizer=tokenizer)
    baseline_model = PeftModel.from_pretrained(base_model2, str(BASELINE_ADAPTER))
    baseline_model.eval()
    baseline_outputs = {}
    for e in manifest:
        masked_ids, *_ = load_sample(e["stub"], tokenizer, text_vocab)
        world_model2._model = baseline_model  # noqa: SLF001
        batch = world_model2.build_prompt(e["instruction"], masked_ids.to(device))
        with torch.no_grad():
            resolved_ids = world_model2.denoise(batch, timesteps=18)
        baseline_outputs[e["stub"]] = tokenizer.decode(resolved_ids[0].cpu().numpy())

    print("building contact sheets...", flush=True)
    for e in manifest:
        stub = e["stub"]
        masked_ids, _, _, _, armvis_512, armfree_512, mask_512, objmask_512 = load_sample(stub, tokenizer, text_vocab)
        inpainted = cv2.inpaint(armvis_512, mask_512.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        overlay = armvis_512.copy()
        overlay[mask_512 & ~objmask_512] = [255, 0, 255]
        overlay[objmask_512] = [0, 255, 255]

        panels = [armvis_512, overlay, inpainted, baseline_outputs[stub], lambda10_outputs[stub], armfree_512]
        sheet = Image.new("RGB", (512 * 6, 512))
        for j, img in enumerate(panels):
            sheet.paste(Image.fromarray(np.asarray(img).astype(np.uint8)), (j * 512, 0))
        sheet.save(OUT_DIR / f"{stub}.png")

        ys, xs = np.where(objmask_512)
        if len(ys):
            pad = 40
            y0, y1 = max(0, ys.min() - pad), min(512, ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(512, xs.max() + pad)
            zoom = Image.new("RGB", ((x1 - x0) * 6 * 3, (y1 - y0) * 3))
            sheet_arr = np.array(sheet)
            for j in range(6):
                crop = sheet_arr[y0:y1, j * 512 + x0 : j * 512 + x1]
                up = Image.fromarray(crop).resize(((x1 - x0) * 3, (y1 - y0) * 3), Image.NEAREST)
                zoom.paste(up, (j * (x1 - x0) * 3, 0))
            zoom.save(OUT_DIR / f"{stub}_objzoom.png")

    print(f"done -- {len(manifest)} contact sheets + object zooms in {OUT_DIR}/")


if __name__ == "__main__":
    main()
