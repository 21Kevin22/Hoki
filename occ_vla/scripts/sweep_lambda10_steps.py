"""H1 (training-insufficiency) vs H2 (architectural ceiling) sweep, per
user's 2026-07-21 request: does more training (100/300/1000 steps vs.
the smoke test's 20) on the SAME 9 grounded_holdout_frames/ samples make
lambda_obj=10's object-region generation visibly sharpen (angular
handle emerging from the round blob), or does it stay a blob regardless
of step count?

Deliberately still an overfit check (9 samples, no held-out split) --
the question here is only "CAN r=16 + this MaskGIT decoder represent
the object's fine geometry at all, given unlimited gradient steps on
these exact samples," not generalization. If it can't even memorize the
shape with 1000 steps on 9 samples, that's strong evidence for H2
(architectural ceiling), not just "needs more data" -- see CLAUDE.md's
"Object-weighted loss (lambda_obj) smoke test" entry for the H1/H2
framing this resolves.

One continuous model load (not reloaded per checkpoint): trains to
CHECKPOINT_STEPS[-1], pausing at each checkpoint to run real denoise()
on 3 representative frames and save adapter + zoomed contact sheets,
then resumes training -- same eval-without-reload pattern as
train_arm_removal_lora_full.py's run_validation().

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/sweep_lambda10_steps.py
"""

import json
import sys
import time
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
CHECKPOINT_STEPS = [100, 300, 1000]
LAMBDA_OBJ = 10.0
LORA_RANK = 16
LORA_ALPHA = 32
OBJECT_MASK_DILATE_PX = 4
EVAL_STUBS = ["mug_in_microwave_ep19_s017", "mug_in_microwave_ep19_s018", "mug_in_microwave_ep19_s019"]
OUT_DIR = _ROOT / "texture_ceiling_probe" / "lambda10_step_sweep"
ADAPTER_OUT_ROOT = _ROOT / "lambda10_step_sweep_adapters"
RESULTS_PATH = _ROOT / "texture_ceiling_probe" / "lambda10_step_sweep_results.json"

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


def save_objzoom(stub, armvis, objmask, panels_dict, step_label):
    ys, xs = np.where(objmask)
    if not len(ys):
        return
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(512, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(512, xs.max() + pad)
    names = list(panels_dict.keys())
    imgs = list(panels_dict.values())
    zoom = Image.new("RGB", ((x1 - x0) * 3 * len(imgs), (y1 - y0) * 3))
    for j, img in enumerate(imgs):
        crop = np.asarray(img)[y0:y1, x0:x1]
        up = Image.fromarray(crop.astype(np.uint8)).resize(((x1 - x0) * 3, (y1 - y0) * 3), Image.NEAREST)
        zoom.paste(up, (j * (x1 - x0) * 3, 0))
    zoom.save(OUT_DIR / f"{stub}_{step_label}_objzoom.png")
    print(f"  saved {stub}_{step_label}_objzoom.png (panels: {names})", flush=True)


def main():
    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text())
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    device = "cuda"

    from peft import prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from training.prompting_utils import UniversalPrompting  # noqa: PLC0415
    from models import MMadaModelLM  # noqa: PLC0415
    from occ_vla.world_model.tokenizer import SUBGOAL_NUM_TOKENS  # noqa: PLC0415

    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    text_tokenizer = AutoTokenizer.from_pretrained(world_model.checkpoint_path, padding_side="left")
    world_model._uni_prompting = UniversalPrompting(text_tokenizer, max_text_len=128, max_seq_len=SUBGOAL_NUM_TOKENS + 8)  # noqa: SLF001
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
    eval_cache = {}
    for e in manifest:
        masked_ids, masked_positions, active_labels, is_object, armvis_512, armfree_512, mask_512, objmask_512 = load_sample(e["stub"], tokenizer, text_vocab)
        prompt_batch = world_model.build_prompt(e["instruction"], masked_ids.to(device))
        seq_len = prompt_batch.input_ids.shape[1]
        image_start = seq_len - num_vq_tokens - 1
        global_positions = masked_positions + image_start
        weights = is_object.float() * (LAMBDA_OBJ - 1.0) + 1.0
        batch.append((e["stub"], prompt_batch.input_ids.cpu(), prompt_batch.attention_mask.cpu(), global_positions, active_labels, weights, masked_ids))
        if e["stub"] in EVAL_STUBS:
            eval_cache[e["stub"]] = (armvis_512, armfree_512, mask_512, objmask_512, e["instruction"], masked_ids)

    tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
    torch.cuda.empty_cache()

    import bitsandbytes as bnb  # noqa: PLC0415

    optimizer = bnb.optim.Adam8bit([p for p in model.parameters() if p.requires_grad], lr=LR)

    def run_eval(step_label):
        model.eval()
        ff_out_wrapper.active_positions = None
        tokenizer._model = tokenizer._model.to(device)  # noqa: SLF001
        world_model._model = model  # noqa: SLF001
        for stub in EVAL_STUBS:
            armvis_512, armfree_512, mask_512, objmask_512, instruction, masked_ids = eval_cache[stub]
            eval_batch = world_model.build_prompt(instruction, masked_ids.to(device))
            with torch.no_grad():
                resolved_ids = world_model.denoise(eval_batch, timesteps=18)
            image_out = tokenizer.decode(resolved_ids[0].cpu().numpy())
            overlay = armvis_512.copy()
            overlay[mask_512 & ~objmask_512] = [255, 0, 255]
            overlay[objmask_512] = [0, 255, 255]
            save_objzoom(stub, armvis_512, objmask_512, {"armvis": armvis_512, "overlay": overlay, "generated": image_out, "gt": armfree_512}, step_label)
        tokenizer._model = tokenizer._model.cpu()  # noqa: SLF001
        torch.cuda.empty_cache()
        model.train()

    model.train()
    rng = np.random.default_rng(0)
    logs = []
    t_start = time.time()
    for step in range(CHECKPOINT_STEPS[-1]):
        idx = rng.integers(len(batch))
        stub, input_ids, attention_mask, global_positions, active_labels, weights, _ = batch[idx]
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        global_positions, active_labels, weights = global_positions.to(device), active_labels.to(device), weights.to(device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        ff_out_wrapper.active_positions = global_positions
        logits = model(input_ids, attention_bias=attention_bias).logits
        logits = logits[:, :, text_vocab : text_vocab + CODEBOOK_SIZE]
        per_pos_loss = F.cross_entropy(logits.reshape(-1, CODEBOOK_SIZE), active_labels.reshape(-1), reduction="none")
        is_object = weights > 1.0
        loss = (per_pos_loss * weights).sum() / weights.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            obj_loss = per_pos_loss[is_object].mean().item() if is_object.any() else float("nan")
            bg_loss = per_pos_loss[~is_object].mean().item() if (~is_object).any() else float("nan")
        if step % 20 == 0 or step == CHECKPOINT_STEPS[-1] - 1:
            elapsed = time.time() - t_start
            print(f"step {step}: loss={loss.item():.4f} obj_loss={obj_loss:.4f} bg_loss={bg_loss:.4f} ({elapsed:.0f}s elapsed)", flush=True)
        logs.append({"step": step, "loss": loss.item(), "obj_loss": obj_loss, "bg_loss": bg_loss})
        del logits, loss, per_pos_loss

        if (step + 1) in CHECKPOINT_STEPS:
            print(f"\n=== checkpoint @ step {step + 1} ===", flush=True)
            adapter_dir = ADAPTER_OUT_ROOT / f"step{step + 1}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            run_eval(f"step{step + 1}")
            RESULTS_PATH.write_text(json.dumps(logs, indent=2))

    print(f"\ndone -- contact sheets in {OUT_DIR}/, adapters in {ADAPTER_OUT_ROOT}/, loss log: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
