"""n=1 background check (per user decision, 2026-07-16): does masking
only the arm's gripper/terminal end -- instead of its whole diagonal
silhouette (_arm_token_mask) or whole bbox (_arm_bbox_token_mask,
which reintroduced a NEW artifact near an unrelated, correctly-visible
object because the arm's axis-aligned bbox reaches ~43-49% across the
frame) -- fix the blob artifact seen in every prior generation attempt?

This is the untested idea flagged in occ_vla/CLAUDE.md's "MMaDA arm-free
generation quality investigation": bound only the region that actually
needs to look "resolved" (the gripper end, where the held object sits),
not the arm's full diagonal reach. A true target-object-bbox variant
(the other alternative flagged there) isn't tested here because it
needs a fresh target segmentation mask from a live LIBERO rollout at
this exact step, which isn't saved on disk -- only the arm mask is
(see LOG_DIR/{STEP}_mask.png). This script is self-contained against
already-logged artifacts, no new data collection.

Per the user's explicit scope: if this doesn't fix the blob, the
arm-free image-generation path stays dropped (see
integration/runtime.py, enable_subgoal_image_generation=False default)
in favor of the non-generative pipeline (Phase 2 visual overlay +
Phase 4 soft gating + text-only cot_anchor).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/test_gripper_bbox_mask.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_MMADA_ROOT = _ROOT / "third_party" / "mmada"
sys.path.insert(0, str(_MMADA_ROOT))

from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator  # noqa: E402
from occ_vla.world_model.mmada import MASK_TOKEN_ID, MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import MagvitV2Tokenizer  # noqa: E402
from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402

LOG_DIR = _ROOT / "t08_mmada_log"
STEP = "step0361"
INSTRUCTION = "put both moka pots on the stove"
RESULTS_PATH = _ROOT / "t08_gripper_bbox_results.json"


def main():
    device = "cuda"
    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    world_model.load()
    gen = ArmFreeSubgoalGenerator(world_model)
    checker = PlausibilityChecker()

    raw = np.array(Image.open(LOG_DIR / f"{STEP}_raw.png").convert("RGB"))
    mask = np.array(Image.open(LOG_DIR / f"{STEP}_mask.png")) > 127
    image_512 = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    image_ids = tokenizer.encode(image_512)
    offset_image_ids = image_ids + world_model.image_token_offset
    prompt = gen._build_subgoal_prompt(INSTRUCTION, 5)  # noqa: SLF001

    variants = {
        "gripper_end_row35_pad1": gen._gripper_end_bbox_token_mask(mask_512, row_fraction=0.35, pad_tokens=1),  # noqa: SLF001
        "gripper_end_row25_pad0": gen._gripper_end_bbox_token_mask(mask_512, row_fraction=0.25, pad_tokens=0),  # noqa: SLF001
        "gripper_end_row50_pad1": gen._gripper_end_bbox_token_mask(mask_512, row_fraction=0.50, pad_tokens=1),  # noqa: SLF001
    }

    results = []
    for name, token_mask in variants.items():
        print(f"\n=== {name}: {token_mask.sum()}/1024 tokens masked ({token_mask.mean():.3f}) ===", flush=True)
        masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)
        batch = world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))
        t0 = time.time()
        ids = world_model.denoise(batch, timesteps=18)
        gen_time = time.time() - t0
        print(f"{gen_time:.1f}s", flush=True)
        image_out = tokenizer.decode(ids[0].cpu().numpy())
        image_224 = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
        score = checker.score(image_224, {"original_image": raw, "arm_pixel_mask": mask})
        print(f"score={score:.4f}", flush=True)
        Image.fromarray(image_out).save(LOG_DIR / f"gripper_bbox_{name}.png")
        results.append({"variant": name, "tokens_masked": int(token_mask.sum()), "gen_time_s": gen_time, "score": score})
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\ndone -- see {LOG_DIR}/gripper_bbox_*.png and {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
