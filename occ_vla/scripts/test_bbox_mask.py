"""Test whether masking the arm's rectangular pixel bounding box (a
solid, contiguous token-grid block) instead of its thin diagonal
silhouette fixes the blob artifact seen in every prior generation
attempt (schedule fix: test_denoise_fix.py, CFG/prompt: test_cfg_prompt.py
-- both ruled out, 6/6 failures with the same shape/location of blob).

Uses the schedule-fixed denoise() (mmada.py) + the best-scoring
no-CFG/counterfactual-prompt condition from test_cfg_prompt.py, varying
only the mask shape: silhouette (_arm_token_mask) vs bbox
(_arm_bbox_token_mask, pad_tokens=1).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/test_bbox_mask.py
"""

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
    prompt = gen._build_subgoal_prompt(INSTRUCTION, 5)

    variants = {
        "silhouette": gen._arm_token_mask(mask_512),
        "bbox_pad1": gen._arm_bbox_token_mask(mask_512, pad_tokens=1),
        "bbox_pad0": gen._arm_bbox_token_mask(mask_512, pad_tokens=0),
    }

    for name, token_mask in variants.items():
        print(f"\n=== {name}: {token_mask.sum()}/1024 tokens masked ({token_mask.mean():.3f}) ===")
        masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)
        batch = world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))
        t0 = time.time()
        ids = world_model.denoise(batch, timesteps=18)
        print(f"{time.time()-t0:.1f}s")
        image_out = tokenizer.decode(ids[0].cpu().numpy())
        image_224 = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
        score = checker.score(image_224, {"original_image": raw, "arm_pixel_mask": mask})
        print(f"score={score:.4f}")
        Image.fromarray(image_out).save(LOG_DIR / f"bbox_{name}.png")

    print("\ndone -- see t08_mmada_log/bbox_*.png")


if __name__ == "__main__":
    main()
