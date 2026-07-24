"""Quick zero-shot (no LoRA) generation on the exact same held-out
sample used in train_arm_removal_lora_full.py's validation, for a
direct apples-to-apples visual comparison against the trained
checkpoints already saved in arm_removal_pairs_policy/eval_progress/
-- per lesson learned earlier this session, don't trust the
PlausibilityChecker score alone to judge whether training actually
improved anything; look at the images against a real zero-shot
baseline on the SAME sample.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=1 python3 scripts/zero_shot_compare_sample.py
"""

import sys
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
from occ_vla.world_model.mmada import MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import MagvitV2Tokenizer, SUBGOAL_IMAGE_SIDE  # noqa: E402

PAIRS_DIR = _ROOT / "arm_removal_pairs_policy"
STUB = "moka_pots_ep19_s000"
INSTRUCTION = "put both moka pots on the stove"
OUT_PATH = PAIRS_DIR / "eval_progress" / "zeroshot_moka_pots_ep19_s000.png"


def main():
    device = "cuda"
    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    world_model.load()  # plain bf16, no LoRA
    gen = ArmFreeSubgoalGenerator(world_model)

    armvis = np.array(Image.open(PAIRS_DIR / f"{STUB}_armvis.png").convert("RGB"))
    armmask = np.array(Image.open(PAIRS_DIR / f"{STUB}_armmask.png")) > 127
    armvis_512 = cv2.resize(armvis, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(armmask.astype(np.uint8), (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_NEAREST) > 0

    result = gen.sample_arm_free_image(armvis_512, mask_512, INSTRUCTION, horizon=5)
    OUT_PATH.parent.mkdir(exist_ok=True)
    Image.fromarray(result.image).save(OUT_PATH)
    print(f"saved zero-shot generation to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
