"""Mask-area sweep (5%/10%/15%/20%) across the multi-frame, multi-task
sample collected by collect_mask_area_frames.py -- follow-up to the
single-frame (step0361) finding that ~8.8% masking looked qualitatively
different from every prior "blob" failure while ~19-25% did not. This
sweep is meant to find where that transition actually sits, and whether
it's consistent across frames/tasks with different object geometry, not
just the one frame it was first observed on.

Uses `_gripper_end_area_token_mask` (controlled area, decoupled from
S_occ per the user's design correction -- see occ_vla/CLAUDE.md).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/test_mask_area_sweep.py
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

FRAMES_DIR = _ROOT / "mask_area_sweep_frames"
OUT_DIR = FRAMES_DIR / "generated"
AREA_TARGETS = [0.05, 0.10, 0.15, 0.20]
RESULTS_PATH = _ROOT / "t08_mask_area_sweep_results.json"


def main():
    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text())
    OUT_DIR.mkdir(exist_ok=True)

    device = "cuda"
    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    world_model.load()
    gen = ArmFreeSubgoalGenerator(world_model)
    checker = PlausibilityChecker()

    results = []
    for entry in manifest:
        stub = f"{entry['label']}_step{entry['step']:03d}"
        raw = np.array(Image.open(FRAMES_DIR / f"{stub}_raw.png").convert("RGB"))
        mask = np.array(Image.open(FRAMES_DIR / f"{stub}_mask.png")) > 127
        image_512 = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        image_ids = tokenizer.encode(image_512)
        offset_image_ids = image_ids + world_model.image_token_offset
        prompt = gen._build_subgoal_prompt(entry["instruction"], 5)  # noqa: SLF001

        for target_area in AREA_TARGETS:
            token_mask = gen._gripper_end_area_token_mask(mask_512, target_area_fraction=target_area)  # noqa: SLF001
            actual_area = token_mask.sum() / 1024
            print(f"\n=== {stub} area={target_area:.0%} (actual {actual_area:.1%}, {token_mask.sum()} tokens) ===", flush=True)

            masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)
            batch = world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))
            t0 = time.time()
            ids = world_model.denoise(batch, timesteps=18)
            gen_time = time.time() - t0
            image_out = tokenizer.decode(ids[0].cpu().numpy())
            image_224 = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
            score = checker.score(image_224, {"original_image": cv2.resize(raw, (224, 224)), "arm_pixel_mask": cv2.resize(mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST) > 0})
            out_name = f"{stub}_area{int(target_area*100):02d}.png"
            Image.fromarray(image_out).save(OUT_DIR / out_name)
            print(f"{gen_time:.1f}s score={score:.4f} -> {out_name}", flush=True)

            results.append(
                {
                    "label": entry["label"],
                    "step": entry["step"],
                    "target_area": target_area,
                    "actual_area": actual_area,
                    "tokens_masked": int(token_mask.sum()),
                    "gen_time_s": gen_time,
                    "score": score,
                    "file": out_name,
                }
            )
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\ndone -- {len(results)} generations, see {OUT_DIR} and {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
