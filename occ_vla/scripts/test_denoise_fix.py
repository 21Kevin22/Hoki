"""One-off A/B test: old (vendored t2i_generate, schedule driven by the
full 1024-token seq_len) vs new (MMaDAWorldModel.denoise, schedule driven
by the actual masked-token count) on the SAME real arm-occlusion frame
saved from a past t08 rollout (t08_mmada_log/step0361_raw.png + _mask.png,
the highest-plausibility-score example on record, arm_frac=0.131).

Run in .venv_mmada, one GPU, no LIBERO/pi0.5 needed:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/test_denoise_fix.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_MMADA_ROOT = _ROOT / "third_party" / "mmada"
sys.path.insert(0, str(_MMADA_ROOT))

from occ_vla.world_model.arm_free_subgoal import DEFAULT_SUBGOAL_HORIZON, ArmFreeSubgoalGenerator  # noqa: E402
from occ_vla.world_model.mmada import MASK_TOKEN_ID, MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import CODEBOOK_SIZE, MagvitV2Tokenizer  # noqa: E402

LOG_DIR = _ROOT / "t08_mmada_log"
STEP = "step0361"  # best whole_frame_score on record (0.0935) -- still a blob, per visual inspection
INSTRUCTION = "put both moka pots on the stove"


def main():
    device = "cuda"
    tokenizer = MagvitV2Tokenizer("showlab/magvitv2", device=device)
    tokenizer.load()
    world_model = MMaDAWorldModel("Gen-Verse/MMaDA-8B-MixCoT", tokenizer, device=device)
    world_model.load()
    gen = ArmFreeSubgoalGenerator(world_model)

    raw = np.array(Image.open(LOG_DIR / f"{STEP}_raw.png").convert("RGB"))
    mask = np.array(Image.open(LOG_DIR / f"{STEP}_mask.png")) > 127
    print(f"loaded {STEP}: raw {raw.shape}, arm_mask frac={mask.mean():.3f}")

    # --- shared setup: encode + mask tokens, exactly as sample_arm_free_image does ---
    import cv2

    image_512 = cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    image_ids = tokenizer.encode(image_512)
    token_mask = gen._arm_token_mask(mask_512)
    offset_image_ids = image_ids + world_model.image_token_offset
    masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)
    print(f"masked tokens: {token_mask.sum()}/{len(token_mask)} ({token_mask.mean():.3f})")

    prompt = gen._build_subgoal_prompt(INSTRUCTION, DEFAULT_SUBGOAL_HORIZON)
    batch = world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))

    # --- OLD: vendored t2i_generate, schedule driven by full seq_len=1024 ---
    print("\n=== OLD (vendored t2i_generate, full-seq_len schedule) ===")
    t0 = time.time()
    old_ids = world_model._model.t2i_generate(
        input_ids=batch.input_ids.clone(),
        attention_mask=batch.attention_mask,
        timesteps=18,
        seq_len=1024,
        mask_token_id=MASK_TOKEN_ID,
        uni_prompting=world_model._uni_prompting,
    )
    print(f"old: {time.time()-t0:.1f}s")
    old_image = tokenizer.decode(old_ids[0].cpu().numpy())

    # --- NEW: MMaDAWorldModel.denoise, schedule driven by actual masked-token count ---
    print("\n=== NEW (masked-count-driven schedule) ===")
    t0 = time.time()
    new_ids = world_model.denoise(batch, timesteps=18)
    print(f"new: {time.time()-t0:.1f}s")
    new_image = tokenizer.decode(new_ids[0].cpu().numpy())

    out_dir = LOG_DIR
    Image.fromarray(image_512).save(out_dir / "ab_raw_512.png")
    Image.fromarray(old_image).save(out_dir / "ab_old_generated.png")
    Image.fromarray(new_image).save(out_dir / "ab_new_generated.png")

    # score both with the existing PlausibilityChecker, at 224 (matching the
    # pipeline's real comparison resolution) -- resize both back down.
    from occ_vla.integration.uncertainty import PlausibilityChecker

    checker = PlausibilityChecker()
    raw_224 = raw
    mask_224 = mask
    old_224 = cv2.resize(old_image, (224, 224), interpolation=cv2.INTER_AREA)
    new_224 = cv2.resize(new_image, (224, 224), interpolation=cv2.INTER_AREA)
    old_score = checker.score(old_224, {"original_image": raw_224, "arm_pixel_mask": mask_224})
    new_score = checker.score(new_224, {"original_image": raw_224, "arm_pixel_mask": mask_224})
    print(f"\nold_score={old_score:.4f}  new_score={new_score:.4f}")
    print(f"saved: {out_dir/'ab_raw_512.png'}, {out_dir/'ab_old_generated.png'}, {out_dir/'ab_new_generated.png'}")


if __name__ == "__main__":
    main()
