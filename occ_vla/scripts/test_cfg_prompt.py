"""Isolate two hypotheses for why MMaDA's arm-free subgoal generation
consistently produces a blob instead of a coherent held-pot render
(t08_mmada_log/calibration_summary.json: 70/70 logged generations,
schedule-bug fix in mmada.py::denoise already ruled out as the cause --
see scripts/test_denoise_fix.py):

  A. classifier-free guidance is off (guidance_scale=0 in the current
     call) -- text conditioning may be too weak to fight the mask-edge
     spatial context ("there's a gripper here").
  B. the current prompt asks for an abstract counterfactual ("Imagine 5
     steps from now... arm out of the way...") that this model may not
     handle as well as a direct, literal description of the target
     content.

Runs all 4 combinations (CFG x prompt) against the SAME real masked
frame (t08_mmada_log/step0361), using the vendored t2i_generate directly
(not the custom schedule-fixed denoise -- keeps this test orthogonal to
that already-tested change) with the CFG-relevant `resolution` kwarg
corrected to match this project's actual prefix length (max_text_len=128
+ 1 soi token = 129), since t2i_generate's own default (512) assumes a
different sequence layout than UniversalPrompting(max_text_len=128) here
-- confirmed by reading training/prompting_utils.py::t2i_gen_prompt's
token layout: [max_text_len text][<|soi|>][image tokens][<|eoi|>].

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/test_cfg_prompt.py
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
COUNTERFACTUAL_PROMPT_INSTRUCTION = "put both moka pots on the stove"
DIRECT_PROMPT = "A silver moka pot resting on a wooden table, no robot arm present, photorealistic."


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
    token_mask = gen._arm_token_mask(mask_512)
    offset_image_ids = image_ids + world_model.image_token_offset
    masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)
    masked_ids_t = torch.from_numpy(masked_ids).long().unsqueeze(0)

    # prefix length in THIS project's sequence layout: max_text_len (128) + <|soi|> (1)
    prefix_len = world_model._uni_prompting.max_text_len

    counterfactual_prompt = gen._build_subgoal_prompt(COUNTERFACTUAL_PROMPT_INSTRUCTION, 5)
    conditions = [
        ("counterfactual_noCFG", counterfactual_prompt, 0),
        ("counterfactual_CFG3", counterfactual_prompt, 3),
        ("direct_noCFG", DIRECT_PROMPT, 0),
        ("direct_CFG3", DIRECT_PROMPT, 3),
    ]

    results = {}
    for name, prompt, guidance_scale in conditions:
        print(f"\n=== {name} (guidance_scale={guidance_scale}) ===")
        print(f"prompt: {prompt!r}")
        batch = world_model.build_prompt(prompt, masked_ids_t)
        t0 = time.time()
        kwargs = dict(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            timesteps=18,
            seq_len=1024,
            mask_token_id=MASK_TOKEN_ID,
            uni_prompting=world_model._uni_prompting,
        )
        if guidance_scale > 0:
            uncond_batch = world_model.build_prompt("", masked_ids_t)
            kwargs["uncond_input_ids"] = uncond_batch.input_ids
            kwargs["uncond_attention_mask"] = uncond_batch.attention_mask
            kwargs["guidance_scale"] = guidance_scale
            kwargs["resolution"] = prefix_len  # see module docstring: default 512 is wrong for this project's layout
        ids = world_model._model.t2i_generate(**kwargs)
        print(f"{time.time()-t0:.1f}s")
        image_out = tokenizer.decode(ids[0].cpu().numpy())
        image_224 = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
        score = checker.score(image_224, {"original_image": raw, "arm_pixel_mask": mask})
        print(f"score={score:.4f}")
        Image.fromarray(image_out).save(LOG_DIR / f"cfg_{name}.png")
        results[name] = score

    print("\n=== summary ===")
    for name, score in results.items():
        print(f"{name}: score={score:.4f}  -> {LOG_DIR / f'cfg_{name}.png'}")


if __name__ == "__main__":
    main()
