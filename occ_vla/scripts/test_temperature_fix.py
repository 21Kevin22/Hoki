"""Before/after check for the compounding-temperature fix in
world_model/mmada.py::_denoise_impl (found by diffing against
third_party/mmada/models/modeling_mmada.py::t2i_generate, which decays
`temperature` from its own *running* value each step -- `temperature =
temperature * (1.0 - ratio)` -- not fresh from a constant every step,
which is what this project's denoise() did before this fix). By step
8/18 the reference (compounding) schedule is roughly 10x smaller than
the non-compounding one this project used for every one of the 70+
generations logged so far in t08_mmada_log/ -- i.e. every prior
generation attempt (schedule fix, CFG, prompt rewrite, bbox mask) used
a temperature ~2-10x too high in mask_by_random_topk for most of the
schedule, injecting far more Gumbel-noise randomness into which tokens
get confidently locked in each round than the reference model was
actually trained/tuned to use.

Runs both schedules (OLD = pre-fix, reproduced here by monkeypatching;
NEW = current denoise()) against two mask variants already known from
this session: the original arm silhouette mask (_arm_token_mask, the
one behind every prior "blob" failure) and the small gripper-end bbox
mask that showed a qualitatively different, more coherent result today
(_gripper_end_bbox_token_mask, row_fraction=0.25).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/test_temperature_fix.py
"""

import json
import sys
import time
from pathlib import Path
from unittest import mock

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
RESULTS_PATH = _ROOT / "t08_temperature_fix_results.json"


def old_noncompounding_schedule(timesteps: int, initial_temperature: float = 1.0) -> list:
    """Reproduces the pre-fix behavior exactly (see module docstring):
    step_temperature recomputed fresh from a constant each step,
    instead of compounding from the running value."""
    schedule = []
    for step in range(timesteps):
        ratio = 1.0 * (step + 1) / timesteps
        schedule.append(initial_temperature * (1.0 - ratio))
    return schedule


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

    mask_variants = {
        "silhouette": gen._arm_token_mask(mask_512),  # noqa: SLF001
        "gripper_end_row25": gen._gripper_end_bbox_token_mask(mask_512, row_fraction=0.25, pad_tokens=0),  # noqa: SLF001
    }
    schedule_variants = {
        "OLD_noncompounding": old_noncompounding_schedule,
        "NEW_compounding": None,  # None -> use the real (fixed) denoise() unpatched
    }

    results = []
    for mask_name, token_mask in mask_variants.items():
        masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)
        batch = world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))

        for schedule_name, schedule_fn in schedule_variants.items():
            print(f"\n=== mask={mask_name} schedule={schedule_name} ===", flush=True)
            t0 = time.time()
            if schedule_fn is None:
                ids = world_model.denoise(batch, timesteps=18)
            else:
                with mock.patch("occ_vla.world_model.mmada._compounding_temperature_schedule", schedule_fn):
                    ids = world_model.denoise(batch, timesteps=18)
            gen_time = time.time() - t0
            print(f"{gen_time:.1f}s", flush=True)
            image_out = tokenizer.decode(ids[0].cpu().numpy())
            image_224 = cv2.resize(image_out, (224, 224), interpolation=cv2.INTER_AREA)
            score = checker.score(image_224, {"original_image": raw, "arm_pixel_mask": mask})
            print(f"score={score:.4f}", flush=True)
            out_name = f"tempfix_{mask_name}_{schedule_name}.png"
            Image.fromarray(image_out).save(LOG_DIR / out_name)
            results.append(
                {"mask": mask_name, "schedule": schedule_name, "tokens_masked": int(token_mask.sum()), "gen_time_s": gen_time, "score": score, "file": out_name}
            )
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\ndone -- see {LOG_DIR}/tempfix_*.png and {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
