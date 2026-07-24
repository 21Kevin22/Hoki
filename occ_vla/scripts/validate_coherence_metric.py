"""Offline (no GPU needed) cross-check of coherence_metric.py's
prototype score against already-generated images from this session,
alongside informal visual verdicts recorded while reviewing them
in-session. Reconstructs the exact pixel-space mask each image was
generated with (all mask-construction functions are pure/deterministic
given the same arm-mask input, so this doesn't need MMaDA reloaded).

This is a sanity check, not a rigorous validation (n is small and the
"visual verdict" labels below are the author's own informal judgments
made once each, not independently re-verified) -- see
coherence_metric.py's module docstring for what would be needed before
this replaces PlausibilityChecker anywhere real.

Run: python3 scripts/validate_coherence_metric.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from occ_vla.integration.coherence_metric import coherence_score  # noqa: E402
from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator, TOKEN_GRID_SIDE  # noqa: E402

gen = ArmFreeSubgoalGenerator(world_model=None)


def token_mask_to_pixel_mask(token_mask: np.ndarray, side: int = 512) -> np.ndarray:
    ph = side // TOKEN_GRID_SIDE
    return np.kron(token_mask.reshape(TOKEN_GRID_SIDE, TOKEN_GRID_SIDE), np.ones((ph, ph), dtype=bool))


# (image_path, arm_mask_path, mask_builder, verdict, old_score)
# old_score is whatever PlausibilityChecker/manual score was recorded for that
# exact image earlier in-session (t08_gripper_bbox_results.json /
# t08_temperature_fix_results.json / t08_mask_area_sweep_results.json).
CASES = [
    (
        "t08_mmada_log/gripper_bbox_gripper_end_row25_pad0.png",
        "t08_mmada_log/step0361_mask.png",
        lambda m: gen._gripper_end_bbox_token_mask(m, row_fraction=0.25, pad_tokens=0),  # noqa: SLF001
        "good",
        0.016,
    ),
    (
        "t08_mmada_log/gripper_bbox_gripper_end_row35_pad1.png",
        "t08_mmada_log/step0361_mask.png",
        lambda m: gen._gripper_end_bbox_token_mask(m, row_fraction=0.35, pad_tokens=1),  # noqa: SLF001
        "bad_smear",
        0.0,
    ),
    (
        "t08_mmada_log/gripper_bbox_gripper_end_row50_pad1.png",
        "t08_mmada_log/step0361_mask.png",
        lambda m: gen._gripper_end_bbox_token_mask(m, row_fraction=0.50, pad_tokens=1),  # noqa: SLF001
        "bad_smear",
        0.0,
    ),
    (
        "t08_mmada_log/tempfix_silhouette_OLD_noncompounding.png",
        "t08_mmada_log/step0361_mask.png",
        lambda m: gen._arm_token_mask(m),  # noqa: SLF001
        "ok_blocky_but_recognizable",
        0.0797,
    ),
    (
        "t08_mmada_log/tempfix_silhouette_NEW_compounding.png",
        "t08_mmada_log/step0361_mask.png",
        lambda m: gen._arm_token_mask(m),  # noqa: SLF001
        "ok_blocky_but_recognizable",
        0.0809,
    ),
    (
        "t08_mmada_log/tempfix_gripper_end_row25_OLD_noncompounding.png",
        "t08_mmada_log/step0361_mask.png",
        lambda m: gen._gripper_end_bbox_token_mask(m, row_fraction=0.25, pad_tokens=0),  # noqa: SLF001
        "good",
        0.0108,
    ),
    (
        "mask_area_sweep_frames/generated/mugs_step060_area05.png",
        "mask_area_sweep_frames/mugs_step060_mask.png",
        lambda m: gen._gripper_end_area_token_mask(m, target_area_fraction=0.05),  # noqa: SLF001
        "good",
        0.286,
    ),
    (
        "mask_area_sweep_frames/generated/mugs_step120_area05.png",
        "mask_area_sweep_frames/mugs_step120_mask.png",
        lambda m: gen._gripper_end_area_token_mask(m, target_area_fraction=0.05),  # noqa: SLF001
        "bad_garbled",
        0.0045,
    ),
    (
        "mask_area_sweep_frames/generated/moka_pots_step060_area05.png",
        "mask_area_sweep_frames/moka_pots_step060_mask.png",
        lambda m: gen._gripper_end_area_token_mask(m, target_area_fraction=0.05),  # noqa: SLF001
        "arm_not_actually_removed",  # known blind spot -- see module docstring
        0.250,
    ),
    (
        "mask_area_sweep_frames/generated/moka_pots_step060_area20.png",
        "mask_area_sweep_frames/moka_pots_step060_mask.png",
        lambda m: gen._gripper_end_area_token_mask(m, target_area_fraction=0.20),  # noqa: SLF001
        "arm_not_actually_removed",  # known blind spot
        0.090,
    ),
]


def main():
    rows = []
    for image_path, arm_mask_path, mask_builder, verdict, old_score in CASES:
        image = np.array(Image.open(_ROOT / image_path).convert("RGB"))
        arm_mask_raw = np.array(Image.open(_ROOT / arm_mask_path)) > 127
        side = image.shape[0]
        arm_mask_512 = cv2.resize(arm_mask_raw.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        token_mask = mask_builder(arm_mask_512)
        pixel_mask = token_mask_to_pixel_mask(token_mask, side=512)
        if side != 512:
            pixel_mask = cv2.resize(pixel_mask.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST) > 0

        new_score = coherence_score(image, pixel_mask)
        rows.append(
            {
                "image": Path(image_path).name,
                "verdict": verdict,
                "old_mse_score": old_score,
                "new_coherence_score": round(new_score, 4),
            }
        )

    print(f"{'image':<52} {'verdict':<28} {'old_mse':>8} {'new_coherence':>14}")
    for r in rows:
        print(f"{r['image']:<52} {r['verdict']:<28} {r['old_mse_score']:>8.4f} {r['new_coherence_score']:>14.4f}")

    out_path = _ROOT / "coherence_metric_validation.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
