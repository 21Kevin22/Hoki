"""One-off calibration analysis: recompute PlausibilityChecker's
background-MSE score against every (raw, generated, mask) triple saved
across past t08 rollouts in t08_mmada_log/, without needing pi0.5 or
MMaDA loaded (the checker is a pure numpy heuristic over saved images).

Also computes a region-limited variant (background restricted to a
dilated ring around the arm mask, approximating "near the target
object" without needing a separate target-object mask on disk) for
comparison against the current whole-frame version.

Produces:
  - t08_mmada_log/calibration_summary.json (per-event scores)
  - t08_mmada_log/calibration_contact_sheet.png (raw/generated/mask
    grid, sorted by whole-frame MSE, for visual good/bad labeling)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402

LOG_DIR = _ROOT / "t08_mmada_log"
RING_DILATION_PX = 24  # ~10% of 224px frame; "near the arm/target" band


def dilate_bool_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Cheap square-kernel dilation without a scipy/cv2 dependency."""
    out = mask.copy()
    for _ in range(px // 4 or 1):
        shifted = np.zeros_like(out)
        shifted[:-1, :] |= out[1:, :]
        shifted[1:, :] |= out[:-1, :]
        shifted[:, :-1] |= out[:, 1:]
        shifted[:, 1:] |= out[:, :-1]
        out = out | shifted
    return out


def main():
    events = sorted({p.name.split("_")[0] for p in LOG_DIR.glob("step*_raw.png")})
    print(f"found {len(events)} generation events")

    checker = PlausibilityChecker()
    results = []
    for step in events:
        raw = np.array(Image.open(LOG_DIR / f"{step}_raw.png"))
        gen = np.array(Image.open(LOG_DIR / f"{step}_generated.png"))
        mask_img = np.array(Image.open(LOG_DIR / f"{step}_mask.png"))
        arm_mask = mask_img > 127

        whole_frame_score = checker.score(gen, {"original_image": raw, "arm_pixel_mask": arm_mask})

        ring_mask = dilate_bool_mask(arm_mask, RING_DILATION_PX) & ~arm_mask
        if ring_mask.any():
            diff = gen.astype(np.float64) - raw.astype(np.float64)
            ring_mse = float((diff[ring_mask] ** 2).mean())
            ring_score = float(np.exp(-ring_mse / checker.mse_scale))
        else:
            ring_mse = None
            ring_score = None

        diff_full = gen.astype(np.float64) - raw.astype(np.float64)
        whole_mse = float((diff_full[~arm_mask] ** 2).mean())

        results.append(
            {
                "step": step,
                "arm_mask_frac": float(arm_mask.mean()),
                "whole_frame_mse": whole_mse,
                "whole_frame_score": whole_frame_score,
                "ring_mse": ring_mse,
                "ring_score": ring_score,
            }
        )
        print(f"{step}: whole_mse={whole_mse:7.2f} whole_score={whole_frame_score:.4f}  "
              f"ring_mse={ring_mse if ring_mse is None else round(ring_mse,2)} "
              f"ring_score={ring_score if ring_score is None else round(ring_score,4)}")

    results.sort(key=lambda r: r["whole_frame_mse"])
    (LOG_DIR / "calibration_summary.json").write_text(json.dumps(results, indent=2))

    # contact sheet: raw | generated | mask, one row per event, sorted by whole_frame_mse ascending
    n = len(results)
    cell = 224
    pad = 4
    label_h = 18
    sheet = Image.new("RGB", (cell * 3 + pad * 4, (cell + label_h + pad) * n + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(results):
        step = r["step"]
        y = pad + i * (cell + label_h + pad)
        raw_im = Image.open(LOG_DIR / f"{step}_raw.png").resize((cell, cell))
        gen_im = Image.open(LOG_DIR / f"{step}_generated.png").resize((cell, cell))
        mask_im = Image.open(LOG_DIR / f"{step}_mask.png").convert("L").resize((cell, cell))
        sheet.paste(raw_im, (pad, y))
        sheet.paste(gen_im, (pad * 2 + cell, y))
        sheet.paste(mask_im.convert("RGB"), (pad * 3 + cell * 2, y))
        draw.text(
            (pad, y + cell + 2),
            f"{step}  whole_mse={r['whole_frame_mse']:.1f} score={r['whole_frame_score']:.3f}  "
            f"ring_mse={r['ring_mse'] and round(r['ring_mse'],1)} ring_score={r['ring_score'] and round(r['ring_score'],3)}",
            fill="black",
        )
    sheet.save(LOG_DIR / "calibration_contact_sheet.png")
    print(f"\nwrote {LOG_DIR / 'calibration_summary.json'}")
    print(f"wrote {LOG_DIR / 'calibration_contact_sheet.png'} ({n} rows)")


if __name__ == "__main__":
    main()
