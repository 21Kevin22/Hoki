"""Cheap (no new GPU generation) follow-up to generate_vs_inpaint.py:
does compositing a "background scaffold" with MMaDA's generated object
region beat MMaDA's own full-mask generation, on the same 15 held-out
frames? User's proposed "Hybrid Structural-Semantic Restoration": PKLP
(geometry) fills static background, MMaDA (semantics) fills the actual
object, DINOv2-guided sharpening polishes the seam.

Important scoping note before running this: real PKLP
(pklp/optical_flow.py + kinematics.py) does per-patch *target-position*
kinematic extrapolation, not general background-texture reprojection --
there's no implemented "reproject static background from a past
keyframe" capability to plug in here. Two honest stand-ins instead,
both labeled clearly, neither claimed to BE PKLP:

  - "inpaint-bg" (realistic-today): cv2.inpaint as the background-fill
    engine -- what's actually buildable right now without new PKLP
    engineering.
  - "oracle-bg" (ceiling test): the real ground-truth (armfree) pixels
    for the background sub-region -- the best ANY geometric
    reprojection could ever do (a static LIBERO camera means a perfect
    background reprojection converges to the true pixels). If even this
    doesn't clearly beat MMaDA-alone, no real (imperfect) PKLP
    implementation will either -- decisive evidence either way, without
    building the real reprojection engine first.

Object-region sub-mask (which pixels get MMaDA content vs. background
fill) reuses the existing, already-tested
`ArmFreeSubgoalGenerator._gripper_end_bbox_token_mask` heuristic --
no new mask logic invented for this probe.

Zero new MMaDA calls: reuses the already-generated `mmada` and
`inpaint` panels cropped directly out of
texture_ceiling_probe/mmada_vs_inpaint_holdout/*.png (contact sheets
are armvis|overlay|inpaint|mmada|ground-truth, 512px columns).

Run (CPU only, no GPU needed, but uses .venv_mmada for its torch/cv2/
scipy install):
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES="" python3 scripts/probe_pklp_mmada_collage.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_MMADA_ROOT = _ROOT / "third_party" / "mmada"
sys.path.insert(0, str(_MMADA_ROOT))

from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator, TOKEN_GRID_SIDE  # noqa: E402

PAIRS_DIR = _ROOT / "arm_removal_pairs_policy"
SHEETS_DIR = _ROOT / "texture_ceiling_probe" / "mmada_vs_inpaint_holdout"
SAMPLE_PATH = _ROOT / "texture_ceiling_probe" / "mmada_compare_sample_holdout.json"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "collage_probe"
RESULTS_PATH = _ROOT / "texture_ceiling_probe" / "collage_probe_results.json"


def windowed_ssim(a, b, mask, win=7):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = np.zeros(a.shape[:2])
    for c in range(3):
        ac, bc = a[..., c], b[..., c]
        mu_a = uniform_filter(ac, win)
        mu_b = uniform_filter(bc, win)
        sigma_a = uniform_filter(ac * ac, win) - mu_a**2
        sigma_b = uniform_filter(bc * bc, win) - mu_b**2
        sigma_ab = uniform_filter(ac * bc, win) - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
        ssim_map += num / den
    ssim_map /= 3
    if not mask.any():
        return float("nan")
    return float(ssim_map[mask].mean())


def mse_inside(a, b, mask):
    if not mask.any():
        return float("nan")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float((diff[mask] ** 2).mean())


def token_mask_to_pixel_mask(token_mask: np.ndarray, side_px: int = 512) -> np.ndarray:
    grid = token_mask.reshape(TOKEN_GRID_SIDE, TOKEN_GRID_SIDE)
    return cv2.resize(grid.astype(np.uint8), (side_px, side_px), interpolation=cv2.INTER_NEAREST) > 0


def main():
    sample = json.loads(SAMPLE_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    gen = ArmFreeSubgoalGenerator(world_model=None)  # pure-numpy mask methods only, no model needed

    results = []
    for entry in sample:
        stub = entry["stub"]
        sheet = np.array(Image.open(SHEETS_DIR / f"{stub}.png").convert("RGB"))
        armvis = sheet[:, 0 * 512 : 1 * 512]
        inpainted = sheet[:, 2 * 512 : 3 * 512]
        mmada_out = sheet[:, 3 * 512 : 4 * 512]
        gt = sheet[:, 4 * 512 : 5 * 512]

        mask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png")) > 127
        mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0

        object_token_mask = gen._gripper_end_bbox_token_mask(mask_512, row_fraction=0.35, pad_tokens=1)  # noqa: SLF001
        object_px_mask = token_mask_to_pixel_mask(object_token_mask) & mask_512
        bg_px_mask = mask_512 & ~object_px_mask

        collage_inpaint_bg = armvis.copy()
        collage_inpaint_bg[bg_px_mask] = inpainted[bg_px_mask]
        collage_inpaint_bg[object_px_mask] = mmada_out[object_px_mask]

        collage_oracle_bg = armvis.copy()
        collage_oracle_bg[bg_px_mask] = gt[bg_px_mask]
        collage_oracle_bg[object_px_mask] = mmada_out[object_px_mask]

        row = {
            "stub": stub,
            "label": entry["label"],
            "object_px_frac": float(object_px_mask.mean()),
            "bg_px_frac": float(bg_px_mask.mean()),
            "inpaint_mse": mse_inside(inpainted, gt, mask_512),
            "mmada_mse": mse_inside(mmada_out, gt, mask_512),
            "collage_inpaint_bg_mse": mse_inside(collage_inpaint_bg, gt, mask_512),
            "collage_oracle_bg_mse": mse_inside(collage_oracle_bg, gt, mask_512),
            "inpaint_ssim": windowed_ssim(inpainted, gt, mask_512),
            "mmada_ssim": windowed_ssim(mmada_out, gt, mask_512),
            "collage_inpaint_bg_ssim": windowed_ssim(collage_inpaint_bg, gt, mask_512),
            "collage_oracle_bg_ssim": windowed_ssim(collage_oracle_bg, gt, mask_512),
        }
        results.append(row)
        print(
            f"{stub:<30} obj_frac={row['object_px_frac']:.2f}  mse mmada/collage_inpaintbg/collage_oraclebg = "
            f"{row['mmada_mse']:6.0f}/{row['collage_inpaint_bg_mse']:6.0f}/{row['collage_oracle_bg_mse']:6.0f}  "
            f"ssim = {row['mmada_ssim']:.3f}/{row['collage_inpaint_bg_ssim']:.3f}/{row['collage_oracle_bg_ssim']:.3f}",
            flush=True,
        )

        overlay = armvis.copy()
        overlay[bg_px_mask] = [255, 0, 255]
        overlay[object_px_mask] = [0, 255, 255]
        out_sheet = Image.new("RGB", (512 * 6, 512))
        for j, img in enumerate([armvis, overlay, mmada_out, collage_inpaint_bg, collage_oracle_bg, gt]):
            out_sheet.paste(Image.fromarray(img.astype(np.uint8)), (j * 512, 0))
        out_sheet.save(OUT_DIR / f"{stub}.png")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    def agg(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(vals)), float(np.median(vals))

    print(f"\n{len(results)} frames, aggregate:")
    print(f"{'metric':<25}{'mean':>10}{'median':>10}")
    for key in [
        "inpaint_mse", "mmada_mse", "collage_inpaint_bg_mse", "collage_oracle_bg_mse",
        "inpaint_ssim", "mmada_ssim", "collage_inpaint_bg_ssim", "collage_oracle_bg_ssim",
    ]:
        mean, median = agg(key)
        print(f"{key:<25}{mean:>10.3f}{median:>10.3f}")

    print(f"\ncontact sheets (armvis | region-overlay(cyan=object/magenta=bg) | mmada-alone | "
          f"collage-inpaint-bg | collage-oracle-bg | ground-truth): {OUT_DIR}/")
    print(f"full results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
