"""Two cheap, zero-new-dependency style-domain-adaptation ideas (user
request, 2026-07-23) applied to SD inpainting on the EXACT frame already
used for the pi0.5 injection test (texture_ceiling_probe/pi05_injection_test/,
bowl_top_drawer step 44, arm_s_occ=0.339):

1. Prompt steering toward LIBERO's flat-shaded synthetic render style
   (negative prompt against photorealistic/ornate detail).
2. Post-hoc edge-preserving smoothing (bilateral filter) to knock down
   the high-frequency real-photo texture SD tends to add.

No arm mask was saved for this specific frame (collect_occlusion_moment_with_state.py
only saved images/state/poses) -- derived here from the pixel difference
between occluded_agentview_256.png and gt_agentview_256.png (same sim
state, robot geoms alpha-zeroed for the GT one -- any large color delta
between them is exactly the arm's footprint).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_sd_style_adapted.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

FRAME_DIR = _ROOT / "texture_ceiling_probe" / "pi05_injection_test"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "sd_style_adapted"
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"

FLAT_SHADING_PROMPT = (
    "simple flat-shaded 3D render, clean synthetic simulation graphics, "
    "matte plastic surface, low detail, MuJoCo robotics simulator style"
)
NEGATIVE_PROMPT = (
    "photorealistic, ornate, intricate detail, metallic texture, "
    "steampunk, engraved, realistic photo, high detail, rust, patina"
)


def derive_arm_mask(occluded, gt, thresh=30):
    diff = np.abs(occluded.astype(np.int16) - gt.astype(np.int16)).sum(axis=-1)
    mask = (diff > thresh).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def main():
    from diffusers import StableDiffusionInpaintPipeline  # noqa: PLC0415

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    occluded = np.array(Image.open(FRAME_DIR / "occluded_agentview_256.png").convert("RGB"))
    gt = np.array(Image.open(FRAME_DIR / "gt_agentview_256.png").convert("RGB"))
    mask = derive_arm_mask(occluded, gt)
    Image.fromarray(mask).save(OUT_DIR / "derived_arm_mask.png")
    print(f"derived arm mask: {mask.mean() / 255:.3f} of frame", flush=True)

    occluded_512 = cv2.resize(occluded, (512, 512), interpolation=cv2.INTER_AREA)
    gt_512 = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    print("loading SD inpainting...", flush=True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda:0")

    instruction = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"

    print("generating: raw (photorealistic-leaning prompt, matches earlier test)...", flush=True)
    raw_result = pipe(
        prompt=f"{instruction}, photorealistic robot workspace, wooden table",
        image=Image.fromarray(occluded_512),
        mask_image=Image.fromarray(mask_512),
        num_inference_steps=30,
    ).images[0]
    raw_out = np.array(raw_result.resize((512, 512)))

    print("generating: style-adapted (flat-shading prompt + negative prompt)...", flush=True)
    styled_result = pipe(
        prompt=f"{instruction}, {FLAT_SHADING_PROMPT}",
        negative_prompt=NEGATIVE_PROMPT,
        image=Image.fromarray(occluded_512),
        mask_image=Image.fromarray(mask_512),
        num_inference_steps=30,
    ).images[0]
    styled_out = np.array(styled_result.resize((512, 512)))

    # post-hoc domain smoothing: bilateral filter on the styled output,
    # only inside the mask (leave the real unmasked pixels untouched)
    smoothed = cv2.bilateralFilter(styled_out, d=9, sigmaColor=75, sigmaSpace=75)
    styled_smoothed_out = styled_out.copy()
    mask_bool = mask_512 > 127
    styled_smoothed_out[mask_bool] = smoothed[mask_bool]

    def mse_inside(a, b, m):
        diff = a.astype(np.float64) - b.astype(np.float64)
        return float((diff[m] ** 2).mean())

    print("\n=== inside-mask MSE vs ground truth ===", flush=True)
    print(f"raw SD:              {mse_inside(raw_out, gt_512, mask_bool):.1f}", flush=True)
    print(f"styled (prompt only): {mse_inside(styled_out, gt_512, mask_bool):.1f}", flush=True)
    print(f"styled + smoothed:    {mse_inside(styled_smoothed_out, gt_512, mask_bool):.1f}", flush=True)

    overlay = occluded_512.copy()
    overlay[mask_bool] = [255, 0, 255]
    sheet = Image.new("RGB", (512 * 6, 512))
    for j, img in enumerate([occluded_512, overlay, raw_out, styled_out, styled_smoothed_out, gt_512]):
        sheet.paste(Image.fromarray(img.astype(np.uint8)), (j * 512, 0))
    sheet.save(OUT_DIR / "comparison_sheet.png")
    print(f"\nsaved: occluded | mask | raw_sd | styled_sd | styled+smoothed | gt -> {OUT_DIR / 'comparison_sheet.png'}", flush=True)

    # save 224x224 versions for pi0.5 injection
    for name, img in [("raw_sd", raw_out), ("styled_sd", styled_out), ("styled_smoothed_sd", styled_smoothed_out)]:
        Image.fromarray(img).resize((224, 224)).save(FRAME_DIR / f"{name}_224.png")
    print(f"saved 224x224 versions to {FRAME_DIR}/ for pi0.5 injection test", flush=True)


if __name__ == "__main__":
    main()
