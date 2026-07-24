"""Cheap, no-GPU, no-training probe: before committing to the 3-stage
"Texture-Conditioned World Model" roadmap (DINOv2 features -> MAGVIT-v2
tokens -> MMaDA-LoRA VIM/Goal reasoning -> Cross-Attention UNet texture
decoder, proposed 2026-07-21, logged as NOT built in CLAUDE.md), test the
architecture's central premise in isolation: does re-injecting real
*texture* into the occluded region actually recover plausible content, or
is the bottleneck something texture injection can't fix (not knowing
*what* was occluded, i.e. content/structure, not surface appearance)?

Uses `arm_removal_pairs/` -- real ground-truth (arm-visible, arm-free,
arm-mask) triples from `collect_arm_removal_pairs.py` (same MuJoCo state
rendered twice, robot geoms alpha-zeroed the second time; zero generative
guessing, perfectly aligned). This lets us score reconstruction quality
*inside* the actually-occluded region against real ground truth, unlike
PlausibilityChecker's background-only proxy (already known to not measure
generation quality -- see CLAUDE.md "MMaDA arm-free generation quality
investigation").

Classical `cv2.inpaint` (Telea) is the proxy for "texture reinjection
without content reasoning": it only extrapolates from surrounding real
pixels, has zero knowledge that a specific object (pot/bowl/mug) should
be there. If it already matches or beats MMaDA's actual blob-collapse
output on ground-truth accuracy, that's evidence the LLM's structural
content isn't adding value here and a texture-focused fix (DINOv2
Cross-Attention decoder) is unlikely to be the missing lever. If it's
clearly worse (smeared/wrong-object/missing-object), that's evidence the
hard part is content reasoning, and texture injection alone -- however
implemented -- can't fix it.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
PAIRS_DIR = _ROOT / "arm_removal_pairs"
OUT_DIR = _ROOT / "texture_ceiling_probe"
INPAINT_RADIUS = 5
CONTACT_SHEET_N = 8


def load(stub: str, suffix: str) -> np.ndarray:
    return np.array(Image.open(PAIRS_DIR / f"{stub}_{suffix}.png").convert("RGB"))


def windowed_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray, win: int = 7) -> float:
    """Standard windowed SSIM (Wang et al. 2004 constants), restricted to
    `mask`. Hand-rolled via scipy.ndimage.uniform_filter since skimage
    isn't installed here -- avoids adding a new dependency for a one-off
    probe script."""
    from scipy.ndimage import uniform_filter

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


def mse_inside(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float((diff[mask] ** 2).mean())


def main():
    manifest = json.loads((PAIRS_DIR / "manifest.json").read_text())
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    contact_examples = []
    for entry in manifest:
        stub = entry["stub"]
        armvis = load(stub, "armvis")
        armfree = load(stub, "armfree")
        mask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png")) > 127

        if not mask.any():
            continue

        inpainted = cv2.inpaint(armvis, mask.astype(np.uint8) * 255, INPAINT_RADIUS, cv2.INPAINT_TELEA)

        # dumb floor: mean color of the surrounding (unmasked) ring, flat-filled
        ring = armvis.copy()
        mean_color = armvis[~mask].mean(axis=0)
        flatfill = armvis.copy()
        flatfill[mask] = mean_color

        row = {
            "stub": stub,
            "label": entry["label"],
            "arm_px": entry["arm_px"],
            "mask_frac": float(mask.mean()),
            "leave_as_arm_mse": mse_inside(armvis, armfree, mask),
            "flatfill_mse": mse_inside(flatfill, armfree, mask),
            "inpaint_mse": mse_inside(inpainted, armfree, mask),
            "leave_as_arm_ssim": windowed_ssim(armvis, armfree, mask),
            "flatfill_ssim": windowed_ssim(flatfill, armfree, mask),
            "inpaint_ssim": windowed_ssim(inpainted, armfree, mask),
        }
        rows.append(row)

        if len(contact_examples) < CONTACT_SHEET_N and entry["label"] in {"moka_pots", "bowl_in_drawer", "mug_in_microwave"}:
            contact_examples.append((stub, armvis, inpainted, armfree, mask))

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))

    def agg(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)), float(np.median(vals))

    print(f"{len(rows)} pairs scored\n")
    print(f"{'metric':<20}{'mean':>10}{'median':>10}")
    for key in ["leave_as_arm_mse", "flatfill_mse", "inpaint_mse", "leave_as_arm_ssim", "flatfill_ssim", "inpaint_ssim"]:
        mean, median = agg(key)
        print(f"{key:<20}{mean:>10.3f}{median:>10.3f}")

    # per-task breakdown for inpaint (the interesting condition)
    print("\nper-task inpaint_mse / inpaint_ssim:")
    labels = sorted({r["label"] for r in rows})
    for label in labels:
        sub = [r for r in rows if r["label"] == label]
        mse_vals = [r["inpaint_mse"] for r in sub]
        ssim_vals = [r["inpaint_ssim"] for r in sub]
        print(f"  {label:<20} mse={np.mean(mse_vals):>8.2f}  ssim={np.mean(ssim_vals):>6.3f}  n={len(sub)}")

    # contact sheet
    tile = 256
    n = len(contact_examples)
    if n:
        sheet = Image.new("RGB", (tile * 4, tile * n))
        for i, (stub, armvis, inpainted, armfree, mask) in enumerate(contact_examples):
            overlay = armvis.copy()
            overlay[mask] = [255, 0, 255]
            for j, img in enumerate([armvis, overlay, inpainted, armfree]):
                sheet.paste(Image.fromarray(img.astype(np.uint8)), (j * tile, i * tile))
        sheet.save(OUT_DIR / "contact_sheet.png")
        print(f"\ncontact sheet (armvis | mask overlay | cv2.inpaint | ground-truth armfree): {OUT_DIR / 'contact_sheet.png'}")

    print(f"\nfull results: {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
