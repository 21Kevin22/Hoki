"""Does a genuinely different decoding paradigm (continuous latent
diffusion, iterative denoising) avoid the failure mode root-caused for
MMaDA-8B (MaskGIT-style parallel non-autoregressive token decoding --
see CLAUDE.md's H1/H2 step-count sweep)? User's hypothesis (2026-07-23):
"maybe a different generative model works" -- the root cause found isn't
"generative models can't do this," it's specifically "this decoding
scheme can't do this," so a model with a different decoding procedure
is a real, distinct hypothesis, not just retrying the same failure mode
under a new name.

Uses Stable Diffusion inpainting via `diffusers` (already installed in
.venv_mmada, no new dependency) -- a standard continuous-latent
iterative-denoising inpainter, architecturally unrelated to MaskGIT/
discrete-token parallel decoding.

Same rigorous methodology as generate_vs_inpaint.py: real ground-truth
(armvis, armfree, armmask) triples from arm_removal_pairs_policy/
(bowl_top_drawer, the task where MMaDA-LoRA's geometric dust3r recovery
already succeeded cleanly -- picked by the user to compare against a
known-good case, not a known-hard one), inside-mask MSE/SSIM against
real ground truth, visual inspection required before trusting scores.

Caveat: MMaDA-LoRA was fine-tuned on this same task's non-held-out
episodes (no held-out bowl_top_drawer object-overlap frames exist in
this dataset -- checked). Stable Diffusion gets no equivalent fine-
tuning/memorization advantage here (zero-shot). If SD still matches or
beats MMaDA despite this asymmetry, that's stronger evidence against
MMaDA's architecture, not weaker.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_sd_inpaint_vs_mmada.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from scipy.ndimage import uniform_filter

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_MMADA_ROOT = _ROOT / "third_party" / "mmada"
sys.path.insert(0, str(_MMADA_ROOT))

from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator  # noqa: E402
from occ_vla.world_model.mmada import MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import MagvitV2Tokenizer  # noqa: E402

PAIRS_DIR = _ROOT / "arm_removal_pairs_policy"
SAMPLE_PATH = _ROOT / "texture_ceiling_probe" / "sd_compare_sample.json"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "sd_vs_mmada"
ADAPTER_PATH = _ROOT / "arm_removal_lora_full_adapter"
RESULTS_PATH = _ROOT / "texture_ceiling_probe" / "sd_vs_mmada_results.json"
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"  # gated stabilityai/stable-diffusion-2-inpainting 401'd with no HF token configured


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


def main():
    from diffusers import StableDiffusionInpaintPipeline  # noqa: PLC0415

    sample = json.loads(SAMPLE_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    print("loading MMaDA-LoRA...", flush=True)
    tokenizer = MagvitV2Tokenizer(checkpoint_path="showlab/magvitv2", device="cuda:0")
    tokenizer.load()
    world_model = MMaDAWorldModel(checkpoint_path="Gen-Verse/MMaDA-8B-MixCoT", tokenizer=tokenizer, device="cuda:0")
    world_model.load()
    world_model._model = PeftModel.from_pretrained(world_model._model, str(ADAPTER_PATH))  # noqa: SLF001
    generator = ArmFreeSubgoalGenerator(world_model)

    print("loading Stable Diffusion inpainting...", flush=True)
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    sd_pipe = sd_pipe.to("cuda:0")
    print("models loaded", flush=True)

    results = []
    for entry in sample:
        stub = entry["stub"]
        armvis = np.array(Image.open(PAIRS_DIR / f"{stub}_armvis.png").convert("RGB"))
        armfree = np.array(Image.open(PAIRS_DIR / f"{stub}_armfree.png").convert("RGB"))
        mask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png")) > 127

        armvis_512 = cv2.resize(armvis, (512, 512), interpolation=cv2.INTER_AREA)
        armfree_512 = cv2.resize(armfree, (512, 512), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0

        inpainted = cv2.inpaint(armvis_512, mask_512.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

        t0 = time.time()
        mmada_result = generator.sample_arm_free_image(image=armvis_512, arm_pixel_mask=mask_512, instruction=entry["instruction"])
        mmada_time = time.time() - t0
        mmada_out = mmada_result.image
        if mmada_out.shape[:2] != (512, 512):
            mmada_out = cv2.resize(mmada_out, (512, 512), interpolation=cv2.INTER_AREA)

        # Stable Diffusion inpainting: standard usage, mask=white where
        # content should be generated (matches arm_pixel_mask convention)
        t0 = time.time()
        sd_result = sd_pipe(
            prompt=f"{entry['instruction']}, photorealistic robot workspace, wooden table",
            image=Image.fromarray(armvis_512),
            mask_image=Image.fromarray((mask_512 * 255).astype(np.uint8)),
            num_inference_steps=30,
        ).images[0]
        sd_time = time.time() - t0
        sd_out = np.array(sd_result.resize((512, 512)))

        row = {
            "stub": stub,
            "label": entry["label"],
            "mmada_time_s": mmada_time,
            "sd_time_s": sd_time,
            "inpaint_mse": mse_inside(inpainted, armfree_512, mask_512),
            "mmada_mse": mse_inside(mmada_out, armfree_512, mask_512),
            "sd_mse": mse_inside(sd_out, armfree_512, mask_512),
            "inpaint_ssim": windowed_ssim(inpainted, armfree_512, mask_512),
            "mmada_ssim": windowed_ssim(mmada_out, armfree_512, mask_512),
            "sd_ssim": windowed_ssim(sd_out, armfree_512, mask_512),
        }
        results.append(row)
        print(
            f"{stub:<30} mse inpaint/mmada/sd = {row['inpaint_mse']:7.0f}/{row['mmada_mse']:7.0f}/{row['sd_mse']:7.0f}  "
            f"ssim = {row['inpaint_ssim']:.3f}/{row['mmada_ssim']:.3f}/{row['sd_ssim']:.3f}  "
            f"time mmada/sd = {mmada_time:.1f}s/{sd_time:.1f}s",
            flush=True,
        )
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

        overlay = armvis_512.copy()
        overlay[mask_512] = [255, 0, 255]
        sheet = Image.new("RGB", (512 * 6, 512))
        for j, img in enumerate([armvis_512, overlay, inpainted, mmada_out, sd_out, armfree_512]):
            sheet.paste(Image.fromarray(img.astype(np.uint8)), (j * 512, 0))
        sheet.save(OUT_DIR / f"{stub}.png")

    def agg(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(vals)), float(np.median(vals))

    print(f"\n{len(results)} frames, aggregate:")
    print(f"{'metric':<15}{'mean':>10}{'median':>10}")
    for key in ["inpaint_mse", "mmada_mse", "sd_mse", "inpaint_ssim", "mmada_ssim", "sd_ssim"]:
        mean, median = agg(key)
        print(f"{key:<15}{mean:>10.3f}{median:>10.3f}")

    print(f"\nper-frame contact sheets (armvis | mask | inpaint | mmada | sd | ground-truth): {OUT_DIR}/")
    print(f"full results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
