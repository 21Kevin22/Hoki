"""Direct comparison: does MMaDA-LoRA's generation carry more object
content than classical cv2.inpaint, on the same real occluded frames?

Follow-up to scripts/probe_texture_ceiling.py, which found classical
inpainting (pure texture extrapolation, zero content reasoning) fails to
reconstruct occluded objects (a drawer knob became a dashed noise
pattern) despite improving background-fidelity MSE/SSIM. That result
argues against "texture quality" being the missing ingredient in the
proposed DINOv2 Cross-Attention decoder -- but doesn't yet tell us
whether MMaDA's LLM-derived structure prior does any better. This script
answers that directly: same frames, same ground truth, three-way
comparison (leave-arm-in-place / cv2.inpaint / MMaDA-LoRA r=16).

Sample: texture_ceiling_probe/mmada_compare_sample.json (24 frames,
stratified across the 4 tasks that have real arm-over-object overlap in
arm_removal_pairs_policy/, one per episode, middle-of-episode sample).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/generate_vs_inpaint.py
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
SAMPLE_PATH = _ROOT / "texture_ceiling_probe" / "mmada_compare_sample_holdout.json"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "mmada_vs_inpaint_holdout"
ADAPTER_PATH = _ROOT / "arm_removal_lora_full_adapter"
RESULTS_PATH = _ROOT / "texture_ceiling_probe" / "mmada_vs_inpaint_holdout_results.json"


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
    sample = json.loads(SAMPLE_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    tokenizer = MagvitV2Tokenizer(checkpoint_path="showlab/magvitv2", device="cuda:0")
    tokenizer.load()
    world_model = MMaDAWorldModel(checkpoint_path="Gen-Verse/MMaDA-8B-MixCoT", tokenizer=tokenizer, device="cuda:0")
    world_model.load()
    world_model._model = PeftModel.from_pretrained(world_model._model, str(ADAPTER_PATH))  # noqa: SLF001
    generator = ArmFreeSubgoalGenerator(world_model)
    print(f"models loaded, adapter={ADAPTER_PATH}", flush=True)

    results = []
    for entry in sample:
        stub = entry["stub"]
        armvis = np.array(Image.open(PAIRS_DIR / f"{stub}_armvis.png").convert("RGB"))
        armfree = np.array(Image.open(PAIRS_DIR / f"{stub}_armfree.png").convert("RGB"))
        mask = np.array(Image.open(PAIRS_DIR / f"{stub}_armmask.png")) > 127

        # everything compared in 512x512 space (MMaDA's native resolution
        # -- see arm_free_subgoal.py's SUBGOAL_IMAGE_SIDE requirement)
        armvis_512 = cv2.resize(armvis, (512, 512), interpolation=cv2.INTER_AREA)
        armfree_512 = cv2.resize(armfree, (512, 512), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0

        inpainted = cv2.inpaint(armvis_512, mask_512.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

        t0 = time.time()
        result = generator.sample_arm_free_image(
            image=armvis_512, arm_pixel_mask=mask_512, instruction=entry["instruction"]
        )
        gen_time = time.time() - t0
        mmada_out = result.image
        if mmada_out.shape[:2] != (512, 512):
            mmada_out = cv2.resize(mmada_out, (512, 512), interpolation=cv2.INTER_AREA)

        row = {
            "stub": stub,
            "label": entry["label"],
            "gen_time_s": gen_time,
            "leave_mse": mse_inside(armvis_512, armfree_512, mask_512),
            "inpaint_mse": mse_inside(inpainted, armfree_512, mask_512),
            "mmada_mse": mse_inside(mmada_out, armfree_512, mask_512),
            "leave_ssim": windowed_ssim(armvis_512, armfree_512, mask_512),
            "inpaint_ssim": windowed_ssim(inpainted, armfree_512, mask_512),
            "mmada_ssim": windowed_ssim(mmada_out, armfree_512, mask_512),
        }
        results.append(row)
        print(
            f"{stub:<30} {gen_time:5.1f}s  mse leave/inpaint/mmada = "
            f"{row['leave_mse']:7.0f}/{row['inpaint_mse']:7.0f}/{row['mmada_mse']:7.0f}  "
            f"ssim = {row['leave_ssim']:.3f}/{row['inpaint_ssim']:.3f}/{row['mmada_ssim']:.3f}",
            flush=True,
        )
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

        overlay = armvis_512.copy()
        overlay[mask_512] = [255, 0, 255]
        sheet = Image.new("RGB", (512 * 5, 512))
        for j, img in enumerate([armvis_512, overlay, inpainted, mmada_out, armfree_512]):
            sheet.paste(Image.fromarray(img.astype(np.uint8)), (j * 512, 0))
        sheet.save(OUT_DIR / f"{stub}.png")

    def agg(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(vals)), float(np.median(vals))

    print(f"\n{len(results)} frames, aggregate:")
    print(f"{'metric':<15}{'mean':>10}{'median':>10}")
    for key in ["leave_mse", "inpaint_mse", "mmada_mse", "leave_ssim", "inpaint_ssim", "mmada_ssim"]:
        mean, median = agg(key)
        print(f"{key:<15}{mean:>10.3f}{median:>10.3f}")

    print(f"\nper-frame contact sheets (armvis | mask | inpaint | mmada | ground-truth): {OUT_DIR}/")
    print(f"full results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
