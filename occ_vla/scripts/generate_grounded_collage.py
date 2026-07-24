"""Grounding-corrected collage test (user proposal, 2026-07-21): object
sub-region = the target's TRUE clear-baseline segmentation footprint
intersected with the arm mask (env.obj_of_interest, same mechanism
S_occ already uses), not the `_gripper_end_bbox_token_mask` geometric
heuristic already shown to invert object/background for these tasks'
reach poses.

Runs MMaDA-LoRA generation fresh on grounded_holdout_frames/ (9
mug_in_microwave samples; moka_pots had 0 samples where the arm actually
overlapped the target's clear footprint in this rollout -- see
collect_grounded_holdout_frames.py's output), then builds both the
grounded collage (realistic: cv2.inpaint background; oracle: ground-truth
background) and scores against MMaDA-alone / inpaint-alone, both over
the full arm mask AND restricted to just the true object sub-region
(the object footprint here is tiny -- <=352px of a 256x256 frame, ~0.5%
-- so a full-mask aggregate could hide whether grounding actually fixed
anything at the object itself).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/generate_grounded_collage.py
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

FRAMES_DIR = _ROOT / "grounded_holdout_frames"
ADAPTER_PATH = _ROOT / "arm_removal_lora_full_adapter"
OUT_DIR = _ROOT / "texture_ceiling_probe" / "grounded_collage"
RESULTS_PATH = _ROOT / "texture_ceiling_probe" / "grounded_collage_results.json"


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
        return None
    return float(ssim_map[mask].mean())


def mse_inside(a, b, mask):
    if not mask.any():
        return None
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float((diff[mask] ** 2).mean())


def main():
    manifest = json.loads((FRAMES_DIR / "manifest.json").read_text())
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    tokenizer = MagvitV2Tokenizer(checkpoint_path="showlab/magvitv2", device="cuda:0")
    tokenizer.load()
    world_model = MMaDAWorldModel(checkpoint_path="Gen-Verse/MMaDA-8B-MixCoT", tokenizer=tokenizer, device="cuda:0")
    world_model.load()
    world_model._model = PeftModel.from_pretrained(world_model._model, str(ADAPTER_PATH))  # noqa: SLF001
    generator = ArmFreeSubgoalGenerator(world_model)
    print(f"models loaded, adapter={ADAPTER_PATH}", flush=True)

    results = []
    for entry in manifest:
        stub = entry["stub"]
        armvis = np.array(Image.open(FRAMES_DIR / f"{stub}_armvis.png").convert("RGB"))
        armfree = np.array(Image.open(FRAMES_DIR / f"{stub}_armfree.png").convert("RGB"))
        arm_mask = np.array(Image.open(FRAMES_DIR / f"{stub}_armmask.png")) > 127
        obj_mask = np.array(Image.open(FRAMES_DIR / f"{stub}_objmask.png")) > 127  # true grounded object region

        armvis_512 = cv2.resize(armvis, (512, 512), interpolation=cv2.INTER_AREA)
        armfree_512 = cv2.resize(armfree, (512, 512), interpolation=cv2.INTER_AREA)
        arm_mask_512 = cv2.resize(arm_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        obj_mask_512 = cv2.resize(obj_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        bg_mask_512 = arm_mask_512 & ~obj_mask_512

        inpainted = cv2.inpaint(armvis_512, arm_mask_512.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

        t0 = time.time()
        result = generator.sample_arm_free_image(image=armvis_512, arm_pixel_mask=arm_mask_512, instruction=entry["instruction"])
        gen_time = time.time() - t0
        mmada_out = result.image
        if mmada_out.shape[:2] != (512, 512):
            mmada_out = cv2.resize(mmada_out, (512, 512), interpolation=cv2.INTER_AREA)

        collage_inpaint_bg = armvis_512.copy()
        collage_inpaint_bg[bg_mask_512] = inpainted[bg_mask_512]
        collage_inpaint_bg[obj_mask_512] = mmada_out[obj_mask_512]

        collage_oracle_bg = armvis_512.copy()
        collage_oracle_bg[bg_mask_512] = armfree_512[bg_mask_512]
        collage_oracle_bg[obj_mask_512] = mmada_out[obj_mask_512]

        row = {
            "stub": stub, "label": entry["label"], "gen_time_s": gen_time,
            "obj_px_512": int(obj_mask_512.sum()), "arm_px_512": int(arm_mask_512.sum()),
            # full-arm-mask scores
            "full_inpaint_mse": mse_inside(inpainted, armfree_512, arm_mask_512),
            "full_mmada_mse": mse_inside(mmada_out, armfree_512, arm_mask_512),
            "full_collage_inpaintbg_mse": mse_inside(collage_inpaint_bg, armfree_512, arm_mask_512),
            "full_collage_oraclebg_mse": mse_inside(collage_oracle_bg, armfree_512, arm_mask_512),
            "full_inpaint_ssim": windowed_ssim(inpainted, armfree_512, arm_mask_512),
            "full_mmada_ssim": windowed_ssim(mmada_out, armfree_512, arm_mask_512),
            # object-sub-region-only scores (the actually-occluded object pixels)
            "obj_inpaint_mse": mse_inside(inpainted, armfree_512, obj_mask_512),
            "obj_mmada_mse": mse_inside(mmada_out, armfree_512, obj_mask_512),
        }
        results.append(row)
        print(
            f"{stub:<30} obj_px={row['obj_px_512']:4d}/{row['arm_px_512']:5d}  "
            f"full_mse inpaint/mmada={row['full_inpaint_mse']:.0f}/{row['full_mmada_mse']:.0f}  "
            f"obj_mse inpaint/mmada={row['obj_inpaint_mse']}/{row['obj_mmada_mse']}",
            flush=True,
        )
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

        overlay = armvis_512.copy()
        overlay[bg_mask_512] = [255, 0, 255]
        overlay[obj_mask_512] = [0, 255, 255]
        sheet = Image.new("RGB", (512 * 6, 512))
        for j, img in enumerate([armvis_512, overlay, mmada_out, collage_inpaint_bg, collage_oracle_bg, armfree_512]):
            sheet.paste(Image.fromarray(img.astype(np.uint8)), (j * 512, 0))
        sheet.save(OUT_DIR / f"{stub}.png")

    def agg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return (float(np.mean(vals)), float(np.median(vals))) if vals else (None, None)

    print(f"\n{len(results)} frames, aggregate:")
    for key in [
        "full_inpaint_mse", "full_mmada_mse", "full_collage_inpaintbg_mse", "full_collage_oraclebg_mse",
        "full_inpaint_ssim", "full_mmada_ssim", "obj_inpaint_mse", "obj_mmada_mse",
    ]:
        mean, median = agg(key)
        print(f"{key:<28} mean={mean}  median={median}")

    print(f"\ncontact sheets: {OUT_DIR}/")
    print(f"full results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
