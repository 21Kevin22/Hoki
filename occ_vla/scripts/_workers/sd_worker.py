"""Runs in .venv_mmada. Serves Stable Diffusion inpainting requests over
the same file-based RPC used by pi05_worker/mmada_worker, so the LIBERO
rollout loop (base python env -- robosuite/mujoco, incompatible with
.venv_mmada's diffusers stack, per this project's own documented
three-stack constraint) can call it without merging environments.

Uses the flat-shading/LIBERO-style prompt steering validated in
test_sd_style_adapted.py (2026-07-23): raw photorealistic-leaning
prompts produced ornate, out-of-domain content; style-steered + a
post-hoc bilateral smoothing pass moved pi0.5's action measurably closer
to ground truth on a single frame. This worker always generates the
style-adapted version -- there is no "raw" mode here, since raw was
already shown worse than doing nothing.

Run:
  source .venv_mmada/bin/activate
  CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/_workers/sd_worker.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpc  # noqa: E402

RPC_DIR = os.environ.get("SD_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "sd_worker"))
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"
# 2026-07-23: swapped default PNDM scheduler for DPMSolver++ (karras
# sigmas) so fewer steps are needed for comparable quality -- cuts
# per-call latency, which matters since every call blocks the rollout
# loop synchronously. Not applied mid-run to any already-started
# experiment (would mix generation quality within one condition); only
# takes effect the next time this worker process is launched.
NUM_INFERENCE_STEPS = 15

FLAT_SHADING_PROMPT = (
    "simple flat-shaded 3D render, clean synthetic simulation graphics, "
    "matte plastic surface, low detail, MuJoCo robotics simulator style"
)
NEGATIVE_PROMPT = (
    "photorealistic, ornate, intricate detail, metallic texture, "
    "steampunk, engraved, realistic photo, high detail, rust, patina"
)


def main():
    from diffusers import StableDiffusionInpaintPipeline  # noqa: PLC0415

    from diffusers import DPMSolverMultistepScheduler  # noqa: PLC0415

    pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    pipe = pipe.to("cuda:0")
    print(f"[sd worker] loaded (DPMSolver++, {NUM_INFERENCE_STEPS} steps), serving {RPC_DIR}", flush=True)

    def handler(arrays, fields):
        image = arrays["image"]  # HWC uint8, any size
        mask = arrays["mask"]  # HW uint8 (0/255) or bool, same size as image
        instruction = fields["instruction"]
        out_size = image.shape[0]

        image_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        mask_512 = cv2.resize((mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        result = pipe(
            prompt=f"{instruction}, {FLAT_SHADING_PROMPT}",
            negative_prompt=NEGATIVE_PROMPT,
            image=image_512,
            mask_image=mask_512,
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]
        out = np.array(result.resize((512, 512)))

        # post-hoc domain smoothing, masked region only (test_sd_style_adapted.py)
        smoothed = cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)
        mask_bool = mask_512 > 127
        out[mask_bool] = smoothed[mask_bool]

        out_resized = cv2.resize(out, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return {"image": out_resized}, {}

    rpc.serve(RPC_DIR, handler)


if __name__ == "__main__":
    main()
