"""Quick, cheap feasibility check for the user's proposal: generate a
"pre-occlusion state" reference image directly from the task's
instruction text (bypassing the need for a real historical clear frame,
which the prerequisite check found doesn't reliably exist for LIBERO-Occ
tasks -- the static occluder blocks the target from near frame 1 in
every task checked so far).

IMPORTANT clarification for the record: V-JEPA2 (encoder + AC predictor)
has NO text-conditioning capability whatsoever -- confirmed from the
checkpoint's own real API (`vjepa2_ac_vit_giant()` takes no text input,
only image/video + actions + states). Generating an image FROM TEXT
requires a genuinely different model family (text-to-image diffusion),
not V-JEPA2. This script uses Stable Diffusion (sd-turbo, single-step,
fast) for the generation step; V-JEPA2 would only come in AFTER this,
to encode whatever image results.

Known related caution (informational only, NOT proof for this project --
per this project's own file-scoping convention, a different codebase's
findings are not automatically evidence here): a related investigation
tried exactly this category of thing (full-scene text-to-image
generation for a simulated robot-manipulation scene, no image
conditioning) with two different models (a MaskGIT-style model and
Stable Diffusion itself) and found neither produced recognizable scene
content -- abstract color fields / unrelated furniture scenes. This is
why this check is being done small and cheap FIRST (one real generation,
real visual inspection) before any further investment, not skipped.
"""
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "thirdparty" / "openvla-oft" / "text_to_goal_check"
OUT_DIR.mkdir(exist_ok=True)

# Real task instructions from libero_10_occluded (already used elsewhere
# this session), not invented.
TASKS = {
    "task1": "put the black bowl in the bottom drawer of the cabinet and close it",
    "task9": "pick up the book and place it in the back compartment of the caddy",
}

STYLE_SUFFIX = (
    ", flat-shaded 3D render, simple matte materials, MuJoCo robotics "
    "simulator style, wooden kitchen table, overhead-angled camera, "
    "robot arm visible, simple synthetic lighting, no photorealism"
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoPipelineForText2Image

    print("[load] downloading/loading stabilityai/sd-turbo (single-step, fast)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe = pipe.to(device)
    print("[load] done")

    for name, instruction in TASKS.items():
        prompt = instruction + STYLE_SUFFIX
        print(f"[gen] {name}: {prompt}")
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        out_path = OUT_DIR / f"{name}_text_goal.png"
        image.save(out_path)
        print(f"  saved {out_path}")

    print("\n=== DONE. Inspect text_to_goal_check/*.png directly before "
          "trusting this as usable goal-image content. ===")


if __name__ == "__main__":
    main()
