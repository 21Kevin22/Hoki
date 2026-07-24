"""First look at the new full-frame "subgoal" generation mode
(2026-07-21, per user decision to move on from arm-removal inpainting):
generates N independent samples of "the ideal resulting scene, K steps
from now" for the mug_in_microwave instruction, with NO current-frame
content pinned at all (unlike sample_arm_free_image, which only ever
masked ~13% of tokens). Purely a visual/qualitative check before
building any injection or rollout harness around this -- mirrors the
"look before you leap" approach used for the arm-removal Best-of-N
check.

Requires only the mmada-lora RPC worker (full adapter). Run:
  python3 scripts/preview_subgoal_mode.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

import os  # noqa: E402

MMADA_LORA_RPC_DIR = os.environ.get("OCC_VLA_MMADA_LORA_RPC_DIR", str(_ROOT / ".rpc" / "mmada_lora_full"))
INSTRUCTION = "put the yellow and white mug in the microwave and close it"
N_SAMPLES = 5
OUT_DIR = _ROOT / "subgoal_mode_frames"


def call_mmada_subgoal(instruction: str, horizon: int = 5):
    resp_arrays, _ = rpc.call(
        MMADA_LORA_RPC_DIR,
        {},
        {"instruction": instruction, "horizon": horizon, "mode": "subgoal"},
        timeout_s=180,
    )
    return resp_arrays["image"]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    samples = []
    for i in range(N_SAMPLES):
        img = call_mmada_subgoal(INSTRUCTION)
        samples.append(img)
        cv2.imwrite(str(OUT_DIR / f"sample_{i:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[sample {i}] generated, shape={img.shape}", flush=True)

    cell = samples[0].shape[0]
    cols = min(N_SAMPLES, 3)
    rows = (N_SAMPLES + cols - 1) // cols
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for idx, img in enumerate(samples):
        r, c = idx // cols, idx % cols
        tile = cv2.resize(img, (cell, cell))
        cv2.putText(tile, f"s{idx}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, f"s{idx}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        grid[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = tile
    cv2.imwrite(str(OUT_DIR / "grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {OUT_DIR / 'grid.png'}", flush=True)


if __name__ == "__main__":
    main()
