"""A second, targeted demo: since scripts/run_demo.py's pi0.5 (running
on franka norm_stats, not a LIBERO-trained policy — see README/session
notes on why no LIBERO-finetuned pi0.5 checkpoint exists publicly)
doesn't reliably move the arm over the target, natural self-occlusion
never triggered in that rollout. This script forces the SELF-occlusion
path on real LIBERO frames (not synthetic noise) so the actual
mechanism — arm-mask detection -> MMaDA sample_arm_free_image() ->
PlausibilityChecker -> right_wrist_0_rgb injection — is visible on
camera, and writes a side-by-side comparison video.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402
from occ_vla.integration.uncertainty import PlausibilityChecker  # noqa: E402

MMADA_RPC_DIR = "/tmp/occ_vla_rpc/mmada"
INSTRUCTION = "pick up the black bowl and place it on the plate"


def call_mmada(image, arm_pixel_mask, instruction, horizon=5):
    resp_arrays, _ = rpc.call(
        MMADA_RPC_DIR,
        {"image": image, "arm_pixel_mask": arm_pixel_mask.astype(np.uint8)},
        {"instruction": instruction, "horizon": horizon},
        timeout_s=180,
    )
    return resp_arrays["image"]


def label(img, text):
    img = img.copy()
    # wrap onto two lines so it fits within one panel's width instead of
    # overflowing into the next panel when frames are concatenated
    words = text.split(" ")
    mid = len(words) // 2 + (len(words) % 2)
    lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    for i, line in enumerate(lines):
        y = 20 + i * 20
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    config = LiberoOccEnvConfig(benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.MEDIUM)
    occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
    obs = occ_env.reset()
    print(f"occluder placed, S_occ={occ_env.last_s_occ:.3f}", flush=True)

    raw_frame = obs[AGENTVIEW_KEY][::-1]

    # Force a self-occlusion mask over the scene's actual target region,
    # sized to exceed ARM_OCC_THRESHOLD, rather than relying on an
    # untrained policy's arm motion to create one naturally.
    h, w = raw_frame.shape[:2]
    arm_mask = np.zeros((h, w), dtype=bool)
    arm_mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = True

    masked_preview = raw_frame.copy()
    masked_preview[arm_mask] = (40, 40, 40)

    print("calling MMaDA sample_arm_free_image on a real LIBERO frame...", flush=True)
    t0 = time.time()
    subgoal_image = call_mmada(raw_frame, arm_mask, INSTRUCTION)
    print(f"generated in {time.time() - t0:.1f}s, shape={subgoal_image.shape}", flush=True)

    checker = PlausibilityChecker()
    score = checker.score(
        cv2.resize(subgoal_image, (w, h)), {"original_image": raw_frame, "arm_pixel_mask": arm_mask}
    )
    print(f"plausibility score={score:.3f} (< 0.5 -> PKLP fallback would trigger)", flush=True)

    raw_labeled = label(raw_frame, "1. raw observation")
    masked_labeled = label(masked_preview, f"2. arm_s_occ > 0.30 -> masked")
    subgoal_resized = cv2.resize(subgoal_image, (w, h))
    subgoal_labeled = label(subgoal_resized, f"3. MMaDA arm-free subgoal (plausibility={score:.2f})")

    panel = np.concatenate([raw_labeled, masked_labeled, subgoal_labeled], axis=1)
    out_png = str(_ROOT / "demo_mechanism.png")
    cv2.imwrite(out_png, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_png}", flush=True)

    # Also a short video: hold on each panel for a couple seconds.
    out_mp4 = str(_ROOT / "demo_mechanism.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))
    for frame, hold_frames in [(raw_labeled, 6), (masked_labeled, 6), (subgoal_labeled, 10)]:
        for _ in range(hold_frames):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
