"""Minimal feasibility check for the REAL Meta V-JEPA2-AC model
(facebookresearch/vjepa2, MIT license) -- NOT this project's own
VJEPA_LatentDynamicsPredictor, which is unrelated.

Scope, deliberately narrow (per explicit user choice): does the real
checkpoint download, load, and run one forward pass (encoder -> AC
predictor) on a real LIBERO agentview frame, on this hardware? No
training, no CEM planning loop, no wiring into any existing pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv_openvla_oft/bin/python3 -u \
        scripts/test_real_vjepa2_load.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
VJEPA2_DIR = REPO_ROOT / "thirdparty" / "vjepa2"
sys.path.insert(0, str(VJEPA2_DIR))

FRAME_PATH = (
    REPO_ROOT
    / "thirdparty"
    / "openvla-oft"
    / "distillation_pairs_task1_smoke"
    / "task1_ep0_t00050_agentview.png"
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/5] device={device}, torch={torch.__version__}")

    t0 = time.time()
    from src.hub.backbones import vjepa2_ac_vit_giant

    print("[2/5] instantiating architecture + downloading/loading checkpoint "
          "(11.7GB, first run will take a while)...")
    encoder, predictor = vjepa2_ac_vit_giant(pretrained=True)
    print(f"      done in {time.time() - t0:.1f}s")

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"[3/5] encoder params={n_enc/1e6:.1f}M, predictor params={n_pred/1e6:.1f}M "
          f"(expect ~1000M / ~300M)")

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    # Real LIBERO agentview frame, duplicated into a 2-frame clip so the
    # 3D (video) patch embed (tubelet_size=2) sees exactly one full tubelet
    # -- avoids relying on undocumented single-image (ndim==4) branch
    # behavior for a model instantiated with num_frames=64 (is_video=True).
    assert FRAME_PATH.exists(), f"missing reference frame: {FRAME_PATH}"
    img = Image.open(FRAME_PATH).convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
    arr = (arr - 0.5) / 0.5  # roughly match ImageNet-style [-1,1] normalization
    frame = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    clip = frame.unsqueeze(0).unsqueeze(2).repeat(1, 1, 2, 1, 1)  # B,C,T=2,H,W
    clip = clip.to(device)
    print(f"[4/5] input clip shape={tuple(clip.shape)}, dtype={clip.dtype}, "
          f"real frame={FRAME_PATH.name}")

    with torch.no_grad():
        t1 = time.time()
        z = encoder(clip)
        print(f"      encoder forward OK in {time.time()-t1:.2f}s, "
              f"output shape={tuple(z.shape)}, "
              f"finite={torch.isfinite(z).all().item()}")

        # AC predictor expects (x=context tokens, actions, states),
        # both actions/states last-dim=7 per ac_predictor.py's
        # action_embed_dim=7 default -- matches LIBERO's own 7-dim
        # OSC_POSE action convention (dx,dy,dz,drx,dry,drz,gripper)
        # in DIMENSIONALITY only; units/sign convention NOT verified here.
        B, N, D = z.shape
        dummy_action = torch.zeros(B, 1, 7, device=device)
        dummy_state = torch.zeros(B, 1, 7, device=device)
        t2 = time.time()
        z_pred = predictor(z, dummy_action, dummy_state)
        print(f"[5/5] AC predictor forward OK in {time.time()-t2:.2f}s, "
              f"output shape={tuple(z_pred.shape)}, "
              f"finite={torch.isfinite(z_pred).all().item()}")

    print("\n=== RESULT: real V-JEPA2-AC loads and runs a full "
          "encoder->predictor forward pass on this hardware. ===")


if __name__ == "__main__":
    main()
