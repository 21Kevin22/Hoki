"""occ_vla addition (2026-08-22), per user's "Representation Alignment"
proposal (Option 1 of their 3 elegant-fine-tuning ideas, message
beginning "現在のVLA...最前線において"): trains OpenVLA-OFT's vision
backbone + projector ONLY (language_model fully frozen) so that
Encoder(I_occluded) matches Encoder(I_clean) via a simple MSE loss on
the projected patch embeddings -- no action-head loss, no autoregressive
generation, exactly per the user's own design:

    L = || Encoder(I_clean) - Encoder(I_occ) ||^2

Data: paired (I_clean, I_occ) frames from `collect_clean_occluded_pairs.py`
(same sim state rendered twice via the occluder alpha=0 hide/reveal
technique already used throughout this project -- zero generative
guessing, real geometry).

This is a genuine v0/smoke-scale script: no dataloader/batching
sophistication, meant to verify the plumbing (freeze LLM, only
vision_backbone+projector get gradients, loss decreases on real data)
before any larger investment, matching this project's own established
"verify a few real steps before trusting/scaling" discipline.
"""
import argparse
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import get_vla, get_processor  # noqa: E402


def load_pair(manifest_entry, data_dir, processor, device):
    occ_img = Image.open(os.path.join(data_dir, manifest_entry["occ_path"])).convert("RGB")
    clean_img = Image.open(os.path.join(data_dir, manifest_entry["clean_path"])).convert("RGB")
    prompt = "In: What action should the robot take?\nOut:"
    occ_inputs = processor(prompt, occ_img).to(device, dtype=torch.bfloat16)
    clean_inputs = processor(prompt, clean_img).to(device, dtype=torch.bfloat16)
    return occ_inputs["pixel_values"], clean_inputs["pixel_values"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    ap.add_argument("--data-dir", default="clean_occluded_pairs")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out-adapter", default="representation_alignment_smoke")
    args = ap.parse_args()
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(SCRIPTS_DIR, args.data_dir)
    if not os.path.isabs(args.out_adapter):
        args.out_adapter = os.path.join(SCRIPTS_DIR, args.out_adapter)

    manifest = json.load(open(os.path.join(args.data_dir, "manifest.json")))
    print(f"loaded manifest: {len(manifest)} pairs")

    # occ_vla note: reuse the SAME real GenerateConfig class + field
    # values already established and validated by
    # run_libero_occluded_oracle_headroom.py for this exact checkpoint
    # (num_images_in_input=2, use_proprio=True, etc.) rather than a
    # hand-rolled stub -- a first attempt at a minimal stub class missed
    # several fields (`use_film` etc.) that get_vla() actually needs,
    # caught by a real AttributeError, not by inspection.
    # occ_vla note: num_images_in_input=1 here (unlike the production
    # rollout config's 2, base+wrist) -- this smoke test only has
    # agentview frames (no paired wrist image collected), and
    # representation alignment doesn't need the wrist camera at all
    # for its own purpose (isolating the vision encoder's occluded-vs-
    # clean feature gap). Caught by a real shape-mismatch RuntimeError
    # on the first attempt (num_images_in_input=2 but only 1 image's
    # worth of pixel_values supplied), not by inspection.
    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=1, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    device = vla.device

    # occ_vla design choice (2026-08-22): freeze EVERYTHING first, then
    # explicitly unfreeze only vision_backbone + projector -- the
    # language_model (the 7B LLM) never gets a gradient, matching the
    # user's own "LLM側は一切触らず" requirement exactly, not just
    # approximately (e.g. via a low LR on it).
    for p in vla.parameters():
        p.requires_grad = False
    trainable_params = []
    for name, p in vla.named_parameters():
        if name.startswith("vision_backbone.") or name.startswith("projector."):
            p.requires_grad = True
            trainable_params.append(p)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in vla.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.3f}%) "
          f"[vision_backbone + projector only, language_model fully frozen]")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    losses = []
    for step in range(args.n_steps):
        entry = manifest[step % len(manifest)]
        occ_px, clean_px = load_pair(entry, args.data_dir, processor, device)

        occ_feat = vla._process_vision_features(occ_px, use_film=False)
        with torch.no_grad():
            clean_feat = vla._process_vision_features(clean_px, use_film=False)

        loss = torch.nn.functional.mse_loss(occ_feat.float(), clean_feat.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        print(f"step {step}: loss={loss.item():.6f} (uid={entry['uid']})")

    print(f"\nloss trend: first={losses[0]:.6f} last={losses[-1]:.6f}")
    os.makedirs(args.out_adapter, exist_ok=True)
    torch.save(
        {name: p.detach().cpu() for name, p in vla.named_parameters() if p.requires_grad},
        os.path.join(args.out_adapter, "vision_projector_weights.pt"),
    )
    with open(os.path.join(args.out_adapter, "loss_log.json"), "w") as f:
        json.dump({"losses": losses, "n_trainable_params": n_trainable}, f, indent=2)
    print(f"saved to {args.out_adapter}/")


if __name__ == "__main__":
    main()
