"""
Month 2 (2026-09-03): confirm the REAL model wiring reproduces byte-identical
output at adapter init -- extends test_zero_init_adapter_smoke.py's isolated-
module check to the full real checkpoint + real predict_action() call path.

Loads the real openvla-7b-oft-libero10-vjepa checkpoint, attaches a fresh
ObjectCentricZeroInitAdapter, and compares predict_action()'s output with vs.
without the adapter+object_mask on the SAME real observation. Since the
adapter is zero-init, these must be identical -- any difference indicates a
wiring bug (e.g. object_mask not reaching _process_vision_features, or the
adapter's forward path having some non-zero side effect at init).
"""
import os
import sys

import numpy as np
import torch

OCC_VLA_ROOT = os.path.expanduser("~/slocal1/Hoki/occ_vla")
sys.path.insert(0, os.path.join(OCC_VLA_ROOT, "thirdparty", "openvla-oft"))
os.chdir(os.path.join(OCC_VLA_ROOT, "thirdparty", "openvla-oft"))

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import get_vla, get_processor, get_action_head, get_proprio_projector, get_vla_action  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import ObjectCentricZeroInitAdapter  # noqa: E402


def main():
    cfg = GenerateConfig(
        pretrained_checkpoint=os.path.join(OCC_VLA_ROOT, "checkpoints", "openvla-7b-oft-libero10-vjepa"),
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    print("loading model...")
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    cfg.unnorm_key = next(iter(vla.norm_stats.keys())) if len(vla.norm_stats) == 1 else "libero_10_no_noops"

    # Real-shaped dummy observation (a real image isn't needed for this
    # wiring check -- what matters is that the SAME inputs produce the SAME
    # output regardless of whether object_mask/adapter are engaged).
    rng = np.random.default_rng(0)
    fake_image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    obs = {
        "full_image": fake_image,
        "wrist_image": fake_image.copy(),
        "state": rng.random(8).astype(np.float32),
    }
    task_label = "pick up the mug"

    print("call 1: no adapter attached (baseline)")
    action1 = get_vla_action(
        cfg, vla, processor, obs, task_label,
        action_head=action_head, proprio_projector=proprio_projector, use_film=False,
    )

    print("attaching fresh zero-init adapter + building a real-shaped object_mask")
    vla.object_centric_adapter = ObjectCentricZeroInitAdapter(vla.llm_dim).to(vla.device, dtype=torch.bfloat16)
    n_patches_total = 256 * cfg.num_images_in_input
    object_mask = torch.rand(1, n_patches_total, 1, device=vla.device, dtype=torch.bfloat16)

    print("call 2: adapter attached, real object_mask passed (should be IDENTICAL since zero-init)")
    action2 = get_vla_action(
        cfg, vla, processor, obs, task_label,
        action_head=action_head, proprio_projector=proprio_projector, use_film=False,
        object_mask=object_mask,
    )

    a1 = np.array(action1)
    a2 = np.array(action2)
    max_abs_diff = np.abs(a1 - a2).max()
    identical = np.array_equal(a1, a2)
    print(f"action1[0] = {a1[0]}")
    print(f"action2[0] = {a2[0]}")
    print(f"max |action1 - action2| = {max_abs_diff:.2e}")
    print(f"exactly equal: {identical}")

    assert identical, "FAIL: real end-to-end wiring is NOT identity-preserving at adapter init"
    print("PASS: real model wiring confirmed identity-preserving at adapter init.")


if __name__ == "__main__":
    main()
