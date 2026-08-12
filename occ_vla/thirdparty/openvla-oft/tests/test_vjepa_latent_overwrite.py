"""
test_vjepa_latent_overwrite.py

Verifies the VJEPA_LatentDynamicsPredictor module and its integration into
OpenVLA-OFT's predict_action() path.

Two tiers, per occ_vla's own review of the first draft of this module:
  1. Unit tests (module in isolation, no checkpoint needed): shape, latency,
     and -- the fixed correctness property -- that the residual formulation
     makes `f_final` an EXACT no-op vs. `f_vla_original` at zero-init, for
     every token, occluded or not (the original `(1-mask)*orig + mask*pred`
     replace formula did NOT have this property: it zero-filled occluded
     tokens even before training).
  2. A real end-to-end regression check against the actual 7B checkpoint and
     the real LIBERO eval loop (moka_pots / libero_10 task 8) -- shape/latency
     tests alone can't catch "the wiring is correct end-to-end" or "this
     doesn't change real deployment behavior yet", which is exactly the kind
     of gap that made occ_vla's other occlusion-recovery mechanisms this
     session look fine in isolation but misbehave once actually run.

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    thirdparty/openvla-oft/tests/test_vjepa_latent_overwrite.py
"""

import os
import sys
import time

OFT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from prismatic.extern.hf.vjepa_latent_predictor import VJEPA_LatentDynamicsPredictor  # noqa: E402

FEATURE_DIM = 2176  # DINOv2 (1024) + SigLIP (1152) fused, OpenVLA-OFT's actual dim
NUM_IMAGES = 2  # agentview + wrist, OFT's default LIBERO config
NUM_PATCHES_PER_IMAGE = 256  # 224/14 = 16 -> 16*16, patch14 (NOT 196/patch16)
N = NUM_PATCHES_PER_IMAGE * NUM_IMAGES  # 512
PROPRIO_DIM = 8


def test_shape():
    predictor = VJEPA_LatentDynamicsPredictor(feature_dim=FEATURE_DIM, proprio_dim=PROPRIO_DIM)
    B = 1
    f_current = torch.randn(B, N, FEATURE_DIM)
    past_latents = torch.randn(B, N, FEATURE_DIM)
    proprio = torch.randn(B, PROPRIO_DIM)

    residual = predictor(f_current, past_latents, proprio)
    assert residual.shape == (B, N, FEATURE_DIM), f"expected {(B, N, FEATURE_DIM)}, got {residual.shape}"
    print(f"[PASS] test_shape: residual shape = {tuple(residual.shape)}")


def test_latency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = VJEPA_LatentDynamicsPredictor(feature_dim=FEATURE_DIM, proprio_dim=PROPRIO_DIM).to(device).eval()
    B = 1
    f_current = torch.randn(B, N, FEATURE_DIM, device=device)
    past_latents = torch.randn(B, N, FEATURE_DIM, device=device)
    proprio = torch.randn(B, PROPRIO_DIM, device=device)

    with torch.no_grad():
        for _ in range(3):  # warmup
            predictor(f_current, past_latents, proprio)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            predictor(f_current, past_latents, proprio)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) / 20 * 1000

    print(f"[PASS] test_latency: {elapsed_ms:.3f} ms/call on {device} (target: <10ms)")
    assert elapsed_ms < 10.0, f"latency {elapsed_ms:.3f}ms exceeds 10ms budget"


def test_zero_init_is_exact_noop():
    """The property the ORIGINAL (round-1/round-2) `(1-mask)*orig + mask*pred`
    replace formula did NOT have: at zero-init, f_final must equal
    f_vla_original EXACTLY, at every token position, including occluded ones.
    """
    predictor = VJEPA_LatentDynamicsPredictor(feature_dim=FEATURE_DIM, proprio_dim=PROPRIO_DIM)
    B = 1
    f_vla_original = torch.randn(B, N, FEATURE_DIM)
    past_latents = torch.randn(B, N, FEATURE_DIM)  # arbitrary -- must not matter at init
    proprio = torch.randn(B, PROPRIO_DIM)  # arbitrary -- must not matter at init

    # Occlusion mask covering roughly half the tokens (not all-zero -- a
    # trivial all-zero mask would make this test vacuous).
    occlusion_mask = torch.zeros(B, N, 1)
    occlusion_mask[:, : N // 2, :] = 1.0
    assert occlusion_mask.sum() > 0, "test setup bug: mask must be non-trivial"

    with torch.no_grad():
        residual = predictor(f_vla_original, past_latents, proprio)
        assert torch.all(residual == 0), "residual must be exactly zero at init (out_proj is zero-initialized)"

        f_final = f_vla_original + occlusion_mask * residual

    assert torch.equal(f_final, f_vla_original), (
        "f_final must EXACTLY equal f_vla_original at init, at every token "
        "(occluded or not) -- this is what the residual formulation "
        "guarantees and the original replace-formula (`(1-mask)*orig + "
        "mask*pred`) did not: that one hard-zeroed occluded tokens even "
        "before any training."
    )
    print("[PASS] test_zero_init_is_exact_noop: f_final == f_vla_original exactly, all", N, "tokens")


def test_end_to_end_real_model_regression():
    """Loads the real 7B checkpoint (local editable copy, so the patched
    modeling_prismatic.py / vjepa_latent_predictor.py are actually what's
    running -- not the volatile HF cache blob) and confirms that engaging
    occlusion_mask (mask has 1s, so the module is NOT bypassed) produces
    BIT-IDENTICAL predict_action() output vs. not passing occlusion_mask at
    all, on the same real input. This is the real "did we break anything"
    check -- shape/latency/toy-tensor tests can't catch a wiring bug in
    predict_action's new code path, a device/dtype mismatch, or a mask-shape
    mismatch against the real 512-token layout.
    """
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        normalize_proprio,
        prepare_images_for_vla,
        resize_image_for_policy,
    )
    from experiments.robot.robot_utils import get_model

    local_ckpt = os.path.expanduser(
        "~/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
    )
    assert os.path.isdir(local_ckpt), (
        f"Local editable checkpoint not found at {local_ckpt} -- this test needs a local-directory "
        "checkpoint (not a bare HF Hub id) so check_model_logic_mismatch() syncs our patched "
        "modeling_prismatic.py / vjepa_latent_predictor.py into it before loading."
    )

    cfg = GenerateConfig(
        pretrained_checkpoint=local_ckpt,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=NUM_IMAGES,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=8,
        task_suite_name="libero_10",
    )

    model = get_model(cfg)
    assert hasattr(model, "vjepa_predictor"), "patched checkpoint did not load vjepa_predictor -- sync failed"
    assert hasattr(model, "reset_vjepa_state"), "patched checkpoint missing reset_vjepa_state()"

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)

    # Real observation from a real LIBERO env (not synthetic noise -- occ_vla's
    # own history repeatedly found that noise-input sanity checks can be
    # misleadingly inconclusive vs. real, in-distribution frames).
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(8)  # moka_pots
    initial_states = task_suite.get_task_init_states(8)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    env.reset()
    obs = env.set_init_state(initial_states[0])
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle

    img_resized = resize_image_for_policy(get_libero_image(obs), 224)
    wrist_resized = resize_image_for_policy(get_libero_wrist_image(obs), 224)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    all_images = [observation["full_image"], observation["wrist_image"]]
    from experiments.robot.openvla_utils import prepare_images_for_vla
    all_images = prepare_images_for_vla(all_images, cfg)
    primary_image = all_images.pop(0)
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, primary_image).to("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, all_images[0]).to(inputs["pixel_values"].device, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

    proprio = normalize_proprio(observation["state"], model.norm_stats[cfg.unnorm_key]["proprio"])

    NUM_PATCHES = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
    assert NUM_PATCHES == N, f"expected {N} vision tokens (256 * {NUM_IMAGES} images), got {NUM_PATCHES}"

    # --- Call A: no occlusion_mask at all (pre-existing behavior, byte-for-byte) ---
    model.reset_vjepa_state()
    with torch.inference_mode():
        action_a, _ = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=cfg.use_film,
        )

    # --- Call B: occlusion_mask covering the entire agentview block (first
    # 256 tokens), module ENGAGED (not bypassed) -- but zero-init, so the
    # residual must still be exactly 0 and the output must match Call A.
    model.reset_vjepa_state()
    occlusion_mask = torch.zeros(1, N, 1)
    occlusion_mask[:, :NUM_PATCHES_PER_IMAGE, :] = 1.0  # agentview block
    with torch.inference_mode():
        action_b, _ = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=cfg.use_film,
            occlusion_mask=occlusion_mask,
        )

    assert np.allclose(action_a, action_b, atol=0.0), (
        "occlusion_mask engaged (module NOT bypassed) at zero-init produced a "
        f"DIFFERENT action than no occlusion_mask at all.\n  no-mask: {action_a}\n  "
        f"masked:  {action_b}\nThis means the residual formulation / zero-init "
        "isn't actually a no-op end-to-end -- a real regression, not just a toy-tensor issue."
    )
    print("[PASS] test_end_to_end_real_model_regression: action_a == action_b exactly (zero-init is a true no-op)")

    # Second real check: with occlusion_mask engaged on the SECOND call of an
    # episode (so `past_latents` is real, not None -- exercises the non-cold-start
    # branch), still a no-op at zero-init.
    model.reset_vjepa_state()
    with torch.inference_mode():
        model.predict_action(  # call 1: seeds _vjepa_past_latents
            **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio,
            proprio_projector=proprio_projector, action_head=action_head, use_film=cfg.use_film,
            occlusion_mask=occlusion_mask,
        )
        assert model._vjepa_past_latents is not None, "past_latents should be seeded after call 1"
        action_c, _ = model.predict_action(  # call 2: past_latents now real
            **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio,
            proprio_projector=proprio_projector, action_head=action_head, use_film=cfg.use_film,
            occlusion_mask=occlusion_mask,
        )
    assert np.allclose(action_a, action_c, atol=0.0), (
        "with real (non-None) past_latents on the 2nd call, zero-init should still be an exact no-op"
    )
    print("[PASS] test_end_to_end_real_model_regression: still a no-op with real (non-cold-start) past_latents")


if __name__ == "__main__":
    test_shape()
    test_latency()
    test_zero_init_is_exact_noop()
    test_end_to_end_real_model_regression()
    print("\nALL TESTS PASSED")
