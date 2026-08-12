"""
smoke_test_cross_view_predictor.py

Targeted regression check for the 2026-08-10 cross-view context addition to
VJEPA_LatentDynamicsPredictor / modeling_prismatic.py's two-phase featurizer
split. Not a replacement for tests/test_vjepa_latent_overwrite.py -- that
file predates the DINO/SigLIP-split, per-image-list refactor and has two
stale top-level attribute assertions (`model.vjepa_predictor`,
`model._vjepa_past_latents`) that don't match the real attribute paths
(`model.vision_backbone.vjepa_predictor_dino`/`_siglip`,
`model.vision_backbone._vjepa_past_latents_dino`/`_siglip`) -- confirmed via
direct hasattr() check, unrelated to today's change, not fixed here (out of
scope; flagged for whoever next touches that file).

What this checks, using a real LIBERO frame (not synthetic noise):
  1. Model loads with the edited modeling_prismatic.py / vjepa_latent_predictor.py.
  2. Call A (no occlusion_mask) vs Call B (AGENTVIEW block occluded --
     engages vjepa_predictor_{dino,siglip} on the agentview image, which now
     also receives the WRIST image's own current tokens as cross_view_context)
     -- must be byte-identical, since out_proj is zero-init regardless of
     cross_view_context's presence/shape.
  3. Same check with the WRIST block occluded instead (cross-view context now
     flows the other direction: wrist's predictor receives agentview's tokens).
  4. Same check across two consecutive calls (real, non-None past_latents on
     the second call -- exercises the non-cold-start branch together with
     cross-view context).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
OFT_ROOT = os.path.join(os.path.dirname(__file__), "..", "thirdparty", "openvla-oft")
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_wrist_image, get_libero_image, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, get_libero_dummy_action,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402
import torch  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"

cfg = GenerateConfig(
    pretrained_checkpoint=CHECKPOINT,
    use_l1_regression=True, use_diffusion=False, use_film=False,
    num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
    center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
)
set_seed_everywhere(cfg.seed)
model = get_model(cfg)
assert hasattr(model.vision_backbone, "vjepa_predictor_dino"), "sync failed: vjepa_predictor_dino missing"
assert hasattr(model.vision_backbone, "vjepa_predictor_siglip"), "sync failed: vjepa_predictor_siglip missing"
assert hasattr(model, "reset_vjepa_state"), "reset_vjepa_state missing"

processor = get_processor(cfg)
check_unnorm_key(cfg, model)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
action_head = get_action_head(cfg, model.llm_dim)
resize_size = get_image_resize_size(cfg)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task = task_suite.get_task(8)  # moka_pots
env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)
init_states = task_suite.get_task_init_states(8)
env.reset()
obs = env.set_init_state(init_states[0])
for _ in range(cfg.num_steps_wait):
    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

from experiments.robot.openvla_utils import prepare_images_for_vla, normalize_proprio  # noqa: E402

full_image = get_libero_image(obs).copy()
wrist_image = get_libero_wrist_image(obs).copy()
state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))

all_images = prepare_images_for_vla([full_image, wrist_image], cfg)
primary_image = all_images.pop(0)
prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
inputs = processor(prompt, primary_image).to("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
wrist_inputs = processor(prompt, all_images[0]).to(inputs["pixel_values"].device, dtype=torch.bfloat16)
inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

proprio = normalize_proprio(state, model.norm_stats[cfg.unnorm_key]["proprio"])

num_patches_per_image = model.vision_backbone.get_num_patches()
N = num_patches_per_image * 2


def call(occlusion_mask, seed_first=False):
    model.reset_vjepa_state()
    with torch.inference_mode():
        if seed_first:
            model.predict_action(
                **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio,
                proprio_projector=proprio_projector, action_head=action_head, use_film=cfg.use_film,
                occlusion_mask=occlusion_mask,
            )
        action, _ = model.predict_action(
            **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio,
            proprio_projector=proprio_projector, action_head=action_head, use_film=cfg.use_film,
            occlusion_mask=occlusion_mask,
        )
    return action


action_a = call(None)

mask_agentview = torch.zeros(1, N, 1)
mask_agentview[:, :num_patches_per_image, :] = 1.0
action_b = call(mask_agentview)
assert np.allclose(action_a, action_b, atol=0.0), (
    f"AGENTVIEW-occluded call (cross-view context now populated from wrist) diverged from no-mask call.\n"
    f"  no-mask: {action_a}\n  masked:  {action_b}"
)
print("[PASS] agentview-occluded, cross-view context from wrist -> exact no-op at zero-init")

mask_wrist = torch.zeros(1, N, 1)
mask_wrist[:, num_patches_per_image:, :] = 1.0
action_c = call(mask_wrist)
assert np.allclose(action_a, action_c, atol=0.0), (
    f"WRIST-occluded call (cross-view context now populated from agentview) diverged from no-mask call.\n"
    f"  no-mask: {action_a}\n  masked:  {action_c}"
)
print("[PASS] wrist-occluded, cross-view context from agentview -> exact no-op at zero-init")

action_d = call(mask_agentview, seed_first=True)
assert np.allclose(action_a, action_d, atol=0.0), (
    "second call with real (non-cold-start) past_latents + cross-view context diverged"
)
print("[PASS] non-cold-start past_latents + cross-view context -> still exact no-op")

print("\nALL CROSS-VIEW SMOKE CHECKS PASSED")
