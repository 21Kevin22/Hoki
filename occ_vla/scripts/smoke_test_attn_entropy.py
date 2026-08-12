"""
smoke_test_attn_entropy.py

Minimal smoke test for the new output_attentions=True / attn_entropy plumbing
(modeling_prismatic.py::_compute_action_patch_attn_entropy,
openvla_utils.py::get_vla_action(return_attn_entropy=True)). Loads the real
checkpoint, resets one real task's env, and calls get_vla_action TWICE on the
identical first observation: once with return_attn_entropy=False (baseline,
must be byte-identical to before this change), once with True (checks (a) no
crash -- the real risk being whatever attention implementation this model's
Llama backbone actually resolves to at runtime may not support
output_attentions=True, e.g. if it silently uses SDPA/flash and errors or
returns None attentions instead of falling back to eager), (b) the entropy
value is finite and in [0, 1] as designed, (c) the action output is unchanged
between the two calls (same input, same seed -- output_attentions must be a
pure side-channel, not change what's returned as the action itself).
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
    get_processor, get_vla_action, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_model, get_image_resize_size, set_seed_everywhere,
)
import torch  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
VJEPA_CKPT = "/home/ubuntu/slocal/occ_vla/thirdparty/openvla-oft/vjepa_predictor_multitask_3task_6000steps.pt"

cfg = GenerateConfig(
    pretrained_checkpoint=CHECKPOINT,
    use_l1_regression=True, use_diffusion=False, use_film=False,
    num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
    center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10",
    seed=7,
)
set_seed_everywhere(cfg.seed)
model = get_model(cfg)
processor = get_processor(cfg)
check_unnorm_key(cfg, model)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
action_head = get_action_head(cfg, model.llm_dim)
resize_size = get_image_resize_size(cfg)

ckpt = torch.load(VJEPA_CKPT, map_location=model.device)
model.vision_backbone.vjepa_predictor_dino.load_state_dict(ckpt["dino"])
model.vision_backbone.vjepa_predictor_siglip.load_state_dict(ckpt["siglip"])

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task = task_suite.get_task(5)
env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)
init_states = task_suite.get_task_init_states(5)
env.reset()
obs = env.set_init_state(init_states[0])
for _ in range(cfg.num_steps_wait):
    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

observation = {
    "full_image": get_libero_image(obs).copy(),
    "wrist_image": get_libero_wrist_image(obs).copy(),
    "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
}

print("\n=== call 1: return_attn_entropy=False (baseline) ===")
actions_baseline, _ = get_vla_action(
    cfg, model, processor, observation, task_description,
    action_head=action_head, proprio_projector=proprio_projector,
    noisy_action_projector=None, use_film=cfg.use_film,
    occlusion_mask=None, return_hidden_states=True,
)
print("baseline action[0]:", actions_baseline[0])

print("\n=== call 2: return_attn_entropy=True ===")
try:
    actions_with_entropy, attn_entropy = get_vla_action(
        cfg, model, processor, observation, task_description,
        action_head=action_head, proprio_projector=proprio_projector,
        noisy_action_projector=None, use_film=cfg.use_film,
        occlusion_mask=None, return_attn_entropy=True,
    )
    print("SUCCESS -- no crash")
    print("attn_entropy:", attn_entropy, "(expect finite, in [0,1])")
    print("action[0] with entropy call:", actions_with_entropy[0])
    same = np.allclose(actions_baseline[0], actions_with_entropy[0])
    print("action output identical to baseline call:", same)
    assert attn_entropy is not None, "attn_entropy is None -- attentions likely not returned by language_model()"
    assert 0.0 <= attn_entropy <= 1.0 + 1e-6, f"entropy out of [0,1] range: {attn_entropy}"
    print("\nALL CHECKS PASSED")
except Exception as e:
    print("FAILED:", type(e).__name__, str(e))
    raise
