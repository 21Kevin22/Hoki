"""
collect_feature_distribution_reference.py

Per user request (2026-08-18): reference/"training-time" distribution for
the distribution-shift measurement -- does the oracle mid-layer correction
push OpenVLA-OFT's internal representation out of what it normally operates
on? Compares two things against this reference:
  (A) the injected patch_clean tensor itself (already saved by
      run_libero_occluded_oracle_headroom.py's --save-oracle-features-dir,
      keys "dino"/"siglip")
  (B) the FINAL representation after patch_clean has been carried through
      the remaining transformer blocks alongside the rest of the (still
      occluded) sequence (same script, keys "dino_final"/"siglip_final",
      added 2026-08-18 specifically for this measurement)

This script produces the REFERENCE side: real, plain `libero_10` (NOT
`_occluded` -- no occluder object ever placed, closer to the finetuning
distribution than the deliberately-occluded benchmark) baseline rollouts
of the SAME underlying task (matched by bddl_file, same convention as
find_occluder_body_names), with features extracted at every replan step
via the model's own featurizer -- no splicing, no masking, exactly the
"else" branch of patched_forward -- at the SAME split layer and final
layer as the oracle mechanism, so the two sides are directly comparable.

Usage: python collect_feature_distribution_reference.py --task-id 1 --n-episodes 5
"""
import argparse
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import run_libero_occluded_oracle_headroom as oh  # noqa: E402  (reuses its module-level sys.path/chdir setup)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from collections import deque  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402


def _run_vit_plain_capture(featurizer, x_pixels, split_layer, out, key):
    """Same block-iteration structure as oh._run_vit_with_midlayer_splice,
    but with NO clean/corrupted split and NO masking -- a genuinely plain
    forward pass, capturing the intermediate (split_layer) and final
    (extraction_layer) patch tensors for comparison against the oracle
    mechanism's (A)/(B) samples."""
    num_blocks = len(featurizer.blocks)
    extraction_layer = num_blocks - 2
    num_prefix = featurizer.num_prefix_tokens
    x = oh._vit_prep(featurizer, x_pixels)
    for i, blk in enumerate(featurizer.blocks):
        x = blk(x)
        if i == split_layer:
            out[key] = x[:, num_prefix:].detach().to(torch.float32).cpu().numpy()
        if i == extraction_layer:
            out[f"{key}_final"] = x[:, num_prefix:].detach().to(torch.float32).cpu().numpy()
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=1, help="libero_10_occluded task id to match by bddl_file")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    parser.add_argument("--midlayer-split-frac", type=float, default=0.67)
    parser.add_argument("--out-dir", default="feature_distribution_reference")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = oh.GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    oh.set_seed_everywhere(cfg.seed)
    model = oh.get_model(cfg)
    processor = oh.get_processor(cfg)
    oh.check_unnorm_key(cfg, model)
    proprio_projector = oh.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = oh.get_action_head(cfg, model.llm_dim)
    resize_size = oh.get_image_resize_size(cfg)
    max_steps = oh.TASK_MAX_STEPS["libero_10"]

    occluded_suite = oh.benchmark.get_benchmark_dict()[oh.OCCLUDED_SUITE]()
    stock_suite = oh.benchmark.get_benchmark_dict()[oh.STOCK_SUITE]()
    occluded_task = occluded_suite.get_task(args.task_id)
    stock_task = None
    for t in stock_suite.tasks:
        if t.bddl_file == occluded_task.bddl_file:
            stock_task = t
            break
    assert stock_task is not None, f"no stock libero_10 task matches bddl_file={occluded_task.bddl_file!r}"
    print(f"[ref] occluded task{args.task_id} '{occluded_task.language}' -> stock task '{stock_task.language}' "
          f"(bddl_file={stock_task.bddl_file})")

    vision_backbone = model.vision_backbone
    nb_dino = len(vision_backbone.featurizer.blocks)
    nb_siglip = len(vision_backbone.fused_featurizer.blocks)
    sl_dino = int(nb_dino * args.midlayer_split_frac)
    sl_siglip = int(nb_siglip * args.midlayer_split_frac)

    env, _ = get_libero_env(stock_task, cfg.model_family, resolution=resize_size)
    stock_task_id = stock_suite.get_task_names().index(stock_task.name)
    init_states = stock_suite.get_task_init_states(stock_task_id)
    n = min(args.n_episodes, len(init_states))
    prompt = f"In: What action should the robot take to {stock_task.language.lower()}?\nOut:"

    n_samples = 0
    for ep in range(n):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        if hasattr(model, "reset_vjepa_state"):
            model.reset_vjepa_state()
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        t = 0
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(oh.get_libero_dummy_action(cfg.model_family))
            t += 1
        success = False
        while t < max_steps + cfg.num_steps_wait:
            if len(action_queue) == 0:
                agentview_color = oh.get_libero_image(obs).copy()
                wrist_img = oh.get_libero_wrist_image(obs).copy()
                observation = {
                    "full_image": agentview_color,
                    "wrist_image": wrist_img,
                    "state": np.concatenate((obs["robot0_eef_pos"], oh.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                }
                with torch.no_grad():
                    pixel_values = oh.build_pixel_values(agentview_color, wrist_img, processor, prompt, model.device, torch.bfloat16)
                    agentview_pixels, wrist_pixels = torch.split(pixel_values, [6, 6], dim=1)
                    agentview_regular, agentview_fused = torch.split(agentview_pixels, [3, 3], dim=1)
                    out = {}
                    _run_vit_plain_capture(vision_backbone.featurizer, agentview_regular, sl_dino, out, "dino")
                    _run_vit_plain_capture(vision_backbone.fused_featurizer, agentview_fused, sl_siglip, out, "siglip")
                    fname = f"task{args.task_id}_ep{ep}_t{t}_reference.npz"
                    np.savez_compressed(os.path.join(args.out_dir, fname),
                                         dino=out["dino"], siglip=out["siglip"],
                                         dino_final=out["dino_final"], siglip_final=out["siglip_final"], t=t)
                    n_samples += 1

                actions = oh.get_vla_action(
                    cfg, model, processor, observation, stock_task.language,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=None,
                )
                action_queue.extend(actions)
            action = action_queue.popleft()
            action = oh.process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
        print(f"[ref] ep{ep}: success={success} done_step={t} samples_so_far={n_samples}")

    print(f"[ref] DONE: {n_samples} reference feature samples saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
