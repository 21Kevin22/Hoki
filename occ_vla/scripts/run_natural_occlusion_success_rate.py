"""
run_natural_occlusion_success_rate.py

Converts the single-frame cos-sim finding (2026-08-09: gray_pixel Semantic
Blanking on a real natural/scene-induced occlusion mask makes the corrected
representation WORSE, not better, -0.045 delta) into real task SUCCESS-RATE
numbers -- the evidence a paper reviewer actually wants, not just a
feature-space proxy.

Three conditions, same real LIBERO-Occ task (real 3D occluder object, not
apply_partial_patch), same init_states across conditions for a paired
comparison:
  - baseline: no correction at all. The real occluded wrist/agentview images
    go straight to the model, unmodified.
  - oracle: at every real VLA call, the true occlusion mask is derived from
    the ACTUAL scene geometry (body-position-translate GT-render-diff trick,
    see test_natural_occlusion_generalization.py's docstring for the
    force_update=True bug this depends on having found/fixed), the masked
    region is Semantic-Blanked (painted gray, matching training
    distribution) in the wrist image, and occlusion_mask is passed to
    get_vla_action so the vjepa predictor's residual splices in there --
    this is the SAME mechanism run_libero_eval_occlusion.py's oracle mode
    uses for apply_partial_patch, just fed a real, irregular, per-step-
    varying mask instead of a fixed synthetic rectangle.
  - (baseline/oracle only for the first pass -- a "proposed/dynamic" third
    condition needing a real-occlusion-domain-adapted trigger classifier is
    intentionally deferred, see bottom of this docstring)

Expected result if the single-frame finding replicates at the rollout level:
oracle should NOT clearly beat baseline (the geometric-overfitting failure
found in feature space should translate to no success-rate benefit, or
active harm) -- unlike this project's OWN in-domain apply_partial_patch
result (moka_pots 6.0%->42.0% oracle, a real, large, replicated win) or
LIBERO-Occ's own reported VIM improvement, which imply working occlusion
recovery genuinely should help when the underlying mask/correction
mechanism actually generalizes.

Per-step cost note: the GT-render-diff trick (2 extra full offscreen
renders: translate object away, render, restore, but importantly only run
once per real VLA call -- i.e. once per up-to-8 env steps via
num_open_loop_steps, NOT once per env step) adds real wall-clock time to
oracle episodes vs. baseline. Expect oracle episodes to run measurably
slower than baseline, independent of any success/failure difference.

Usage (openvla-oft conda env), ALWAYS pilot with a small --num-episodes
first (this project's own established discipline -- every "promising"
small-n result in the sibling investigation that skipped this step needed
walking back at larger n):
  python scripts/run_natural_occlusion_success_rate.py \
    --checkpoint checkpoints/openvla-7b-oft-libero10-vjepa \
    --vjepa-checkpoint vjepa_predictor_multitask_3task_6000steps.pt \
    --task-suite libero_10_occluded --task-id 7 \
    --occluder-body-substrings wooden_cabinet \
    --conditions baseline oracle --num-episodes 3 \
    --out-prefix natural_occ_success_rate_task7_pilot
"""

import argparse
import json
import os
import sys
from collections import deque

OCC_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
SCRIPTS_DIR = os.path.join(OCC_VLA_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402

import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env, get_libero_wrist_image, get_libero_image, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, get_libero_dummy_action, process_action, TASK_MAX_STEPS,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_vla_action, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_model, get_image_resize_size, set_seed_everywhere,
)
from train_vjepa_predictor_scaled import GRAY_FILL, PARTIAL_PATCH_FRAC  # noqa: E402


def _training_reference_mask_256(image_side=224):
    """The FIXED mask geometry vjepa_predictor_multitask_3task_6000steps.pt was
    actually trained on: apply_partial_patch's centered square,
    PARTIAL_PATCH_FRAC=0.59 side fraction (~35% area), no jitter (--mask_jitter
    was never used for this checkpoint -- see train_vjepa_predictor_multitask.py's
    own comment: 'apply_partial_patch uses a FIXED geometry ... that doesn't
    depend on image content'). Pooled to the same 16x16 patch grid as
    derive_natural_mask_and_gt, same >50%-of-patch-area rule, so the two are
    directly comparable."""
    h = w = image_side
    ph, pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    mask_px = np.zeros((h, w), dtype=bool)
    mask_px[y0:y0 + ph, x0:x0 + pw] = True
    bh, bw = h // 16, w // 16
    mask_256 = np.zeros(256, dtype=bool)
    for r in range(16):
        for c in range(16):
            if mask_px[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw].mean() > 0.5:
                mask_256[r * 16 + c] = True
    return mask_256


def _mask_stats(mask_256):
    """area (patch count), centroid (row, col in 0..15 grid units), solidity
    (mask_area / bounding_box_area, 1.0 = a solid filled rectangle -- the
    training mask's own shape -- lower = scattered/irregular)."""
    grid = mask_256.reshape(16, 16)
    rows, cols = np.nonzero(grid)
    area = len(rows)
    if area == 0:
        return 0, (7.5, 7.5), 0.0
    centroid = (rows.mean(), cols.mean())
    bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    solidity = area / bbox_area
    return area, centroid, solidity


_REF_MASK_256 = _training_reference_mask_256()
_REF_AREA, _REF_CENTROID, _REF_SOLIDITY = _mask_stats(_REF_MASK_256)


def shape_confidence_gate(mask_256, area_ratio_range=(0.5, 1.8), max_centroid_dist=3.0, min_solidity=0.55):
    """Root cause established this session (LIBERO-Occ investigation, sec.
    15.2/15.3c): the correction module was trained on ONLY the fixed, centered,
    ~35%-area square above and does not generalize to real irregular masks --
    on similar-severity real tasks it produced a thin rescue on one (task5)
    and a new, directly-harmful failure on another (task4). This gate does NOT
    try to fix the correction's generalization -- it tries to catch, before
    applying the correction, when the derived mask looks nothing like what the
    module was ever trained to correct, and skip correction (fall back to raw
    occluded pixels, i.e. the 'baseline' behavior for that call) rather than
    risk injecting an unreliable residual. A cheap, training-free heuristic --
    no calibration data, no learned threshold -- deliberately the lightweight
    option (see project discussion) before committing to a full ReconVLA-style
    learned SMD/CQR gate.

    Returns (gate_pass: bool, stats: dict) -- stats always returned for logging/
    auditing every call, not just gated-out ones."""
    area, centroid, solidity = _mask_stats(mask_256)
    area_ratio = area / _REF_AREA if _REF_AREA else 0.0
    centroid_dist = float(np.hypot(centroid[0] - _REF_CENTROID[0], centroid[1] - _REF_CENTROID[1]))
    gate_pass = (
        area_ratio_range[0] <= area_ratio <= area_ratio_range[1]
        and centroid_dist <= max_centroid_dist
        and solidity >= min_solidity
    )
    stats = {"area": int(area), "area_ratio": area_ratio, "centroid_dist": centroid_dist,
              "solidity": float(solidity), "gate_pass": bool(gate_pass)}
    return gate_pass, stats


def find_occluder_body_ids(sim, substrings):
    ids = []
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if name and any(s in name.lower() for s in substrings):
            ids.append(i)
    return ids


def derive_natural_mask_and_gt(env, sim, occluder_body_ids, diff_threshold=25):
    """Real occluded render + GT-clean render (occluder body translated
    +5m, force_update=True render, restored) + derived 16x16 patch mask,
    for the CURRENT sim state. See module docstring for why force_update
    is required."""
    obs = env.env._get_observations(force_update=True)
    wrist_occ = get_libero_wrist_image(obs).copy()
    agent_occ = get_libero_image(obs).copy()

    orig_pos = {bid: sim.model.body_pos[bid].copy() for bid in occluder_body_ids}
    for bid in occluder_body_ids:
        sim.model.body_pos[bid] = orig_pos[bid] + np.array([5.0, 5.0, 5.0])
    sim.forward()
    obs_gt = env.env._get_observations(force_update=True)
    wrist_gt = get_libero_wrist_image(obs_gt).copy()
    for bid in occluder_body_ids:
        sim.model.body_pos[bid] = orig_pos[bid]
    sim.forward()

    diff = np.abs(wrist_occ.astype(int) - wrist_gt.astype(int)).sum(axis=-1)
    mask_px = diff > diff_threshold
    H, W = mask_px.shape
    ph, pw = H // 16, W // 16
    mask_256 = np.zeros(256, dtype=bool)
    for r in range(16):
        for c in range(16):
            if mask_px[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw].mean() > 0.5:
                mask_256[r * 16 + c] = True
    return wrist_occ, agent_occ, mask_px, mask_256


def run_episode_natural(cfg, env, task_description, model, resize_size, processor, action_head,
                         proprio_projector, initial_state, condition, occluder_body_ids, base_suite,
                         record_video=False, max_record_steps=None, max_steps_override=None,
                         attn_entropy_threshold=0.5):
    """record_video: if True, saves an agentview replay MP4 (official
    convention, save_rollout_video) AND a separate MP4 of the actual WRIST
    frame fed to the model (real occluded pixels for baseline; gray
    Semantic-Blanked for oracle when engaged) -- every real env step, via a
    fresh force_update render, so the video shows genuine per-step
    occlusion/blanking, not just the once-per-VLA-call snapshot. Adds real
    cost (an extra render every step) -- only intended for a single
    diagnostic episode, not full success-rate runs.

    max_steps_override: bypass TASK_MAX_STEPS[base_suite] (520 for libero_10)
    -- tests the "just ran out of budget on this 2-object task" hypothesis
    raised from the record_video contact sheet (2026-08-09): both baseline
    and oracle showed real task progress (grasping/moving the yellow mug by
    step ~150-199) rather than paralysis, and every pilot episode hit
    done_step==520 exactly (full timeout, never an early drop/error) --
    consistent with, but not proof of, a budget problem rather than a
    capability problem."""
    env.reset()
    obs = env.set_init_state(initial_state)
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    max_steps = max_steps_override if max_steps_override is not None else TASK_MAX_STEPS[base_suite]
    if record_video and max_record_steps is not None:
        max_steps = min(max_steps, max_record_steps)
    success = False
    sim = env.env.sim
    n_calls = 0
    n_engaged = 0
    n_gated_out = [0]  # mutable single-element list so the inner while-loop body can increment it
    agent_frames, wrist_frames = [], []
    last_wrist_fed = None  # persists across the 8 env steps between VLA calls, for video continuity

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        if len(action_queue) == 0:
            n_calls += 1
            occlusion_mask = None
            attn_entropy_val = None
            actions = None  # set directly (attn_gated_oracle's own get_vla_action calls) or below
            if condition in ("oracle", "gated_oracle", "attn_gated_oracle"):
                wrist_occ, agent_occ, mask_px, mask_256_np = derive_natural_mask_and_gt(env, sim, occluder_body_ids)
                apply_correction = mask_256_np.any()
                if apply_correction and condition == "gated_oracle":
                    gate_pass, gate_stats = shape_confidence_gate(mask_256_np)
                    apply_correction = gate_pass
                    if os.environ.get("GATE_DEBUG"):
                        print(f"    [gate] area={gate_stats['area']} area_ratio={gate_stats['area_ratio']:.2f} "
                              f"centroid_dist={gate_stats['centroid_dist']:.2f} solidity={gate_stats['solidity']:.2f} "
                              f"pass={gate_pass}")
                    if not gate_pass:
                        n_gated_out[0] += 1
                if apply_correction:
                    wrist_img_resized = wrist_occ.copy()
                    wrist_img_resized[mask_px] = GRAY_FILL
                    occlusion_mask = torch.from_numpy(mask_256_np).to(
                        device=model.device, dtype=torch.bfloat16).reshape(1, -1, 1)
                else:
                    wrist_img_resized = wrist_occ
                img_resized = agent_occ

                if apply_correction and condition == "attn_gated_oracle":
                    observation = {
                        "full_image": img_resized, "wrist_image": wrist_img_resized,
                        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                  obs["robot0_gripper_qpos"])),
                    }
                    actions_corrected, attn_entropy_val = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                        occlusion_mask=occlusion_mask, return_attn_entropy=True,
                    )
                    trust_correction = attn_entropy_val is not None and attn_entropy_val <= attn_entropy_threshold
                    if os.environ.get("GATE_DEBUG"):
                        print(f"    [attn_gate] entropy={attn_entropy_val} threshold={attn_entropy_threshold} "
                              f"trust={trust_correction}")
                    if trust_correction:
                        n_engaged += 1
                        actions = actions_corrected
                    else:
                        n_gated_out[0] += 1
                        # fall back to the RAW occluded image, no correction (second real call --
                        # a real per-rejected-step compute cost, see module docstring)
                        obs_now = env.env._get_observations(force_update=True)
                        fallback_observation = {
                            "full_image": get_libero_image(obs_now).copy(),
                            "wrist_image": get_libero_wrist_image(obs_now).copy(),
                            "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                      obs["robot0_gripper_qpos"])),
                        }
                        actions, _ = get_vla_action(
                            cfg, model, processor, fallback_observation, task_description,
                            action_head=action_head, proprio_projector=proprio_projector,
                            noisy_action_projector=None, use_film=cfg.use_film,
                            occlusion_mask=None, return_hidden_states=True,
                        )
                        wrist_img_resized = fallback_observation["wrist_image"]
                elif apply_correction:  # plain oracle / gated_oracle with correction applied
                    n_engaged += 1
            else:  # baseline: real occluded render, no intervention
                obs_now = env.env._get_observations(force_update=True)
                wrist_img_resized = get_libero_wrist_image(obs_now).copy()
                img_resized = get_libero_image(obs_now).copy()

            if actions is None:
                observation = {
                    "full_image": img_resized,
                    "wrist_image": wrist_img_resized,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                actions, _ = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film,
                    occlusion_mask=occlusion_mask, return_hidden_states=True,
                )
            action_queue.extend(actions)
            last_wrist_fed = wrist_img_resized

        if record_video:
            obs_vid = env.env._get_observations(force_update=True)
            agent_frames.append(get_libero_image(obs_vid))
            wrist_frames.append(last_wrist_fed if last_wrist_fed is not None else get_libero_wrist_image(obs_vid))

        action = action_queue.popleft()
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    if record_video:
        from experiments.robot.libero.libero_utils import save_rollout_video
        agent_path = save_rollout_video(agent_frames, f"natocc_{condition}_agentview", success, task_description)
        # separate writer for the wrist video (save_rollout_video's naming
        # convention is reused via a distinct idx string, not a second helper)
        import imageio
        wrist_path = agent_path.replace("agentview", "wrist")
        w = imageio.get_writer(wrist_path, fps=30)
        for img in wrist_frames:
            w.append_data(img)
        w.close()
        print(f"Saved wrist rollout MP4 at path {wrist_path}")

    return success, t - cfg.num_steps_wait, n_calls, n_engaged, n_gated_out[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--occluder-body-substrings", nargs="+", default=[],
                         help="required for oracle/baseline conditions on an '_occluded' suite (used to find "
                              "the real occluder body for GT-mask derivation); not needed for --conditions clean "
                              "on the plain (non-'_occluded') suite, where no such body exists.")
    parser.add_argument("--conditions", nargs="+", default=["baseline", "oracle"],
                         choices=["baseline", "oracle", "clean", "gated_oracle", "attn_gated_oracle"])
    parser.add_argument("--attn-implementation", default=None,
                         help="e.g. 'eager' -- forces a single consistent attention implementation "
                              "for the WHOLE model/rollout, required for attn_gated_oracle so the "
                              "SDPA<->eager fallback triggered by output_attentions=True doesn't "
                              "itself perturb behavior on gated-vs-ungated steps differently "
                              "(2026-08-09 finding). None (default) leaves the transformers-library "
                              "default (SDPA in this environment) untouched -- fully backward-compatible.")
    parser.add_argument("--attn-entropy-threshold", type=float, default=0.5,
                         help="attn_gated_oracle only: reject (fall back to baseline) a correction "
                              "call whose normalized [0,1] action-to-vision-patch attention entropy "
                              "exceeds this. Pass 1.1 to never reject (pure calibration/logging mode, "
                              "combine with GATE_DEBUG=1 to see the real per-call distribution before "
                              "picking a real threshold).")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--start-episode-idx", type=int, default=0,
                         help="first episode_idx to run (episode_idx still indexes init_states via "
                              "%% len(init_states), so this resumes/extends a prior run's episode "
                              "sequence deterministically instead of re-running episodes already done "
                              "-- e.g. --start-episode-idx 10 --num-episodes 40 continues an existing "
                              "n=10 result out to n=50; merge the two output JSONs' per_episode lists "
                              "afterward)")
    parser.add_argument("--out-prefix", default="natural_occ_success_rate")
    parser.add_argument("--record-video", action="store_true",
                         help="qualitative mode: run exactly ONE episode per condition (ignores "
                              "--num-episodes) with agentview + wrist-as-fed MP4s saved, instead of "
                              "a success-rate sweep")
    parser.add_argument("--record-episode-idx", type=int, default=0)
    parser.add_argument("--max-steps-override", type=int, default=None,
                         help="override TASK_MAX_STEPS[base_suite] (520 for libero_10) -- e.g. 600, "
                              "to test whether a failure was a step-budget problem rather than a "
                              "capability problem")
    parser.add_argument("--record-max-steps", type=int, default=200,
                         help="cap episode length for the video (full 520-step episodes are slow to "
                              "render every-step; 200 covers the early phase where the pilot's failure "
                              "mode should already be visible)")
    args = parser.parse_args()

    base_suite = args.task_suite[: -len("_occluded")] if args.task_suite.endswith("_occluded") else args.task_suite

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=base_suite,
        seed=7,
    )
    if args.attn_implementation:
        cfg.attn_implementation = args.attn_implementation
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    resize_size = get_image_resize_size(cfg)

    ckpt = torch.load(args.vjepa_checkpoint, map_location=model.device)
    model.vision_backbone.vjepa_predictor_dino.load_state_dict(ckpt["dino"])
    model.vision_backbone.vjepa_predictor_siglip.load_state_dict(ckpt["siglip"])
    print(f"Loaded vjepa predictor weights from {args.vjepa_checkpoint}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    print(f"Task: {task.language} ({args.task_suite} task_id={args.task_id})")
    env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)
    init_states = task_suite.get_task_init_states(args.task_id)

    sim = env.env.sim
    needs_occluder = any(c in ("baseline", "oracle", "gated_oracle", "attn_gated_oracle") for c in args.conditions) and args.occluder_body_substrings
    if needs_occluder:
        occluder_body_ids = find_occluder_body_ids(sim, args.occluder_body_substrings)
        assert occluder_body_ids, f"no bodies matched {args.occluder_body_substrings}"
        print(f"Occluder bodies: {[sim.model.body_id2name(i) for i in occluder_body_ids]}")
    else:
        occluder_body_ids = []
        print("No occluder-body-substrings given -- skipping occluder lookup (fine for --conditions clean "
              "on a plain, non-'_occluded' suite; 'clean' never calls derive_natural_mask_and_gt).")

    if args.record_video:
        init_state = init_states[args.record_episode_idx % len(init_states)]
        for condition in args.conditions:
            print(f"\n=== recording condition={condition} episode_idx={args.record_episode_idx} ===")
            success, done_step, n_calls, n_engaged, n_gated_out = run_episode_natural(
                cfg, env, task_description, model, resize_size, processor, action_head,
                proprio_projector, init_state, condition, occluder_body_ids, base_suite,
                record_video=True, max_record_steps=args.record_max_steps,
                attn_entropy_threshold=args.attn_entropy_threshold,
            )
            print(f"  success={success} done_step={done_step} n_calls={n_calls} n_engaged={n_engaged} n_gated_out={n_gated_out}")
        return

    results = {}
    for condition in args.conditions:
        print(f"\n=== condition={condition} ===")
        successes, episodes = 0, 0
        ep_results = []
        ep_range = range(args.start_episode_idx, args.start_episode_idx + args.num_episodes)
        for ep_idx in tqdm.tqdm(ep_range):
            init_state = init_states[ep_idx % len(init_states)]
            success, done_step, n_calls, n_engaged, n_gated_out = run_episode_natural(
                cfg, env, task_description, model, resize_size, processor, action_head,
                proprio_projector, init_state, condition, occluder_body_ids, base_suite,
                max_steps_override=args.max_steps_override,
                attn_entropy_threshold=args.attn_entropy_threshold,
            )
            successes += int(success)
            episodes += 1
            ep_results.append({"episode_idx": ep_idx, "success": success, "done_step": done_step,
                                "n_calls": n_calls, "n_engaged": n_engaged, "n_gated_out": n_gated_out})
            print(f"  ep{ep_idx}: success={success} done_step={done_step} n_calls={n_calls} "
                  f"n_engaged={n_engaged} n_gated_out={n_gated_out}")
        rate = successes / episodes if episodes else 0.0
        print(f"{condition}: {successes}/{episodes} ({rate*100:.1f}%)")
        results[condition] = {"successes": successes, "episodes": episodes, "rate": rate, "per_episode": ep_results}

    out = {
        "task_suite": args.task_suite, "task_id": args.task_id, "task_name": task.name,
        "occluder_body_substrings": args.occluder_body_substrings, "checkpoint": args.checkpoint,
        "vjepa_checkpoint": args.vjepa_checkpoint, "num_episodes": args.num_episodes,
        "max_steps_override": args.max_steps_override,
        "results": results,
    }
    out_path = f"{args.out_prefix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
