"""
run_libero_eval_occlusion.py

Occlusion-aware sibling of the OFFICIAL run_libero_eval.py: same CLI shape
(draccus @wrap, GenerateConfig fields), same full-suite loop (ALL tasks in
task_suite_name, cfg.num_trials_per_task episodes each), same logging/video
conventions (setup_logging/log_message/save_rollout_video reused verbatim
from run_libero_eval.py) -- the ONLY behavioral difference is that each
episode's wrist camera gets a wrist_partial-style patch occlusion starting
at --delay-steps, and, depending on --correction-mode, an occlusion_mask is
(or isn't) built and passed through to the VLA call so the vjepa correction
does (or doesn't) engage.

This exists so the occluded-condition numbers are collected with the exact
same harness (task iteration, init states, max_steps, logging format) as the
clean baseline already run via the official script
(occ_vla/thirdparty/openvla-oft/experiments/logs/EVAL-*--eval20.txt,
2026-08-05, 97.6% avg over 4 suites) -- apples-to-apples, suite-wide,
n=cfg.num_trials_per_task/task, not a single cherry-picked task.

--correction-mode:
  none    : occlusion happens, occlusion_mask is never built/passed --
            measures the raw cost of the occlusion with NO recovery
            mechanism (matches run_dynamic_gating_eval.py's
            "never_corrected", generalized to all tasks in a suite).
  oracle  : occlusion_mask is passed from the instant occlusion begins
            (control_step >= delay_steps) -- upper bound / oracle-timed
            correction (matches "always_corrected"). Requires
            --vjepa-checkpoint.
  dynamic : occlusion_classifier.py's P(occluded), scored on the PREVIOUS
            VLA call's hidden state, gates when correction engages (sticky
            once triggered) -- the actual deployable trigger, with its
            known >=1-call detection latency (matches run_dynamic_gating_
            eval.py's "dynamic"). Requires --vjepa-checkpoint and
            --classifier-path.

Run with the openvla-oft conda env, e.g. (mirrors the just-completed clean
n=20 baseline, spatial suite, oracle-corrected, occluded from step 0):
  python scripts/run_libero_eval_occlusion.py \
    --pretrained_checkpoint <path> --task_suite_name libero_spatial \
    --num_trials_per_task 20 --vjepa_checkpoint <path> \
    --correction_mode oracle --delay_steps 0 \
    --run_id_note occ-oracle-eval20
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OCC_VLA_ROOT = os.path.dirname(SCRIPTS_DIR)
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import occlusion_classifier as oc  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig,
    TASK_MAX_STEPS,
    TaskSuite,
    check_unnorm_key,
    load_initial_states,
    log_message,
    process_action,
    setup_logging,
    validate_config,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from run_oft_camera_dropout_eval import _apply_partial_patch, _build_patch_token_mask  # noqa: E402

CORRECTION_MODES = ("none", "oracle", "dynamic")
CAMERA_BLOCK_INDEX = {"agentview": 0, "wrist": 1}  # num_images_in_input==2 order: [full_image, wrist_image]


@dataclass
class GenerateConfigOcclusion(GenerateConfig):
    # fmt: off
    vjepa_checkpoint: str = ""                # required for correction_mode in {oracle, dynamic}
    classifier_path: str = ""                 # required for correction_mode == dynamic
    threshold: float = 0.5                    # dynamic-mode trigger threshold on P(occluded)
    correction_mode: str = "none"             # one of CORRECTION_MODES
    occlude_camera: str = "wrist"             # one of CAMERA_BLOCK_INDEX ("wrist" or "agentview")
    delay_steps: int = 0                      # control-loop steps before occlusion begins (0 = from episode start)
    task_ids: str = ""                        # comma-separated task indices to restrict to; "" = all tasks in suite
    assist_alpha: float = 1.0                 # blend weight for the correction when engaged: action =
                                               # (1-alpha)*uncorrected + alpha*corrected. 1.0 (default) = old
                                               # behavior, single corrected-only forward, no extra compute.
                                               # <1.0 = "assistive" mode, dual forward pass + blend, base stays
                                               # closer to the uncorrected/baseline action.
    whitelist_task_ids: str = ""              # "Do No Harm" gate (2026-08-07): comma-separated task indices
                                               # the vjepa predictor is known/trusted to help on. "" (default)
                                               # = no restriction, correction_mode applies to every task as
                                               # before. When set, any task NOT in this list is forced to
                                               # behave as correction_mode="none" regardless of the configured
                                               # mode -- real n=20 data (2026-08-07) showed the OOD predictor
                                               # doesn't just fail to help on untested tasks, it can actively
                                               # break ones that worked fine uncorrected (task2: 45%->0%,
                                               # task3: 55%->10%, libero_10). This gate is the corresponding
                                               # safety default: never apply an unvalidated correction.
    debounce_calls: int = 1                   # "afterglow" gate (2026-08-07, correction_mode=dynamic only):
                                               # require the classifier's P(occluded) >= threshold on this many
                                               # CONSECUTIVE VLA calls before actually triggering (sticky)
                                               # engagement -- a single call dropping back below threshold
                                               # resets the streak to 0. 1 (default) = old behavior, trigger on
                                               # the very first crossing, fully backward-compatible. >1 trades
                                               # detection latency for robustness against a transient/one-call
                                               # classifier spike -- doesn't reintroduce a real momentum/decay
                                               # blend (that's a bigger change to the splice mechanism itself,
                                               # not attempted here), just delays commitment until occlusion
                                               # looks sustained rather than a blip.
    clean_scoring_only: bool = False          # false-positive-rate probe (2026-08-07, correction_mode=dynamic
                                               # only): NEVER apply the visual occlusion patch and NEVER engage
                                               # correction, but still run occlusion_classifier scoring +
                                               # debounce-streak tracking every VLA call on the genuinely clean
                                               # wrist image, and log whether/when a spurious trigger occurs.
                                               # Answers "how often does this real, unoccluded episode still
                                               # cross the trigger threshold" -- the debounce cost eval
                                               # (debounce_cost_eval_summary_n20.json) only measured the COST
                                               # side (delayed correction on real occlusion); this measures the
                                               # mechanism's actual motivation (suppressing false triggers).
    log_trajectories: bool = False            # trajectory eval (2026-08-08): save one .npz per episode
                                               # (eef_pos over time, gripper state, success, done_step,
                                               # engage flag per step) to --trajectory_dir. Off by default --
                                               # zero cost/behavior change for every existing invocation.
    trajectory_dir: str = "trajectory_logs"   # where log_trajectories writes -- plain relative name, resolves
                                               # inside thirdparty/openvla-oft like every other --out-dir here.
    # fmt: on


def run_episode_occluded(cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                          initial_state, log_file, classifier_params, effective_correction_mode=None,
                          task_id=None, episode_idx=None):
    """Same structure as run_libero_eval.run_episode, plus: wrist_partial-style
    patch occlusion from cfg.delay_steps onward, and correction-mode-gated
    occlusion_mask construction/passing.

    effective_correction_mode: the mode actually used for THIS episode's
    engage decision -- defaults to cfg.correction_mode, but run_task
    overrides it to "none" for any task not in cfg.whitelist_task_ids (the
    "Do No Harm" gate, see GenerateConfigOcclusion.whitelist_task_ids).

    task_id/episode_idx: only used to name the saved .npz when
    cfg.log_trajectories is set -- no effect on the rollout itself."""
    if effective_correction_mode is None:
        effective_correction_mode = cfg.correction_mode
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()

    from collections import deque
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    camera_block_index = CAMERA_BLOCK_INDEX[cfg.occlude_camera]

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False
    triggered = False   # dynamic mode: sticky once the debounce streak below completes
    consecutive_above = 0  # dynamic mode: consecutive VLA calls with P(occluded) >= threshold ("afterglow" debounce)
    last_hidden = None
    false_trigger_step = None  # clean_scoring_only: first control_step a spurious trigger occurred, if any
    n_calls_scored = 0         # clean_scoring_only: how many VLA calls this episode actually scored
    eef_traj, gripper_traj, engaged_traj = [], [], []  # log_trajectories: per-control-step log

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            control_step = t - cfg.num_steps_wait
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
            replay_images.append(img)  # unoccluded agentview frame, same convention as official script's replay video

            if cfg.log_trajectories:
                eef_traj.append(obs["robot0_eef_pos"].copy())
                gripper_traj.append(obs["robot0_gripper_qpos"].copy())

            occluded = (not cfg.clean_scoring_only) and (control_step >= cfg.delay_steps)
            occlusion_mask_np = None
            engage = False
            if occluded:
                if cfg.occlude_camera == "wrist":
                    wrist_img_resized, pixel_bounds = _apply_partial_patch(wrist_img_resized)
                else:
                    img_resized, pixel_bounds = _apply_partial_patch(img_resized)

                if effective_correction_mode == "oracle":
                    engage = True
                elif effective_correction_mode == "dynamic":
                    if triggered:
                        engage = True
                    elif last_hidden is not None:
                        p = oc.score(classifier_params, last_hidden)
                        if p >= cfg.threshold:
                            consecutive_above += 1
                        else:
                            consecutive_above = 0  # "afterglow": a single below-threshold call resets the streak
                        if os.environ.get("OCC_VLA_DEBUG_DEBOUNCE"):
                            print(f"[debounce-debug] control_step={control_step} p={p:.4f} "
                                  f"consecutive_above={consecutive_above}/{cfg.debounce_calls}", flush=True)
                        if consecutive_above >= cfg.debounce_calls:
                            triggered = True
                            engage = True
                            if os.environ.get("OCC_VLA_DEBUG_DEBOUNCE"):
                                print(f"[debounce-debug] TRIGGERED at control_step={control_step}", flush=True)
                # effective_correction_mode == "none": engage stays False
                # (either cfg.correction_mode=="none", or the whitelist gate
                # forced this task's correction off)

                if engage:
                    occlusion_mask_np = _build_patch_token_mask(
                        pixel_bounds, camera_block_index=camera_block_index, num_images=cfg.num_images_in_input
                    )
            elif cfg.clean_scoring_only and effective_correction_mode == "dynamic" and last_hidden is not None:
                # False-positive-rate probe: score the classifier on this
                # genuinely clean call's hidden state and track the debounce
                # streak exactly like the real dynamic branch above, but
                # NEVER build occlusion_mask_np / NEVER engage -- we only
                # want to know whether this clean episode's activations
                # would have spuriously crossed the trigger threshold.
                n_calls_scored += 1
                if not triggered:
                    p = oc.score(classifier_params, last_hidden)
                    if p >= cfg.threshold:
                        consecutive_above += 1
                    else:
                        consecutive_above = 0
                    if consecutive_above >= cfg.debounce_calls:
                        triggered = True
                        false_trigger_step = control_step

            if cfg.log_trajectories:
                engaged_traj.append(bool(engage))

            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            occlusion_mask = None
            if occlusion_mask_np is not None:
                occlusion_mask = torch.from_numpy(occlusion_mask_np).to(
                    device=model.device, dtype=torch.bfloat16
                ).reshape(1, -1, 1)

            if len(action_queue) == 0:
                if occlusion_mask is not None and cfg.assist_alpha < 1.0:
                    # Assistive mode: blend a corrected forward pass with an
                    # uncorrected one, instead of fully committing to the
                    # correction -- base action stays closer to what the
                    # uncorrected/baseline policy would have done.
                    #
                    # model.vision_backbone caches _vjepa_past_latents_{dino,siglip}
                    # as INTERNAL, sequential, one-forward-per-real-step state
                    # (read at the start of forward(), overwritten unconditionally
                    # at the end -- see modeling_prismatic.py's own docstring).
                    # Two forward calls per replan step would otherwise have the
                    # 2nd call read/extend the state the 1st call *just* produced
                    # (not the real previous env step's state), and leave a
                    # doubled-up, cross-contaminated history for every subsequent
                    # step -- silently invalidating both branches. Snapshot/restore
                    # around the pair so each call sees the same real prior-step
                    # history the other does, and the corrected branch's resulting
                    # state (not the uncorrected branch's, which is otherwise
                    # discarded) is what persists for the next real step.
                    vb = model.vision_backbone
                    dino_before = vb._vjepa_past_latents_dino
                    siglip_before = getattr(vb, "_vjepa_past_latents_siglip", None)

                    actions_corrected, hidden_state = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                        occlusion_mask=occlusion_mask, return_hidden_states=True,
                    )
                    dino_after_corrected = vb._vjepa_past_latents_dino
                    siglip_after_corrected = getattr(vb, "_vjepa_past_latents_siglip", None)

                    vb._vjepa_past_latents_dino = dino_before
                    if siglip_before is not None or hasattr(vb, "_vjepa_past_latents_siglip"):
                        vb._vjepa_past_latents_siglip = siglip_before
                    actions_uncorrected, _ = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                        occlusion_mask=None, return_hidden_states=True,
                    )

                    vb._vjepa_past_latents_dino = dino_after_corrected
                    if siglip_before is not None or hasattr(vb, "_vjepa_past_latents_siglip"):
                        vb._vjepa_past_latents_siglip = siglip_after_corrected

                    alpha = cfg.assist_alpha
                    if os.environ.get("OCC_VLA_DEBUG_ASSIST"):
                        diff = np.abs(np.asarray(actions_corrected) - np.asarray(actions_uncorrected))
                        print(f"[assist-debug] control_step={control_step} "
                              f"max|corrected-uncorrected|={diff.max():.6f} mean={diff.mean():.6f} "
                              f"corrected[0]={np.asarray(actions_corrected)[0]} "
                              f"uncorrected[0]={np.asarray(actions_uncorrected)[0]}", flush=True)
                    actions = [
                        (1.0 - alpha) * a_un + alpha * a_co
                        for a_un, a_co in zip(actions_uncorrected, actions_corrected)
                    ]
                else:
                    actions, hidden_state = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                        occlusion_mask=occlusion_mask, return_hidden_states=True,
                    )
                action_queue.extend(actions)
                last_hidden = hidden_state

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if cfg.clean_scoring_only:
        log_message(
            f"[clean-scoring] false_trigger={'yes' if false_trigger_step is not None else 'no'} "
            f"first_trigger_step={false_trigger_step} n_calls_scored={n_calls_scored} "
            f"debounce_calls={cfg.debounce_calls}",
            log_file,
        )

    if cfg.log_trajectories:
        os.makedirs(cfg.trajectory_dir, exist_ok=True)
        fname = (
            f"task{task_id}_ep{episode_idx}_{effective_correction_mode}"
            f"_{cfg.task_suite_name}.npz"
        )
        np.savez_compressed(
            os.path.join(cfg.trajectory_dir, fname),
            eef_pos=np.asarray(eef_traj, dtype=np.float32),        # (T, 3)
            gripper_qpos=np.asarray(gripper_traj, dtype=np.float32),  # (T, 2)
            engaged=np.asarray(engaged_traj, dtype=bool),           # (T,) -- correction active this control_step?
            success=success,
            done_step=t - cfg.num_steps_wait,
            correction_mode=effective_correction_mode,
            task_id=task_id if task_id is not None else -1,
            episode_idx=episode_idx if episode_idx is not None else -1,
        )

    return success, replay_images


def run_task(cfg, task_suite, task_id, model, resize_size, processor, action_head, proprio_projector,
             classifier_params, total_episodes, total_successes, log_file):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # "Do No Harm" whitelist gate: if set, only whitelisted tasks get
    # cfg.correction_mode's real behavior -- every other task is forced to
    # "none" (correction never engages) regardless of what was requested.
    effective_correction_mode = cfg.correction_mode
    if cfg.whitelist_task_ids:
        whitelist = {int(x) for x in cfg.whitelist_task_ids.split(",") if x.strip() != ""}
        if task_id not in whitelist:
            effective_correction_mode = "none"
            log_message(
                f"[whitelist gate] task_id={task_id} not in {sorted(whitelist)} -- "
                f"forcing correction_mode 'none' for this task (requested: {cfg.correction_mode})",
                log_file,
            )

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images = run_episode_occluded(
            cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
            initial_state, log_file, classifier_params, effective_correction_mode=effective_correction_mode,
            task_id=task_id, episode_idx=episode_idx,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero_occlusion(cfg: GenerateConfigOcclusion) -> float:
    validate_config(cfg)
    assert cfg.correction_mode in CORRECTION_MODES, f"correction_mode must be one of {CORRECTION_MODES}"
    assert cfg.occlude_camera in CAMERA_BLOCK_INDEX, f"occlude_camera must be one of {list(CAMERA_BLOCK_INDEX)}"
    if cfg.correction_mode in ("oracle", "dynamic"):
        assert cfg.vjepa_checkpoint, "correction_mode requires --vjepa_checkpoint (zero-init weights are a no-op)"
    if cfg.correction_mode == "dynamic" and not cfg.classifier_path:
        # Domain routing (2026-08-08): occlusion_classifier.npz was fit ONLY on
        # libero_10 activations -- applying it to a different suite's hidden-
        # state distribution silently starved the trigger (libero_goal
        # middle_drawer: engaged_frac 15.6% vs oracle's 100%, production_n50_
        # eval_summary.json). If --classifier_path isn't given explicitly,
        # auto-resolve occlusion_classifier_{task_suite_name}.npz next to the
        # default classifier, falling back to the original only if no
        # suite-specific one has been fit yet -- an explicit --classifier_path
        # always wins over this convention.
        # NOTE: these live at OCC_VLA_ROOT (occ_vla/occlusion_classifier*.npz),
        # not OFT_ROOT -- fit_occlusion_classifier.py never chdirs, unlike
        # most other scripts in this project, and the original
        # occlusion_classifier.npz was already saved there. Verified by
        # locating both files directly before trusting this path (2026-08-08).
        suite_specific = os.path.join(OCC_VLA_ROOT, f"occlusion_classifier_{cfg.task_suite_name}.npz")
        default_path = os.path.join(OCC_VLA_ROOT, "occlusion_classifier.npz")
        if os.path.isfile(suite_specific):
            cfg.classifier_path = suite_specific
        elif os.path.isfile(default_path):
            cfg.classifier_path = default_path
        assert cfg.classifier_path, (
            "correction_mode=dynamic requires --classifier_path, and no "
            f"{suite_specific} or {default_path} was found to auto-route to"
        )

    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)
    if cfg.vjepa_checkpoint:
        state_dicts = torch.load(cfg.vjepa_checkpoint, map_location=model.device)
        model.vision_backbone.vjepa_predictor_dino.load_state_dict(state_dicts["dino"])
        model.vision_backbone.vjepa_predictor_dino.to(dtype=torch.bfloat16)
        model.vision_backbone.vjepa_predictor_siglip.load_state_dict(state_dicts["siglip"])
        model.vision_backbone.vjepa_predictor_siglip.to(dtype=torch.bfloat16)

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    resize_size = get_image_resize_size(cfg)

    classifier_params = oc.load(cfg.classifier_path) if cfg.classifier_path else None

    log_file, local_log_filepath, run_id = setup_logging(cfg)
    if cfg.correction_mode == "dynamic":
        log_message(f"[domain routing] classifier_path resolved to: {cfg.classifier_path}", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_id_list = (
        [int(x) for x in cfg.task_ids.split(",") if x.strip() != ""]
        if cfg.task_ids
        else list(range(num_tasks))
    )

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Task ids: {task_id_list} (of {num_tasks} total)", log_file)
    log_message(
        f"Occlusion: camera={cfg.occlude_camera} delay_steps={cfg.delay_steps} "
        f"correction_mode={cfg.correction_mode} vjepa_checkpoint={cfg.vjepa_checkpoint or '(none)'} "
        f"classifier_path={cfg.classifier_path or '(none)'} threshold={cfg.threshold} "
        f"assist_alpha={cfg.assist_alpha} "
        f"whitelist_task_ids={cfg.whitelist_task_ids or '(none -- no restriction)'} "
        f"debounce_calls={cfg.debounce_calls} clean_scoring_only={cfg.clean_scoring_only}",
        log_file,
    )

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_id_list):
        total_episodes, total_successes = run_task(
            cfg, task_suite, task_id, model, resize_size, processor, action_head, proprio_projector,
            classifier_params, total_episodes, total_successes, log_file,
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero_occlusion()
