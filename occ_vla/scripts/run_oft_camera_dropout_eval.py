"""
run_oft_camera_dropout_eval.py

Tests whether OpenVLA-OFT's existing dual-camera input (agentview + wrist) already
tolerates a single camera going fully dark ("camera lost"), before building any
occlusion-recovery module. Mirrors occ_vla's own earlier finding for pi0.5
(CLAUDE.md, 2026-07-15: "pi0.5 does not need an occlusion countermeasure to avoid
*failing* ... it already succeeds under full, clean target occlusion via its
existing wrist-camera input") and the later "Attention gating" finding that full-
frame agentview zeroing (not just target-region masking) made a moka_pots episode
fail outright -- this script re-tests that exact scenario (same task, same style of
full-camera blackout) on OpenVLA-OFT instead of pi0.5.

Three conditions, same init_states across conditions (paired comparison):
  - baseline: both cameras real
  - agentview_occluded: agentview replaced with flat gray for the whole episode,
    wrist real
  - wrist_occluded: wrist replaced with flat gray for the whole episode,
    agentview real

Run with the openvla-oft conda env:
  /home/ubuntu/.pyenv/versions/miniforge3-latest/envs/openvla-oft/bin/python \
    scripts/run_oft_camera_dropout_eval.py --num-trials 3
"""

import argparse
import json
import os
import sys
import time
from collections import deque

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)  # for midlayer_oracle_splice
# Derived from __file__ (was hardcoded to the original project server's
# path, "/home/ubuntu/slocal1/Hoki/occ_vla/thirdparty/openvla-oft" -- broke
# on any other machine, e.g. a Kaggle clone under /root/oft_work/Hoki/...).
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
# experiments/robot/libero/libero_utils.py resolves bddl paths via the
# installed `libero` package, not cwd; chdir just matches upstream's own
# expected invocation location.
os.chdir(OFT_ROOT)
# isolated from the base-python LIBERO install used elsewhere in occ_vla
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, TaskSuite, check_unnorm_key  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Logging-spec + debounce-gate infrastructure (2026-08-17). All four modules
# are deliberately GPU/torch-import-order-independent and unit-tested
# without this file's heavy env (see scripts/tests/test_oft_*.py) -- keep
# new gating/logging logic there, not inline here, so it stays testable off
# the GPU server.
from oft_occlusion_gate import OcclusionGate  # noqa: E402
from oft_occlusion_gt import occ_gt_for_camera_block  # noqa: E402
from oft_step_logger import StepLogWriter  # noqa: E402
from oft_timing import CallTimer, StepTimer, unwrap_forward, wrap_forward_with_timer  # noqa: E402

GRAY_FILL = 127  # flat mid-gray -- "camera signal lost", not a content-aware occlusion

# Centered square patch for "partial occlusion" conditions: ~35% of frame area,
# simulating a gripper/object blocking part of (not all of) one camera's view.
# resize_size is 224 for OpenVLA -- patch spans rows/cols [46:178) (132px side).
PARTIAL_PATCH_FRAC = 0.59  # side length as a fraction of image side (~35% area)
NUM_PATCHES_PER_IMAGE = 256  # 224/14 = 16 -> 16*16, matches vision_backbone.get_num_patches()
GRID_SIDE = 16
PATCH_PX = 14


def _apply_partial_patch(img_resized):
    # img_resized: [H, W, 3] uint8
    h, w = img_resized.shape[:2]
    ph, pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
    r0, c0 = (h - ph) // 2, (w - pw) // 2
    out = img_resized.copy()
    out[r0 : r0 + ph, c0 : c0 + pw] = GRAY_FILL  # [ph, pw, 3] region blanked, rest stays real
    return out, (r0, r0 + ph, c0, c0 + pw)


def _build_patch_token_mask(pixel_bounds, camera_block_index, num_images):
    """Boolean (256*num_images,) array: True for tokens whose 14x14 pixel patch
    CENTER falls inside the occluded pixel region, restricted to one camera's
    256-token block -- same convention as train_vjepa_predictor_smoke_test.py's
    build_patch_token_mask, duplicated here to avoid a training-script import
    at eval time."""
    r0, r1, c0, c1 = pixel_bounds
    mask_grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
    for i in range(GRID_SIDE):
        for j in range(GRID_SIDE):
            center_r, center_c = i * PATCH_PX + PATCH_PX / 2, j * PATCH_PX + PATCH_PX / 2
            if r0 <= center_r < r1 and c0 <= center_c < c1:
                mask_grid[i, j] = True
    full_mask = np.zeros(NUM_PATCHES_PER_IMAGE * num_images, dtype=bool)
    full_mask[camera_block_index * NUM_PATCHES_PER_IMAGE : (camera_block_index + 1) * NUM_PATCHES_PER_IMAGE] = mask_grid.reshape(-1)
    return full_mask


def prepare_observation(obs, resize_size, occlude=None, hold_state=None, num_images=2):
    # img/wrist_img: [H, W, 3] uint8, real camera frames from LIBERO's OffScreenRenderEnv
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)  # [224, 224, 3] uint8
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)  # [224, 224, 3] uint8
    clean_wrist_img_resized = wrist_img_resized.copy()  # pre-occlusion, only used by *_oracle conditions
    occlusion_mask_np = None  # only set for *_vjepa / *_oracle conditions
    # (r0, r1, c0, c1) of the WRIST camera's occluded pixel region this step,
    # or None if the wrist camera isn't occluded this step -- this is what
    # occ_gt_for_camera_block() (oft_occlusion_gt.py) needs to compute a
    # ground-truth occlusion label. Deliberately scoped to the wrist camera
    # only: the correction mechanism this whole harness evaluates only ever
    # operates on wrist-camera patches (camera_block_index=1 below), so an
    # agentview_* occlusion is correctly "not occluded" from occ_gt's point
    # of view even though the frame itself is degraded.
    wrist_pixel_bounds = None

    if occlude in (None, "baseline"):
        pass
    elif occlude == "agentview_full":
        img_resized = np.full_like(img_resized, GRAY_FILL)
    elif occlude == "wrist_full":
        h, w = wrist_img_resized.shape[:2]
        wrist_img_resized = np.full_like(wrist_img_resized, GRAY_FILL)
        wrist_pixel_bounds = (0, h, 0, w)  # entire wrist frame is non-real -> occ_gt=1.0
    elif occlude == "agentview_hold":
        # Check A: freeze the first real frame seen this episode instead of
        # zero-filling -- tests whether failure is genuine info loss or OOD
        # shock from an unseen-during-training flat-gray input.
        if "agentview" not in hold_state:
            hold_state["agentview"] = img_resized.copy()
        img_resized = hold_state["agentview"]
    elif occlude == "wrist_hold":
        h, w = wrist_img_resized.shape[:2]
        if "wrist" not in hold_state:
            hold_state["wrist"] = wrist_img_resized.copy()
        wrist_img_resized = hold_state["wrist"]
        wrist_pixel_bounds = (0, h, 0, w)  # frozen frame is not the true current scene either
    elif occlude == "agentview_partial":
        # Check B: blank only a centered ~35%-area patch each step (real
        # content elsewhere stays live) -- simulates local occlusion by a
        # gripper/object rather than total camera loss.
        img_resized, _ = _apply_partial_patch(img_resized)
    elif occlude == "wrist_partial":
        wrist_img_resized, wrist_pixel_bounds = _apply_partial_patch(wrist_img_resized)
    elif occlude in ("wrist_partial_vjepa", "wrist_partial_vjepa_gated"):
        # Same pixel-level occlusion as wrist_partial, but ALSO builds the
        # token-level occlusion_mask so the VJEPA predictor actually engages
        # (only meaningful if model.vjepa_predictor has trained weights loaded
        # -- with zero-init weights this is mathematically identical to
        # wrist_partial, per the residual formulation's no-op guarantee).
        # "_gated" is the SAME occlusion, but run_episode only actually
        # forwards this mask to the model when the OcclusionGate says
        # correction_applied=True this step -- see run_episode.
        wrist_img_resized, wrist_pixel_bounds = _apply_partial_patch(wrist_img_resized)
        occlusion_mask_np = _build_patch_token_mask(wrist_pixel_bounds, camera_block_index=1, num_images=num_images)
    elif occlude == "wrist_partial_prevframe":
        # B3 control (zero learned parameters): fill the occluded patch with
        # the REAL wrist frame from the immediately preceding step (a
        # rolling copy), not frozen at episode start like wrist_hold. Must
        # be beaten by the trained vjepa predictor to claim its learned
        # dynamics are adding value over "just reuse the last real pixels" --
        # see the user's own plan, 2026-08-17: "追加パラメータ0の手法に勝てて
        # 初めて『V-JEPAの予測能力が効いている』と言える".
        wrist_img_resized, wrist_pixel_bounds = _apply_partial_patch(wrist_img_resized)
        prev_clean_wrist = hold_state.get("wrist_prev")
        if prev_clean_wrist is not None:
            r0, r1, c0, c1 = wrist_pixel_bounds
            wrist_img_resized[r0:r1, c0:c1] = prev_clean_wrist[r0:r1, c0:c1]
        # else: no real previous frame yet (first occluded step of the
        # episode) -- falls back to the same gray fill as wrist_partial,
        # which is the correct behavior (nothing real to copy from yet).
    elif occlude == "wrist_partial_midlayer_oracle":
        # DIAGNOSTIC: same pixel-level occlusion as wrist_partial, but the
        # actual splice happens INSIDE vision_backbone.forward (see
        # midlayer_oracle_splice.py) via a monkey-patch installed in main() --
        # a ground-truth ceiling check, standalone from the model's own
        # trained vjepa_predictor_dino/_siglip. occlusion_mask_np is returned
        # here only so run_episode knows this is a masked-diagnostic
        # condition; the model's own occlusion_mask kwarg is NOT used for
        # this condition (kept as None -- see run_episode).
        wrist_img_resized, wrist_pixel_bounds = _apply_partial_patch(wrist_img_resized)
        occlusion_mask_np = _build_patch_token_mask(wrist_pixel_bounds, camera_block_index=1, num_images=num_images)
    else:
        raise ValueError(f"Unknown occlude condition: {occlude}")

    # Always track the last REAL (pre-occlusion) wrist frame, regardless of
    # `occlude` -- keeps wrist_partial_prevframe's rolling-copy state
    # updated for every condition uniformly, harmless (unused) for any
    # condition that doesn't read hold_state["wrist_prev"].
    if hold_state is not None:
        hold_state["wrist_prev"] = clean_wrist_img_resized

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, occlusion_mask_np, clean_wrist_img_resized, wrist_pixel_bounds


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def _build_pixel_values(agentview_img, wrist_img, processor, prompt, device, dtype):
    """DIAGNOSTIC-only helper (wrist_partial_midlayer_oracle): builds the same
    pixel_values tensor get_vla_action would, for a direct model.vision_backbone(...)
    call to compute ground-truth clean features."""
    from experiments.robot.openvla_utils import prepare_images_for_vla

    class _CfgStub:
        center_crop = True

    images = prepare_images_for_vla([agentview_img, wrist_img], _CfgStub())
    primary, wrist = images
    inputs_primary = processor(prompt, primary).to(device, dtype=dtype)
    inputs_wrist = processor(prompt, wrist).to(device, dtype=dtype)
    return torch.cat([inputs_primary["pixel_values"], inputs_wrist["pixel_values"]], dim=1)


def _compute_s_occ(occ_gt, s_occ_source):
    """S_occ: the (in principle independently-ESTIMATED) occlusion severity
    signal the gate/A2/A3 operate on. `s_occ_source="oracle"` sets it equal
    to occ_gt -- this validates the logging/gate PIPELINE end-to-end and is
    legitimate for A1/A2/A4/B1 (their engagement-rate / latency / occluded-
    ADE numbers don't depend on the detector being non-trivial), but makes
    A3's precision/recall comparison (S_occ thresholded vs. occ_gt) exactly
    perfect BY CONSTRUCTION -- do not report an A3 PR curve computed this
    way as a real detector-quality result. `s_occ_source="probe"` is the
    real, independent detector (a linear probe on the VLA's own hidden
    state -- see thirdparty/openvla-oft/CLAUDE.md "In progress") and is not
    implemented yet: the probe itself hasn't been trained
    (collect_failure_probe_data.py collects data for it, but no training
    script exists yet as of this writing)."""
    if s_occ_source == "oracle":
        return occ_gt
    if s_occ_source == "probe":
        raise NotImplementedError(
            "s_occ_source='probe' needs a trained failure-prediction probe "
            "(see thirdparty/openvla-oft/CLAUDE.md 'In progress') -- not "
            "built yet. Use --s-occ-source oracle for now (pipeline "
            "validation / A1 / A2 / A4 / B1 only, NOT a real A3 result)."
        )
    raise ValueError(f"Unknown s_occ_source: {s_occ_source}")


def run_episode(
    cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
    initial_state, occlude, max_steps,
    episode=0, task_id=0, seed=0,
    s_occ_source="oracle", gate=None, step_log_writer=None,
    measure_latency=False, predictor_timer=None,
):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()  # don't leak temporal context across episodes
    if gate is not None:
        gate.reset()  # debounce state must not leak across episodes either

    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    hold_state = {}  # per-episode cache for *_hold/*_prevframe conditions, reset every episode
    t = 0
    n_calls = 0
    success = False
    log_step_idx = 0  # separate counter from `t`: skips the num_steps_wait warmup steps
    step_timer = StepTimer() if measure_latency else None
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, occlusion_mask_np, clean_wrist_img, wrist_pixel_bounds = prepare_observation(
                obs, resize_size, occlude=occlude, hold_state=hold_state, num_images=cfg.num_images_in_input
            )

            occ_gt = occ_gt_for_camera_block(wrist_pixel_bounds)
            s_occ = _compute_s_occ(occ_gt, s_occ_source)
            gate_result = gate.step(s_occ) if gate is not None else None

            occlusion_mask = None
            if occlusion_mask_np is not None:
                if occlude == "wrist_partial_vjepa":
                    # unconditional -- always engages the predictor whenever
                    # the pixel-level occlusion is present, no gating
                    occlusion_mask = torch.from_numpy(occlusion_mask_np).to(
                        device=model.device, dtype=torch.bfloat16
                    ).reshape(1, -1, 1)
                elif occlude == "wrist_partial_vjepa_gated":
                    # gated -- only forward the mask (i.e. only let the
                    # predictor actually overwrite patches) once the
                    # debounce gate says correction_applied=True; otherwise
                    # this step is IDENTICAL to a plain wrist_partial step
                    # (raw occluded pixels, model sees them unmodified).
                    if gate_result is not None and gate_result.correction_applied:
                        occlusion_mask = torch.from_numpy(occlusion_mask_np).to(
                            device=model.device, dtype=torch.bfloat16
                        ).reshape(1, -1, 1)
                elif occlude == "wrist_partial_midlayer_oracle":
                    # occlusion_mask stays None -- the splice happens
                    # transparently inside the monkey-patched
                    # vision_backbone.forward (see midlayer_oracle_splice.py).
                    # Just stash the clean wrist pixels it needs to read.
                    with torch.no_grad():
                        pixel_values_clean = _build_pixel_values(
                            observation["full_image"], clean_wrist_img, processor, prompt, model.device, torch.bfloat16
                        )
                        _, clean_wrist_pixel_values = torch.split(pixel_values_clean, [6, 6], dim=1)
                        model.vision_backbone._diagnostic_clean_wrist_pixel_values = clean_wrist_pixel_values

            t_vla_ms = 0.0
            t_predictor_ms = 0.0
            if len(action_queue) == 0:
                if predictor_timer is not None:
                    predictor_timer.reset()
                if step_timer is not None:
                    step_timer.start()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    use_film=cfg.use_film,
                    occlusion_mask=occlusion_mask,
                )
                if step_timer is not None:
                    t_vla_ms = step_timer.stop_ms()
                if predictor_timer is not None:
                    t_predictor_ms = predictor_timer.total_ms
                action_queue.extend(actions)
                n_calls += 1

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)

            if step_log_writer is not None:
                # t_vla_ms/t_predictor_ms are only nonzero on steps that
                # triggered a fresh VLA call (len(action_queue)==0 above,
                # i.e. every cfg.num_open_loop_steps-th logged step) --
                # steps that reused a queued action have zero LLM compute
                # this step. Filter to t_vla_ms > 0 for A1's real per-call
                # latency statistics; ee_position/action/S_occ/occ_gt are
                # meaningful on every logged row regardless.
                step_log_writer.log_step(
                    step=log_step_idx,
                    s_occ=s_occ,
                    occ_flag=gate_result.occ_flag if gate_result is not None else (s_occ > (gate.threshold if gate else 0.3)),
                    debounce_counter=gate_result.debounce_counter if gate_result is not None else 0,
                    correction_applied=gate_result.correction_applied if gate_result is not None else (occlusion_mask is not None),
                    occ_gt=occ_gt,
                    ee_position=obs["robot0_eef_pos"],
                    action=action,
                    t_vla_ms=t_vla_ms,
                    t_predictor_ms=t_predictor_ms,
                    t_total_ms=t_vla_ms,  # this harness's per-step overhead IS the VLA call; env.step (physics) is not counted as "our system"
                )
                log_step_idx += 1

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        print(f"  Episode error: {e}")

    if step_log_writer is not None:
        step_log_writer.close(success=success, steps_to_success=t if success else None)

    return success, t, n_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8, help="8 = moka_pots -- same task as CLAUDE.md's earlier full-agentview-zeroing pi0.5 finding")
    parser.add_argument("--num-trials", type=int, default=3, help="episodes per condition (pilot default; scale to 10 after a pilot looks sane, per project convention)")
    parser.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-10")
    parser.add_argument("--vjepa-checkpoint", default=None, help="path to a {'dino': state_dict, 'siglip': state_dict} .pt file (e.g. saved by train_vjepa_predictor_smoke_test.py) to load into vision_backbone.vjepa_predictor_dino/_siglip after model init; omit to use zero-init/untrained weights")
    parser.add_argument("--midlayer-split-frac", type=float, default=0.67, help="fraction of each backbone's own depth at which to splice for wrist_partial_midlayer_oracle -- 0.67 is the empirically-confirmed best value (occ_vla, 2026-07-31: 0.33->6/10, 0.50->7/10, 0.67->8/10, matching baseline); also the default the real vjepa_predictor_dino/_siglip modules are wired to (PrismaticVisionBackbone.midlayer_split_frac)")
    parser.add_argument("--results-path", default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "agentview_full", "wrist_full"],
        help="condition names double as the `occlude` value passed to prepare_observation; "
        "'baseline' occludes nothing. Options: baseline, agentview_full, wrist_full, "
        "agentview_hold, wrist_hold, agentview_partial, wrist_partial, "
        "wrist_partial_vjepa (engages the model's own trained/zero-init "
        "vjepa_predictor_dino/_siglip at split_frac=0.67, UNCONDITIONALLY "
        "whenever the pixel occlusion is present), "
        "wrist_partial_vjepa_gated (same predictor, but only actually "
        "engages once --debounce-k consecutive occ_flag=True steps have "
        "fired -- see oft_occlusion_gate.py; B1's k sweep varies this via "
        "--debounce-k without needing a different condition name), "
        "wrist_partial_prevframe (B3 zero-parameter control: fills the "
        "occluded patch with the real previous-step wrist frame instead of "
        "any learned prediction), "
        "wrist_partial_midlayer_oracle (diagnostic: standalone ground-truth "
        "ceiling check via monkeypatch, per --midlayer-split-frac, independent "
        "of the model's own predictor)",
    )
    parser.add_argument("--log-steps-dir", default=None, help="if set, writes one JSONL step log per (condition, episode) here -- see oft_step_logger.py for the schema. Omit to skip per-step logging entirely (episode-level results-path summary is always written regardless).")
    parser.add_argument("--s-occ-source", default="oracle", choices=["oracle", "probe"], help="oracle: S_occ = ground-truth wrist occlusion fraction (pipeline validation; makes A3 precision/recall tautological, NOT a real detector result -- see _compute_s_occ docstring). probe: NOT YET IMPLEMENTED, raises NotImplementedError.")
    parser.add_argument("--debounce-threshold", type=float, default=0.3, help="S_occ strictly above this counts as a single-step occ_flag trigger")
    parser.add_argument("--debounce-k", type=int, default=3, help="consecutive occ_flag steps required before correction_applied fires (B1 sweep axis: 1/3/5/10). Only affects the *_gated condition's actual model behavior; every condition still LOGS what correction_applied would have been at this k, via the gate object, so a log-only k re-sweep (B1) needs no rerun regardless of which k the live run used.")
    parser.add_argument("--gate-mode", default="threshold", choices=["threshold", "always", "never"], help="threshold: normal debounced gating. always: unconditional correction (B1's '無条件' arm) -- forces *_gated conditions to behave like *_vjepa. never: correction never applied (B1's '補正なし' arm) -- forces *_gated conditions to behave like plain wrist_partial.")
    parser.add_argument("--gate-no-latch", action="store_true", help="disable latching (correction_applied drops back to False as soon as debounce_counter falls below k) -- default is latched (stays on for the rest of the episode once triggered, matching this project's own established pi0.5-track gate convention)")
    parser.add_argument("--measure-latency", action="store_true", help="wrap vjepa_predictor_dino/_siglip forward calls with CUDA-synced timers for t_predictor_ms (A1). Adds real overhead (a GPU sync per predictor call) -- leave off for non-A1 runs.")
    parser.add_argument("--seed", type=int, default=None, help="value to log in the seed column; defaults to cfg.seed (the single global seed GenerateConfig/set_seed_everywhere already use) if not given")
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=8,
        task_suite_name=args.task_suite,
    )

    set_seed_everywhere(cfg.seed)

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    t0 = time.time()
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    print(f"Model loaded in {time.time() - t0:.1f}s, unnorm_key={cfg.unnorm_key}")

    if args.vjepa_checkpoint:
        state_dicts = torch.load(args.vjepa_checkpoint, map_location=model.device)
        model.vision_backbone.vjepa_predictor_dino.load_state_dict(state_dicts["dino"])
        model.vision_backbone.vjepa_predictor_dino.to(dtype=torch.bfloat16)
        model.vision_backbone.vjepa_predictor_siglip.load_state_dict(state_dicts["siglip"])
        model.vision_backbone.vjepa_predictor_siglip.to(dtype=torch.bfloat16)
        print(f"Loaded vjepa_predictor_dino/_siglip weights from {args.vjepa_checkpoint}")

    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    task_names = task_suite.get_task_names()
    print(f"Task suite: {args.task_suite}, task_id={args.task_id}, name={task_names[args.task_id]}")
    print(f"Task description: {task.language}")

    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = TASK_MAX_STEPS[TaskSuite(args.task_suite)]

    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    original_vision_backbone_forward = model.vision_backbone.forward

    run_seed = args.seed if args.seed is not None else cfg.seed

    if args.s_occ_source == "oracle":
        print(
            "  NOTE: --s-occ-source oracle => S_occ == occ_gt exactly. Valid "
            "for A1/A2/A4/B1 and for validating this logging/gate pipeline, "
            "but A3 precision/recall computed from this run's logs will be "
            "perfect by construction -- NOT a real detector-quality result."
        )

    predictor_timer = None
    if args.measure_latency:
        predictor_timer = CallTimer()
        _orig_dino_forward = wrap_forward_with_timer(model.vision_backbone.vjepa_predictor_dino, predictor_timer)
        _orig_siglip_forward = wrap_forward_with_timer(model.vision_backbone.vjepa_predictor_siglip, predictor_timer)
        print("  Wrapped vjepa_predictor_dino/_siglip forward calls with CUDA-synced timers (--measure-latency)")

    gate = OcclusionGate(
        threshold=args.debounce_threshold,
        k=args.debounce_k,
        mode=args.gate_mode,
        latch=not args.gate_no_latch,
    )

    results = {}
    for condition in args.conditions:
        occlude = condition  # condition names double as prepare_observation's `occlude` values

        if condition == "wrist_partial_midlayer_oracle":
            from midlayer_oracle_splice import make_midlayer_splice_forward

            # _apply_partial_patch uses the same fixed centered bounds every
            # step -- compute the (256,) single-image mask once, up front.
            h = w = resize_size
            ph, pw = int(h * PARTIAL_PATCH_FRAC), int(w * PARTIAL_PATCH_FRAC)
            r0, c0 = (h - ph) // 2, (w - pw) // 2
            fixed_bounds = (r0, r0 + ph, c0, c0 + pw)
            wrist_mask_256_np = _build_patch_token_mask(fixed_bounds, camera_block_index=0, num_images=1)
            wrist_mask_256 = torch.from_numpy(wrist_mask_256_np)

            patched_forward = make_midlayer_splice_forward(model.vision_backbone, args.midlayer_split_frac, wrist_mask_256)
            model.vision_backbone.forward = patched_forward
            print(f"  Installed mid-layer splice monkeypatch (split_frac={args.midlayer_split_frac})")
        else:
            model.vision_backbone.forward = original_vision_backbone_forward

        print(f"\n=== Condition: {condition} ===")
        cond_results = []
        for ep in range(args.num_trials):
            init_state = initial_states[ep]

            step_log_writer = None
            if args.log_steps_dir:
                log_path = os.path.join(args.log_steps_dir, f"{condition}_ep{ep:03d}.jsonl")
                step_log_writer = StepLogWriter(log_path, episode=ep, task_id=args.task_id, seed=run_seed)

            t0 = time.time()
            success, done_step, n_calls = run_episode(
                cfg, env, task_description, model, resize_size, processor, action_head, proprio_projector,
                init_state, occlude, max_steps,
                episode=ep, task_id=args.task_id, seed=run_seed,
                s_occ_source=args.s_occ_source, gate=gate, step_log_writer=step_log_writer,
                measure_latency=args.measure_latency, predictor_timer=predictor_timer,
            )
            wall = time.time() - t0
            print(f"  ep{ep}: success={success} done_step={done_step} n_calls={n_calls} wall={wall:.1f}s")
            cond_results.append({"episode": ep, "success": success, "done_step": done_step, "n_calls": n_calls, "wall_s": wall})
        n_success = sum(r["success"] for r in cond_results)
        print(f"  {condition}: {n_success}/{args.num_trials} success")
        results[condition] = cond_results

    if predictor_timer is not None:
        unwrap_forward(model.vision_backbone.vjepa_predictor_dino, _orig_dino_forward)
        unwrap_forward(model.vision_backbone.vjepa_predictor_siglip, _orig_siglip_forward)

    out_path = args.results_path or f"oft_camera_dropout_results_{args.task_suite}_task{args.task_id}_n{args.num_trials}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "s_occ_source_caveat": (
                    "S_occ == occ_gt exactly under --s-occ-source oracle; any A3 "
                    "precision/recall derived from this run's step logs is "
                    "tautologically perfect, not a real detector-quality result."
                    if args.s_occ_source == "oracle" else None
                ),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")
    if args.log_steps_dir:
        print(f"Saved per-step logs to {args.log_steps_dir}/<condition>_ep<NNN>.jsonl")

    print("\n=== Summary ===")
    for condition, cond_results in results.items():
        n_success = sum(r["success"] for r in cond_results)
        print(f"  {condition}: {n_success}/{len(cond_results)}")


if __name__ == "__main__":
    main()
