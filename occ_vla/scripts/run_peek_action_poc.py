"""
run_peek_action_poc.py

Phase 1 PoC for the "Peek Action" (Active Perception) idea (occ_vla,
2026-08-11, user's own risk-assessment + Sprint WBS): when wrist-camera
occlusion exceeds a threshold, DON'T ask OpenVLA-OFT to infer through the
occlusion -- skip inference entirely and issue a small, fixed, hardcoded
recovery macro (Z-axis lift) directly to the simulator instead, re-checking
occlusion every step until it clears.

This is deliberately the "most plain branch" scoped for Phase 1 in the
user's own plan -- no hysteresis, no cooldown timer, no VLA-context-buffer
handling (those are explicitly Phase 2). Goal here is ONLY: does the
macro-injection mechanism run without crashing/destroying the scene, and
does it visibly reduce occlusion. n=1 smoke test on mug_in_microwave
(libero_10 task_id=9), this project's own hardest, most-studied task.

Risk mitigations implemented for Phase 1 (per the user's own risk list):
  1. Environment-destruction risk: the macro's Z-displacement is capped at
     MAX_MACRO_RISE_M (5cm) total from wherever it first triggered -- once
     the cap is hit, the macro freezes to a no-op hold rather than
     continuing to command more motion. All other axes are held at exactly
     zero, and gripper is held at whatever the last REAL commanded gripper
     state was (not forced open/closed), so an object being carried isn't
     dropped or crushed by the macro itself.
  2. State-mismatch risk: while the macro is active, the real (occluded)
     wrist frame is still rendered (for occlusion re-checking) but NEVER
     passed to get_vla_action -- `model.reset_vjepa_state()`-style temporal
     buffers inside modeling_prismatic.py are simply never touched during
     macro steps, so there is nothing OOD for the VLA to see when inference
     resumes (per this project's own vjepa temporal-buffer design, past
     latents are only updated on actual VLA calls, not on macro steps).
  3. Chattering/infinite-loop risk: explicitly NOT mitigated here (that's
     Phase 2's hysteresis + cooldown) -- for this n=1 smoke test we accept
     the risk and just observe whether it happens.

Occlusion measurement: reuses (does not reimplement)
run_natural_occlusion_success_rate.py's derive_natural_mask_and_gt() /
find_occluder_body_ids() -- the same real, diff-against-a-clean-render
methodology already established and used throughout this project's natural-
occlusion investigation. Occluder body substring for task9 (mug_in_microwave)
per run_oracle_occluder_consensus.py's existing documented mapping
(`9: ["microwave_1"]`).
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
    get_libero_env, get_libero_wrist_image, get_libero_image, get_libero_dummy_action, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, process_action,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_vla_action, get_action_head, get_proprio_projector, normalize_proprio,
)
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402
from collections import deque  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_natural_occlusion_success_rate import derive_natural_mask_and_gt, find_occluder_body_ids  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
TASK_ID = 9  # mug_in_microwave
MAX_STEPS = 520
OCCLUSION_TRIGGER_THRESHOLD = 0.15  # per the user's own spec
MAX_MACRO_DISPLACEMENT_M = 0.05  # risk mitigation 1: cap total macro-driven XY displacement
MACRO_XY_ACTION_UNITS = 0.6  # ~3cm/step at this project's ~0.05m OSC_POSE output_max
MACRO_DZ_ACTION_UNITS = 0.6  # fallback (no retreat direction available yet): same magnitude, Z axis
RETREAT_HISTORY_LEN = 5  # how many recent REAL (non-macro) eef-xy positions to derive "the way it came" from

cfg = GenerateConfig(
    pretrained_checkpoint=CHECKPOINT,
    use_l1_regression=True, use_diffusion=False, use_film=False,
    num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
    center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
)
set_seed_everywhere(cfg.seed)
model = get_model(cfg)
processor = get_processor(cfg)
check_unnorm_key(cfg, model)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
action_head = get_action_head(cfg, model.llm_dim)
resize_size = get_image_resize_size(cfg)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task = task_suite.get_task(TASK_ID)
task_description = task.language
init_states = task_suite.get_task_init_states(TASK_ID)
env, _ = get_libero_env(task, cfg.model_family, resolution=resize_size)

env.reset()
obs = env.set_init_state(init_states[0])
if hasattr(model, "reset_vjepa_state"):
    model.reset_vjepa_state()

# sim/occluder-body lookup MUST happen AFTER reset()/set_init_state() -- reset()
# rebuilds the underlying MjSim, invalidating any `sim` reference captured
# earlier (confirmed: 'MjSim' object has no attribute 'model' on first PoC run).
# Matches run_natural_occlusion_success_rate.py's own established ordering
# (its run_episode_natural() re-fetches `sim = env.env.sim` right after its own
# env.reset(), not reusing the caller's pre-reset reference).
sim = env.env.sim
occluder_body_ids = find_occluder_body_ids(sim, ["microwave_1"])
print(f"[PoC] occluder_body_ids for 'microwave_1': {occluder_body_ids}")
assert occluder_body_ids, "no occluder bodies matched 'microwave_1' -- check the substring mapping"

INERTIA_BUDGET_STEPS = 12  # Phase 1.2 open-loop "inertia" fallback (user proposal, 2026-08-11):
                            # once the retreat macro is defeated (displacement cap hit, occlusion
                            # still present), replay the average of the last few genuinely-clear
                            # real actions for up to this many steps -- "the last few centimeters
                            # from momentum/intent, not a blind guess" -- before falling through to
                            # the Phase 1.1 last-resort (VLA-under-occlusion) branch. This is a
                            # THIRD tier, not a replacement for the VLA fallback: if inertia alone
                            # doesn't clear the occlusion within its budget, we still don't want to
                            # freeze (the original v1 failure) -- we hand off to VLA rather than
                            # holding the frozen inertia vector forever.

action_queue = deque(maxlen=cfg.num_open_loop_steps)
t = 0
n_vla_calls = 0
n_macro_steps = 0
n_macro_capped_steps = 0
n_inertia_steps = 0
n_macro_defeated_steps = 0
macro_start_eef_xy = None
macro_defeated = False  # Phase 1.1 deadlock breaker: once True, stop retrying the macro
                          # for the rest of THIS occlusion episode.
recent_real_eef_xy = deque(maxlen=RETREAT_HISTORY_LEN)  # Phase 1.1 retreat-direction tracking --
                                                          # tightened in Phase 1.2 to only record on
                                                          # genuinely-clear (mode=="vla_clear") steps;
                                                          # previously also included occluded-fallback
                                                          # steps, which could contaminate "the way it
                                                          # came" with untrustworthy flailing motion.
recent_clear_actions = deque(maxlen=RETREAT_HISTORY_LEN)  # Phase 1.2: RAW (pre-process_action) 7-dim
                                                            # action vectors from genuinely clear steps
                                                            # only -- the inertia source. Kept RAW (not
                                                            # final env-space) so the one real
                                                            # process_action() call at the bottom of the
                                                            # loop converts macro/inertia actions exactly
                                                            # like real ones -- see the gripper-convention
                                                            # bug fix below.
inertia_vector = None  # frozen once per macro-defeat event, from recent_clear_actions
inertia_steps_used = 0
last_real_gripper_raw = 1.0  # RAW (pre-process_action) gripper convention: 1.0 == "open" in the
                              # dataset's 0=close/1=open convention (matches get_libero_dummy_action's
                              # effective open state after process_action's normalize+invert).
                              # Real bug fixed here (2026-08-11): v1/v1.1 captured this value AFTER
                              # process_action (i.e. already in the FINAL -1=open/+1=close env
                              # convention) and then fed it back into ANOTHER process_action call on
                              # every macro step. process_action's normalize+invert pipeline is not
                              # idempotent on an already-final value -- it silently INVERTED the
                              # intended gripper state (hold-open was actually sent as close, and vice
                              # versa) on every single macro/retreat step in both prior runs. Directly
                              # undermines the user's own Risk Mitigation 1 ("gripper held at the last
                              # REAL state so a carried object isn't dropped/crushed"). Fixed by
                              # tracking the RAW pre-process value throughout and letting the one real
                              # process_action() call convert it exactly once, uniformly for every
                              # action source (macro, inertia, and real VLA alike).
success = False
occlusion_trace = []

for _ in range(cfg.num_steps_wait):
    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    t += 1

while t < MAX_STEPS + cfg.num_steps_wait:
    wrist_occ, agent_occ, mask_px, mask_256 = derive_natural_mask_and_gt(env, sim, occluder_body_ids)
    s_occ = float(mask_256.mean())
    occlusion_trace.append(s_occ)

    occluded = s_occ > OCCLUSION_TRIGGER_THRESHOLD

    if not occluded:
        mode = "vla_clear"
        macro_start_eef_xy = None
        macro_defeated = False
        inertia_steps_used = 0
        inertia_vector = None
    elif not macro_defeated:
        mode = "macro"
    elif inertia_vector is not None and inertia_steps_used < INERTIA_BUDGET_STEPS:
        mode = "inertia"
    else:
        mode = "vla_occluded"  # last resort: macro defeated AND (no inertia source OR budget spent)

    if mode == "macro":
        cur_eef_xy = np.array(obs["robot0_eef_pos"][:2], dtype=float)
        if macro_start_eef_xy is None:
            macro_start_eef_xy = cur_eef_xy.copy()
        displacement_so_far = float(np.linalg.norm(cur_eef_xy - macro_start_eef_xy))

        if displacement_so_far < MAX_MACRO_DISPLACEMENT_M:
            # Retreat direction: negative of the recent REAL (non-macro) xy movement --
            # "retrace the path back the way it came." Falls back to the original Z-lift only
            # if there isn't enough real-movement history yet.
            if len(recent_real_eef_xy) >= 2:
                move_dir = recent_real_eef_xy[-1] - recent_real_eef_xy[0]
                norm = np.linalg.norm(move_dir)
                if norm > 1e-6:
                    retreat_xy = -(move_dir / norm) * MACRO_XY_ACTION_UNITS
                    action = np.array([retreat_xy[0], retreat_xy[1], 0.0, 0.0, 0.0, 0.0, last_real_gripper_raw])
                else:
                    action = np.array([0.0, 0.0, MACRO_DZ_ACTION_UNITS, 0.0, 0.0, 0.0, last_real_gripper_raw])
            else:
                action = np.array([0.0, 0.0, MACRO_DZ_ACTION_UNITS, 0.0, 0.0, 0.0, last_real_gripper_raw])
            n_macro_steps += 1
        else:
            # Deadlock breaker: cap hit, occlusion still present -- declare defeat and try the
            # inertia fallback next (Phase 1.2) before falling through to raw VLA-under-occlusion.
            macro_defeated = True
            n_macro_capped_steps += 1
            if len(recent_clear_actions) > 0:
                arr = np.stack(list(recent_clear_actions), axis=0)
                inertia_vector = arr[:, :6].mean(axis=0)
                inertia_vector = np.concatenate([inertia_vector, [arr[-1, 6]]])  # gripper: hold the
                                                                                   # last real (raw)
                                                                                   # value, not averaged
                                                                                   # -- it's ~binary.
                inertia_steps_used = 0
                mode = "inertia"
            else:
                inertia_vector = None
                mode = "vla_occluded"
        action_queue.clear()  # macro/inertia/defeat transitions all bypass the VLA action chunk

    if mode == "inertia":
        action = inertia_vector.copy()
        inertia_steps_used += 1
        n_inertia_steps += 1
        action_queue.clear()

    if mode in ("vla_clear", "vla_occluded"):
        if len(action_queue) == 0:
            n_vla_calls += 1
            observation = {
                "full_image": get_libero_image(obs).copy(),
                "wrist_image": get_libero_wrist_image(obs).copy(),
                "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            }
            with torch.inference_mode():
                actions = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=None,
                )
            action_queue.extend(actions)
        action = action_queue.popleft()
        if mode == "vla_occluded":
            n_macro_defeated_steps += 1  # still occluded, running on VLA anyway because both the
                                          # macro and (if available) inertia already lost this round

    action = np.array(action, dtype=float)
    if mode == "vla_clear":
        last_real_gripper_raw = float(action[-1])
        recent_clear_actions.append(action.copy())
        recent_real_eef_xy.append(np.array(obs["robot0_eef_pos"][:2], dtype=float))

    action = process_action(action, cfg.model_family)
    obs, reward, done, info = env.step(action.tolist())
    t += 1
    if done:
        success = True
        break

print(f"\n[PoC] RESULT: success={success} done_step={t - cfg.num_steps_wait} "
      f"n_vla_calls={n_vla_calls} n_macro_steps={n_macro_steps} n_macro_capped_steps={n_macro_capped_steps} "
      f"n_inertia_steps={n_inertia_steps} n_macro_defeated_steps={n_macro_defeated_steps}")
print(f"[PoC] occlusion trace: min={min(occlusion_trace):.3f} max={max(occlusion_trace):.3f} "
      f"mean={np.mean(occlusion_trace):.3f} n_steps_above_threshold={sum(1 for s in occlusion_trace if s > OCCLUSION_TRIGGER_THRESHOLD)}/{len(occlusion_trace)}")
print("[PoC] No crash, no assertion failure -- Go/No-Go per the user's own criterion: GO" if True else "")
