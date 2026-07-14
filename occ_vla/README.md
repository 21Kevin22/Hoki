# occ_vla — Occlusion-aware VLA system

Integrated system for robot manipulation under occlusion, combining a
generalist control policy with two complementary occlusion-resolution
modules and a dedicated evaluation benchmark.

## Modules

| # | Module | Path | Role |
|---|--------|------|------|
| 1 | Control backbone (pi0.5) | `src/occ_vla/control/` | Gemma-2.6B + SigLIP-400M open-world VLA. Predicts high-level subtasks, then 50Hz continuous actions. |
| 2 | World model (MMaDA-8B) | `src/occ_vla/world_model/` | Discrete-diffusion image/text model. Generates an "arm-free" subgoal image via masked token prediction when the arm self-occludes the target, and provides a CoT logical anchor when vision is fully unavailable. |
| 3 | PKLP | `src/occ_vla/pklp/` | RAFT optical flow → per-patch velocity/acceleration → constant-acceleration kinematic extrapolation for scene-induced (dynamic) occlusion. |
| 4 | LIBERO-Occ | `src/occ_vla/eval/` | LIBERO extended with physically-placed occluder boxes; Light/Medium/Heavy difficulty by target occlusion ratio (S_occ). |
| 5 | Integration / runtime | `src/occ_vla/integration/` | Adaptive routing between WM and PKLP, uncertainty-based fallback, main control loop. |

## Switching logic (see `integration/runtime.py::ControlLoop.step`)

- `arm_s_occ > 0.30` (self-occlusion) → `world_model.sample_arm_free_image()`
  generates an arm-free subgoal image; injected into pi0.5's observation
  via `control/observation_injection.py`.
- If that image is judged physically implausible
  (`integration/uncertainty.py`), fall through to PKLP instead of trusting it.
- Scene-induced dynamic occlusion (`scene_dyn_occ`) → PKLP
  (`pklp/optical_flow.py` + `pklp/kinematics.py`) extrapolates target
  position K steps ahead from the last clear observation, fed to the
  policy as a CoT anchor string.
- Pseudo-anchor: if PKLP has never seen the target clearly
  (`_last_clear_state is None`) when self-occlusion resolves, the WM's
  arm-free image seeds PKLP's frame history so a *later* scene occlusion
  still has something to extrapolate from.

## Status

Module interfaces are grounded against the real upstream APIs (not just
placeholder shapes) — see each module's docstring for exactly which
upstream class/method it wraps and why:

- `control/pi05_policy.py` — `openpi.policies.policy.Policy`
- `control/observation_injection.py` — repurposes LIBERO's always-zero
  `right_wrist_0_rgb` slot to carry the WM's subgoal image (a copy of
  `openpi`'s `LiberoInputs`, per that class's own "copy and modify" docstring)
- `world_model/mmada.py`, `tokenizer.py`, `arm_free_subgoal.py`, `cot_anchor.py`
  — `MMadaModelLM` / `MAGVITv2` (gen-verse/mmada)
- `world_model/action_tokenizer.py` — the `physical-intelligence/fast`
  action discretizer, remapped into MMaDA's own vocab space (MMaDA has
  no action modality upstream; this is occ_vla's own extension)
- `pklp/optical_flow.py` — `torchvision.models.optical_flow.raft_large`
- `pklp/latent_predictor.py`, `pklp/adaptive_horizon.py` — original to
  occ_vla (Euler-step convention mirrored from `openpi`'s own
  `Pi0.sample_actions`, not from either vendored repo)

Everything needing real trained weights (pi0.5, MMaDA-8B) still raises
`NotImplementedError` or `RuntimeError` until wired up against those
weights (`uncertainty.py`'s heuristic plausibility score doesn't need
weights and *is* implemented; `pklp/latent_predictor.py`'s `velocity_fn`
does, and has none — the analytic kinematic extrapolation in
`kinematics.py` is what `runtime.py` actually calls today). The
pure-logic pieces — occlusion routing, kinematic extrapolation, patch
flow pooling, flow-matching integration, adaptive-horizon variance
stopping, action-token offset arithmetic, observation injection, S_occ
difficulty bands, arm-token masking, and the full `ControlLoop.step`
branching — are implemented and unit-tested (55 tests, `tests/`).

**`eval/occluder.py`'s placement search is verified against a live
LIBERO simulation**, not just unit tests against a fake env (2026-07-14,
`libero_spatial` task 0, `libero_root` as below): searched and landed
inside all three difficulty bands — LIGHT S_occ=0.283, MEDIUM
S_occ=0.502, HEAVY S_occ=0.995 — each in 11-18s wall-clock (dominated by
MuJoCo model reload per binary-search trial, not by anything occ_vla
does). `LiberoOccEnv.reset()`/`.step()` run end-to-end. See "Setup"
below for the three real-environment issues that run surfaced (none of
them occ_vla bugs) and how they're worked around.

**Not implemented**: cross-timestep KV/activation caching (VLA-Cache-style
skipping of unchanged background tokens across control-loop iterations,
for the <200ms latency target). `Pi0.sample_actions` already caches the
prefix KV *within* one inference call (`third_party/openpi/src/openpi/models/pi0.py`),
but caching *across* control-loop steps would mean patching openpi's
model/serving code directly, which this wrapper-level integration
doesn't do.

Upstream references (see `scripts/setup_third_party.sh`):
- pi0.5: https://github.com/Physical-Intelligence/openpi (`src/openpi`)
- MMaDA-8B: https://github.com/gen-verse/mmada
- LIBERO (base benchmark LIBERO-Occ extends): https://github.com/Lifelong-Robot-Learning/LIBERO
  — pulled in transitively as `third_party/openpi`'s own `third_party/libero`
  git submodule, not cloned separately

## Layout

```
configs/             per-module YAML configs (hydra/omegaconf)
src/occ_vla/
  control/            pi0.5 adapter (pi05_policy.py) + observation_injection.py (feature reinjection)
  world_model/         MMaDA-8B adapter (mmada.py, tokenizer.py), arm_free_subgoal.py (Visual CoT),
                        cot_anchor.py, action_tokenizer.py (FAST action tokens)
  pklp/                optical_flow.py (RAFT), kinematics.py (V/A + extrapolation),
                        latent_predictor.py (conditional flow matching), adaptive_horizon.py
  integration/         occlusion_router.py, uncertainty.py, runtime.py (ControlLoop)
  eval/                 LIBERO-Occ: occluder.py, metrics.py, libero_occ_env.py
scripts/              entry points (train_*, run_policy, eval_libero_occ, setup_third_party.sh)
tests/                unit tests mirroring src/occ_vla layout
third_party/          openpi + mmada, cloned by setup_third_party.sh (gitignored)
```

## Setup

```bash
cd occ_vla
./scripts/setup_third_party.sh          # clones third_party/openpi (+ its libero submodule), third_party/mmada

pip install -e ".[pklp,world-model,eval,dev]"

# pi0.5: openpi has its own uv-managed dependency set (jax/flax + optional
# pytorch backend) — follow third_party/openpi/README.md ("uv sync" is the
# upstream-recommended path); or, for the pytorch-only inference path used
# by control/pi05_policy.py, `pip install -e third_party/openpi` and
# `pip install -e third_party/openpi/packages/openpi-client`.

# MMaDA: its own pinned stack (transformers==4.46.0, deepspeed, lightning, ...)
pip install -r third_party/mmada/requirements.txt
```

LIBERO itself (the base benchmark that LIBERO-Occ extends) comes in as
`third_party/openpi/third_party/libero` via the openpi submodule that
`setup_third_party.sh` initializes — `configs/eval/libero_occ.yaml`'s
`libero_root` already points at it.

```bash
pip install -e third_party/openpi/third_party/libero
pip install -e ".[eval]"   # robosuite==1.4.1, mujoco==3.0.0, cloudpickle, gym, thop — see pyproject.toml comment for why these exact pins
```

Three more things a live run needed, none of them occ_vla-side bugs:

1. **Offscreen rendering backend**: `export MUJOCO_GL=egl` (no display in
   a headless/cloud environment; EGL uses the GPU directly and worked
   without any other setup here).
2. **LIBERO's first-import config prompt**: `libero/libero/__init__.py`
   interactively asks for a dataset path on first import and hangs
   waiting for stdin in any non-interactive run. Pre-seed
   `~/.libero/config.yaml` (see `get_default_path_dict()` in that file
   for the keys) before importing `libero.libero` anywhere.
3. **`torch.load` default changed in PyTorch 2.6**: `benchmark.get_task_init_states()`
   calls `torch.load()` on LIBERO's bundled `.pt` init-state files
   without `weights_only=False`; PyTorch >=2.6 defaults to
   `weights_only=True` and rejects them. These files are from the
   official LIBERO repo (trusted), so wrap the call site with
   `weights_only=False` — occ_vla doesn't do this itself since it never
   calls `torch.load` directly; whatever script drives `LiberoOccEnv`
   needs to.

One thing occ_vla *does* work around, in `eval/libero_occ_env.py`
itself: `SegmentationRenderEnv.reset()` hardcodes the robot's
segmentation-instance name as `"Panda0"` to compute
`segmentation_robot_id`, but robosuite 1.4.1 names it `"MountedPanda0"`
— `LiberoOccEnv._fix_segmentation_robot_id()` patches this after
`reset()` instead of editing the vendored file.

Checkpoints (also not pip-installable, fetched on first use):
- pi0.5 base: `gs://openpi-assets/checkpoints/pi05_base` (auto-downloaded
  by `openpi.shared.download`, see `configs/control_backbone/pi05.yaml`)
- MMaDA-8B: `Gen-Verse/MMaDA-8B-MixCoT` on the HF Hub (CoT-tuned variant,
  see `configs/world_model/mmada.yaml`)
- MAGVIT-v2 tokenizer: `showlab/magvitv2` on the HF Hub
