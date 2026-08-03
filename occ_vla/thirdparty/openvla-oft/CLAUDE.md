# OpenVLA-OFT vjepa_predictor — project notes for Claude

Scoped to this directory (`occ_vla/thirdparty/openvla-oft/`) so it doesn't
collide with the unrelated `occ_vla/CLAUDE.md` one level up, which belongs
to a completely separate investigation (pi0.5 + MMaDA + PKLP + dust3r +
UniVLA, different model/codebase) that happens to share the same parent
`occ_vla/` directory. Do not conflate the two — never cite that file's
findings as evidence for anything here, and vice versa.

## What this project is

A mid-layer occlusion-recovery module for OpenVLA-OFT
(`moojink/openvla-7b-oft-finetuned-*`, one checkpoint per LIBERO suite).
`VJEPA_LatentDynamicsPredictor` (`prismatic/extern/hf/vjepa_latent_predictor.py`)
splices a FiLM(proprio)+cross-attention residual correction into the vision
backbone at `split_frac=0.67` of each ViT's depth (DINOv2 + SigLIP, fused
backbone), overwriting occluded wrist-camera patch tokens in place. Not
real V-JEPA (no self-supervised pretraining, no EMA target encoder) — see
that file's own docstring for the naming rationale.

Empirical motivation: OpenVLA-OFT's baseline collapses to ~0% success when
the wrist camera is even partially occluded (a fixed centered ~35%-area
patch, `apply_partial_patch`/`PARTIAL_PATCH_FRAC=0.59` in
`train_vjepa_predictor_scaled.py`), despite having a second (agentview)
camera. This predictor is trained to recover from that.

## Environment setup (new server / fresh clone)

1. `git clone https://github.com/21Kevin22/Hoki.git && cd Hoki/occ_vla`
2. Set up the `openvla-oft` conda env per `thirdparty/openvla-oft/SETUP.md`.
   Separately clone+`pip install -e` `Lifelong-Robot-Learning/LIBERO`.
3. **Download base checkpoints to a LOCAL DIRECTORY, not a bare HF repo ID**:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="moojink/openvla-7b-oft-finetuned-libero-10",  # or -spatial/-object/-goal
       local_dir="checkpoints/openvla-7b-oft-libero10-vjepa",
   )
   ```
   Critical: `experiments/robot/openvla_utils.py`'s `update_auto_map`/
   `check_model_logic_mismatch` (which wire OUR custom
   `modeling_prismatic.py`/`vjepa_latent_predictor.py` into the checkpoint)
   both silently no-op via `if not os.path.isdir(pretrained_checkpoint): return`.
   Pass a bare HF repo ID string as `--checkpoint` and you get the STOCK
   OFT model with zero occlusion-recovery capability, no error, no warning.
4. First script run against that local dir auto-syncs our code in — look
   for `Created backup of original config...` / `Copied current version to
   checkpoint...` in the log, and confirm `unnorm_key=libero_XX_no_noops`
   resolves. `checkpoints/` itself is gitignored (each suite's download is
   ~15GB, trivially reproducible, not worth versioning or backing up).
5. Each downloaded checkpoint's `dataset_statistics.json` has norm_stats
   for ONLY the suite it was fine-tuned on — a libero_10 checkpoint cannot
   run libero_spatial tasks (`check_unnorm_key` asserts and fails). Suites
   need their own separate local checkpoint directory.

## Directory map

- `prismatic/extern/hf/vjepa_latent_predictor.py` — the predictor module.
- `prismatic/extern/hf/modeling_prismatic.py` — mid-layer splice wiring
  (`PrismaticVisionBackbone._run_featurizer_with_optional_predictor`),
  `reset_vjepa_state()`.
- `experiments/robot/openvla_utils.py` — `get_vla_action(...,
  return_hidden_states=True)` optionally returns the mean-pooled
  action-token final-layer LLM hidden state `predict_action` already
  computes internally (added 2026-08-03 for failure-prediction-probe work,
  see below) — default False is fully backward-compatible.
- `../../scripts/train_vjepa_predictor_multitask.py` — THE active training
  script (multi-task, `--sampling balanced|pooled`, optional loss
  reweighting flags — see "Findings" below for why those default off).
  Earlier `train_vjepa_predictor_{scaled,midlayer,smoke_test}.py` are
  single-task/earlier-phase scripts, kept for reference, not the current
  path.
- `../../scripts/collect_oft_onpolicy_rollout_data.py` — collects
  (agentview, wrist, proprio) triples from real unoccluded rollouts, for
  training data. `--out-dir` results are gitignored (large, reproducible).
- `../../scripts/run_oft_camera_dropout_eval.py` — the eval harness.
  `--conditions` include `baseline`, `wrist_partial` (raw occlusion, no
  correction), `wrist_partial_vjepa` (engages the trained/zero-init
  predictor), `wrist_partial_midlayer_oracle` (ground-truth ceiling check).
- `../../scripts/collect_failure_probe_data.py` — new (2026-08-03), see
  "In progress" below.
- `vjepa_predictor_*.pt` (repo root of this dir) — trained predictor
  checkpoints, committed to git (small, ~5.3MB each). **Best:
  `vjepa_predictor_multitask_3task_6000steps.pt`** (libero_10 moka_pots +
  mug_in_microwave + book_in_caddy, plain architecture, balanced sampling,
  no reweighting — 60%/20%/80% final eval, n=10).

## Established findings — read before proposing new interventions

1. **Loss reweighting is net-harmful at this data scale (~30 episodes/task).**
   Tried timestep-based precision weighting, then a data-driven
   spatio-temporal adaptive loss (temporal weight from measured error-by-
   decile curve + spatial boundary-patch boost). Both underperformed plain
   uniform loss. moka_pots degraded monotonically as reweighting got more
   sophisticated (pooled 50% → balanced 40% → precision-weighted 20% →
   spatio-temporal 0%). **Do not reweight the loss without a lot more
   training data to justify it.**
2. **Architecture additions (occlusion-mask channel into the predictor's
   `in_proj`, proprio velocity conditioning, mask-position/size jittering
   as augmentation) were also net-harmful**, independently confirmed via
   ablation (removing jitter alone didn't change the result — the
   mask-channel/velocity changes themselves were the regression, not
   jitter). Reverted; current `vjepa_latent_predictor.py` is the plain
   8-dim-proprio, no-mask-channel version. **Don't re-add these without
   more data.**
3. **What actually works: training directly on the target task's own
   on-policy data, multi-task, no reweighting.** Adding a 3rd task's real
   data improved ALL tasks simultaneously (libero_10: moka 40%→60%, mug
   10%→20%, book steady at 80%). This is the one lever that's reliably
   helped so far.
4. **Zero-shot cross-task/cross-suite transfer of a trained predictor is
   unreliable — always pair `wrist_partial` (no correction) with
   `wrist_partial_vjepa` (with correction) when testing a new task.** A
   single condition's success rate can't distinguish "the predictor
   helped" from "this task doesn't need correction anyway" (libero_10
   task_id=5: 80% either way, zero-init included — inconclusive) from
   "the predictor actively hurts" (libero_10 task_id=3: 60%→10%, a real
   regression using the libero_10-trained predictor cross-task).
5. **libero_spatial's "pick up bowl, place on plate" tasks are much more
   occlusion-tolerant than libero_10's longer, multi-stage tasks** — a
   3-task libero_spatial multitask run (tasks 1/4/7, same recipe as the
   successful libero_10 one) landed at 60%/80%/90%, but these numbers are
   barely different from each task's own no-recovery baseline. Likely
   because short (~110-150 step) single-stage tasks give corrupted signal
   less time to compound into failure. **libero_spatial may not be a
   useful testbed for this mechanism at all** — don't assume a suite-level
   win just because libero_10 showed one.
6. **Other suites ARE usable, just need their own downloaded checkpoint**
   (see Setup step 3) — an earlier claim in this thread that
   libero_spatial/object/goal were "impossible" was WRONG (only true for
   one specific local checkpoint copy, not in general). Checked-in
   checkpoints: `checkpoints/openvla-7b-oft-libero-{spatial,object,goal}-vjepa/`
   are NOT committed (see Setup) but were downloaded and verified working
   this session — re-download if needed on a new server.

## In progress (started 2026-08-03, not yet complete)

**Failure-prediction probe**, motivated by arXiv:2606.29699 ("Early Warning
Signals for OpenVLA Failure under Visual Distribution Shift" — found
near-term OpenVLA failure under occlusion is linearly decodable from
feedforward activations). Goal: a linear probe on `predict_action`'s own
action-token LLM hidden state (already exposed via `get_vla_action(...,
return_hidden_states=True)`) to detect "should the occlusion-recovery
predictor engage" WITHOUT needing a ground-truth occlusion mask (which our
controlled experiments always supply, but a real deployment wouldn't have).

**Important scope correction, already made once in this thread**: gating
answers "when to correct," not "is the correction any good" — it does NOT
fix finding 4 above (bad cross-task transfer). It's a real, separate,
useful capability (safe deployment without ground-truth occlusion signal),
not a fix for generalization.

Plan: `collect_failure_probe_data.py` runs `wrist_partial` (occluded, no
correction) rollouts, logs (per-step hidden state, episode success) to
`failure_probe_data_*/` (gitignored, not yet committed). Simplification
vs. the cited paper: using episode-level success/failure as the label
(coarser than their step-level near-term-failure labels) for tractability
— note this if revisiting. Not yet done: the actual probe training script,
and wiring a trained probe into the eval loop as a real gate.

## Gotchas hit this session

| Issue | Fix |
|---|---|
| `--save-path`/`--out-dir` prefixed with `thirdparty/openvla-oft/` crashes on save | Scripts `os.chdir(OFT_ROOT)` internally — always pass plain filenames/relative paths, cwd IS already this directory by execution time |
| Bash cwd drifts between tool calls | Always `cd` explicitly to `Hoki/occ_vla` before invoking scripts |
| Parallel multi-GPU launch via a bash loop with `set --` positional-param tricks silently produced empty `CUDA_VISIBLE_DEVICES` | Launch each GPU's command explicitly, one per call; verify with `/proc/<pid>/environ` before trusting placement |
| Shared machine's GPUs can be full of other users'/sessions' processes invisible to `ps aux` (different container namespace) | Check `nvidia-smi` memory/util directly before assuming a GPU is free; don't try to kill unrecognized processes |
| A vendored `thirdparty/openvla-oft/.git` (from how it was originally cloned) made git treat the whole directory as a submodule gitlink, silently dropping all our file-level changes from a commit | Removed the nested `.git`; verify with `git ls-files -s <path>` (mode `160000` = gitlink, means your edits aren't actually tracked) before trusting a commit that touches a vendored directory |
