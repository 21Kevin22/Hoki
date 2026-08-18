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
6. **Kaggle infra (occ_vla, 2026-08-18)**: this whole pipeline was also
   validated end-to-end on Kaggle (4-bit-quantized 7B model, T4 GPU) — see
   `occ_vla/kaggle/README.md` for the full chain of environment/version-pin
   fixes that took, and `occ_vla/kaggle/oft_kaggle_bootstrap.ipynb` for a
   ready-to-run setup notebook. `--load-in-4bit` (bitsandbytes NF4) cuts the
   base model to ~4GB, needed on a 16GB card.

## Directory map

- `prismatic/extern/hf/vjepa_latent_predictor.py` — the predictor module.
  Also takes an optional `cross_view_context` (added 2026-08-10,
  StereoPolicy-inspired, arXiv:2605.09989) — the OTHER camera's own current
  patch tokens at the same split layer, fused into the same cross-attention
  key/value set. `None` (default) reproduces the exact prior (temporal-only)
  behavior byte-for-byte; `VJEPA_DISABLE_CROSS_VIEW=1` env var disables it
  at inference/training time for a clean ablation.
- `prismatic/extern/hf/modeling_prismatic.py` — mid-layer splice wiring
  (`PrismaticVisionBackbone._featurize_to_split`/`_featurize_from_split`,
  a two-phase split so each image's predictor can see the OTHER image's
  own current tokens as cross-view context before either image's own
  correction is applied), `reset_vjepa_state()`. Also:
  `PrismaticForConditionalGeneration.predict_action(...,
  output_attentions=True)` requests attention weights from the same
  forward pass (no extra compute) and stashes a computed action-token-to-
  vision-patch attention-entropy scalar on `self._last_action_attn_entropy`
  (see "Failure-prediction probe / dynamic gating status" below).
- `experiments/robot/openvla_utils.py` — `get_vla_action(...,
  return_hidden_states=True, return_attn_entropy=True)`: `return_hidden_states`
  optionally returns the mean-pooled action-token final-layer LLM hidden
  state `predict_action` already computes internally (added 2026-08-03 for
  failure-prediction-probe work); `return_attn_entropy` optionally returns
  the attention-entropy scalar above (added 2026-08-09). Both default False,
  fully backward-compatible. `get_vla()` also accepts
  `cfg.attn_implementation` (getattr-guarded, unset by default) to force
  eager attention for a WHOLE rollout — only needed when actually MIXING
  `output_attentions=True/False` calls within one episode (see Gotchas).
- `../../scripts/train_vjepa_predictor_multitask.py` — THE active training
  script (multi-task, `--sampling balanced|pooled`, optional loss
  reweighting flags — see "Findings" below for why those default off; also
  `--mask-diverse`/`--adr-curriculum` for irregular-edge-anchored occlusion
  augmentation, see finding 12-adjacent work below).
  Earlier `train_vjepa_predictor_{scaled,midlayer,smoke_test}.py` are
  single-task/earlier-phase scripts, kept for reference, not the current
  path.
- `../../scripts/collect_oft_onpolicy_rollout_data.py` — collects
  (agentview, wrist, proprio) triples from real unoccluded rollouts, for
  training data. `--out-dir` results are gitignored (large, reproducible).
- `../../scripts/run_oft_camera_dropout_eval.py` — the eval harness.
  `--conditions` include `baseline`, `wrist_partial` (raw occlusion, no
  correction), `wrist_partial_vjepa` (engages the trained/zero-init
  predictor, unconditionally whenever the pixel occlusion is present),
  `wrist_partial_vjepa_gated` (same predictor, only engages once
  `--debounce-k` consecutive occ_flag=True steps have fired — see
  `oft_occlusion_gate.py`), `wrist_partial_prevframe` (B3 zero-learned-
  parameter control: fills the occluded patch with the real previous-step
  wrist frame), `wrist_partial_midlayer_oracle` (ground-truth ceiling
  check). Also: `--log-steps-dir` (per-step JSONL logging: S_occ raw value,
  occ_flag, debounce_counter, correction_applied, occ_gt, ee_position,
  action, t_vla_ms, t_predictor_ms — see `oft_step_logger.py`),
  `--s-occ-source oracle` (S_occ == occ_gt exactly, for pipeline validation;
  `probe` raises `NotImplementedError` — NOT the same thing as the trained
  failure-prediction probe below, which is wired via `run_dynamic_gating_eval.py`'s
  own `dynamic` condition, not this flag), `--load-in-4bit`/`--load-in-8bit`,
  `--start-episode` (resumability across Kaggle GPU-quota-limited sessions).
- `../../scripts/collect_failure_probe_data.py`, `train_failure_probe.py`,
  `occlusion_classifier.py`, `fit_occlusion_classifier.py`,
  `clean_manifold_detector.py`, `run_dynamic_gating_eval.py`,
  `collect_trajectory_data.py` — the failure-prediction-probe /
  dynamic-gating thread (started 2026-08-03, real results by 2026-08-09,
  see "Failure-prediction probe / dynamic gating status" below — NOT still
  "in progress" the way earlier notes here implied).
- `../../scripts/run_peek_action_poc.py` — single-episode PoC for "Peek
  Action" (Active Perception): natural-occlusion-triggered 3-tier fallback
  (retreat macro → open-loop inertia → last-resort raw VLA), gated by
  `derive_natural_mask_and_gt` (reused from `run_natural_occlusion_success_rate.py`).
- `../../scripts/run_peek_action_eval.py` — n-episode batch eval for Peek
  Action. `--condition {baseline, peek_v3, vjepa_oracle}` (`vjepa_oracle`
  loads a trained `vjepa_predictor_*.pt` and engages correction on the REAL
  natural occlusion mask, not synthetic). Also records attention-entropy
  (whole-episode AND occluded-vs-clear-sliced) and path-length/jitter
  (whole-episode AND occluded-steps-only) diagnostics — see finding 9 below
  for why the sliced versions matter.
- `../../scripts/run_libero_occ_benchmark.py` — batch natural SELF-occlusion
  (arm-hides-target) baseline scan across all `libero_10` tasks, via a live
  per-step hide-and-reveal segmentation render (see finding 10 for the two
  bugs this went through). **Tests the PLAIN `libero_10` suite's incidental
  self-occlusion only — see finding 12, this is NOT the real LIBERO-Occ
  benchmark.**
- `../../scripts/register_libero_occ_suites.py` — registers the 4 real
  LIBERO-Occ (arXiv:2606.10862, github.com/litsh/Libero-Occ) occluded task
  suites (`libero_{spatial,object,goal,10}_occluded`) with
  `libero.libero.benchmark`, purely additively (extends `libero_task_map` +
  `register_benchmark`, no vendored-source edits). Prerequisite:
  `third_party/Libero-Occ/scripts/setup/install_libero_occ_assets.sh` must
  already have copied `bddl_files`/`init_files` into the LIBERO checkout.
  This benchmark's occlusion is REAL SCENE GEOMETRY (e.g. `wooden_cabinet_1`,
  `short_cabinet_1`, `desk_caddy_1` — real, deliberately-placed 3D objects
  at fixed positions that physically block camera line-of-sight), confirmed
  by diffing `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`
  (`libero_10`) against its `libero_10_occluded` counterpart (occ_vla
  2026-08-09) — fundamentally different from, and much harder than, both
  our own synthetic `apply_partial_patch` and plain `libero_10`'s incidental
  self-occlusion (see finding 12).
- `../../scripts/run_libero_occluded_fast_scan.py` — fast (no per-step
  occlusion measurement, plain `get_libero_env`) baseline success-rate scan
  on the REAL `libero_10_occluded` suite. Use this, not
  `run_libero_occ_benchmark.py`, for anything meant to compare against the
  published LIBERO-Occ paper's own numbers.
- `../../scripts/record_task8_failure_videos.py` — records agentview+wrist
  MP4 (S_occ value burned into each frame) for specific episode_idx values,
  for qualitative failure-mode diagnosis (deterministic re-run: same
  seed=7 + same init_state reproduces the identical failure).
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
   8-dim-proprio, no-mask-channel version (plus the separately-added,
   backward-compatible `cross_view_context` — see directory map above,
   an architecturally different addition tested independently, not part
   of this reverted batch). **Don't re-add the mask-channel/velocity
   changes without more data.**
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
7. **Real (not synthetic) natural occlusion on `mug_in_microwave` (libero_10
   task_id=9) does NOT collapse OpenVLA-OFT baseline.** Measured baseline
   90-95% (n=20, `occlusion_mask=None`, real per-step natural mask via
   `derive_natural_mask_and_gt`) — a genuine ceiling case for this task/
   checkpoint. Both a Peek Action-style retreat+inertia intervention (85%)
   and the trained VJEPA predictor fed the REAL natural mask as oracle
   correction (70-80%) **underperformed doing nothing**, n=20-confirmed.
   Likely causes: little headroom (ceiling effect) + the VJEPA predictor's
   synthetic-mask training distribution not generalizing to real natural
   mask shapes (first time it was ever tested against a real, not
   synthetic, mask). **Don't assume `wrist_partial`-style synthetic-mask
   results (finding 3/4 above) predict real-natural-mask performance.**
   **Important scope note (occ_vla, 2026-08-18): this finding is about
   PLAIN `libero_10`'s incidental self-occlusion specifically (see finding
   12) — it does NOT mean oracle correction has no headroom on the real,
   harder `libero_10_occluded` benchmark, which has not yet been tested
   with VJEPA correction as of this writing. Don't cite finding 7 as
   evidence against the real-benchmark condition without actually running
   it.**
8. **Real bug: forcing `cfg.attn_implementation = "eager"` for a whole
   rollout caused baseline success to collapse from ~95% to ~0%** (every
   episode timed out) — introduced by over-applying a caution meant only
   for MIXING `return_attn_entropy=True`/`False` calls within one episode
   (`run_natural_occlusion_success_rate.py`'s `attn_gated_oracle` docstring).
   `smoke_test_attn_entropy.py` — already-validated, already in this repo —
   never sets `attn_implementation` at all and explicitly checks
   `return_attn_entropy=True` gives a byte-identical action to `False` on
   the same input. **Leave `attn_implementation` unset unless you are
   actually mixing True/False calls in the same episode.**
9. **Whole-episode-averaged diagnostics dilute real local signal, and can
   mislead in the opposite direction of what's assumed.** Slicing attention
   entropy by occluded-vs-clear step (rather than one whole-episode mean)
   revealed a real pattern invisible in the aggregate: attention gets MORE
   focused (lower entropy) during occlusion for both baseline and
   `vjepa_oracle`, with `vjepa_oracle`'s effect ~2.5x baseline's — plausibly
   linked to its worse success rate (over-narrowing onto the wrong signal).
   Separately, **low occluded-step jitter is NOT a smoothness/quality
   signal here — it's an anti-correlate of success**: in every condition
   tested, failed episodes show markedly LOWER occluded-step jitter than
   successful ones (stalling reads as "smooth" by a naive jitter metric).
   Don't cite low jitter as evidence an intervention "helped" without
   checking whether the episode actually progressed.
10. **Two real bugs found building a task-agnostic natural self-occlusion
    scanner** (`run_libero_occ_benchmark.py`): (a) measuring "any
    wrist-camera pixel that changes when the robot is hidden" is dominated
    by the wrist camera's OWN gripper-in-frame mounting geometry (a
    near-constant ~0.19-0.23 regardless of task/scene) — must restrict the
    diff to the TARGET object's own segmentation footprint instead; (b) a
    "clear baseline captured once at episode start, compared in
    screen-space against later frames" (the formula already correct for a
    STATIC agentview camera) is invalid for the WRIST camera specifically,
    since it moves with the arm — the same screen-space region means a
    different 3D point every step. Fix: a LIVE per-step hide-and-reveal
    (two renders/step: current target-pixel count vs. robot-hidden
    target-pixel count, from the SAME current camera pose each time).
11. **A 10-task natural-self-occlusion baseline scan (n=20, plain
    `libero_10`) found only ONE task with real headroom**: task_id=8
    ("put both moka pots on the stove"), 40% success (every other task
    90-100%). Occlusion severity (mean S_occ) does NOT discriminate
    success from failure WITHIN this task (0.621 success eps vs. 0.603 fail
    eps — essentially identical). Video inspection of 3 failed episodes
    (deterministic re-run, `record_task8_failure_videos.py`) found a
    consistent pattern: the model successfully grasps+places the FIRST pot
    even through ~160 steps of sustained ~100% self-occlusion, then
    **stalls on the SECOND pot during a LOWER-occlusion window** — a
    visible, imprecise grasp attempt around step ~320, followed by the arm
    drifting to a pose where the target isn't even in the wrist camera's
    view, no recovery for the rest of the 520-step budget. **This looks
    like a grasp-precision/recovery problem, not a vision-occlusion
    problem, on this specific task under natural self-occlusion —
    occlusion-triggered interventions (Peek Action, VJEPA correction) are
    unlikely to be the right lever here, on the PLAIN `libero_10` version
    of this task specifically.**
12. **`libero_10` (plain) and `libero_10_occluded` are NOT the same
    benchmark, and look identical from task language alone.**
    `libero_10_occluded` (registered by `register_libero_occ_suites.py`)
    adds REAL, deliberately-placed 3D occluder objects (e.g.
    `wooden_cabinet_1`) at fixed positions that physically block camera
    line-of-sight — the actual published LIBERO-Occ benchmark
    (arXiv:2606.10862) condition. Finding 11 (and the whole natural-
    occlusion investigation through 2026-08-12) tested only the PLAIN
    suite's incidental self-occlusion, measurably easier than the real
    benchmark (this project's own scan: 90-100% on 9/10 tasks; the paper's
    own reported OpenVLA-OFT average on the real benchmark: 47.95%). Task
    indices also differ between the two suites (`_occluded` is sorted
    alphabetically by BDDL filename, not the stock `task_order`
    permutation) — e.g. "both moka pots" is index 3 in `libero_10_occluded`,
    not 8. **Always confirm which suite string a script actually passes to
    `benchmark.get_benchmark_dict()` before trusting a baseline number.**
13. **The `configs/eval/libero_occ.yaml`-style shipped eval config for the
    real benchmark defaults to `num_trials_per_task: 1`** (occ_vla,
    2026-08-18) — this is NOT the protocol that produced the LIBERO-Occ
    paper's own Table 2/3 numbers (their repo's own `RELEASE_MANIFEST.md`
    explicitly excludes "paper reproduction scripts"). Any number reported
    against this benchmark needs its own explicit n/seed stated — don't
    assume a single default config's n is meaningful.

## Failure-prediction probe / dynamic gating status (started 2026-08-03,
correcting an earlier "not yet complete" note — real results existed by
2026-08-09, just hadn't been folded into this file)

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
— note this if revisiting.

**Actual status, confirmed 2026-08-12 by re-checking the repo directly**
(this section previously said "not yet done" past this point — wrong,
stale): the probe WAS trained and DOES gate a real eval loop.
`train_failure_probe.py` (occlusion-vs-clean classifier, synthetic
`wrist_partial` masks, moka_pots+mug_in_microwave): row-level val AUC
0.9997, episode-level AUC 1.0 (`failure_probe_results.json`). A second
classifier distinguishing VJEPA-corrected from uncorrected activations:
val AUC 0.99, episode AUC 1.0 (`failure_probe_results_vjepa_vs_uncorrected.json`)
— confirms correction measurably shifts internal representations, though
this is a separate claim from correction being *helpful* (see finding 7
above — it wasn't, on real natural masks, though scoped only to PLAIN
`libero_10`'s self-occlusion — see finding 12/13). A third probe predicting
eventual episode SUCCESS from mid-episode hidden state (`label_from_success=True`):
episode-level val AUC 1.0 (`failure_probe_results_recovery_quality_n70.json`).
`occlusion_classifier.py`/`fit_occlusion_classifier.py` wrap the trained
classifier as a real-time trigger, used by `run_dynamic_gating_eval.py`'s
`dynamic` condition (engages correction once `P(occluded)` crosses a
threshold, sticky). **Negative result also on record**:
`clean_manifold_detector.py` (naive PCA+Mahalanobis distance from a
"clean" activation manifold, the more obvious first approach) INVERTS —
AUC 0.12-0.20, occlusion reads as MORE typical than clean — kept in the
codebase as a documented dead end, not deleted. **Not yet done (occ_vla,
2026-08-18): none of this dynamic-gating machinery — including the
classifier's own AUC — has been validated against the REAL
`libero_10_occluded` benchmark's occlusion (only synthetic `wrist_partial`
so far); same generalization gap already flagged for finding 7's
VJEPA-oracle result. Required before trusting any image-based-detection
result on the real benchmark.**

## Gotchas hit this session

| Issue | Fix |
|---|---|
| `--save-path`/`--out-dir` prefixed with `thirdparty/openvla-oft/` crashes on save | Scripts `os.chdir(OFT_ROOT)` internally — always pass plain filenames/relative paths, cwd IS already this directory by execution time |
| Bash cwd drifts between tool calls | Always `cd` explicitly to `Hoki/occ_vla` before invoking scripts |
| Parallel multi-GPU launch via a bash loop with `set --` positional-param tricks silently produced empty `CUDA_VISIBLE_DEVICES` | Launch each GPU's command explicitly, one per call; verify with `/proc/<pid>/environ` before trusting placement |
| Shared machine's GPUs can be full of other users'/sessions' processes invisible to `ps aux` (different container namespace) | Check `nvidia-smi` memory/util directly before assuming a GPU is free; don't try to kill unrecognized processes |
| A vendored `thirdparty/openvla-oft/.git` (from how it was originally cloned) made git treat the whole directory as a submodule gitlink, silently dropping all our file-level changes from a commit | Removed the nested `.git`; verify with `git ls-files -s <path>` (mode `160000` = gitlink, means your edits aren't actually tracked) before trusting a commit that touches a vendored directory |
| Forcing `cfg.attn_implementation = "eager"` for a whole rollout, reasoning by analogy from a caution that only applies to MIXING `return_attn_entropy=True/False` calls within one episode | Leave it unset; `smoke_test_attn_entropy.py` already proves `return_attn_entropy=True` alone is a safe, byte-identical-action no-op — collapsed baseline success 95%→0% when misapplied (2026-08-12) |
| Self-occlusion via the WRIST camera: diffing "robot hidden vs visible" over the whole frame just measures the gripper's own near-constant screen presence; a "clear baseline captured once" (valid for a static agentview camera) silently becomes meaningless once the wrist camera moves | Restrict the diff to the TARGET's own segmentation footprint, and re-derive it LIVE every step (two renders/step), not from a stale start-of-episode baseline |
| `libero_10` and `libero_10_occluded` are registered as separate suites but look identical from task language/description alone (2026-08-12) | Always check which suite string a script passes to `benchmark.get_benchmark_dict()` — `libero_10` has no deliberate occluders at all; only `libero_10_occluded` matches the published LIBERO-Occ benchmark. Task indices differ between the two (alphabetical-by-BDDL-filename vs. stock `task_order`) |
| Multiple GPU processes sharing one `--results-dir`, each writing its OWN local `summary.json` after every task | The shared `summary.json` gets silently overwritten by whichever process finishes last (only its own task subset survives) — per-task JSON files (distinct filenames) are safe; recompute the combined summary from those, don't trust the shared aggregation file after a multi-process parallel scan |
| bitsandbytes 4-bit/8-bit quantization structurally converts EVERY `nn.Linear` found by `named_modules()` to `Linear4bit`, including newly-added (not-in-checkpoint) submodules like `vjepa_predictor_dino/_siglip` | Pass `llm_int8_skip_modules=["vjepa_predictor_dino", "vjepa_predictor_siglip"]` in `BitsAndBytesConfig` |
| `nn.Module.to()`'s `_apply` only forwards `dtype` to a leaf tensor already `.is_floating_point()` — a `torch.uint8` meta tensor (seen for these submodules specifically under 4-bit quantization) silently no-ops on `.to(dtype=...)` regardless of call order | Bypass `Module.to()` for the dtype cast: reassign each parameter's `.data` directly via a small `_force_dtype()` helper, after `to_empty()` gives real (if wrong-dtype) storage |
