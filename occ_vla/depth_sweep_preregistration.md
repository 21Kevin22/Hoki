# Depth sweep — pre-registered hypothesis (2026-08-19, before any depth-sweep rollout is run)

## Design (per user specification, 2026-08-19)

Not a hyperparameter search. This is a falsification test of the "mixing
hypothesis": ΔSR (oracle − baseline) degrades with how many transformer
blocks the spliced patch_clean must propagate through alongside the rest
of the (still-occluded) sequence — i.e. degrades with N_effective − L,
the number of remaining shared-context blocks after the splice point.

- **Effective N** (confirmed by direct measurement, not nominal block
  count): DINO = 22, SigLIP = 25 — the vendored OpenVLA-OFT backbone's own
  `get_intermediate_layers(n=num_blocks-2)` convention means the last 2
  blocks of each tower are never invoked by this checkpoint's real
  inference path at all, so this — not the nominal 24/27 — is the true
  depth the sweep is measured against.
- **Levels, task1 (primary, full 6-point sweep)**: split_frac ∈
  {0.0, 0.25, 0.50, 16/22 ("current", the pre-existing n=20×2 result),
  0.75, 1.0}, giving (dino, siglip) layer pairs:
  (0,0), (6,6), (11,12), (16,18)="current", (16,19), (22,25).
- **Levels, task6/task8 (replication, endpoints + current only)**:
  split_frac ∈ {0.0, 16/22, 1.0} → (0,0), (16,18), (22,25).
- **L=0 semantics**: splicing before block 0 (pixel-level) is
  architecturally equivalent to feeding the fully clean (occluder
  alpha-zeroed) image — outside the occluded region clean and corrupted
  pixels are already identical, so masked pixel-splice at this point
  reduces exactly to "use the clean image everywhere" (user's own
  derivation, implemented as a distinct code path, not an approximation).
- **L=N_effective semantics**: patch_clean is computed through the FULL
  depth the model actually uses (same as the corrupted branch), then
  substituted only at the very end — zero shared-context mixing in the
  transformer itself, matching the code's own splice-then-break structure
  when split_layer == extraction_layer exactly.
- **Same 20 seeds (task1: init_states[0:20], the ORIGINAL n=20 gate run's
  seeds — baseline=30%, chi2=6.40 there) across every depth level** —
  baseline does not depend on split_frac at all, so the already-recorded
  per-episode baseline outcomes for this seed set are reused/paired
  against each new depth level's oracle-only rollout, rather than
  re-running baseline (which would reintroduce run-to-run noise into the
  comparison for no reason, given baseline's own outcomes are already on
  record for this exact seed set).

## Pre-registered prediction (written BEFORE any depth-sweep rollout)

**ΔSR is monotonically non-increasing in (N_effective − L)** — degrades
as the remaining shared-context block count increases from the splice
point. Specifically:

- **ΔSR ≥ 0 at L=0** (full clean-image substitution — no occlusion
  information ever reaches the model for the corrupted region at all).
- **ΔSR ≥ 0 at L=N_effective** (patch_clean substituted only after full
  independent processing — zero propagated mixing).
- **ΔSR < 0 at intermediate L** (the "current" L=16/18 result, ΔSR was
  measured going one direction — +40pt on the original seed set — but
  did NOT replicate on a second, independent 20-seed set of the same
  task, where ΔSR was -10pt; task6/task8 at this same intermediate depth
  both showed ΔSR < 0). The mixing hypothesis predicts this
  inconsistency is itself explained by intermediate-depth mixing being
  the worst regime, not evidence the whole mechanism is directionless.

## What would falsify this

If ΔSR at L=0 or L=N_effective is NOT better than (or comparable to) the
already-measured intermediate-depth ΔSR, the mixing hypothesis is not
supported — the degradation isn't explained by propagation distance
through shared context, and a different explanation is needed. Per the
recommended execution order: run L=0 and L=N_effective FIRST (task1,
existing seeds[0:20]) — this alone is the cheapest possible test of
whether the hypothesis survives at all, before spending budget on the
middle levels or the replication tasks.

## Confound checklist (verified before treating any run as trustworthy)

- [ ] Same occlusion mask across all levels (only splice position changes)
- [ ] Injected clean features are that level's own clean activations
      (never reused across levels)
- [ ] Same injection method for both towers at every level
- [ ] L=N_effective's post-splice handling documented (see semantics above)
- [ ] Same trigger condition (patch_mask.any()) at every level
- [ ] `n_correction_applied > 0` empirically confirmed at every level
      (not inferred from code reading — the same discipline that caught
      the earlier always-None `occlusion_mask` gating bug)
