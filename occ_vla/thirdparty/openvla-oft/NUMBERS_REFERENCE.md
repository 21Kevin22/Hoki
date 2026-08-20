# Numbers reference table (2026-08-20 night session)

All numbers below, with exact source directory / n / episode range /
condition, so every slide can cite this table instead of a bare
percentage. **Any number without an entry here should not be presented.**

## task1 (`put the black bowl in the bottom drawer of the cabinet and close it`, KITCHEN_SCENE4)

| label | value | n | episodes | source dir | condition |
|---|---|---|---|---|---|
| baseline (n=20 slice, original) | 30% (6/20) | 20 | 0-19 | `libero_occluded_oracle_task1_n20/` | baseline |
| baseline (n=20 slice, reproducibility check) | 30% (6/20) | 20 | 0-19 | `baseline_reproducibility_check_seeds0_20/` | baseline |
| baseline (n=20 slice, factorial v2 run) | 35% (7/20) | 20 | 0-19 | `factorial_task1_n20_v2/` | baseline |
| **baseline (n=50, full distribution)** | **54% (27/50)** | 50 | 0-49 | `baseline_all50_task1/` | baseline |
| oracle(16,18) [WRONG depth (15,17), historical, do not cite] | 70% (14/20) | 20 | 0-19 | `libero_occluded_oracle_task1_n20/` | oracle |
| **oracle(16,18) [CORRECT depth, n=50]** | **68% (34/50)** | 50 | 0-49 | `current_correct_1618_all50/` | oracle |
| **L=0 (privileged full-frame clean render, n=50)** | **76% (38/50)** | 50 | 0-49 | `depth_sweep_task1_L0_all50/` | oracle |
| no_collision [BUGGY free-fall, RETRACTED, never cite] | 100% (20/20) | 20 | 0-19 | `factorial_task1_n20/` | no_collision |
| **no_collision [CORRECTED bit-separation fix]** | **95% (19/20)** | 20 | 0-19 | `factorial_task1_n20_v2/` | no_collision |
| **non-occluded plain LIBERO-10 baseline (stock task_id=3)** | **95% (19/20)** | 20 | 0-19 | `stock_libero10_baseline_task1equiv_n20/` | baseline (--use-stock-suite) |

**Why baseline varies 30%/35%/50%/54% across rows above**: the n=20
slice (episodes 0-19) is itself a genuinely harder-than-average subset
of the full 50-state distribution (confirmed reproducible, not policy
noise, via the repeated 30%/30% measurement on the identical 20
episodes) -- the 35% row is a THIRD independent measurement on the
SAME 20 episodes that came back different again (35% vs 30%/30%),
which is real evidence the underlying policy is not perfectly
deterministic run-to-run even on identical init_states (openvla-oft's
`use_l1_regression` head is not guaranteed bit-identical across
process restarts -- not yet root-caused, note for future work). **The
n=50 baseline (54%) is the only number that should be used as "task1's
real baseline rate"** in any slide; the n=20 numbers exist only because
some paired comparisons (oracle(16,18)-historical, no_collision) were
run against that specific 20-episode slice before the n=50 baseline
existed, and are kept for internal paired-comparison bookkeeping, not
as standalone baseline citations.

## Cross-task screening (mid-layer oracle(16,18), n=20 each, CORRECT depth)

| task | baseline | oracle(16,18) | effect | b/c (McNemar) |
|---|---|---|---|---|
| task1 | 54% (n=50) | 68% (n=50) | +14pt | 6/13 (chi2=2.58, p=0.108) |
| task6 | 30% (6/20) | 20% (4/20) | -10pt | 3/1 (chi2=1.0) |
| task8 | 35% (7/20) | 30% (6/20) | -5pt | 3/2 (chi2=0.2) |
| task2 | 70% (14/20) | 65% (13/20) | -5pt | 1/0 (chi2=1.0) |
| task0 | 95% (19/20) | 100% (20/20) | +5pt (ceiling-limited) | 0/1 (chi2=1.0) |
| task4 | in progress | in progress | -- | -- |
| task3 | preempted, not completed | -- | -- | -- |
| task5 | preempted, not completed | -- | -- | -- |

Sign count so far (5 tasks): 2 positive (task1, task0-ceiling-limited),
3 negative (task6, task8, task2). **Not yet a consistent direction.**

## 2x2 factorial (visual x physical), task1

| | physical: collision | physical: no collision |
|---|---|---|
| visual: occluded | 54% (baseline, n=50) | **95%** (no_collision, n=20, corrected) |
| visual: clean | 76% (L=0, n=50, privileged) | not yet measured (4th cell, `oracle_no_collision` implemented, not run) |

## Mobility sweep (task1, mass x0.2 / friction x0.1 on occluder body, physical collision left ON)

| condition | success | n | episodes | source dir |
|---|---|---|---|---|
| baseline | 30% (6/20) | 20 | 0-19 | `mobility_sweep_task1_n20/` |
| **low_mobility (occluder 5x lighter, 10x lower friction, still collidable)** | **50% (10/20)** | 20 | 0-19 | `mobility_sweep_task1_n20/` |

McNemar: b=2 (baseline-only success), c=6 (low_mobility-only success),
both-succeed=4, both-fail=8, chi2=2.00, **p≈0.157 (not significant at
alpha=0.05, needs chi2>3.84)**. Directionally positive (+20pt, same
magnitude as `no_collision`'s ceiling-style jump was large; this one is
smaller/noisier) and the discordant pairs favor low_mobility 6:2 (3x),
but n=20 is underpowered to confirm at conventional significance. This
is the **realistic, real-robot-buildable analog** of the
`no_collision` finding (a real object the arm can physically push/
displace, e.g. a light cardboard box, vs. `no_collision`'s
non-physical "arm passes through it" idealization) -- baseline here
(30%) matches the noise-floor-confirmed task1 n=20 reproducibility
number exactly, so this is a clean, well-matched pair, not a
baseline-drift artifact.

**Reading**: consistent in direction with the `no_collision`
factorial finding (physical interference/stagnation-upon-contact hurts
success rate; removing or reducing that interference helps) but weaker
in magnitude and not statistically confirmed at n=20. Present as
suggestive/consistent-with, not as a second confirmed result, unless
n is increased.

## 2x2 factorial (visual x physical), task6/task8 -- IN PROGRESS

| task | baseline | no_collision | status |
|---|---|---|---|
| task6 | (re-measuring, see `factorial_task6_n20/`) | in progress | running |
| task8 | (re-measuring, see `factorial_task8_n20/`) | in progress | running |

## Notes for slide-writing

- Every number in this table has a `run_config.json` in its source dir
  (except a few pre-dating that convention: `libero_occluded_oracle_
  task{1,6,8}_n20`, `baseline_reproducibility_check_seeds0_20`,
  `baseline_all50_task1`, `factorial_task1_n20` (retracted)) -- check
  it before trusting depth/condition if reusing a number from here.
- "oracle" and "L=0" both use PRIVILEGED information (true occluder-
  free render) -- present as an upper-bound ceiling, never as a
  deployable method.
- "no_collision" is a diagnostic/counterfactual intervention (disabling
  physical collision is not deployable on a real robot) -- present as
  a benchmark-decomposition finding, not a method result.
