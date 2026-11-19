# Hoki — making frozen vision-language-action policies robust to occlusion and collision

Research code by [21Kevin22](https://github.com/21Kevin22). Work in progress.

Vision-language-action (VLA) policies such as [OpenVLA-OFT](https://openvla-oft.github.io/) and
[π0.5](https://github.com/Physical-Intelligence/openpi) solve many tabletop manipulation tasks, but they fail when the
target is hidden from a camera, or when the arm has to work in tight spaces next to other objects. This repository studies
how to add that robustness **around a frozen policy**, without retraining it: modules that see what the policy cannot,
and a safety layer that keeps the robot from hitting things.

The experiments run in simulation on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and on
[LIBERO-Occ](https://arxiv.org/abs/2606.10862), a LIBERO variant in which objects are hidden by occluders.

## Research threads

| Thread | Where | What it asks |
|---|---|---|
| Occlusion recovery for OpenVLA-OFT | [`occ_vla/thirdparty/openvla-oft/`](occ_vla/thirdparty/openvla-oft/) | Why does OpenVLA-OFT fail under occlusion, and can a mid-layer module that rewrites occluded image tokens recover it? Separates visual occlusion from physical interference with the occluder (e.g. rendering the occluder without its collision). |
| Occlusion-aware π0.5 system | [`occ_vla/src/occ_vla/`](occ_vla/src/occ_vla/) | π0.5 as the control backbone, with a world model (MMaDA-8B) that imagines the arm-free scene, kinematic extrapolation of hidden targets (PKLP), and gated action blending. |
| Hidden objects inside containers | [`occ_vla/next_hidden/`](occ_vla/next_hidden/) (branch `next-hidden-microwave`) | Can a frozen π0.5 retrieve a mug from a closed microwave? Includes a new LIBERO task, a scripted solvability check, and policy-free checks of depth-map signals and of a safety layer at a narrow opening. Pre-registered (`PREREG_next.md`), results in `RESULTS_next.md`. |
| VLM-based replanning | [`vlm_replan_lab/`](vlm_replan_lab/) | Perception, reasoning and replanning loop around a manipulation policy. |
| LLM task planning (inherited) | repository root, [`baselines/`](baselines/) | The DELTA code this repository was forked from: LLM-generated PDDL and 3D-scene-graph task planning. Usage: [`docs/DELTA.md`](docs/DELTA.md). |

Each thread keeps its own notes, scripts and result files in its directory. **Results are preliminary**; numbers should be
cited only from the result files they come from (for example `occ_vla/thirdparty/openvla-oft/NUMBERS_REFERENCE.md`), not
from this overview.

## Getting started

Each thread has its own environment, because the policies need incompatible dependency stacks (OpenVLA-OFT, openpi/JAX
for π0.5, MMaDA). Start from the README of the thread you need:

- OpenVLA-OFT: [`occ_vla/thirdparty/openvla-oft/SETUP.md`](occ_vla/thirdparty/openvla-oft/SETUP.md), [`LIBERO.md`](occ_vla/thirdparty/openvla-oft/LIBERO.md)
- π0.5 system: [`occ_vla/README.md`](occ_vla/README.md)
- Closed-container exploration: `occ_vla/next_hidden/README.md` (on branch `next-hidden-microwave`)
- DELTA planning code: [`docs/DELTA.md`](docs/DELTA.md)

## Credits and license

This repository started as a fork of **DELTA: Decomposed Efficient Long-Term Robot Task Planning using Large Language
Models** (Yuchen Liu, Luigi Palmieri et al., Robert Bosch GmbH; [paper](https://arxiv.org/abs/2404.03275),
[project page](https://delta-llm.github.io/)), released under the GNU Affero General Public License v3.0.
The DELTA code, its [`LICENSE`](LICENSE), copyright holders ([`NOTICE`](NOTICE)) and third-party notices
([`3rd_party_licenses.md`](3rd_party_licenses.md)) are kept unchanged.

Vendored third-party code keeps its own license in its directory, including OpenVLA-OFT
(`occ_vla/thirdparty/openvla-oft/LICENSE`), Discrete Diffusion VLA (`DiscreteDiffusionVLA/LICENSE`) and LIBERO (`LIBERO/`).
