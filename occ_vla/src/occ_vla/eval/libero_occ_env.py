"""LIBERO task env wrapper that injects an OccluderSpec at reset and
reports S_occ alongside the usual LIBERO observation/reward.

Grounded on Lifelong-Robot-Learning/LIBERO (third_party/openpi/third_party/libero,
see scripts/setup_third_party.sh):
- `libero.libero.benchmark.get_benchmark(suite_name)()` returns a
  `Benchmark` (e.g. libero_spatial, libero_10); `.get_task_bddl_file_path(i)`
  and `.get_task_init_states(i)` are what construct a specific task's env.
- `libero.libero.envs.SegmentationRenderEnv(bddl_file_name=..., camera_names=[...])`
  (libero/libero/envs/env_wrapper.py) is `OffScreenRenderEnv` plus
  per-instance segmentation masks (`get_segmentation_instances`) —
  occluder.py's S_occ search needs exactly that, so this wraps
  `SegmentationRenderEnv`, not the plain render-only class.
- The task's target object comes from the *env instance*, not the
  `Task` NamedTuple the benchmark returns: `env.obj_of_interest` (set
  in libero/libero/envs/bddl_base_domain.py from the BDDL problem's
  `obj_of_interest` field) is a list of object names; this wrapper
  occludes the first one, since occluder.py targets a single object.
- Control runs at 20Hz (`control_freq=20` default in ControlEnv) —
  pi0.5's 50Hz action chunks need resampling/subselection before
  `env.step()`, not a 1:1 timestep match.
- `SegmentationRenderEnv.reset()` (not `set_init_state()` alone) is what
  builds `segmentation_id_mapping`/`segmentation_robot_id` — confirmed
  against a live env: skipping it left `segmentation_robot_id` at its
  `None` default and `get_segmentation_instances()` raised a `TypeError`
  trying to add 1 to it. `reset()` below always calls the real
  `.reset()` before `set_init_state()`.
- Upstream compatibility bug, also found by running against a live env:
  `SegmentationRenderEnv.reset()` hardcodes `instance_name == "Panda0"`
  to find the robot's segmentation id, but the installed robosuite
  (1.4.1) names it `"MountedPanda0"` — so `segmentation_robot_id` stays
  `None` even after `reset()`, and the same `TypeError` recurs. Rather
  than edit the vendored file, `_fix_segmentation_robot_id()` below
  patches it after `reset()` if still `None`, by matching any instance
  name containing "Panda" instead of the exact hardcoded string.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from occ_vla.eval.metrics import Difficulty, SoccMetric
from occ_vla.eval.occluder import OccluderPlacer

AGENTVIEW_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"
AGENTVIEW_SEGMENTATION_KEY = "agentview_segmentation_instance"
LIBERO_CONTROL_HZ = 20
# openpi's own LIBERO eval (third_party/openpi/examples/libero/main.py)
# renders at 256 and only then resize_with_pad's down to the policy's
# 224 input -- not a direct 224 render -- so this wrapper renders at the
# same 256 by default and leaves the resize step to the caller.
DEFAULT_CAMERA_RESOLUTION = 256


@dataclass
class LiberoOccEnvConfig:
    benchmark_suite: str  # e.g. "libero_spatial", "libero_10"
    task_id: int
    difficulty: Difficulty
    init_state_idx: int = 0
    camera_resolution: int = DEFAULT_CAMERA_RESOLUTION
    seed: int | None = None
    # False skips OccluderPlacer entirely -- no box is inserted into the
    # scene, so there is nothing for the robot to physically contact.
    # Added after finding (2026-07-15 session) that the placed occluder
    # box is a real collidable MuJoCo body sitting on the camera-target
    # line, on the table, in the robot's own workspace -- pi0.5's
    # gripper was observed repeatedly approaching/resting against the
    # box itself instead of the task's real target object, confounding
    # any conclusion about vision-only occlusion robustness with
    # physical-obstacle and OOD-object-attraction effects. Callers
    # wanting a *clean* visual-occlusion test should mask pixels in the
    # rendered image directly (e.g. via the target's clear/baseline
    # segmentation footprint) instead of relying on this env to inject
    # a physical body.
    place_occluder: bool = True
    # Pixel-space clean occlusion test (2026-07-15 finding, see
    # place_occluder above): blackens the target's clear/baseline
    # segmentation footprint directly in the rendered agentview RGB
    # every step, with zero 3D-scene footprint -- nothing to collide
    # with or mistake for a real object, unlike OccluderPlacer's
    # physical body. Independent of place_occluder; the standard clean
    # config is place_occluder=False, pixel_mask=True. Requires calling
    # capture_clear_baseline() once (after any settle-wait steps) before
    # it takes effect -- see that method's docstring.
    pixel_mask: bool = False
    pixel_mask_dilate_px: int = 0


class LiberoOccEnv:
    """Wraps one LIBERO task's `SegmentationRenderEnv` with an
    S_occ-targeted occluder injected at reset."""

    def __init__(self, config: LiberoOccEnvConfig, libero_root: str):
        self.config = config
        self.libero_root = libero_root  # third_party/openpi/third_party/libero
        self.occluder_placer = OccluderPlacer()
        self.metric = SoccMetric()
        self._env: Any = None
        self._benchmark: Any = None
        self.last_s_occ: float | None = None
        self.target_body_name: str | None = None
        self._target_mask_clear: np.ndarray | None = None
        self._pixel_mask_region: np.ndarray | None = None

    def _build_env(self):
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        libero_pkg = str(Path(self.libero_root))
        if libero_pkg not in sys.path:
            sys.path.insert(0, libero_pkg)
        from libero.libero import benchmark  # noqa: PLC0415
        from libero.libero.envs import SegmentationRenderEnv  # noqa: PLC0415

        self._benchmark = benchmark.get_benchmark(self.config.benchmark_suite)()
        bddl_file = self._benchmark.get_task_bddl_file_path(self.config.task_id)
        self._env = SegmentationRenderEnv(
            bddl_file_name=bddl_file,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=self.config.camera_resolution,
            camera_widths=self.config.camera_resolution,
        )
        if self.config.seed is not None:
            # openpi's eval script notes the seed affects object initial
            # positions even under a fixed init_state -- see
            # examples/libero/main.py's env.seed(seed) call.
            self._env.seed(self.config.seed)

    def _fix_segmentation_robot_id(self) -> None:
        """Work around the "Panda0" vs "MountedPanda0" mismatch (see
        module docstring) without editing the vendored file."""
        if self._env.segmentation_robot_id is not None:
            return
        instance_names = list(self._env.env.model.instances_to_ids.keys())
        for i, instance_name in enumerate(instance_names):
            if "Panda" in instance_name:
                self._env.segmentation_robot_id = i
                return
        raise RuntimeError(f"no Panda-like instance found in {instance_names}; cannot fix segmentation_robot_id")

    def reset(self) -> dict:
        if self._env is None:
            self._build_env()

        # SegmentationRenderEnv.reset() (not just set_init_state) is what
        # builds segmentation_id_mapping / segmentation_robot_id (see
        # libero/libero/envs/env_wrapper.py) — skipping it leaves
        # segmentation_robot_id as None and get_segmentation_instances()
        # crashes. set_init_state() afterwards overrides to this task's
        # specific init state.
        self._env.reset()
        self._fix_segmentation_robot_id()
        init_states = self._benchmark.get_task_init_states(self.config.task_id)
        self._env.set_init_state(init_states[self.config.init_state_idx])
        self.target_body_name = self._env.obj_of_interest[0]
        # A new episode invalidates any previously captured baseline.
        self._target_mask_clear = None
        self._pixel_mask_region = None

        if not self.config.place_occluder:
            self.last_s_occ = None
            baseline_state = self._env.get_sim_state()
            return self._env.regenerate_obs_from_state(baseline_state)

        spec = self.occluder_placer.place(self._env, self.target_body_name, self.config.difficulty)
        self.last_s_occ = spec.achieved_s_occ

        # occluder_placer.search() leaves self._env reset onto the winning
        # trial XML already (see occluder.py::OccluderPlacer.place
        # docstring) — regenerate the observation from the held-fixed
        # init state one more time to hand back a clean obs dict.
        baseline_state = self._env.get_sim_state()
        return self._env.regenerate_obs_from_state(baseline_state)

    def capture_clear_baseline(self, obs: dict) -> None:
        """Snapshot the target's clear/unoccluded segmentation footprint,
        to diff future frames against (CLAUDE.md item 7: clear-baseline
        diff, not live-vs-live intersection -- MuJoCo segmentation is
        single-layer, so a pixel where the arm occludes the target gets
        only the *arm's* id, never both).

        Call this once per episode, after any settle-wait steps (a few
        dummy `step()` calls to let dropped objects finish falling) --
        not right at `reset()` -- so the "clear" footprint reflects the
        object's actual resting pose, not its initial drop position.
        Required before `pixel_mask`, `compute_arm_s_occ`, or
        `compute_total_occ` are used.
        """
        if self._env is None or self.target_body_name is None:
            raise RuntimeError("call reset() first")
        seg_dict = self._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])
        target_mask_raw = seg_dict.get(self.target_body_name)
        if target_mask_raw is None:
            self._target_mask_clear = np.zeros(obs[AGENTVIEW_SEGMENTATION_KEY].shape[:2], dtype=bool)
        else:
            self._target_mask_clear = target_mask_raw.squeeze(-1) != 0

        region = self._target_mask_clear
        if self.config.pixel_mask_dilate_px > 0:
            kernel = np.ones((self.config.pixel_mask_dilate_px, self.config.pixel_mask_dilate_px), np.uint8)
            region = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
        self._pixel_mask_region = region

    def compute_arm_s_occ(self, obs: dict) -> float:
        """Fraction of the target's clear/baseline footprint currently
        occluded specifically by the arm (not any other occluder)."""
        if self._target_mask_clear is None:
            raise RuntimeError("call capture_clear_baseline() first")
        seg_dict = self._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])
        target_mask_now_raw = seg_dict.get(self.target_body_name)
        if target_mask_now_raw is None:
            return 0.0
        target_mask_now = target_mask_now_raw.squeeze(-1) != 0
        arm_mask = seg_dict["robot"].squeeze(-1) != 0
        return self.metric.compute(self._target_mask_clear, (~target_mask_now) & arm_mask)

    def compute_total_occ(self, obs: dict) -> float:
        """Fraction of the target's clear/baseline footprint no longer
        visible right now, regardless of what's blocking it (arm,
        placed scene occluder, or both)."""
        if self._target_mask_clear is None:
            raise RuntimeError("call capture_clear_baseline() first")
        seg_dict = self._env.get_segmentation_instances(obs[AGENTVIEW_SEGMENTATION_KEY])
        target_mask_now_raw = seg_dict.get(self.target_body_name)
        if target_mask_now_raw is None:
            return 0.0
        target_mask_now = target_mask_now_raw.squeeze(-1) != 0
        return self.metric.compute(self._target_mask_clear, ~target_mask_now)

    def step(self, action):
        if self._env is None:
            raise RuntimeError("call reset() first")
        obs, reward, done, info = self._env.step(action)
        if self.config.pixel_mask and self._pixel_mask_region is not None:
            obs[AGENTVIEW_KEY] = obs[AGENTVIEW_KEY].copy()
            obs[AGENTVIEW_KEY][self._pixel_mask_region] = 0
        return obs, reward, done, info
