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

from occ_vla.eval.metrics import Difficulty, SoccMetric
from occ_vla.eval.occluder import OccluderPlacer

AGENTVIEW_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"
AGENTVIEW_SEGMENTATION_KEY = "agentview_segmentation_instance"
LIBERO_CONTROL_HZ = 20


@dataclass
class LiberoOccEnvConfig:
    benchmark_suite: str  # e.g. "libero_spatial", "libero_10"
    task_id: int
    difficulty: Difficulty


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
            bddl_file_name=bddl_file, camera_names=["agentview", "robot0_eye_in_hand"]
        )

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
        self._env.set_init_state(init_states[0])

        target_body_name = self._env.obj_of_interest[0]
        spec = self.occluder_placer.place(self._env, target_body_name, self.config.difficulty)
        self.last_s_occ = spec.achieved_s_occ

        # occluder_placer.search() leaves self._env reset onto the winning
        # trial XML already (see occluder.py::OccluderPlacer.place
        # docstring) — regenerate the observation from the held-fixed
        # init state one more time to hand back a clean obs dict.
        baseline_state = self._env.get_sim_state()
        return self._env.regenerate_obs_from_state(baseline_state)

    def step(self, action):
        if self._env is None:
            raise RuntimeError("call reset() first")
        return self._env.step(action)
