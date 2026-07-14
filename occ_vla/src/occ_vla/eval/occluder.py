"""Automatic placement of a physically-plausible occluder box between
the segmentation camera and the task's target object.

Grounded on Lifelong-Robot-Learning/LIBERO's env API:
- `ControlEnv.sim` exposes the underlying MuJoCo sim, using the
  mujoco_py-compatible binding robosuite provides — `sim.model.body_name2id`
  / `sim.data.body_xpos` (used throughout LIBERO itself, e.g.
  libero/libero/envs/bddl_base_domain.py:415-512) and, by the same
  convention, `sim.model.camera_name2id` / `sim.data.cam_xpos` for camera
  pose, and `sim.model.get_xml()` for the current scene MJCF.
- `ControlEnv.reset_from_xml_string(xml_string)` reloads the scene from a
  modified XML — the injection point: insert a *static* (no `<joint>`,
  so it doesn't add DOF and `set_state()` with the pre-edit state vector
  stays valid) `<body>` with a box `<geom>`, then reset from the edited
  XML.
- `SegmentationRenderEnv.get_segmentation_instances(seg_image)` (same
  file) returns a `{instance_name: mask}` dict directly — exactly what
  S_occ needs, so `libero_occ_env.py` uses `SegmentationRenderEnv`, not
  the plain `OffScreenRenderEnv`.
- `ControlEnv.get_sim_state()` / `set_state()` / `regenerate_obs_from_state()`
  let the search below hold everything but the occluder fixed across
  trial resets (same object/robot pose each time), so only the occluder
  placement varies.

Placement is a binary search over occluder size (fixed position: 85% of
the way from camera to target, i.e. just in front of the target from
the camera's view), not a closed form — S_occ is measured by rendering,
not computed geometrically, so it accounts for the target's real shape
and the occluder's real silhouette rather than a bounding-box estimate.
Each candidate is also rejected if MuJoCo reports it in contact with
anything at the (otherwise unchanged) initial pose — a cheap,
reset-time-only safety check; it does not guarantee the occluder stays
clear of the robot for the whole episode, since the arm moves.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from occ_vla.eval.metrics import DIFFICULTY_BANDS, Difficulty, SoccMetric

CAMERA_TARGET_FRACTION = 0.85  # how far from camera to target the occluder sits (closer to 1.0 = nearer the target)
MIN_HALF_EXTENT = 0.01  # meters
MAX_HALF_EXTENT = 0.15  # meters
MAX_SEARCH_ITERS = 12


@dataclass
class OccluderSpec:
    size: tuple[float, float, float]  # half-extents, MuJoCo box geom convention
    position: tuple[float, float, float]  # world frame, same frame as body_xpos
    target_difficulty: Difficulty
    achieved_s_occ: float = 0.0  # measured S_occ at the winning trial; set by search()


class OccluderPlacer:
    def __init__(self, metric: SoccMetric | None = None, max_search_iters: int = MAX_SEARCH_ITERS):
        self.metric = metric or SoccMetric()
        self.max_search_iters = max_search_iters

    def build_occluder_xml_fragment(self, spec: OccluderSpec) -> str:
        """MJCF `<body>` snippet for a static box geom at `spec.position`.
        No `<joint>`, deliberately: a free body adds 7 DOF (3 pos + 4
        quat) to qpos, which would break `set_state()` with a
        pre-occluder state vector; a static body adds none."""
        sx, sy, sz = spec.size
        x, y, z = spec.position
        return (
            f'<body name="occluder" pos="{x} {y} {z}">'
            f'<geom name="occluder_geom" type="box" size="{sx} {sy} {sz}" '
            f'rgba="0.3 0.3 0.3 1" group="1" density="500"/>'
            f"</body>"
        )

    def _insert_before_worldbody_close(self, xml: str, fragment: str) -> str:
        marker = "</worldbody>"
        idx = xml.rfind(marker)
        if idx == -1:
            raise ValueError("scene XML has no </worldbody> closing tag")
        return xml[:idx] + fragment + xml[idx:]

    def _target_position(self, env: Any, target_body_name: str) -> np.ndarray:
        domain = env.env  # ControlEnv wraps the BDDL domain env as .env
        return np.array(domain.sim.data.body_xpos[domain.obj_body_id[target_body_name]])

    def _camera_position(self, env: Any, camera_name: str) -> np.ndarray:
        sim = env.env.sim
        return np.array(sim.data.cam_xpos[sim.model.camera_name2id(camera_name)])

    def _target_mask(self, env: Any, target_body_name: str, segmentation_image: np.ndarray) -> np.ndarray:
        return env.get_segmentation_instances(segmentation_image)[target_body_name] != 0

    def _occluder_in_contact(self, env: Any) -> bool:
        """MuJoCo doesn't generate contacts between two bodies that are
        both static (no joint, i.e. welded to the world) — verified
        directly against a fresh `mujoco` MjModel: two overlapping
        jointless spheres give `ncon == 0`, but the same pair with a
        `<freejoint/>` on one gives `ncon == 1`. Since the occluder body
        is deliberately jointless (see build_occluder_xml_fragment), this
        check only ever fires against the robot or a movable object (both
        jointed in LIBERO's MJCF) — which is exactly what matters: the
        occluder resting against a static table/wall isn't a problem, the
        occluder overlapping the robot or a manipulable object is."""
        sim = env.env.sim
        occluder_geom_id = sim.model.geom_name2id("occluder_geom")
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if occluder_geom_id in (contact.geom1, contact.geom2):
                return True
        return False

    def search(
        self,
        env: Any,
        target_body_name: str,
        target_difficulty: Difficulty,
        camera_name: str = "agentview",
    ) -> OccluderSpec:
        target_pos = self._target_position(env, target_body_name)
        camera_pos = self._camera_position(env, camera_name)
        position = tuple(camera_pos + CAMERA_TARGET_FRACTION * (target_pos - camera_pos))

        baseline_state = env.get_sim_state()
        baseline_obs = env.regenerate_obs_from_state(baseline_state)
        target_mask_clear = self._target_mask(env, target_body_name, baseline_obs[f"{camera_name}_segmentation_instance"])
        if not target_mask_clear.any():
            raise RuntimeError(
                f"{target_body_name!r} isn't visible in {camera_name!r} at this init state; "
                "cannot place an occluder against it"
            )

        base_xml = env.sim.model.get_xml()
        band_lo, band_hi = DIFFICULTY_BANDS[target_difficulty]
        lo, hi = MIN_HALF_EXTENT, MAX_HALF_EXTENT
        last_s_occ = None

        for _ in range(self.max_search_iters):
            half_extent = (lo + hi) / 2
            spec = OccluderSpec(size=(half_extent,) * 3, position=position, target_difficulty=target_difficulty)
            trial_xml = self._insert_before_worldbody_close(base_xml, self.build_occluder_xml_fragment(spec))
            env.reset_from_xml_string(trial_xml)
            trial_obs = env.regenerate_obs_from_state(baseline_state)

            if self._occluder_in_contact(env):
                hi = half_extent  # shrink: too big, touching something at rest pose
                continue

            target_mask_with_occluder = self._target_mask(
                env, target_body_name, trial_obs[f"{camera_name}_segmentation_instance"]
            )
            occluded_pixels = target_mask_clear & ~target_mask_with_occluder
            last_s_occ = self.metric.compute(target_mask_clear, occluded_pixels)

            if band_lo <= last_s_occ < band_hi:
                spec.achieved_s_occ = last_s_occ
                return spec
            if last_s_occ < band_lo:
                lo = half_extent  # not occluding enough: grow
            else:
                hi = half_extent  # occluding too much: shrink

        raise RuntimeError(
            f"occluder search did not converge to the {target_difficulty.value} band "
            f"(S_occ={band_lo}-{band_hi}) after {self.max_search_iters} iterations; "
            f"last S_occ={last_s_occ}"
        )

    def place(
        self, env: Any, target_body_name: str, target_difficulty: Difficulty, camera_name: str = "agentview"
    ) -> OccluderSpec:
        """search()'s last trial reset already leaves `env` in the
        winning configuration (the loop returns immediately on success),
        so this just runs the search — kept as a separate method since
        `libero_occ_env.py` calls it for its name, not its side effect."""
        return self.search(env, target_body_name, target_difficulty, camera_name=camera_name)
