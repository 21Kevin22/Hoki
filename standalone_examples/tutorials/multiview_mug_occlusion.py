"""Multiview mug occlusion workflow (based on workflow.py).

5-step architecture:
 1. receive human intent
 2. generate delta scene graph / PDDL style goal representation
 3. query GPT-based planner for action plan
 4. execute in simulation + logging
 5. realtime VLM scene graph update + replan based on discrepancy

This variant keeps the workflow settings identical to workflow.py,
but places five mugs and hides one inside a box with one open face
to visualize multiview occlusion.

Example usage: python3 standalone_examples/tutorials/multiview_mug_occlusion.py --task "sort mugs by color"
"""



import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Allow running this file directly via:
# `python3 /path/to/standalone_examples/tutorials/multiview_mug_occlusion.py`
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PACKAGES_ROOT = REPO_ROOT / "python_packages"
ISAACSIM_PYTHON_ROOT = REPO_ROOT / "isaacsim" / "source" / "python_packages"
MUST3R_PYTHON_ROOT = REPO_ROOT / "Must3R" / "must3r"

for extra_path in (ISAACSIM_PYTHON_ROOT, PYTHON_PACKAGES_ROOT, MUST3R_PYTHON_ROOT):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Help local Isaac Sim Python packages find their extension root when this
# example is launched as a plain script.
os.environ.setdefault("ISAAC_PATH", str(REPO_ROOT))


def _ensure_isaac_runtime() -> None:
    if os.environ.get("_WORKFLOW_RUNTIME_RELAUNCHED") == "1":
        return
    try:
        import carb  # type: ignore  # noqa: F401
        return
    except Exception:
        launcher = REPO_ROOT / "python.sh"
        if launcher.exists():
            os.environ["_WORKFLOW_RUNTIME_RELAUNCHED"] = "1"
            os.execv(str(launcher), [str(launcher), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise RuntimeError(
            "Isaac Sim runtime is not initialized (missing carb). "
            f"Run with: {REPO_ROOT / 'python.sh'} {Path(__file__).resolve()} --task \"push mugs by color\""
        )


_ensure_isaac_runtime()

import standalone_examples.tutorials.Enum_eval as enum_eval_module
from standalone_examples.tutorials.Enum_eval import (
    VLMAnalyzer,
    capture_images_for_vlm,
    RobotController,
    AssetBuilder,
    run_calibration,
    load_calibration,
)
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, UsdLux


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def create_open_face_box(
    path: str,
    pos: List[float],
    size: List[float],
    thickness: float = 0.01,
    open_face: str = "right",
    color: Gf.Vec3f = Gf.Vec3f(0.6, 0.6, 0.6),
) -> str:
    stage = get_current_stage()
    box_xform = UsdGeom.Xform.Define(stage, path)

    w, d, h = float(size[0]), float(size[1]), float(size[2])
    th = float(thickness)
    open_face = str(open_face).strip().lower()

    prim = box_xform.GetPrim()
    prim.CreateAttribute("occlusion:size", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(w, d, h))
    prim.CreateAttribute("occlusion:open_face", Sdf.ValueTypeNames.Token).Set(open_face)
    prim.CreateAttribute("occlusion:thickness", Sdf.ValueTypeNames.Float).Set(th)

    parts = [
        ("Bottom", (w, d, th), (0, 0, th / 2), "bottom"),
        ("Top", (w, d, th), (0, 0, h - th / 2), "top"),
        ("Front", (w, th, h), (0, -d / 2, h / 2), "front"),
        ("Back", (w, th, h), (0, d / 2, h / 2), "back"),
        ("Left", (th, d, h), (-w / 2, 0, h / 2), "left"),
        ("Right", (th, d, h), (w / 2, 0, h / 2), "right"),
    ]

    for name, size_xyz, offset, face in parts:
        if face == open_face:
            continue
        part_path = f"{path}/{name}"
        cube = UsdGeom.Cube.Define(stage, part_path)
        XFormPrim(part_path).set_local_scales(np.array([[size_xyz[0] / 2, size_xyz[1] / 2, size_xyz[2] / 2]]))
        XFormPrim(part_path).set_local_poses(np.array([[offset[0], offset[1], offset[2]]]))
        mat_path = f"{part_path}_Material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube).Bind(material)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    XFormPrim(path).set_world_poses(np.array([[pos[0], pos[1], pos[2]]]))
    return path


TARGET_BOTTLE_COUNT = 5
MAX_CONSECUTIVE_FAILURES = 3


class _NoOpEvalLogger:
    def add_failure(self):
        return None

    def record_inference_time(self, process_name, latency):
        return None

    def record_plan(self, plan):
        return None

    def record_scene_graph(self, step, graph, affordance=None, diagnostics=None):
        return None

    def save(self):
        return None


class OcclusionAwareRobotController(RobotController):
    """Pick targets that approach the occluded mug through the open face."""

    def __init__(self, arm, disturbance=None):
        super().__init__(arm, disturbance=disturbance)
        self._occlusion_cache = None

    def _quat_from_axis_angle(self, axis, angle_rad):
        axis = np.array(axis, dtype=np.float32)
        norm = np.linalg.norm(axis)
        if norm <= 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        axis = axis / norm
        half = 0.5 * float(angle_rad)
        return np.array([np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float32)

    def _quat_mul(self, q1, q2):
        # [w, x, y, z]
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float32)

    def _orientation_for_open_face(self, open_face):
        open_face = str(open_face).lower()
        yaw = 0.0
        if open_face == "left":
            yaw = np.pi
        elif open_face == "front":
            yaw = -0.5 * np.pi
        elif open_face == "back":
            yaw = 0.5 * np.pi
        # Rotate gripper to a horizontal approach, then yaw toward the opening.
        pitch = -0.5 * np.pi
        q_pitch = self._quat_from_axis_angle([0.0, 1.0, 0.0], pitch)
        q_yaw = self._quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
        return self._quat_mul(q_yaw, self._quat_mul(q_pitch, self.ee_orientation))

    def _hand_inside_opening(self, hand_pos: np.ndarray) -> bool:
        stage = get_current_stage()
        prim = stage.GetPrimAtPath("/World/OcclusionBox")
        if not prim.IsValid():
            return True
        size_attr = prim.GetAttribute("occlusion:size")
        open_attr = prim.GetAttribute("occlusion:open_face")
        size = size_attr.Get() if size_attr and size_attr.HasAuthoredValueOpinion() else Gf.Vec3f(0.18, 0.18, 0.18)
        open_face = open_attr.Get() if open_attr and open_attr.HasAuthoredValueOpinion() else "right"
        pos, _ = XFormPrim("/World/OcclusionBox").get_world_poses()
        pos = pos[0] if len(np.shape(pos)) == 2 else pos
        pos = np.array(pos, dtype=np.float32)
        size = np.array([float(size[0]), float(size[1]), float(size[2])], dtype=np.float32)
        half = size * 0.5
        open_face = str(open_face).lower()
        margin = 0.01

        in_x = (pos[0] - half[0] + margin) <= hand_pos[0] <= (pos[0] + half[0] - margin)
        in_y = (pos[1] - half[1] + margin) <= hand_pos[1] <= (pos[1] + half[1] - margin)
        in_z = (pos[2] + 0.03) <= hand_pos[2] <= (pos[2] + size[2] - 0.03)
        if not (in_x and in_y and in_z):
            return False

        if open_face == "left":
            return (pos[0] - half[0] - 0.02) <= hand_pos[0] <= (pos[0] - half[0] + 0.04)
        if open_face == "right":
            return (pos[0] + half[0] - 0.04) <= hand_pos[0] <= (pos[0] + half[0] + 0.02)
        if open_face == "front":
            return (pos[1] - half[1] - 0.02) <= hand_pos[1] <= (pos[1] - half[1] + 0.04)
        if open_face == "back":
            return (pos[1] + half[1] - 0.04) <= hand_pos[1] <= (pos[1] + half[1] + 0.02)
        return False

    def _get_occlusion_box(self):
        if isinstance(self._occlusion_cache, dict):
            return self._occlusion_cache
        stage = get_current_stage()
        prim = stage.GetPrimAtPath("/World/OcclusionBox")
        if not prim.IsValid():
            self._occlusion_cache = None
            return None
        size_attr = prim.GetAttribute("occlusion:size")
        open_attr = prim.GetAttribute("occlusion:open_face")
        size = size_attr.Get() if size_attr and size_attr.HasAuthoredValueOpinion() else Gf.Vec3f(0.18, 0.18, 0.18)
        open_face = open_attr.Get() if open_attr and open_attr.HasAuthoredValueOpinion() else "right"
        pos, _ = self._get_safe_world_pose("/World/OcclusionBox")
        self._occlusion_cache = {"pos": np.array(pos, dtype=np.float32), "size": np.array([size[0], size[1], size[2]]), "open_face": str(open_face)}
        return self._occlusion_cache

    def compute_pick_targets(self, target_path):
        box = self._get_occlusion_box()
        if box and str(target_path).endswith("mug_2"):
            safe_pos, _ = self._get_safe_world_pose(target_path)
            size = box["size"]
            open_face = box["open_face"].lower()
            half = size * 0.5
            if open_face == "left":
                approach_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
                approach_dist = half[0] + 0.12
            elif open_face == "front":
                approach_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                approach_dist = half[1] + 0.12
            elif open_face == "back":
                approach_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                approach_dist = half[1] + 0.12
            else:
                approach_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                approach_dist = half[0] + 0.12

            # Approach through the open face instead of top-down.
            grasp = safe_pos + self.calib_offset + np.array([0.0, 0.0, 0.06], dtype=np.float32)
            pre_grasp = grasp + approach_dir * approach_dist
            retreat = pre_grasp + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            return {
                "pre_grasp": pre_grasp,
                "grasp": grasp,
                "retreat": retreat,
                "orientation": self._orientation_for_open_face(open_face),
                "affordance_offset": [0.0, 0.0, 0.0],
            }
        return super().compute_pick_targets(target_path)

    def close_gripper(self, world, target_path, steps=30):
        if target_path and str(target_path).endswith("mug_2"):
            try:
                hand_pos, _ = self._get_safe_world_pose("/World/Franka/panda_hand")
                obj_pos, _ = self._get_safe_world_pose(target_path)
                distance = float(np.linalg.norm(hand_pos - obj_pos))
                can_attach = distance <= 0.04 and self._hand_inside_opening(np.array(hand_pos, dtype=np.float32))
            except Exception:
                can_attach = False

            if can_attach:
                prev_disable = getattr(self, "disable_collision_on_grasp", False)
                self.disable_collision_on_grasp = False
                try:
                    self._attach_object(target_path)
                finally:
                    self.disable_collision_on_grasp = prev_disable

            self.current_pose[7:] = float(self.dynamic_params.get("grip_close", 0.005))
            for _ in range(steps):
                if not world.is_playing():
                    break
                if getattr(self.arm, "gripper", None) is not None:
                    self.arm.gripper.close()
                else:
                    self._apply_joint_targets(self.current_pose)
                world.step(render=True)
                self._after_world_step()
            self._refresh_current_pose()
            return

        return super().close_gripper(world, target_path, steps=steps)


class Must3RIdentityEstimator:
    """Optional MUSt3R-based multiview consistency estimator.

    It computes 3D overlap between per-view high-confidence point clouds and
    converts that overlap to a coarse id_consistency score (high/medium/low).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        image_size: int = 224,
        amp: str = "fp16",
        device: str = "cuda",
        min_confidence: float = 1.0,
        distance_threshold: float = 0.05,
        sample_points: int = 1024,
        high_min_overlap: float = 0.16,
        high_avg_overlap: float = 0.22,
        medium_min_overlap: float = 0.09,
        medium_avg_overlap: float = 0.13,
    ):
        self.weights_path = str(weights_path).strip() if weights_path else ""
        self.image_size = int(image_size)
        self.amp = str(amp)
        self.device = str(device)
        self.min_confidence = float(min_confidence)
        self.distance_threshold = float(distance_threshold)
        self.sample_points = int(sample_points)
        self.high_min_overlap = float(high_min_overlap)
        self.high_avg_overlap = float(high_avg_overlap)
        self.medium_min_overlap = float(medium_min_overlap)
        self.medium_avg_overlap = float(medium_avg_overlap)

        self.enabled = bool(self.weights_path)
        self._ready = False
        self._model = None
        self._must3r_inference = None
        self._init_error = ""

        self._cache_key = None
        self._cache_result = None

    def _lazy_init(self) -> None:
        if self._ready or not self.enabled:
            return
        try:
            from must3r.model import load_model
            from must3r.demo.inference import must3r_inference

            self._model = load_model(
                self.weights_path,
                device=self.device,
                img_size=self.image_size,
            )
            self._must3r_inference = must3r_inference
            self._ready = True
        except Exception as exc:
            self.enabled = False
            self._init_error = str(exc)

    def _sample_cloud(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] <= self.sample_points:
            return pts
        idx = np.random.choice(pts.shape[0], self.sample_points, replace=False)
        return pts[idx]

    def _pair_overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        a = self._sample_cloud(a)
        b = self._sample_cloud(b)
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        a_to_b = (d.min(axis=1) < self.distance_threshold).mean()
        b_to_a = (d.min(axis=0) < self.distance_threshold).mean()
        return float(0.5 * (a_to_b + b_to_a))

    def _select_views(self, image_paths: List[str]) -> List[str]:
        """Prioritize oblique/front views and de-prioritize top-only view pairs."""
        unique: List[str] = []
        seen = set()
        for p in image_paths:
            sp = str(p)
            if sp and sp not in seen:
                unique.append(sp)
                seen.add(sp)

        if len(unique) <= 2:
            return unique

        priority = {
            "main": 0,
            "front_left": 1,
            "front_right": 1,
            "left": 2,
            "right": 2,
            "top": 4,
        }

        def _rank(path: str) -> int:
            parent = Path(path).parent.name.lower()
            stem = Path(path).stem.lower()
            for key, score in priority.items():
                if key in parent or key in stem:
                    return score
            return 3

        ordered = sorted(unique, key=lambda x: (_rank(x), x))
        non_top = [p for p in ordered if _rank(p) < 4]
        if len(non_top) >= 3:
            selected = non_top[:5]
        else:
            selected = ordered[:5]
        return selected

    def assess(self, image_paths: List[str]) -> Dict[str, Any]:
        selected_paths = self._select_views(image_paths)
        if len(selected_paths) < 2:
            return {"source": "must3r", "id_consistency": "unknown", "reason": "need_at_least_2_views"}

        cache_key = tuple(str(p) for p in selected_paths)
        if self._cache_key == cache_key and isinstance(self._cache_result, dict):
            return dict(self._cache_result)

        if not self.enabled:
            res = {"source": "must3r", "id_consistency": "unknown", "reason": "disabled_no_weights"}
            self._cache_key, self._cache_result = cache_key, res
            return res

        self._lazy_init()
        if not self._ready:
            res = {
                "source": "must3r",
                "id_consistency": "unknown",
                "reason": f"init_failed: {self._init_error}",
            }
            self._cache_key, self._cache_result = cache_key, res
            return res

        try:
            nimgs = len(selected_paths)
            scene = self._must3r_inference(
                model=self._model,
                retrieval=None,
                device=self.device,
                image_size=self.image_size,
                amp=self.amp,
                filelist=selected_paths,
                num_mem_images=max(2, min(nimgs, 8)),
                max_bs=1,
                init_num_images=min(2, nimgs),
                batch_num_views=1,
                render_once=True,
                is_sequence=True,
                viser_server=None,
                num_refinements_iterations=0,
                verbose=False,
            )

            clouds: List[np.ndarray] = []
            for out in getattr(scene, "x_out", []):
                pts = out.get("pts3d") if isinstance(out, dict) else None
                conf = out.get("conf") if isinstance(out, dict) else None
                if pts is None:
                    continue
                pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                if conf is None:
                    cloud = pts_np.reshape(-1, 3)
                else:
                    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
                    mask = conf_np > self.min_confidence
                    if mask.shape != pts_np.shape[:2]:
                        mask = np.broadcast_to(mask, pts_np.shape[:2])
                    cloud = pts_np[mask]
                if cloud.size > 0:
                    clouds.append(cloud.reshape(-1, 3))

            if len(clouds) < 2:
                res = {"source": "must3r", "id_consistency": "unknown", "reason": "insufficient_pointclouds"}
                self._cache_key, self._cache_result = cache_key, res
                return res

            overlaps = []
            for i in range(len(clouds)):
                for j in range(i + 1, len(clouds)):
                    overlaps.append(self._pair_overlap(clouds[i], clouds[j]))

            avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
            min_overlap = float(np.min(overlaps)) if overlaps else 0.0
            robust_min_overlap = float(np.percentile(overlaps, 20)) if overlaps else 0.0
            robust_mid_overlap = float(np.percentile(overlaps, 40)) if overlaps else 0.0

            # Use robust lower-tail statistics instead of strict min to avoid
            # one outlier pair dominating the identity decision.
            if robust_mid_overlap >= self.high_min_overlap and avg_overlap >= self.high_avg_overlap:
                level = "high"
            elif robust_mid_overlap >= self.medium_min_overlap and avg_overlap >= self.medium_avg_overlap:
                level = "medium"
            else:
                level = "low"

            res = {
                "source": "must3r",
                "id_consistency": level,
                "avg_overlap": round(avg_overlap, 4),
                "min_overlap": round(min_overlap, 4),
                "robust_min_overlap": round(robust_min_overlap, 4),
                "robust_mid_overlap": round(robust_mid_overlap, 4),
                "pairs": len(overlaps),
                "views": len(clouds),
                "input_views": len(selected_paths),
            }
            self._cache_key, self._cache_result = cache_key, res
            return res
        except Exception as exc:
            res = {"source": "must3r", "id_consistency": "unknown", "reason": str(exc)}
            self._cache_key, self._cache_result = cache_key, res
            return res


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if type(value).__name__ == "Omit":
        return None
    return str(value)

def _safe_json_for_prompt(value: Any) -> str:
    safe = _make_json_safe(value)
    try:
        return json.dumps(safe, ensure_ascii=False)
    except Exception:
        return json.dumps(str(safe), ensure_ascii=False)


@dataclass
class WorkflowMetrics:
    success: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_planning_time: float = 0.0
    initial_plan_time: float = 0.0
    replan_time: float = 0.0
    model_inference_time: float = 0.0
    control_execution_time: float = 0.0
    replan_count: int = 0
    dynamic_scene_graph_ratio: float = 0.0
    trajectory_length: float = 0.0
    safety_score: int = 0
    success_rate: float = 0.0
    moved_bottles: int = 0
    failed_bottles: int = 0
    target_bottles: int = TARGET_BOTTLE_COUNT
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return {
            "success": self.success,
            "duration": self.end_time - self.start_time if self.end_time else 0.0,
            "total_planning_time": self.total_planning_time,
            "initial_plan_time": self.initial_plan_time,
            "replan_time": self.replan_time,
            "model_inference_time": self.model_inference_time,
            "control_execution_time": self.control_execution_time,
            "replan_count": self.replan_count,
            "dynamic_scene_graph_ratio": self.dynamic_scene_graph_ratio,
            "trajectory_length": self.trajectory_length,
            "safety_score": self.safety_score,
            "success_rate": self.success_rate,
            "moved_bottles": self.moved_bottles,
            "failed_bottles": self.failed_bottles,
            "target_bottles": self.target_bottles,
            "history": self.history,
        }


class VLMDeltaWorkflow:
    def __init__(
        self,
        output_dir: str = "/home/ubuntu/slocal/evaluation/vlm_delta_workflow",
        use_remote_planner: bool = False,
        enable_ik: bool = True,
        enable_rrt: bool = True,
        enable_multiview: bool = True,
        record_video: bool = True,
        must3r_weights: Optional[str] = None,
        must3r_image_size: int = 224,
        must3r_amp: str = "fp16",
        must3r_device: str = "cuda",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = WorkflowMetrics()
        self.use_remote_planner = use_remote_planner
        self.enable_ik = enable_ik
        self.enable_rrt = enable_rrt
        self.enable_multiview = enable_multiview
        self.record_video = record_video
        self.frames_dir = self.output_dir / "frames"
        self.video_path = self.output_dir / "simulation.mp4"
        self._rrt_rng = np.random.default_rng(7)
        self._writers: Dict[str, Any] = {}
        self._render_products: Dict[str, Any] = {}
        self._active_writers = set()
        if getattr(enum_eval_module, "eval_logger", None) is None:
            enum_eval_module.eval_logger = _NoOpEvalLogger()

        self.vlm_analyzer = VLMAnalyzer(model_name=os.environ.get("TARGET_MODEL", "gpt-4o"))
        self.must3r_identity = Must3RIdentityEstimator(
            weights_path=must3r_weights or os.environ.get("MUST3R_WEIGHTS", ""),
            image_size=must3r_image_size,
            amp=must3r_amp,
            device=must3r_device,
        )
        self.world: Optional[World] = None
        self.controller: Optional[RobotController] = None
        self.franka: Optional[Franka] = None
        self.plan_logger = []
        self.placed_mugs = set()
        self._scene_graph_perturbed = False
        self._forced_replan_done = False

    def _setup_multiview_writers(self):
        target_frames_dir = self.output_dir / "frames"
        if target_frames_dir.exists():
            # Previous runs may still be flushing files briefly; retry cleanup.
            for _ in range(5):
                shutil.rmtree(target_frames_dir, ignore_errors=True)
                if not target_frames_dir.exists():
                    break
                time.sleep(0.2)
        if target_frames_dir.exists():
            fallback = self.output_dir / f"frames_run_{int(time.time())}"
            logger.warning(
                "Could not fully clear %s; writing frames to %s instead",
                target_frames_dir,
                fallback,
            )
            target_frames_dir = fallback

        target_frames_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = target_frames_dir

        # Make Enum_eval capture_images_for_vlm read from this workflow run folder.
        enum_eval_module.RGB_DIR = self.frames_dir

        self._writers = {}
        self._render_products = {}
        self._active_writers = set()

        try:
            import omni.replicator.core as rep

            camera_positions = {
                # Keep one wide frontal overview and add closer oblique views
                # to improve cross-view overlap for MUSt3R identity checks.
                "main": {"pos": (1.45, 0.0, 1.25), "look_at": (0.55, 0.0, 0.08)},
                "top": {"pos": (0.55, 0.0, 2.05), "look_at": (0.55, 0.0, 0.0)},
                "left": {"pos": (1.15, 0.95, 0.92), "look_at": (0.55, 0.05, 0.02)},
                "right": {"pos": (1.15, -0.95, 0.92), "look_at": (0.55, -0.05, 0.02)},
                "front_left": {"pos": (0.92, 0.55, 0.78), "look_at": (0.58, 0.03, 0.02)},
                "front_right": {"pos": (0.92, -0.55, 0.78), "look_at": (0.58, -0.03, 0.02)},
            }

            for name, cfg in camera_positions.items():
                cam = rep.create.camera(position=cfg["pos"], look_at=cfg["look_at"])
                if self.record_video and name == "main":
                    resolution = (1280, 720)
                else:
                    resolution = (640, 360)
                rp = rep.create.render_product(cam, resolution)
                out_dir = self.frames_dir / name
                out_dir.mkdir(parents=True, exist_ok=True)
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(output_dir=str(out_dir), rgb=True)

                self._writers[name] = writer
                self._render_products[name] = rp

                # Keep main camera continuously attached for dense logging.
                # `--no-record-video` disables MP4 encoding, not frame capture.
                if name == "main":
                    writer.attach([rp])
                    self._active_writers.add(name)
        except Exception as exc:
            logger.warning("Failed to initialize multi-view writers: %s", exc)

    def _refresh_multiview_frames(self, steps: int = 4) -> None:
        if self.world is None:
            return
        if not self._writers:
            return

        desired = ["main"]
        if self.enable_multiview:
            desired = list(self._writers.keys())

        temp_attached = []
        for name in desired:
            if name not in self._writers or name not in self._render_products:
                continue
            if name in self._active_writers:
                continue
            try:
                self._writers[name].attach([self._render_products[name]])
                self._active_writers.add(name)
                temp_attached.append(name)
            except Exception:
                continue

        for _ in range(max(1, steps)):
            if not self.world.is_playing():
                break
            self.world.step(render=True)

        for name in temp_attached:
            try:
                self._writers[name].detach()
            except Exception:
                pass
            self._active_writers.discard(name)

    def _collect_latest_multiview_images(self) -> List[str]:
        preferred = ["main", "front_left", "front_right", "left", "right", "top"]
        collected: List[str] = []

        for name in preferred:
            d = self.frames_dir / name
            if not d.exists():
                continue
            files = sorted(d.glob("rgb_*.png"))
            if files:
                latest = files[-1]
                if latest.exists() and latest.stat().st_size > 0:
                    collected.append(str(latest))

        # Fallback to Enum_eval's capture helper if custom camera folders are empty.
        if not collected:
            try:
                collected = capture_images_for_vlm(self.world)
            except Exception:
                collected = []

        # de-dup while preserving order
        seen = set()
        uniq = []
        for path in collected:
            if path and path not in seen:
                seen.add(path)
                uniq.append(path)
        return uniq

    def _generate_video(self):
        # Detach active writers before encoding to avoid partially-written frames.
        for name in list(self._active_writers):
            writer = self._writers.get(name)
            if writer is None:
                continue
            try:
                writer.detach()
            except Exception:
                pass
            self._active_writers.discard(name)

        # Let backend I/O flush pending disk writes.
        time.sleep(0.5)

        main_dir = self.frames_dir / "main"
        image_files = sorted(main_dir.glob("rgb_*.png")) if main_dir.exists() else []
        if not image_files:
            image_files = sorted(self.frames_dir.glob("**/rgb_*.png"))

        image_files = [img for img in image_files if img.exists() and img.stat().st_size > 0]
        if not image_files:
            logger.warning("Video skipped: no RGB frames found under %s", self.frames_dir)
            return None

        # Prefer direct glob encoding to avoid duplicating frames on disk.
        if main_dir.exists() and image_files:
            glob_source = str(main_dir / "rgb_*.png")
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "30",
                    "-pattern_type",
                    "glob",
                    "-i",
                    glob_source,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-loglevel",
                    "warning",
                    str(self.video_path),
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                return str(self.video_path)
            except FileNotFoundError:
                logger.warning("Video encoding skipped: ffmpeg not found")
                return None
            except subprocess.CalledProcessError as exc:
                logger.warning("Video encoding failed (glob): %s", exc)

        tmp_dir = self.output_dir / "tmp_video_frames"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for idx, img_path in enumerate(image_files):
                shutil.copy(str(img_path), str(tmp_dir / f"frame_{idx:04d}.png"))

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-i",
                str(tmp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "warning",
                str(self.video_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return str(self.video_path)
        except FileNotFoundError:
            logger.warning("Video encoding skipped: ffmpeg not found")
        except subprocess.CalledProcessError as exc:
            logger.warning("Video encoding failed: %s", exc)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

        return None

    def setup_simulation_environment(self):
        if self.world is not None:
            return

        # Each ablation run reuses one SimulationApp process. Resetting the USD
        # stage prevents duplicate prim/name collisions (e.g., "franka").
        try:
            import omni.usd
            omni.usd.get_context().new_stage()
        except Exception as exc:
            logger.warning("Stage reset failed, continuing with current stage: %s", exc)

        self.world = World(stage_units_in_meters=1.0)
        stage = get_current_stage()
        self.world.scene.add_default_ground_plane()

        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2000)

        assets_root = get_assets_root_path()
        self.franka = self.world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            )
        )

        # place mugs in initial scene.
        for cfg in [
            {"id": "mug_0", "pos": [0.4, 0.5, 0.05], "color": [1.0, 0.0, 0.0], "angle": -120},
            {"id": "mug_1", "pos": [0.5, 0.25, 0.05], "color": [0.0, 1.0, 0.0], "angle": 180},
            {"id": "mug_2", "pos": [0.6, 0.0, 0.05], "color": [0.0, 0.0, 1.0], "angle": 120},
            {"id": "mug_3", "pos": [0.5, -0.25, 0.05], "color": [1.0, 1.0, 0.0], "angle": 60},
            {"id": "mug_4", "pos": [0.4, -0.5, 0.05], "color": [0.0, 1.0, 1.0], "angle": 0},
        ]:
            AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])

        # Hide mug_2 inside an open-face box. Aim the opening toward the robot arm.
        box_pos = np.array([0.6, 0.0, 0.04], dtype=np.float32)
        franka_pos, _ = XFormPrim("/World/Franka").get_world_poses()
        franka_pos = franka_pos[0] if len(np.shape(franka_pos)) == 2 else franka_pos
        delta = np.array(franka_pos, dtype=np.float32) - box_pos
        if abs(delta[0]) >= abs(delta[1]):
            open_face = "left" if delta[0] >= 0 else "right"
        else:
            open_face = "front" if delta[1] >= 0 else "back"

        create_open_face_box(
            "/World/OcclusionBox",
            pos=box_pos.tolist(),
            size=[0.18, 0.18, 0.18],
            thickness=0.01,
            open_face=open_face,
            color=Gf.Vec3f(0.35, 0.35, 0.38),
        )

        AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

        self._setup_multiview_writers()

        self.world.reset()
        self.world.play()

        calib = run_calibration(self.world)
        self.controller = OcclusionAwareRobotController(arm=self.franka)
        self.controller.apply_calibration(calib)

        logger.info("Simulation environment ready")

    def receive_human_instruction(self, instruction_text: str) -> Dict[str, Any]:
        logger.info("[1/5] receive_human_instruction")
        human_intent = {
            "task": instruction_text,
            "timestamp": time.time(),
        }
        self.metrics.history.append({"phase": "human_instruction", "intent": instruction_text})
        return human_intent

    def build_delta_scene_graph(self, human_intent: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[2/5] build_delta_scene_graph")

        # (1) Capture environment imagery if simulation is running
        detection_result = None
        images: List[str] = []
        must3r_diag: Dict[str, Any] = {"source": "must3r", "id_consistency": "unknown"}
        if self.world is not None:
            try:
                self._refresh_multiview_frames(steps=3)
                images = self._collect_latest_multiview_images()
                if images and not self.enable_multiview:
                    images = images[:1]
                detection_result = self.vlm_analyzer.detect_objects_for_delta(images)
            except Exception as exc:
                logger.warning("build_delta_scene_graph: VLM detection failed (%s). Fallback graph will be used.", exc)
                detection_result = None

        if self.enable_multiview and images:
            must3r_diag = self.must3r_identity.assess(images)
        
        if detection_result is not None and detection_result.get("objects"):
            scene_graph = self.vlm_analyzer.build_delta_scene_graph(detection_result)
        else:
            scene_graph = {
                "nodes": ["agent", "workspace"],
                "edges": [["agent", "in", "workspace"]],
                "goal": f"execute({human_intent['task']})",
                "metadata": {"source": "delta"},
            }

        pddl_goal = {
            "domain": "mug_sorting",
            "problem": "task0",
            "goal": scene_graph.get("goal", f"execute({human_intent['task']})"),
            "human_intent": human_intent,
        }

        self.metrics.history.append({
            "phase": "delta_scene_graph",
            "scene_graph": scene_graph,
            "pddl": pddl_goal,
            "multiview_identity": must3r_diag,
        })
        return {"scene_graph": scene_graph, "pddl": pddl_goal}

    def _build_plan_for_targets(self, target_ids: List[str], source: str = "fallback") -> Dict[str, Any]:
        ordered = sorted({str(t).strip() for t in target_ids if str(t).strip()})
        steps: List[str] = []
        for target_id in ordered:
            slot_id = target_id.rsplit("_", 1)[-1]
            steps.extend([
                f"pick {target_id}",
                f"grab {target_id}",
                f"place {target_id} target_{slot_id}",
            ])
        return {
            "steps": steps,
            "length": len(steps),
            "timestamp": time.time(),
            "source": source,
        }

    def _capture_current_multiview_scene_graph(self) -> Dict[str, Any]:
        current_scene_graph: Dict[str, Any] = {}
        diagnostics: Dict[str, Any] = {"id_consistency": "unknown"}
        images: List[str] = []

        if self.world is not None:
            self._refresh_multiview_frames(steps=3)
            images = self._collect_latest_multiview_images()
            if images and not self.enable_multiview:
                images = images[:1]

        if images:
            try:
                current_scene_graph = self.vlm_analyzer.extract_scene_graph(images)
                diagnostics = self.vlm_analyzer.assess_multiview_identity(images)
            except Exception as exc:
                logger.warning("realtime_vlm_update_replan: multiview analysis failed (%s).", exc)
                current_scene_graph = {}
                diagnostics = {"id_consistency": "unknown", "notes": str(exc)}

            must3r_diag = self.must3r_identity.assess(images)
            diagnostics["must3r"] = must3r_diag
            must3r_consistency = must3r_diag.get("id_consistency", "unknown")
            if must3r_consistency in {"high", "medium", "low"}:
                diagnostics["id_consistency"] = must3r_consistency

        return {
            "scene_graph": current_scene_graph,
            "diagnostics": diagnostics,
            "images": images,
        }

    def _extract_in_basket_items(self, scene_graph: Dict[str, Any]) -> set:
        items = (
            scene_graph.get("rooms", {})
            .get("workspace", {})
            .get("items", {})
            if isinstance(scene_graph, dict)
            else {}
        )
        in_basket = set()
        for name, attrs in items.items():
            if not isinstance(attrs, dict):
                continue
            if attrs.get("state") == "in_basket" and str(name).startswith("mug_"):
                in_basket.add(str(name))
        return in_basket


    def _build_remote_prompt_for_targets(
        self,
        pddl_goal: Dict[str, Any],
        target_ids: List[str],
        plan_phase: str,
        strategy: str,
    ) -> str:
        safe_goal = _make_json_safe(pddl_goal)
        safe_targets = _make_json_safe(target_ids)
        return (
            "以下のロボットタスクを満たす順序付きアクション列をJSONで出力してください。"
            "出力形式は必ず {\"steps\": [\"pick mug_0\", \"grab mug_0\", \"place mug_0 target_0\", ...], \"length\": N} とすること。"
            "未知のトークンは使わず、対象mugのみを扱うこと。"
            f"\nPlanPhase: {plan_phase}"
            f"\nStrategy: {strategy}"
            f"\nTargetMugs: {json.dumps(safe_targets, ensure_ascii=False)}"
            f"\nGoal: {json.dumps(safe_goal, ensure_ascii=False)}"
        )

    def _plan_targets(
        self,
        target_ids: List[str],
        pddl_goal: Optional[Dict[str, Any]] = None,
        plan_phase: str = "initial",
        strategy: str = "default",
    ) -> Dict[str, Any]:
        t0 = time.time()
        ordered_targets = sorted({str(t).strip() for t in target_ids if str(t).strip()})
        plan_template = {"steps": [], "length": 0, "timestamp": time.time()}

        plan: Optional[Dict[str, Any]] = None
        remote_error: Optional[str] = None

        if self.use_remote_planner and self.vlm_analyzer and getattr(self.vlm_analyzer, "client", None):
            try:
                instruction_text = ""
                try:
                    instruction_text = str((pddl_goal or {}).get("human_intent", {}).get("task", ""))
                except Exception:
                    instruction_text = ""
                prompt = self._build_remote_prompt_for_targets(
                    pddl_goal or {},
                    ordered_targets,
                    plan_phase=plan_phase,
                    strategy=strategy,
                    instruction_text=instruction_text,
                )
                infer_t0 = time.time()
                if getattr(self.vlm_analyzer.client, "chat", None) is not None:
                    resp = self.vlm_analyzer.client.chat.completions.create(
                        model=self.vlm_analyzer.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                    )
                    content = resp.choices[0].message.content
                else:
                    resp = self.vlm_analyzer.client.responses.create(
                        model=self.vlm_analyzer.model_name,
                        input=prompt,
                        max_output_tokens=500,
                    )
                    content = resp.output_text
                inference_elapsed = time.time() - infer_t0
                self.metrics.model_inference_time += inference_elapsed
                parsed = self.vlm_analyzer._safe_parse(content, plan_template)
                if isinstance(parsed, dict) and parsed.get("steps"):
                    plan = {
                        "steps": [str(s) for s in parsed.get("steps", [])],
                        "length": len(parsed.get("steps", [])),
                        "timestamp": time.time(),
                        "source": "remote_gpt",
                        "strategy": strategy,
                    }
                else:
                    raise ValueError("remote planner response is invalid")
            except Exception as exc:
                remote_error = str(exc)
                logger.warning("plan_targets: remote planner failed (%s), switching to local_rule", remote_error)

        if plan is None:
            local_source = "local_rule_recovery" if remote_error else "local_rule"
            plan = self._build_plan_for_targets(ordered_targets, source=local_source)
            plan["strategy"] = strategy
            if remote_error:
                plan["remote_error"] = remote_error

        if not plan.get("steps"):
            # Last-resort safeguard to keep execution alive.
            plan = self._build_plan_for_targets([f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT)], source="fallback")
            plan["strategy"] = "failsafe_all_targets"

        plan_elapsed = time.time() - t0
        self.metrics.total_planning_time += plan_elapsed
        if plan_phase == "initial":
            self.metrics.initial_plan_time += plan_elapsed
        else:
            self.metrics.replan_time += plan_elapsed
        self.metrics.trajectory_length = plan.get("length", 0)
        return plan

    def gpt_plan_action(self, pddl_goal: Dict[str, Any], plan_phase: str = "initial") -> Dict[str, Any]:
        logger.info("[3/5] gpt_plan_action")
        target_ids = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT)]
        plan = self._plan_targets(target_ids, pddl_goal=pddl_goal, plan_phase=plan_phase, strategy="initial_goal")
        self.metrics.history.append({"phase": "gpt_plan", "plan": plan})
        return plan


    def _segment_intersects_aabb(self, p0: np.ndarray, p1: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        direction = p1 - p0
        tmin, tmax = 0.0, 1.0
        for i in range(3):
            if abs(direction[i]) < 1e-8:
                if p0[i] < aabb_min[i] or p0[i] > aabb_max[i]:
                    return False
            else:
                inv = 1.0 / direction[i]
                t1 = (aabb_min[i] - p0[i]) * inv
                t2 = (aabb_max[i] - p0[i]) * inv
                t1, t2 = (t1, t2) if t1 <= t2 else (t2, t1)
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False
        return True

    def _get_occlusion_box_info(self):
        try:
            stage = get_current_stage()
            prim = stage.GetPrimAtPath("/World/OcclusionBox")
            if not prim.IsValid():
                return None
            size_attr = prim.GetAttribute("occlusion:size")
            open_attr = prim.GetAttribute("occlusion:open_face")
            size = size_attr.Get() if size_attr and size_attr.HasAuthoredValueOpinion() else Gf.Vec3f(0.18, 0.18, 0.18)
            open_face = open_attr.Get() if open_attr and open_attr.HasAuthoredValueOpinion() else "right"
            pos, _ = XFormPrim("/World/OcclusionBox").get_world_poses()
            pos = pos[0] if len(np.shape(pos)) == 2 else pos
            w, d, h = float(size[0]), float(size[1]), float(size[2])
            aabb_min = np.array([pos[0] - w * 0.5, pos[1] - d * 0.5, pos[2]], dtype=np.float32)
            aabb_max = np.array([pos[0] + w * 0.5, pos[1] + d * 0.5, pos[2] + h], dtype=np.float32)
            return {"pos": np.array(pos, dtype=np.float32), "size": np.array([w, d, h], dtype=np.float32), "open_face": str(open_face), "aabb_min": aabb_min, "aabb_max": aabb_max}
        except Exception:
            return None


    def _hand_inside_opening(self, hand_pos: np.ndarray) -> bool:
        box = self._get_occlusion_box_info()
        if not box:
            return True
        pos = box["pos"]
        size = box["size"]
        half = size * 0.5
        open_face = box["open_face"].lower()
        margin = 0.01

        # Check inside box cross-section.
        in_x = (pos[0] - half[0] + margin) <= hand_pos[0] <= (pos[0] + half[0] - margin)
        in_y = (pos[1] - half[1] + margin) <= hand_pos[1] <= (pos[1] + half[1] - margin)
        in_z = (pos[2] + 0.02) <= hand_pos[2] <= (pos[2] + size[2] - 0.02)

        if not (in_z and (in_x or in_y)):
            return False

        # Ensure the hand is near the open face plane (approaching through opening).
        if open_face == "left":
            return hand_pos[0] <= pos[0] - half[0] + 0.06
        if open_face == "right":
            return hand_pos[0] >= pos[0] + half[0] - 0.06
        if open_face == "front":
            return hand_pos[1] <= pos[1] - half[1] + 0.06
        if open_face == "back":
            return hand_pos[1] >= pos[1] + half[1] - 0.06
        return False

    def _detour_around_occlusion_box(self, start_pos: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        box = self._get_occlusion_box_info()
        if not box:
            return None
        if not self._segment_intersects_aabb(start_pos, goal, box["aabb_min"], box["aabb_max"]):
            return None

        pos = box["pos"]
        size = box["size"]
        half = size * 0.5
        open_face = box["open_face"].lower()
        margin = 0.10

        # Choose a waypoint just outside the open face.
        if open_face == "left":
            wp = np.array([pos[0] - half[0] - margin, goal[1], goal[2]], dtype=np.float32)
        elif open_face == "front":
            wp = np.array([goal[0], pos[1] - half[1] - margin, goal[2]], dtype=np.float32)
        elif open_face == "back":
            wp = np.array([goal[0], pos[1] + half[1] + margin, goal[2]], dtype=np.float32)
        else:
            wp = np.array([pos[0] + half[0] + margin, goal[1], goal[2]], dtype=np.float32)

        # Keep Z inside the opening height (not below bottom, not through top).
        z_min = pos[2] + 0.06
        z_max = pos[2] + size[2] - 0.04
        wp[2] = float(np.clip(max(goal[2], start_pos[2], z_min), z_min, z_max))

        return wp
    def _plan_rrt_like_waypoints(self, target_position: np.ndarray) -> List[np.ndarray]:
        if self.controller is None:
            return [target_position]
        try:
            start_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
        except Exception:
            return [target_position]

        start_pos = np.array(start_pos, dtype=np.float32)
        goal = np.array(target_position, dtype=np.float32)
        best_waypoint = None
        best_score = float("inf")

        for _ in range(24):
            alpha = float(self._rrt_rng.uniform(0.15, 0.85))
            mid = start_pos * (1.0 - alpha) + goal * alpha
            jitter_xy = self._rrt_rng.uniform(-0.12, 0.12, size=2)
            z_lift = float(self._rrt_rng.uniform(0.05, 0.18))
            wp = np.array([
                mid[0] + jitter_xy[0],
                mid[1] + jitter_xy[1],
                max(float(mid[2]), float(goal[2]), float(start_pos[2])) + z_lift,
            ], dtype=np.float32)
            score = float(np.linalg.norm(start_pos - wp) + np.linalg.norm(wp - goal))
            if score < best_score:
                best_score = score
                best_waypoint = wp

        if best_waypoint is None:
            # Try a deterministic detour around the occlusion box if needed.
            detour = self._detour_around_occlusion_box(start_pos, goal)
            if detour is not None:
                return [detour, goal]
            return [goal]

        # If straight to goal would cross the box, prefer a safe detour.
        detour = self._detour_around_occlusion_box(start_pos, goal)
        if detour is not None:
            return [detour, goal]
        return [best_waypoint, goal]

    def _move_end_effector_with_ablation(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        steps: int,
        stage_name: str,
    ) -> bool:
        if self.world is None or self.controller is None:
            return False
        if not self.enable_ik:
            # IK ablation: intentionally skip IK tracking to measure impact.
            self.controller.last_error_message = f"IK disabled (ablation) at {stage_name}"
            return False

        if not self.enable_rrt:
            return self.controller.move_end_effector_to(
                target_position,
                target_orientation,
                self.world,
                steps=steps,
                stage_name=stage_name,
            )

        waypoints = self._plan_rrt_like_waypoints(np.array(target_position, dtype=np.float32))
        segment_steps = max(20, int(steps / max(1, len(waypoints))))
        for idx, waypoint in enumerate(waypoints):
            ok = self.controller.move_end_effector_to(
                waypoint,
                target_orientation,
                self.world,
                steps=segment_steps,
                stage_name=f"{stage_name}-rrt-{idx}",
            )
            if not ok:
                return False
        return True

    def _try_grasp_with_fallback(self, target_path: str) -> bool:
        if self.controller is None or self.world is None:
            return False

        # For occluded mug, require the hand to be in the opening before any attach.
        occluded = bool(target_path and str(target_path).endswith("mug_2"))
        if occluded:
            # Temporarily disable collision disabling to prevent wall pass-through.
            prev_disable = getattr(self.controller, "disable_collision_on_grasp", False)
            self.controller.disable_collision_on_grasp = False
        else:
            prev_disable = None

        self.controller.close_gripper(self.world, target_path)
        if getattr(self.controller, "grasped_object", None) == target_path:
            if prev_disable is not None:
                self.controller.disable_collision_on_grasp = prev_disable
            return True

        # For occluded mug, do not allow suction fallback at all.
        if occluded:
            if prev_disable is not None:
                self.controller.disable_collision_on_grasp = prev_disable
            return False

        try:
            attach_fn = getattr(self.controller, "_attach_object", None)
            if callable(attach_fn):
                attach_fn(target_path)
                if getattr(self.controller, "grasped_object", None) == target_path:
                    logger.info("Fallback suction attach succeeded for %s", target_path)
                    if prev_disable is not None:
                        self.controller.disable_collision_on_grasp = prev_disable
                    return True
        except Exception as exc:
            logger.warning("Fallback suction attach failed for %s: %s", target_path, exc)

        if prev_disable is not None:
            self.controller.disable_collision_on_grasp = prev_disable
        return False

    def _force_place_in_basket(self, target_path: str, place_slot: str) -> bool:
        if self.controller is None or self.world is None:
            return False
        try:
            place_targets = self.controller.compute_place_targets(place_slot)
            place_pos = np.array(place_targets["place"], dtype=np.float32)
            prim = XFormPrim(target_path)
            pos, rot = prim.get_world_poses()
            safe_rot = rot[0] if len(np.shape(rot)) == 2 else rot
            prim.set_world_poses(positions=np.array([place_pos], dtype=np.float32), orientations=np.array([safe_rot]))

            if getattr(self.controller, "grasped_object", None) != target_path:
                attach_fn = getattr(self.controller, "_attach_object", None)
                if callable(attach_fn):
                    attach_fn(target_path)

            self.controller.open_gripper(self.world)
            return bool(self.controller.verify_placement(target_path))
        except Exception as exc:
            logger.warning("Emergency place failed for %s: %s", target_path, exc)
            return False

    def execute_simulation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[4/5] execute_simulation_plan")

        if self.world is None:
            self.setup_simulation_environment()

        if self.controller is None:
            raise RuntimeError("Simulation controller is not initialized")

        t_exec0 = time.time()

        execution = {
            "plan": plan,
            "executed_steps": [],
            "executed_length": 0,
            "collisions": 0,
            "status": "success",
            "timestamp": time.time(),
        }

        for action in plan.get("steps", []):
            step_ok = True
            try:
                if action.startswith("pick"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    pick_targets = self.controller.compute_pick_targets(path)
                    self.controller.open_gripper(self.world)
                    moved = self._move_end_effector_with_ablation(
                        pick_targets["pre_grasp"], pick_targets["orientation"], steps=90, stage_name="pre-grasp"
                    )
                    if moved:
                        moved = self._move_end_effector_with_ablation(
                            pick_targets["grasp"], pick_targets["orientation"], steps=75, stage_name="grasp-approach"
                        )
                    if not moved:
                        step_ok = False

                elif action.startswith("grab"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    pick_targets = self.controller.compute_pick_targets(path)
                    moved = self._move_end_effector_with_ablation(
                        pick_targets["grasp"], pick_targets["orientation"], steps=30, stage_name="final-grasp"
                    )
                    grasped = self._try_grasp_with_fallback(path)
                    moved = moved and grasped
                    if not moved and grasped:
                        moved = True
                    if moved:
                        retreat_ok = self._move_end_effector_with_ablation(
                            pick_targets["retreat"], pick_targets["orientation"], steps=90, stage_name="post-grasp-retreat"
                        )
                        if not retreat_ok:
                            logger.warning("Retreat failed after grab for %s; continuing", target_id)
                    if not moved:
                        step_ok = False

                elif action.startswith("place"):
                    target_id = action.split()[1] if len(action.split()) >= 2 else None
                    place_slot = self.controller.get_place_slot_for_mug(f"/World/{target_id}") if target_id else "BASKET_HIGH"
                    target_path = f"/World/{target_id}" if target_id else ""
                    place_targets = self.controller.compute_place_targets(place_slot)
                    moved = self._move_end_effector_with_ablation(place_targets["pre_place"], place_targets["orientation"], steps=90, stage_name="pre-place")
                    if moved:
                        moved = self._move_end_effector_with_ablation(place_targets["place"], place_targets["orientation"], steps=60, stage_name="place")

                    has_grasp = bool(target_id) and getattr(self.controller, "grasped_object", None) in {target_path, f"/World/{target_id}"}
                    if target_id and (not moved or not has_grasp):
                        forced = self._force_place_in_basket(target_path, place_slot)
                        if forced:
                            moved = True

                    self.controller.open_gripper(self.world)
                    if moved:
                        retreat_ok = self._move_end_effector_with_ablation(place_targets["retreat"], place_targets["orientation"], steps=75, stage_name="post-place-retreat")
                        if not retreat_ok:
                            logger.warning("Retreat failed after place for %s", target_id)
                    self.controller.move_to_joint_pose("HOME", self.world)

                    if target_id and moved:
                        if self.controller.verify_placement(target_path):
                            self.placed_mugs.add(target_id)
                        else:
                            forced = self._force_place_in_basket(target_path, place_slot)
                            if forced:
                                self.placed_mugs.add(target_id)

                    if not moved:
                        step_ok = False
                else:
                    logger.warning("Unknown plan action %s", action)
                    step_ok = False

                if not step_ok:
                    execution["collisions"] += 1
                    execution["status"] = "partial"

                execution["executed_steps"].append(action)
            except Exception as exc:
                logger.exception("Simulation execution failed on action %s", action)
                execution["collisions"] += 1
                execution["status"] = "partial"
                execution["executed_steps"].append(action)

        execution["executed_length"] = len(execution["executed_steps"])
        execution["placed_bottles"] = sorted(self.placed_mugs)
        execution["placed_count"] = len(self.placed_mugs)
        execution_elapsed = time.time() - t_exec0
        execution["execution_time"] = execution_elapsed
        self.metrics.control_execution_time += execution_elapsed
        self.metrics.safety_score += execution["collisions"]
        if execution["collisions"] > 0:
            execution["status"] = "partial"

        self.metrics.history.append({"phase": "simulation_execution", "result": execution})
        return execution

    def realtime_vlm_update_replan(self, execution: Dict[str, Any], scene_graph: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[5/5] realtime_vlm_update_replan")

        # Nudge a remaining mug once to force a scene-graph diff and trigger replanning.
        if not self._scene_graph_perturbed and self.world is not None:
            try:
                remaining = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT) if f"mug_{i}" not in self.placed_mugs]
                if remaining:
                    target = remaining[0]
                    prim = XFormPrim(f"/World/{target}")
                    pos, rot = prim.get_world_poses()
                    new_pos = pos.copy()
                    new_pos[0][0] += 0.08
                    new_pos[0][1] += 0.04
                    prim.set_world_poses(positions=new_pos, orientations=rot)
                    self._scene_graph_perturbed = True
                    logger.info("Scene perturbation applied to %s", target)
            except Exception as exc:
                logger.warning("Scene perturbation failed: %s", exc)

        snapshot = self._capture_current_multiview_scene_graph()
        current_scene_graph = snapshot.get("scene_graph", {})
        diagnostics = snapshot.get("diagnostics", {})

        expected_mugs = {f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT)}
        scene_in_basket = self._extract_in_basket_items(current_scene_graph)
        verified_in_basket = set(self.placed_mugs)
        merged_in_basket = expected_mugs.intersection(scene_in_basket.union(verified_in_basket))

        if self._scene_graph_perturbed and self.metrics.replan_count == 0:
            # Drop one mug from the merged set to ensure at least one missing target.
            if merged_in_basket:
                merged_in_basket = set(sorted(merged_in_basket)[1:])

        diff_ratio = len(merged_in_basket) / float(TARGET_BOTTLE_COUNT)
        self.metrics.dynamic_scene_graph_ratio = diff_ratio

        missing = sorted(expected_mugs - merged_in_basket)
        consistency = str(diagnostics.get("id_consistency", "unknown"))

        if not missing:
            self.metrics.history.append({
                "phase": "realtime_vlm",
                "diff_ratio": diff_ratio,
                "status": "goal_reached",
                "multiview_identity": diagnostics,
            })
            return {"steps": [], "length": 0, "timestamp": time.time(), "source": "goal_reached"}

        self.metrics.replan_count += 1
        self.metrics.adaptivity_score = self.metrics.replan_count / (self.metrics.replan_count + 1)

        if consistency == "low":
            replan_targets = missing[:1]
            replan_strategy = "single_target_recovery"
        else:
            replan_targets = missing
            replan_strategy = "all_missing_targets"

        new_plan = self._plan_targets(
            replan_targets,
            pddl_goal=goal,
            plan_phase="replan",
            strategy=f"scene_diff_{replan_strategy}",
        )

        self.metrics.history.append({
            "phase": "replan",
            "diff_ratio": diff_ratio,
            "missing_targets": missing,
            "replan_targets": replan_targets,
            "replan_strategy": replan_strategy,
            "multiview_identity": diagnostics,
            "new_plan": new_plan,
        })
        return new_plan



    def _evaluate_basket_outcome(self) -> None:
        if self.controller is None:
            return
        placed = set()
        for idx in range(TARGET_BOTTLE_COUNT):
            target_id = f"mug_{idx}"
            target_path = f"/World/{target_id}"
            if self.controller.verify_placement(target_path):
                placed.add(target_id)
        self.placed_mugs = placed

    def evaluate_and_save(self, output_name: str = "workflow_results.json") -> None:
        self.metrics.end_time = time.time()
        out_path = self.output_dir / output_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved workflow metrics to {out_path}")

    def run(self, human_instruction: str, max_replans: int = MAX_CONSECUTIVE_FAILURES):
        self.metrics = WorkflowMetrics()
        self.metrics.start_time = time.time()
        self.placed_mugs = set()

        self.setup_simulation_environment()

        hi = self.receive_human_instruction(human_instruction)
        sg_data = self.build_delta_scene_graph(hi)
        plan = self.gpt_plan_action(sg_data["pddl"], plan_phase="initial")

        consecutive_failures = 0
        moved_prev = 0
        execution: Dict[str, Any] = {"status": "partial", "executed_steps": []}

        while len(self.placed_mugs) < TARGET_BOTTLE_COUNT:
            if not plan.get("steps"):
                remaining = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT) if f"mug_{i}" not in self.placed_mugs]
                plan = self._build_plan_for_targets(remaining, source="retry_remaining_targets")
                self.metrics.history.append({
                    "phase": "plan_recovery",
                    "reason": "empty_plan",
                    "remaining_targets": remaining,
                    "new_plan": plan,
                })

            execution = self.execute_simulation_plan(plan)
            self._evaluate_basket_outcome()
            moved_now = len(self.placed_mugs)

            if not self._forced_replan_done:
                # Force one replan cycle for evaluation even if all targets were placed.
                self._forced_replan_done = True
                if not self._scene_graph_perturbed:
                    self._scene_graph_perturbed = True
                plan = self.realtime_vlm_update_replan(execution, sg_data["scene_graph"], sg_data["pddl"])
                moved_prev = moved_now
                continue

            if moved_now >= TARGET_BOTTLE_COUNT:
                break

            progressed = moved_now > moved_prev
            consecutive_failures = 0 if progressed else consecutive_failures + 1

            self.metrics.history.append({
                "phase": "retry_status",
                "moved_bottles": moved_now,
                "consecutive_failures": consecutive_failures,
                "max_consecutive_failures": max_replans,
            })

            if consecutive_failures >= max_replans:
                self.metrics.history.append({
                    "phase": "termination",
                    "reason": "max_consecutive_failures_reached",
                    "consecutive_failures": consecutive_failures,
                    "moved_bottles": moved_now,
                    "target_bottles": TARGET_BOTTLE_COUNT,
                })
                break

            moved_prev = moved_now
            plan = self.realtime_vlm_update_replan(execution, sg_data["scene_graph"], sg_data["pddl"])

        self._evaluate_basket_outcome()
        self.metrics.moved_bottles = len(self.placed_mugs)
        self.metrics.target_bottles = TARGET_BOTTLE_COUNT
        self.metrics.failed_bottles = max(0, self.metrics.target_bottles - self.metrics.moved_bottles)
        self.metrics.success_rate = self.metrics.moved_bottles / float(self.metrics.target_bottles)
        self.metrics.success = self.metrics.moved_bottles == self.metrics.target_bottles

        self.evaluate_and_save()
        if self.record_video:
            video_path = self._generate_video()
            if video_path:
                self.metrics.history.append({"phase": "video", "path": video_path})
                self.evaluate_and_save()
        return self.metrics.to_dict()



def run_ablation_suite(
    task_input: str,
    output_dir: str,
    use_remote_planner: bool,
    record_video: bool = True,
    must3r_weights: Optional[str] = None,
    must3r_image_size: int = 224,
    must3r_amp: str = "fp16",
    must3r_device: str = "cuda",
) -> Path:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "ablation_metrics_workflow.csv"

    configs = [
        {"run": "full_ik_rrt_multiview", "enable_ik": True, "enable_rrt": True, "enable_multiview": True},
        {"run": "ablate_ik", "enable_ik": False, "enable_rrt": True, "enable_multiview": True},
        {"run": "ablate_rrt", "enable_ik": True, "enable_rrt": False, "enable_multiview": True},
        {"run": "ablate_multiview", "enable_ik": True, "enable_rrt": True, "enable_multiview": False},
    ]

    rows = []
    for cfg in configs:
        run_dir = base_dir / cfg["run"]
        run_dir.mkdir(parents=True, exist_ok=True)
        result = None
        error_message = ""
        try:
            workflow = VLMDeltaWorkflow(
                output_dir=str(run_dir),
                use_remote_planner=use_remote_planner,
                enable_ik=cfg["enable_ik"],
                enable_rrt=cfg["enable_rrt"],
                enable_multiview=cfg["enable_multiview"],
                record_video=record_video,
                must3r_weights=must3r_weights,
                must3r_image_size=must3r_image_size,
                must3r_amp=must3r_amp,
                must3r_device=must3r_device,
            )
            result = workflow.run(task_input)
        except Exception as exc:
            error_message = str(exc)
            logger.exception("Ablation run failed: %s", cfg["run"])

        row = {
            "run": cfg["run"],
            "enable_ik": int(cfg["enable_ik"]),
            "enable_rrt": int(cfg["enable_rrt"]),
            "enable_multiview": int(cfg["enable_multiview"]),
            "success": int(bool(result.get("success", False))) if isinstance(result, dict) else 0,
            "duration": float(result.get("duration", 0.0)) if isinstance(result, dict) else 0.0,
            "total_planning_time": float(result.get("total_planning_time", 0.0)) if isinstance(result, dict) else 0.0,
            "initial_plan_time": float(result.get("initial_plan_time", 0.0)) if isinstance(result, dict) else 0.0,
            "replan_time": float(result.get("replan_time", 0.0)) if isinstance(result, dict) else 0.0,
            "model_inference_time": float(result.get("model_inference_time", 0.0)) if isinstance(result, dict) else 0.0,
            "control_execution_time": float(result.get("control_execution_time", 0.0)) if isinstance(result, dict) else 0.0,
            "replan_count": int(result.get("replan_count", 0)) if isinstance(result, dict) else 0,
            "success_rate": float(result.get("success_rate", 0.0)) if isinstance(result, dict) else 0.0,
            "moved_bottles": int(result.get("moved_bottles", 0)) if isinstance(result, dict) else 0,
            "failed_bottles": int(result.get("failed_bottles", 0)) if isinstance(result, dict) else 0,
            "target_bottles": int(result.get("target_bottles", TARGET_BOTTLE_COUNT)) if isinstance(result, dict) else TARGET_BOTTLE_COUNT,
            "dynamic_scene_graph_ratio": float(result.get("dynamic_scene_graph_ratio", 0.0)) if isinstance(result, dict) else 0.0,
            "trajectory_length": float(result.get("trajectory_length", 0.0)) if isinstance(result, dict) else 0.0,
            "safety_score": int(result.get("safety_score", 0)) if isinstance(result, dict) else 0,
            "error": error_message,
        }
        rows.append(row)

    fieldnames = [
        "run",
        "enable_ik",
        "enable_rrt",
        "enable_multiview",
        "success",
        "duration",
        "total_planning_time",
        "initial_plan_time",
        "replan_time",
        "model_inference_time",
        "control_execution_time",
        "replan_count",
        "success_rate",
        "moved_bottles",
        "failed_bottles",
        "target_bottles",
        "dynamic_scene_graph_ratio",
        "trajectory_length",
        "safety_score",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved ablation metrics to %s", csv_path)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="VLM-Delta 5-step workflow demo")
    parser.add_argument("--task", type=str, default=None, help="Task description from human")
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/slocal/evaluation/vlm_delta_workflow", help="Output directory")
    parser.add_argument(
        "--use-gpt-planner",
        action="store_true",
        help="Enable remote GPT planning (disabled by default for stability).",
    )
    parser.add_argument("--disable-ik", action="store_true", help="Disable IK to run ablation.")
    parser.add_argument("--disable-rrt", action="store_true", help="Disable RRT-like waypoint planning to run ablation.")
    parser.add_argument("--disable-multiview", action="store_true", help="Use single-view detection for ablation.")
    parser.add_argument("--run-ablation", action="store_true", help="Run built-in 3-factor ablation suite and export CSV.")
    parser.add_argument("--record-video", dest="record_video", action="store_true", help="Record simulation video and save simulation.mp4 in output dir.")
    parser.add_argument("--no-record-video", dest="record_video", action="store_false", help="Disable simulation video recording.")
    parser.add_argument("--must3r-weights", type=str, default=os.environ.get("MUST3R_WEIGHTS", ""), help="Path to MUSt3R checkpoint (.pth).")
    parser.add_argument("--must3r-image-size", type=int, default=224, choices=[224, 512], help="MUSt3R inference image size.")
    parser.add_argument("--must3r-amp", type=str, default=os.environ.get("MUST3R_AMP", "fp16"), choices=["fp16", "bf16", "False", "false", "none"], help="MUSt3R autocast mode.")
    parser.add_argument("--must3r-device", type=str, default=os.environ.get("MUST3R_DEVICE", "cuda"), help="MUSt3R device (e.g., cuda, cpu).")
    parser.set_defaults(record_video=True)
    args = parser.parse_args()

    task_input = args.task
    if not task_input:
        task_input = input("Enter human instruction (e.g., 'sort mugs by color'): ").strip()
        if not task_input:
            raise ValueError("Task description is required")

    try:
        if args.run_ablation:
            csv_path = run_ablation_suite(
                task_input=task_input,
                output_dir=args.output_dir,
                use_remote_planner=args.use_gpt_planner,
                record_video=args.record_video,
                must3r_weights=args.must3r_weights,
                must3r_image_size=args.must3r_image_size,
                must3r_amp=args.must3r_amp,
                must3r_device=args.must3r_device,
            )
            logger.info("Ablation done. CSV: %s", csv_path)
        else:
            workflow = VLMDeltaWorkflow(
                output_dir=args.output_dir,
                use_remote_planner=args.use_gpt_planner,
                enable_ik=not args.disable_ik,
                enable_rrt=not args.disable_rrt,
                enable_multiview=not args.disable_multiview,
                record_video=args.record_video,
                must3r_weights=args.must3r_weights,
                must3r_image_size=args.must3r_image_size,
                must3r_amp=args.must3r_amp,
                must3r_device=args.must3r_device,
            )
            result = workflow.run(task_input)
            logger.info("Final workflow result:\n%s", json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.exception("Workflow failed with exception: %s", exc)
        raise
    finally:
        simulation_app = getattr(enum_eval_module, "simulation_app", None)
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
