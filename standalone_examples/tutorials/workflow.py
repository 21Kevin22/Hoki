"""vlm_delta workflow skeleton implementation.

5-step architecture:
 1. receive human intent
 2. generate delta scene graph / PDDL style goal representation
 3. query GPT-based planner for action plan
 4. execute in simulation + logging
 5. realtime VLM scene graph update + replan based on discrepancy

Example usage: python3 standalone_examples/tutorials/workflow.py --task "sort mugs by color"
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

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running this file directly via:
# `python3 /path/to/standalone_examples/tutorials/workflow.py`
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
    wait_steps,
    BASKET_PLACE_SLOTS,
)
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, Usd, UsdGeom, UsdLux, PhysxSchema, UsdPhysics, UsdShade


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

TARGET_BOTTLE_COUNT = 5
MAX_CONSECUTIVE_FAILURES = 3
AFFORDANCE_SCHEMA = {
    "handle": {"graspable": True},
    "body_outer": {"pourable": True},
    "body_inner": {"pourable": True},
    "bottom": {"stable": True},
}
AFFORDANCE_PARTS = ("handle", "body_outer", "body_inner", "bottom")
MIN_GRASP_CLEARANCE = 0.12
HANDLE_MIN_CLEARANCE = 0.08
HANDLE_GRASP_Z_ADJUST = -0.08
DEFAULT_DELTA_SCRIPT = "/home/ubuntu/slocal/Hoki/delta.py"
DEFAULT_SAYPLAN_SCRIPT = "/home/ubuntu/slocal/Hoki/baselines/sayplan.py"


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
        sim_device: str = "cpu",
        ik_method: str = "damped-least-squares",
        step_mode: bool = False,
        replan_on_failure: bool = True,
        replan_backend: str = "auto",
        replan_script: Optional[str] = None,
        replan_actions_json: Optional[str] = None,
        replan_python: Optional[str] = None,
        max_failure_replans: int = 3,
        enable_physics_stabilization: bool = True,
        enable_contact_offsets: bool = True,
        enable_solver_boost: bool = True,
        enable_action_waits: bool = True,
        pick_wait_sec: float = 0.2,
        grab_wait_sec: float = 0.5,
        place_settle_wait_sec: float = 0.5,
        verify_wait_sec: float = 1.0,
        pre_close_wait_sec: float = 1.0,
        enable_ccd: bool = True,
        enable_velocity_capping: bool = True,
        max_linear_velocity: float = 1.2,
        max_angular_velocity: float = 8.0,
        linear_damping: float = 0.2,
        angular_damping: float = 0.35,
        rigid_position_iterations: int = 32,
        rigid_velocity_iterations: int = 8,
        max_depenetration_velocity: float = 1.0,
        enable_grasp_jump_guard: bool = True,
        grasp_jump_stop_threshold: float = 0.10,
        enable_scene_perturbation: bool = False,
        enable_cumotion_style: bool = True,
        gripper_static_friction: float = 2.5,
        gripper_dynamic_friction: float = 2.0,
        mug_static_friction: float = 1.8,
        mug_dynamic_friction: float = 1.4,
        render_decimation: int = 3,
        residual_warn_threshold: float = 0.05,
        residual_stop_threshold: float = 0.12,
        grasp_quality_min: float = 0.35,
        isaac_grasp_file: Optional[str] = None,
        grasp_z_offset: float = -0.015,
        replan_grasp_z_offset: float = -0.03,
        attach_distance_threshold: float = 0.255,
        attach_distance_grace: float = 0.045,
        simple_mode: bool = False,
        disable_attachment: bool = False,
        motion_step_scale: float = 1.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = WorkflowMetrics()
        self.use_remote_planner = use_remote_planner
        self.enable_ik = enable_ik
        self.enable_rrt = enable_rrt
        self.enable_multiview = enable_multiview
        self.record_video = record_video
        self.step_mode = bool(step_mode)
        self.replan_on_failure = bool(replan_on_failure)
        self.replan_backend = str(replan_backend or "auto")
        self.replan_script = str(replan_script) if replan_script else ""
        self.replan_actions_json = str(replan_actions_json) if replan_actions_json else ""
        self.replan_python = str(replan_python) if replan_python else os.environ.get("REPLAN_PYTHON", "/usr/bin/python3")
        self.max_failure_replans = int(max_failure_replans)
        self.enable_physics_stabilization = bool(enable_physics_stabilization)
        self.enable_contact_offsets = bool(enable_contact_offsets)
        self.enable_solver_boost = bool(enable_solver_boost)
        self.enable_action_waits = bool(enable_action_waits)
        self.pick_wait_sec = float(pick_wait_sec)
        self.grab_wait_sec = float(grab_wait_sec)
        self.sim_device = str(sim_device) if str(sim_device) in {"cpu", "cuda"} else "cpu"
        self.ik_method = str(ik_method) if str(ik_method) in {"singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"} else "damped-least-squares"
        self.place_settle_wait_sec = float(place_settle_wait_sec)
        self.verify_wait_sec = float(verify_wait_sec)
        self.pre_close_wait_sec = float(pre_close_wait_sec)
        # Workflow-level grasp stabilization knobs (applied to Enum_eval controller at runtime).
        self.workflow_attach_distance_threshold = float(np.clip(attach_distance_threshold, 0.10, 0.30))
        self.workflow_attach_distance_grace = float(np.clip(attach_distance_grace, 0.0, 0.08))
        self.workflow_grasp_follow_alpha = 0.12
        self.workflow_release_max_depen = 0.6
        self.workflow_release_linear_damping = 1.0
        self.workflow_release_angular_damping = 1.2
        self.enable_ccd = bool(enable_ccd)
        self.enable_velocity_capping = bool(enable_velocity_capping)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.linear_damping = float(linear_damping)
        self.angular_damping = float(angular_damping)
        self.rigid_position_iterations = int(rigid_position_iterations)
        self.rigid_velocity_iterations = int(rigid_velocity_iterations)
        self.max_depenetration_velocity = float(max_depenetration_velocity)
        self.enable_grasp_jump_guard = bool(enable_grasp_jump_guard)
        self.grasp_jump_stop_threshold = float(grasp_jump_stop_threshold)
        self.enable_scene_perturbation = bool(enable_scene_perturbation)
        self.enable_cumotion_style = bool(enable_cumotion_style)
        self.gripper_static_friction = float(gripper_static_friction)
        self.gripper_dynamic_friction = float(gripper_dynamic_friction)
        self.mug_static_friction = float(mug_static_friction)
        self.mug_dynamic_friction = float(mug_dynamic_friction)
        self.render_decimation = max(1, int(render_decimation))
        self.residual_warn_threshold = float(residual_warn_threshold)
        self.residual_stop_threshold = float(residual_stop_threshold)
        self.grasp_quality_min = float(grasp_quality_min)
        self.isaac_grasp_file = str(isaac_grasp_file or "").strip()
        self.grasp_z_offset = float(grasp_z_offset)
        self.replan_grasp_z_offset = float(replan_grasp_z_offset)
        self.simple_mode = bool(simple_mode)
        self.disable_attachment = bool(disable_attachment)
        self.motion_step_scale = float(max(0.5, min(6.0, motion_step_scale)))
        self._isaac_grasp_entries: List[Dict[str, Any]] = []
        self._active_isaac_grasp_by_target: Dict[str, Dict[str, Any]] = {}
        self._grasp_prev_pos_by_target: Dict[str, np.ndarray] = {}
        self._grasp_feedback_by_target: Dict[str, Dict[str, Any]] = {}
        self._grasp_emergency_stop = False
        self._grasp_emergency_reason = ""
        self._failure_replan_counts_by_target: Dict[str, int] = {}
        self._grasp_feedback_by_target.clear()
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
        self.attempted_mugs = set()
        self.failed_mugs = set()
        self.mug_retry_counts: Dict[str, int] = {}
        self._scene_graph_perturbed = False
        self._forced_replan_done = False
        self._affordance_cache: Dict[str, Dict[str, Any]] = {}
        self._last_instruction: str = ""
        self._mug_angles: Dict[str, float] = {}
        self._safety_ui_window = None
        self._safety_ui_models: Dict[str, Any] = {}
        self._original_world_step = None
        self._step_counter = 0
        self._residual_events: List[Dict[str, Any]] = []
        self._load_isaac_grasp_file()

    def _configure_physics_settings(self):
        if self.world is None:
            return
        try:
            ctx = self.world.get_physics_context()
            # smaller dt with more substeps for stable contacts
            ctx.set_physics_dt(dt=1.0 / 60.0, substeps=8)
            try:
                ctx.enable_gpu_dynamics(True)
                ctx.set_broadphase_type("GPU")
            except Exception:
                pass
            try:
                ctx.set_solver_type("TGS")
            except Exception:
                pass
        except Exception:
            pass

        try:
            stage = get_current_stage()
            scene_prim = None
            for candidate in ["/physicsScene", "/PhysicsScene"]:
                prim = stage.GetPrimAtPath(candidate)
                if prim.IsValid():
                    scene_prim = prim
                    break
            if scene_prim is None:
                scene_prim = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
            physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
            if self.enable_ccd:
                try:
                    physx_scene.CreateEnableCCDAttr().Set(True)
                except Exception:
                    pass
        except Exception:
            pass

    def _apply_rigid_body_stabilization(self, prim_paths):
        stage = get_current_stage()
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            try:
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            except Exception:
                continue

            if self.enable_ccd:
                try:
                    rb.CreateEnableCCDAttr().Set(True)
                except Exception:
                    pass
            if self.enable_velocity_capping:
                try:
                    rb.CreateMaxLinearVelocityAttr().Set(float(self.max_linear_velocity))
                except Exception:
                    pass
                try:
                    rb.CreateMaxAngularVelocityAttr().Set(float(self.max_angular_velocity))
                except Exception:
                    pass
            try:
                rb.CreateLinearDampingAttr().Set(float(self.linear_damping))
            except Exception:
                pass
            try:
                rb.CreateAngularDampingAttr().Set(float(self.angular_damping))
            except Exception:
                pass
            try:
                rb.CreateSolverPositionIterationCountAttr().Set(int(self.rigid_position_iterations))
            except Exception:
                pass
            try:
                rb.CreateSolverVelocityIterationCountAttr().Set(int(self.rigid_velocity_iterations))
            except Exception:
                pass
            try:
                rb.CreateMaxDepenetrationVelocityAttr().Set(float(self.max_depenetration_velocity))
            except Exception:
                pass

    def _apply_contact_offsets(self, prim_paths, contact_offset=0.002, rest_offset=0.0):
        stage = get_current_stage()
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            try:
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr().Set(float(contact_offset))
                api.CreateRestOffsetAttr().Set(float(rest_offset))
            except Exception:
                continue

    def _get_runtime_friction_prim_groups(self) -> Dict[str, List[str]]:
        mug_parts: List[str] = []
        for idx in range(TARGET_BOTTLE_COUNT):
            base = f"/World/mug_{idx}"
            mug_parts.extend([f"{base}/Body", f"{base}/Handle"])
        finger_parts = [
            "/World/Franka/panda_leftfinger",
            "/World/Franka/panda_rightfinger",
            "/World/Franka/panda_hand/panda_leftfinger",
            "/World/Franka/panda_hand/panda_rightfinger",
        ]
        basket_parts = [
            "/World/Basket/Bottom",
            "/World/Basket/Front",
            "/World/Basket/Back",
            "/World/Basket/Left",
            "/World/Basket/Right",
        ]
        return {"mug_parts": mug_parts, "finger_parts": finger_parts, "basket_parts": basket_parts}

    def _bind_physics_material(self, prim_paths, material_name: str, static_friction: float, dynamic_friction: float, restitution: float = 0.0):
        stage = get_current_stage()
        material_path = f"/World/PhysicsMaterials/{material_name}"
        material = UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim()
        try:
            physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
            physx_mat.CreateStaticFrictionAttr().Set(float(static_friction))
            physx_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
            physx_mat.CreateRestitutionAttr().Set(float(restitution))
        except Exception:
            return

        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            try:
                UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            except Exception:
                continue

    def _apply_runtime_friction_settings(self) -> None:
        try:
            groups = self._get_runtime_friction_prim_groups()
            self._bind_physics_material(
                groups.get("finger_parts", []),
                material_name="gripper_high_friction",
                static_friction=self.gripper_static_friction,
                dynamic_friction=self.gripper_dynamic_friction,
            )
            self._bind_physics_material(
                groups.get("mug_parts", []),
                material_name="mug_high_friction",
                static_friction=self.mug_static_friction,
                dynamic_friction=self.mug_dynamic_friction,
            )
        except Exception:
            pass

    def _resolve_replan_backend(self) -> str:
        backend = (self.replan_backend or "auto").strip().lower()
        if backend and backend != "auto":
            return backend

        if self.replan_script:
            if "sayplan" in self.replan_script.lower():
                return "sayplan"
            return "delta"

        sayplan_path = os.environ.get("SAYPLAN_SCRIPT", "").strip() or DEFAULT_SAYPLAN_SCRIPT
        if os.path.exists(sayplan_path) and os.environ.get("OPENAI_API_KEY"):
            return "sayplan"

        delta_path = os.environ.get("DELTA_SCRIPT", "").strip() or DEFAULT_DELTA_SCRIPT
        if os.path.exists(delta_path):
            return "delta"

        return "none"

    def _get_replan_script_path(self, backend: str) -> str:
        if self.replan_script:
            return self.replan_script
        if backend == "sayplan":
            return os.environ.get("SAYPLAN_SCRIPT", "").strip() or DEFAULT_SAYPLAN_SCRIPT
        if backend == "delta":
            return os.environ.get("DELTA_SCRIPT", "").strip() or DEFAULT_DELTA_SCRIPT
        return ""

    def _run_external_replan(self, failed_action: str, reason: str) -> Dict[str, Any]:
        backend = self._resolve_replan_backend()
        script_path = self._get_replan_script_path(backend)
        result: Dict[str, Any] = {
            "backend": backend,
            "script": script_path,
            "failed_action": failed_action,
            "reason": reason,
            "status": "skipped",
        }

        if backend in {"none", ""}:
            return result
        if not script_path or not os.path.exists(script_path):
            result["status"] = "missing"
            return result

        # Prefer system Python for external replanners to avoid Isaac-kit stdlib mismatches.
        python_bin = str(self.replan_python or "").strip() or "/usr/bin/python3"
        if ("/kit/python" in python_bin) or (not os.path.exists(python_bin)):
            python_bin = "/usr/bin/python3"

        cmd = [python_bin, script_path]
        if backend == "delta":
            cmd += ["--domain", "pc", "--scene", "office", "--domain-example", "laundry", "--ref-pddl", "office_pc_domain.pddl"]
            if self._last_instruction:
                cmd += ["--instruction", self._last_instruction]
        elif backend == "sayplan":
            cmd += ["-d", "pc", "-s", "office", "--domain-example", "laundry", "--scene-example", "allensville", "--no-search", "--print-plan"]

        env = os.environ.copy()
        env.setdefault("PYTHONWARNINGS", "ignore")
        # Remove Isaac runtime Python overlays for child process.
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "PYTHONNOUSERSITE", "LD_PRELOAD"):
            env.pop(key, None)

        log_path = self.output_dir / f"replan_{backend}_{int(time.time() * 1000)}.log"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                completed = subprocess.run(
                    cmd,
                    cwd=Path(script_path).parent,
                    env=env,
                    check=False,
                    text=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            result["log"] = str(log_path)
            result["returncode"] = int(getattr(completed, "returncode", 1))
            result["python"] = python_bin
            result["status"] = "ok" if int(result["returncode"]) == 0 else "failed"
        except Exception as exc:
            result["status"] = f"error: {exc}"
        return result

    def _load_replan_actions(self) -> List[str]:
        actions: List[str] = []
        json_path = self.replan_actions_json.strip() if self.replan_actions_json else ""
        if not json_path:
            json_path = str(self.output_dir / "actions.json")
        try:
            data = json.loads(Path(json_path).read_text())
            if isinstance(data, dict):
                actions = data.get("actions") or data.get("steps") or []
            elif isinstance(data, list):
                actions = data
        except Exception:
            actions = []
        return [str(a) for a in actions if str(a).strip()]

    def _actions_to_plan(self, actions: List[str], source: str) -> Dict[str, Any]:
        steps: List[str] = []
        for raw in actions:
            stripped = raw.strip()
            if stripped.startswith("(") and stripped.endswith(")"):
                stripped = stripped[1:-1].strip()
            tokens = stripped.split()
            if not tokens:
                continue
            verb = tokens[0]
            target = tokens[1] if len(tokens) > 1 else None
            if verb in {"pick", "grab", "place"}:
                if not target:
                    continue
                slot_id = target.rsplit("_", 1)[-1]
                if self.step_mode:
                    steps.extend([
                        f"open {target}",
                        f"approach {target}",
                        f"grasp {target}",
                        f"close {target}",
                        f"retreat {target}",
                        f"pre_place {target} target_{slot_id}",
                        f"place {target} target_{slot_id}",
                        f"release {target}",
                        f"home {target}",
                    ])
                else:
                    if verb == "place":
                        steps.append(f"place {target} target_{slot_id}")
                    else:
                        steps.append(f"{verb} {target}")
            elif verb in {"open", "approach", "grasp", "close", "retreat", "pre_place", "release", "home"}:
                if target:
                    steps.append(f"{verb} {target}")
            else:
                continue
        return {"steps": steps, "length": len(steps), "timestamp": time.time(), "source": source}

    def _replan_after_failure(self, execution: Dict[str, Any], pddl_goal: Dict[str, Any], target_id: str = "") -> Optional[Dict[str, Any]]:
        if not self.replan_on_failure:
            return None
        key = str(target_id or self._extract_target_id_from_action(execution.get("failed_action") or "") or "global")
        used = int(self._failure_replan_counts_by_target.get(key, 0))
        if used >= max(0, self.max_failure_replans):
            logger.info("Replan skipped for %s: reached limit %d", key, int(self.max_failure_replans))
            return None

        failed_action = execution.get("failed_action") or "unknown"
        reason = execution.get("failed_reason") or "action failed"
        self._failure_replan_counts_by_target[key] = used + 1
        logger.info("Replan trigger for %s (%d/%d)", key, int(self._failure_replan_counts_by_target[key]), int(self.max_failure_replans))

        external = self._run_external_replan(str(failed_action), str(reason))
        actions: List[str] = []
        if str(external.get("status", "")) == "ok":
            actions = self._load_replan_actions()
        else:
            logger.warning(
                "External replan unavailable (backend=%s status=%s). Using internal fallback plan.",
                external.get("backend"),
                external.get("status"),
            )

        if actions:
            plan = self._actions_to_plan(actions, source=f"external_replan_{external.get('backend')}")
        else:
            remaining = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT) if f"mug_{i}" not in self.placed_mugs and f"mug_{i}" not in self.failed_mugs]
            for target in remaining:
                self.attempted_mugs.discard(target)
            plan = self._build_plan_for_targets(remaining, source="failure_recovery_fallback")

        self.metrics.history.append({
            "phase": "failure_replan",
            "target": key,
            "used_replans": int(self._failure_replan_counts_by_target.get(key, 0)),
            "failed_action": failed_action,
            "reason": reason,
            "external": external,
            "new_plan": plan,
        })
        return plan

    def _setup_multiview_writers(self):
        # Always write frames into a fresh run directory to avoid mixing
        # old frames (which causes flicker in the encoded video).
        target_frames_dir = self.output_dir / f"frames_run_{int(time.time())}"
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

        ffmpeg_bin = (
            os.environ.get("FFMPEG_BIN")
            or shutil.which("ffmpeg")
            or ("/home/ubuntu/.pyenv/versions/miniconda3-latest/bin/ffmpeg" if Path("/home/ubuntu/.pyenv/versions/miniconda3-latest/bin/ffmpeg").exists() else "ffmpeg")
        )

        # Prefer direct glob encoding to avoid duplicating frames on disk.
        if main_dir.exists() and image_files:
            glob_source = str(main_dir / "rgb_*.png")
            try:
                cmd = [
                    ffmpeg_bin,
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

    def _apply_workflow_grasp_stabilization(self) -> None:
        if self.controller is None:
            return
        try:
            clamp_fn = getattr(self.controller, "_clamp_attach_distance", None)
            if callable(clamp_fn):
                self.controller.attach_distance_threshold = float(clamp_fn(self.workflow_attach_distance_threshold))
            else:
                self.controller.attach_distance_threshold = float(self.workflow_attach_distance_threshold)
        except Exception:
            self.controller.attach_distance_threshold = float(self.workflow_attach_distance_threshold)

        try:
            self.controller.grasp_follow_alpha = float(self.workflow_grasp_follow_alpha)
        except Exception:
            pass

        try:
            setattr(enum_eval_module, "SAFE_ATTACH_DISTANCE_GRACE", float(self.workflow_attach_distance_grace))
        except Exception:
            pass

        logger.info(
            "[GraspStabilize] attach_th=%.3f grace=%.3f follow_alpha=%.3f",
            float(getattr(self.controller, "attach_distance_threshold", self.workflow_attach_distance_threshold)),
            float(getattr(enum_eval_module, "SAFE_ATTACH_DISTANCE_GRACE", self.workflow_attach_distance_grace)),
            float(getattr(self.controller, "grasp_follow_alpha", self.workflow_grasp_follow_alpha)),
        )

    def _prepare_release_depenetration(self, target_path: str) -> None:
        if not target_path:
            return
        try:
            prim = get_current_stage().GetPrimAtPath(target_path)
            if not prim.IsValid():
                return
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateMaxDepenetrationVelocityAttr().Set(float(self.workflow_release_max_depen))
            rb.CreateLinearDampingAttr().Set(float(self.workflow_release_linear_damping))
            rb.CreateAngularDampingAttr().Set(float(self.workflow_release_angular_damping))
        except Exception:
            pass

    def _plan_uses_replan_grasp_offset(self, plan: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(plan, dict):
            return False
        source = str(plan.get("source") or "").strip().lower()
        return bool(source) and ("replan" in source)

    def _target_retry_count(self, target_path: str) -> int:
        target_id = str(target_path).split("/")[-1]
        try:
            return int(self.mug_retry_counts.get(target_id, 0))
        except Exception:
            return 0

    def _adaptive_grasp_quality_min(self, target_id: str) -> float:
        try:
            retry_count = int(self.mug_retry_counts.get(str(target_id), 0))
        except Exception:
            retry_count = 0
        tuning = self._get_vlm_tuning_for_target(str(target_id))
        quality_relax = float(tuning.get("quality_relax", 0.0))
        base = float(max(0.12, float(self.grasp_quality_min) - 0.08 * retry_count))
        return float(max(0.08, base - quality_relax))

    def _get_or_init_grasp_feedback(self, target_id: str) -> Dict[str, Any]:
        key = str(target_id)
        fb = self._grasp_feedback_by_target.get(key)
        if not isinstance(fb, dict):
            fb = {"xyz": np.zeros(3, dtype=np.float32), "tilt_deg": 0.0}
            self._grasp_feedback_by_target[key] = fb
        if not isinstance(fb.get("xyz"), np.ndarray):
            fb["xyz"] = np.array(fb.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
        fb["tilt_deg"] = float(fb.get("tilt_deg", 0.0))
        return fb

    def _estimate_grasp_residual(self, target_path: str) -> np.ndarray:
        if self.controller is None:
            return np.zeros(3, dtype=np.float32)
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            body_path = f"{target_path}/Body"
            try:
                obj_pos, _ = self.controller._get_safe_world_pose(body_path)
            except Exception:
                obj_pos, _ = self.controller._get_safe_world_pose(target_path)
            return np.array(obj_pos, dtype=np.float32) - np.array(hand_pos, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _update_grasp_feedback(self, target_path: str, score: float, quality_min: float) -> None:
        target_id = str(target_path).split("/")[-1]
        fb = self._get_or_init_grasp_feedback(target_id)
        residual = self._estimate_grasp_residual(target_path)

        if float(score) >= float(quality_min):
            fb["xyz"] = np.array(fb["xyz"], dtype=np.float32) * 0.5
            fb["tilt_deg"] = float(fb["tilt_deg"]) * 0.5
        else:
            delta = np.array([
                np.clip(float(residual[0]) * 0.35, -0.012, 0.012),
                np.clip(float(residual[1]) * 0.35, -0.012, 0.012),
                np.clip(float(residual[2]) * 0.45, -0.018, 0.010),
            ], dtype=np.float32)
            if float(residual[2]) < -0.004:
                delta[2] -= 0.003
            fb["xyz"] = np.array(fb["xyz"], dtype=np.float32) + delta
            fb["tilt_deg"] = min(6.0, float(fb["tilt_deg"]) + 1.2)

        fb_xyz = np.array(fb["xyz"], dtype=np.float32)
        fb_xyz[0] = float(np.clip(fb_xyz[0], -0.05, 0.05))
        fb_xyz[1] = float(np.clip(fb_xyz[1], -0.05, 0.05))
        fb_xyz[2] = float(np.clip(fb_xyz[2], -0.06, 0.02))
        fb["xyz"] = fb_xyz
        fb["tilt_deg"] = float(np.clip(float(fb["tilt_deg"]), 0.0, 6.0))
        self._grasp_feedback_by_target[target_id] = fb

        logger.info(
            "[WF-Feedback] target=%s score=%.3f min=%.3f residual=(%.3f,%.3f,%.3f) corr=(%.3f,%.3f,%.3f) tilt=%.1f",
            target_id,
            float(score),
            float(quality_min),
            float(residual[0]),
            float(residual[1]),
            float(residual[2]),
            float(fb_xyz[0]),
            float(fb_xyz[1]),
            float(fb_xyz[2]),
            float(fb["tilt_deg"]),
        )

    def _adjust_pick_targets_for_execution(self, target_path: str, pick_targets: Dict[str, Any], use_replan_offset: bool) -> Dict[str, Any]:
        if not pick_targets:
            return pick_targets

        grasp = pick_targets.get("grasp")
        if grasp is None:
            return pick_targets

        base_drop = float(self.grasp_z_offset)
        if use_replan_offset:
            base_drop += float(self.replan_grasp_z_offset)

        bbox_min, bbox_max = self._get_target_bbox_world(target_path)
        if bbox_min is None or bbox_max is None:
            obj_min_z = None
            obj_max_z = None
            obj_height = 0.12
            try:
                safe_pos, _ = self.controller._get_safe_world_pose(target_path)
                floor_z = float(safe_pos[2]) + 0.016
            except Exception:
                floor_z = float(np.array(grasp, dtype=np.float32)[2])
        else:
            obj_min_z = float(bbox_min[2])
            obj_max_z = float(bbox_max[2])
            obj_height = max(0.01, obj_max_z - obj_min_z)
            if obj_height < 0.10:
                floor_z = obj_min_z + 0.004
            else:
                floor_z = min(obj_max_z + 0.006, obj_min_z + 0.018)

        retry_count = self._target_retry_count(target_path)
        retry_drop = min(0.06, 0.012 * float(retry_count))
        profile_drop = 0.014 if obj_height < 0.10 else (0.006 if obj_height < 0.13 else 0.0)
        drop = base_drop - retry_drop - profile_drop

        target_id = str(target_path).split("/")[-1]
        tuning = self._get_vlm_tuning_for_target(target_id)
        vlm_drop = float(tuning.get("z_extra_drop", 0.0))
        backoff_scale = float(tuning.get("backoff_scale", 1.0))
        drop += vlm_drop
        if target_id == "mug_1":
            # mug_1 tends to miss when we descend too deep.
            drop += 0.008

        # In simple/no-attachment mode, prioritize a shallower, more repeatable body grasp.
        if self.simple_mode or self.disable_attachment:
            drop = max(drop, -0.004)
            floor_z = max(float(floor_z), float(obj_min_z + min(0.03, 0.35 * obj_height)))

        adjusted = dict(pick_targets)
        grasp_vec = np.array(grasp, dtype=np.float32).copy()
        desired_z = float(grasp_vec[2] + drop)
        if self.simple_mode or self.disable_attachment:
            desired_z = max(desired_z, float(obj_min_z + min(0.03, 0.35 * obj_height)))
        grasp_vec[2] = max(float(floor_z), desired_z)

        feedback = self._get_or_init_grasp_feedback(target_id)
        corr = np.array(feedback.get("xyz", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        grasp_vec = grasp_vec + corr
        grasp_vec[2] = max(float(floor_z), float(grasp_vec[2]))
        adjusted["grasp"] = grasp_vec

        pre = adjusted.get("pre_grasp")
        if pre is not None:
            pre_vec = np.array(pre, dtype=np.float32).copy()
            # Keep pre-grasp above grasp, but bring it slightly down as retries increase.
            pre_vec[2] = float(max(grasp_vec[2] + 0.085, min(pre_vec[2] + 0.5 * drop, grasp_vec[2] + 0.13)))

            # Ensure enough horizontal side-approach before descending to grasp height.
            approach_xy = np.array([-float(grasp_vec[0]), -float(grasp_vec[1])], dtype=np.float32)
            approach_norm = float(np.linalg.norm(approach_xy))
            if approach_norm < 1e-6:
                approach_xy = np.array([0.0, -1.0], dtype=np.float32)
                approach_norm = 1.0
            approach_xy = approach_xy / approach_norm
            backoff = (0.055 + min(0.03, 0.01 * float(retry_count))) * backoff_scale
            target_id = str(target_path).split("/")[-1]
            if target_id == "mug_1":
                backoff = (0.04 + min(0.02, 0.008 * float(retry_count))) * backoff_scale
            pre_vec[0] = float(grasp_vec[0] + approach_xy[0] * backoff)
            pre_vec[1] = float(grasp_vec[1] + approach_xy[1] * backoff)

            if target_id == "mug_0":
                # mug_0 often needs extra side clearance before final descend.
                pre_vec[0] = float(pre_vec[0] + 0.02)

            pre_vec[0] += float(corr[0])
            pre_vec[1] += float(corr[1])
            pre_vec[2] += float(0.5 * corr[2])
            adjusted["pre_grasp"] = pre_vec

        ori = adjusted.get("orientation")
        if ori is not None:
            try:
                ori_q = np.array(ori, dtype=np.float32)
                tilt_deg = float(feedback.get("tilt_deg", 0.0))
                if abs(tilt_deg) > 1e-3:
                    approach_xy = np.array([-float(grasp_vec[0]), -float(grasp_vec[1]), 0.0], dtype=np.float32)
                    n = float(np.linalg.norm(approach_xy[:2]))
                    if n < 1e-6:
                        approach_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        n = 1.0
                    approach_xy = approach_xy / n
                    tilt_axis = np.array([-approach_xy[1], approach_xy[0], 0.0], dtype=np.float32)
                    tilt_q = self._quat_from_axis_angle(tilt_axis, np.deg2rad(tilt_deg))
                    ori_q = self._quat_mul(tilt_q, ori_q)
                    n2 = float(np.linalg.norm(ori_q))
                    if n2 > 1e-8:
                        ori_q = ori_q / n2
                adjusted["orientation"] = ori_q
            except Exception:
                pass

        try:
            target_id = str(target_path).split("/")[-1]
            logger.info(
                "[WF-GraspZ] target=%s retry=%d height=%.3f drop=%.3f floor=%.3f grasp_z=%.3f vlm_drop=%.3f backoff_scale=%.2f",
                target_id,
                retry_count,
                float(obj_height),
                float(drop),
                float(floor_z),
                float(grasp_vec[2]),
                float(vlm_drop),
                float(backoff_scale),
            )
        except Exception:
            pass

        return adjusted

    def _trigger_grasp_jump_emergency(self, target_path: str, jump_dist: float) -> None:
        if self.controller is None:
            return
        if bool(getattr(self.controller, "attach_locked_until_open", False)):
            logger.warning("[SafetyStop] detach suppressed by attach lock for %s (jump=%.3f)", target_path, jump_dist)
            return
        self._grasp_emergency_stop = True
        self._grasp_emergency_reason = (
            f"grasp jump detected on {target_path}: {jump_dist:.3f}m > {self.grasp_jump_stop_threshold:.3f}m"
        )
        self.controller.last_error_message = self._grasp_emergency_reason
        logger.warning("[SafetyStop] %s", self._grasp_emergency_reason)

        try:
            stage = get_current_stage()
            prim = stage.GetPrimAtPath(target_path)
            if prim.IsValid():
                UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
        except Exception:
            pass

        try:
            if hasattr(self.controller, "_set_collision_enabled"):
                self.controller._set_collision_enabled(target_path, True)
            if hasattr(self.controller, "_set_gripper_collision_enabled"):
                self.controller._set_gripper_collision_enabled(True)
        except Exception:
            pass

        try:
            self.controller.grasped_object = None
        except Exception:
            pass
        self._grasp_prev_pos_by_target.clear()

    def _scale_motion_steps(self, steps: int) -> int:
        try:
            return max(1, int(round(float(steps) * float(self.motion_step_scale))))
        except Exception:
            return max(1, int(steps))

    def _configure_attachment_mode(self) -> None:
        if self.controller is None:
            return
        if not self.disable_attachment:
            return
        attach_fn = getattr(self.controller, "_attach_object", None)
        if callable(attach_fn) and not getattr(self.controller, "_attachment_disabled_wrapped", False):
            def _attach_disabled(target_path):
                logger.info("[AttachmentOff] skip virtual attach for %s", target_path)
                return None
            self.controller._attach_object = _attach_disabled
            self.controller._attachment_disabled_wrapped = True

    def _is_physical_grasp_likely(self, target_path: str) -> bool:
        if self.controller is None or (not target_path):
            return False
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            obj_pos, _ = self.controller._get_safe_world_pose(target_path)
            dist = float(np.linalg.norm(np.array(hand_pos, dtype=np.float32) - np.array(obj_pos, dtype=np.float32)))
        except Exception:
            return False

        overlap = 0.0
        try:
            overlap, _ = self.controller._compute_grasp_box_overlap(target_path)
        except Exception:
            overlap = 0.0

        likely = bool(
            (dist <= 0.085 and float(overlap) >= 0.045)
            or (dist <= 0.070)
            or (float(overlap) >= 0.08)
        )
        logger.info(
            "[PhysicalGraspCheck] target=%s dist=%.3f overlap=%.3f likely=%s",
            str(target_path).split('/')[-1],
            dist,
            float(overlap),
            likely,
        )
        return likely

    def _publish_rrt_waypoint_markers(self, waypoints: List[np.ndarray], prefix: str) -> None:
        if not waypoints:
            return
        vh = getattr(enum_eval_module, "VisualHelper", None)
        if vh is None:
            return
        create_fn = getattr(vh, "create_or_move_proxy", None)
        if not callable(create_fn):
            return
        for idx, wp in enumerate(waypoints):
            try:
                create_fn(pos=np.array(wp, dtype=np.float32), name=f"{prefix}_rrt_wp_{idx}")
            except Exception:
                continue

    def _get_finger_world_pose(self, side: str) -> Optional[np.ndarray]:
        if self.controller is None:
            return None
        side = str(side).lower().strip()
        candidates = [
            f"/World/Franka/panda_{side}finger",
            f"/World/Franka/panda_hand/panda_{side}finger",
        ]
        for prim_path in candidates:
            try:
                pos, _ = self.controller._get_safe_world_pose(prim_path)
                pos = np.array(pos, dtype=np.float32)
                if np.all(np.isfinite(pos)):
                    return pos
            except Exception:
                continue
        return None

    def _has_bilateral_finger_grasp(self, target_path: str, distance_tol: float = 0.03, overlap_min: float = 0.04) -> bool:
        if self.controller is None or (not target_path):
            return False
        left = self._get_finger_world_pose("left")
        right = self._get_finger_world_pose("right")
        bbox_min, bbox_max = self._get_target_bbox_world(target_path)
        if left is None or right is None or bbox_min is None or bbox_max is None:
            return False

        bbox_min = np.array(bbox_min, dtype=np.float32)
        bbox_max = np.array(bbox_max, dtype=np.float32)
        obj_center = 0.5 * (bbox_min + bbox_max)

        def point_to_aabb_distance(point: np.ndarray) -> float:
            clamped = np.minimum(np.maximum(point, bbox_min), bbox_max)
            return float(np.linalg.norm(np.array(point, dtype=np.float32) - clamped))

        left_dist = point_to_aabb_distance(np.array(left, dtype=np.float32))
        right_dist = point_to_aabb_distance(np.array(right, dtype=np.float32))
        finger_delta = np.array(right, dtype=np.float32) - np.array(left, dtype=np.float32)
        axis = 0 if abs(float(finger_delta[0])) >= abs(float(finger_delta[1])) else 1
        lo = min(float(left[axis]), float(right[axis])) - 0.015
        hi = max(float(left[axis]), float(right[axis])) + 0.015
        center_between = bool(lo <= float(obj_center[axis]) <= hi)

        overlap = 0.0
        try:
            overlap, _ = self.controller._compute_grasp_box_overlap(target_path)
        except Exception:
            overlap = 0.0

        ok = bool(
            left_dist <= float(distance_tol)
            and right_dist <= float(distance_tol)
            and center_between
            and float(overlap) >= float(overlap_min)
        )
        logger.info(
            "[TwoFingerCheck] target=%s left_dist=%.3f right_dist=%.3f overlap=%.3f axis=%s center_between=%s ok=%s",
            str(target_path).split("/")[-1],
            left_dist,
            right_dist,
            float(overlap),
            "x" if axis == 0 else "y",
            center_between,
            ok,
        )
        return ok

    def _lock_target_for_transport(self, target_path: str) -> None:
        if self.controller is None or (not target_path):
            return
        if self.disable_attachment:
            self._ensure_transport_gripper_hold(target_path)
            if not self._has_bilateral_finger_grasp(target_path):
                logger.info("[TransportLockSkip] target=%s waiting for bilateral finger grasp", str(target_path).split("/")[-1])
                return
        try:
            stage = get_current_stage()
            prim = stage.GetPrimAtPath(target_path)
            if prim.IsValid():
                UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
        except Exception:
            pass
        try:
            self.controller.grasped_object = str(target_path)
            self.controller.attach_locked_until_open = True
        except Exception:
            pass
        self._ensure_transport_gripper_hold(target_path)
        if self.disable_attachment:
            logger.info("[TransportLockBilateral] target=%s bilateral finger grasp confirmed; semi-lock enabled", str(target_path).split("/")[-1])
        else:
            logger.info("[TransportLock] target=%s locked for transport", str(target_path).split("/")[-1])

    def _install_grasp_jump_guard(self) -> None:
        if self.controller is None or not self.enable_grasp_jump_guard:
            return
        if getattr(self.controller, "_jump_guard_wrapped", False):
            return

        original_update = getattr(self.controller, "_update_grasped_object", None)
        if not callable(original_update):
            return

        def wrapped_update():
            if not self.enable_grasp_jump_guard:
                original_update()
                return
            if self._grasp_emergency_stop:
                return
            original_update()
            if self.controller is None:
                return
            target_path = getattr(self.controller, "grasped_object", None)
            if not target_path:
                self._grasp_prev_pos_by_target.clear()
                return
            if bool(getattr(self.controller, "attach_locked_until_open", False)):
                try:
                    pos, _ = self.controller._get_safe_world_pose(target_path)
                    self._grasp_prev_pos_by_target[target_path] = np.array(pos, dtype=np.float32)
                except Exception:
                    pass
                return
            try:
                pos, _ = self.controller._get_safe_world_pose(target_path)
                hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
                follow = getattr(self.controller, "attach_follow_offset", np.array([0.0, 0.0, -0.07], dtype=np.float32))
                target = np.array(hand_pos, dtype=np.float32) + np.array(follow, dtype=np.float32)
                residual = float(np.linalg.norm(np.array(pos, dtype=np.float32) - target))
                if residual > float(self.residual_warn_threshold):
                    evt = {"type": "residual_warn", "target": target_path, "residual": residual, "t": time.time()}
                    self._residual_events.append(evt)
                    logger.warning("[Residual] %s residual=%.3f", target_path, residual)
                if residual > float(self.residual_stop_threshold):
                    self._trigger_grasp_jump_emergency(target_path, residual)
                    return

                prev = self._grasp_prev_pos_by_target.get(target_path)
                if prev is not None:
                    jump_dist = float(np.linalg.norm(pos - prev))
                    if jump_dist > float(self.grasp_jump_stop_threshold):
                        self._trigger_grasp_jump_emergency(target_path, jump_dist)
                        return
                self._grasp_prev_pos_by_target[target_path] = np.array(pos, dtype=np.float32)
            except Exception:
                return

        self.controller._update_grasped_object = wrapped_update
        self.controller._jump_guard_wrapped = True
        logger.info("[SafetyStop] grasp jump guard enabled (threshold=%.3fm)", self.grasp_jump_stop_threshold)

    def _apply_safety_ui_values(self) -> None:
        if not self._safety_ui_models:
            return

        def _f(key: str, default: float) -> float:
            mdl = self._safety_ui_models.get(key)
            if mdl is None:
                return float(default)
            try:
                return float(mdl.as_float)
            except Exception:
                try:
                    return float(mdl.get_value_as_float())
                except Exception:
                    return float(default)

        def _b(key: str, default: bool) -> bool:
            mdl = self._safety_ui_models.get(key)
            if mdl is None:
                return bool(default)
            try:
                return bool(mdl.as_bool)
            except Exception:
                try:
                    return bool(mdl.get_value_as_bool())
                except Exception:
                    return bool(default)

        self.workflow_attach_distance_threshold = _f("attach_th", self.workflow_attach_distance_threshold)
        self.workflow_attach_distance_grace = _f("attach_grace", self.workflow_attach_distance_grace)
        self.workflow_grasp_follow_alpha = _f("follow_alpha", self.workflow_grasp_follow_alpha)
        self.grasp_jump_stop_threshold = _f("jump_stop", self.grasp_jump_stop_threshold)
        self.enable_grasp_jump_guard = _b("jump_guard", self.enable_grasp_jump_guard)
        self.grasp_z_offset = _f("grasp_z_offset", self.grasp_z_offset)
        self.replan_grasp_z_offset = _f("replan_grasp_z_offset", self.replan_grasp_z_offset)
        self.gripper_static_friction = _f("gripper_static_friction", self.gripper_static_friction)
        self.gripper_dynamic_friction = _f("gripper_dynamic_friction", self.gripper_dynamic_friction)
        self.mug_static_friction = _f("mug_static_friction", self.mug_static_friction)
        self.mug_dynamic_friction = _f("mug_dynamic_friction", self.mug_dynamic_friction)
        if self.controller is not None:
            grip_open = _f("grip_open", float(self.controller.dynamic_params.get("grip_open", 0.04)))
            grip_close = _f("grip_close", float(self.controller.dynamic_params.get("grip_close", 0.005)))
            self.controller.update_dynamic_params({"grip_open": grip_open, "grip_close": grip_close})
            self._set_controller_gripper_target(grip_open, hold=False)

        self._apply_workflow_grasp_stabilization()
        self._apply_runtime_friction_settings()
        self._install_grasp_jump_guard()
        self._setup_safety_tuning_ui()
        logger.info(
            "[SafetyUI] applied attach_th=%.3f grace=%.3f follow_alpha=%.3f jump_guard=%s jump_th=%.3f",
            self.workflow_attach_distance_threshold,
            self.workflow_attach_distance_grace,
            self.workflow_grasp_follow_alpha,
            self.enable_grasp_jump_guard,
            self.grasp_jump_stop_threshold,
        )

    def _setup_safety_tuning_ui(self) -> None:
        if self.controller is None:
            return
        if self._safety_ui_window is not None:
            return
        if bool(getattr(enum_eval_module, "HEADLESS", False)):
            return

        try:
            import omni.ui as ui
        except Exception:
            return

        self._safety_ui_window = ui.Window("Workflow Safety", width=420, height=520)
        with self._safety_ui_window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Grasp Safety")
                self._safety_ui_models["attach_th"] = ui.SimpleFloatModel(float(self.workflow_attach_distance_threshold))
                self._safety_ui_models["attach_grace"] = ui.SimpleFloatModel(float(self.workflow_attach_distance_grace))
                self._safety_ui_models["follow_alpha"] = ui.SimpleFloatModel(float(self.workflow_grasp_follow_alpha))
                self._safety_ui_models["jump_stop"] = ui.SimpleFloatModel(float(self.grasp_jump_stop_threshold))
                self._safety_ui_models["jump_guard"] = ui.SimpleBoolModel(bool(self.enable_grasp_jump_guard))
                self._safety_ui_models["grasp_z_offset"] = ui.SimpleFloatModel(float(self.grasp_z_offset))
                self._safety_ui_models["replan_grasp_z_offset"] = ui.SimpleFloatModel(float(self.replan_grasp_z_offset))
                grip_open = float(self.controller.dynamic_params.get("grip_open", 0.04))
                grip_close = float(self.controller.dynamic_params.get("grip_close", 0.005))
                self._safety_ui_models["grip_open"] = ui.SimpleFloatModel(grip_open)
                self._safety_ui_models["grip_close"] = ui.SimpleFloatModel(grip_close)
                self._safety_ui_models["gripper_static_friction"] = ui.SimpleFloatModel(float(self.gripper_static_friction))
                self._safety_ui_models["gripper_dynamic_friction"] = ui.SimpleFloatModel(float(self.gripper_dynamic_friction))
                self._safety_ui_models["mug_static_friction"] = ui.SimpleFloatModel(float(self.mug_static_friction))
                self._safety_ui_models["mug_dynamic_friction"] = ui.SimpleFloatModel(float(self.mug_dynamic_friction))

                with ui.HStack(height=24):
                    ui.Label("Attach Th", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["attach_th"], min=0.08, max=0.30, step=0.005)
                with ui.HStack(height=24):
                    ui.Label("Attach Grace", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["attach_grace"], min=0.0, max=0.08, step=0.001)
                with ui.HStack(height=24):
                    ui.Label("Follow Alpha", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["follow_alpha"], min=0.05, max=0.35, step=0.01)
                with ui.HStack(height=24):
                    ui.Label("Jump Stop (m)", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["jump_stop"], min=0.01, max=0.20, step=0.005)
                with ui.HStack(height=24):
                    ui.Label("Enable Jump Guard", width=150)
                    ui.CheckBox(model=self._safety_ui_models["jump_guard"])
                ui.Separator()
                ui.Label("Grasp Targets")
                with ui.HStack(height=24):
                    ui.Label("Grasp Z Offset", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["grasp_z_offset"], min=-0.06, max=0.02, step=0.001)
                with ui.HStack(height=24):
                    ui.Label("Replan Z Offset", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["replan_grasp_z_offset"], min=-0.08, max=0.0, step=0.001)
                with ui.HStack(height=24):
                    ui.Label("Grip Open", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["grip_open"], min=0.01, max=0.08, step=0.001)
                with ui.HStack(height=24):
                    ui.Label("Grip Close", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["grip_close"], min=0.0, max=0.05, step=0.001)
                ui.Separator()
                ui.Label("Friction")
                with ui.HStack(height=24):
                    ui.Label("Grip Static Fric", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["gripper_static_friction"], min=0.1, max=5.0, step=0.1)
                with ui.HStack(height=24):
                    ui.Label("Grip Dynamic Fric", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["gripper_dynamic_friction"], min=0.1, max=5.0, step=0.1)
                with ui.HStack(height=24):
                    ui.Label("Mug Static Fric", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["mug_static_friction"], min=0.1, max=5.0, step=0.1)
                with ui.HStack(height=24):
                    ui.Label("Mug Dynamic Fric", width=150)
                    ui.FloatDrag(model=self._safety_ui_models["mug_dynamic_friction"], min=0.1, max=5.0, step=0.1)

                with ui.HStack(height=28):
                    ui.Button("Apply", clicked_fn=lambda: self._apply_safety_ui_values())
                    ui.Button("Reset", clicked_fn=lambda: self._reset_safety_ui_values())

    def _reset_safety_ui_values(self) -> None:
        if not self._safety_ui_models:
            return
        defaults = {
            "attach_th": 0.255,
            "attach_grace": 0.045,
            "follow_alpha": 0.12,
            "jump_stop": 0.10,
            "jump_guard": True,
            "grasp_z_offset": -0.015,
            "replan_grasp_z_offset": -0.03,
            "grip_open": 0.04,
            "grip_close": 0.005,
            "gripper_static_friction": 2.5,
            "gripper_dynamic_friction": 2.0,
            "mug_static_friction": 1.8,
            "mug_dynamic_friction": 1.4,
        }
        for k, v in defaults.items():
            mdl = self._safety_ui_models.get(k)
            if mdl is None:
                continue
            try:
                mdl.set_value(v)
            except Exception:
                pass
        self._apply_safety_ui_values()

    def _install_fixed_step_render_gate(self) -> None:
        if self.world is None:
            return
        if self._original_world_step is not None:
            return
        if self.render_decimation <= 1:
            return
        try:
            original = self.world.step

            def wrapped_step(render=True):
                # Keep physics stepping every call; render only every N calls.
                self._step_counter += 1
                do_render = bool(render) and (self._step_counter % self.render_decimation == 0)
                return original(render=do_render)

            self.world.step = wrapped_step
            self._original_world_step = original
            logger.info("[Physics] render decimation enabled: 1/%d", self.render_decimation)
        except Exception as exc:
            logger.warning("render decimation setup skipped: %s", exc)

    def _evaluate_grasp_quality(self, target_path: str) -> (float, str):
        if self.controller is None:
            return 0.0, "controller missing"
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            obj_pos, _ = self.controller._get_safe_world_pose(target_path)
            dist = float(np.linalg.norm(hand_pos - obj_pos))
            # distance-based quality in [0,1]
            dist_score = max(0.0, min(1.0, 1.0 - dist / 0.20))
            attached = 1.0 if getattr(self.controller, "grasped_object", None) == target_path else 0.0
            score = 0.7 * dist_score + 0.3 * attached
            reason = f"dist={dist:.3f}, attached={bool(attached)}"
            return float(score), reason
        except Exception as exc:
            return 0.0, f"grasp-quality-exception: {exc}"

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
        try:
            SimulationManager.set_physics_sim_device(self.sim_device)
        except Exception as exc:
            logger.warning("set_physics_sim_device(%s) failed: %s", self.sim_device, exc)
        self._install_fixed_step_render_gate()
        if self.enable_physics_stabilization:
            self._configure_physics_settings()
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
        mug_configs = [
            {"id": "mug_0", "pos": [0.4, 0.5, 0.05], "color": [1.0, 0.0, 0.0], "angle": -120},
            {"id": "mug_1", "pos": [0.5, 0.25, 0.05], "color": [0.0, 1.0, 0.0], "angle": 180},
            {"id": "mug_2", "pos": [0.6, 0.0, 0.05], "color": [0.0, 0.0, 1.0], "angle": 120},
            {"id": "mug_3", "pos": [0.5, -0.25, 0.05], "color": [1.0, 1.0, 0.0], "angle": 60},
            {"id": "mug_4", "pos": [0.4, -0.5, 0.05], "color": [0.0, 1.0, 1.0], "angle": 0},
        ]
        self._mug_angles = {}
        for cfg in mug_configs:
            AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])
            self._mug_angles[str(cfg["id"])] = float(cfg.get("angle", 0.0))

        AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.0])

        # Strengthen articulation solver iterations for stable grasping
        if self.enable_solver_boost:
            try:
                franka_prim = get_current_stage().GetPrimAtPath("/World/Franka")
                if franka_prim.IsValid():
                    art_api = PhysxSchema.PhysxArticulationAPI.Apply(franka_prim)
                    art_api.CreateSolverPositionIterationCountAttr().Set(24)
                    art_api.CreateSolverVelocityIterationCountAttr().Set(6)
            except Exception:
                pass

        # Contact/rest offsets and friction materials for mug colliders and gripper fingers
        groups = self._get_runtime_friction_prim_groups()
        mug_parts = groups["mug_parts"]
        finger_parts = groups["finger_parts"]
        basket_parts = groups["basket_parts"]
        stabilized_parts = mug_parts + finger_parts
        if self.enable_contact_offsets:
            self._apply_contact_offsets(stabilized_parts + basket_parts, contact_offset=0.002, rest_offset=0.0)
        self._apply_rigid_body_stabilization(stabilized_parts)
        self._bind_physics_material(
            finger_parts,
            material_name="gripper_high_friction",
            static_friction=self.gripper_static_friction,
            dynamic_friction=self.gripper_dynamic_friction,
        )
        self._bind_physics_material(
            mug_parts,
            material_name="mug_high_friction",
            static_friction=self.mug_static_friction,
            dynamic_friction=self.mug_dynamic_friction,
        )
        logger.info(
            "[Physics] substeps=6 ccd=%s vel_cap=%s vmax=(%.2f, %.2f) damping=(%.2f, %.2f) rigid_iters=(%d,%d) max_depen=%.2f",
            self.enable_ccd,
            self.enable_velocity_capping,
            self.max_linear_velocity,
            self.max_angular_velocity,
            self.linear_damping,
            self.angular_damping,
            self.rigid_position_iterations,
            self.rigid_velocity_iterations,
            self.max_depenetration_velocity,
        )

        # Keep handle/body collisions enabled so each mug behaves as one compound rigid body.
        # (Disabling Handle collision can look like the handle is detached during contact.)

        self._setup_multiview_writers()

        self.world.reset()
        self.world.play()

        calib = run_calibration(self.world)
        self.controller = RobotController(arm=self.franka)
        self.controller.apply_calibration(calib)
        self._apply_workflow_grasp_stabilization()
        self._configure_attachment_mode()
        self._install_grasp_jump_guard()

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
        ordered = list(dict.fromkeys([str(t).strip() for t in target_ids if str(t).strip()]))
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

    def _normalize_mug_pick_plan(self, target_id: str, pick_part: str, fallback_parts: List[str]) -> Tuple[str, List[str]]:
        target_name = str(target_id or "")
        if not target_name.startswith("mug_"):
            normalized_part = pick_part if pick_part in AFFORDANCE_SCHEMA else "handle"
            normalized_fallbacks = [p for p in fallback_parts if p in AFFORDANCE_SCHEMA and p != normalized_part]
            return normalized_part, normalized_fallbacks

        allowed_parts = ["body_outer", "body_inner"]
        normalized_part = pick_part if pick_part in allowed_parts else "body_outer"
        normalized_fallbacks = [p for p in fallback_parts if p in allowed_parts and p != normalized_part]
        if not normalized_fallbacks:
            normalized_fallbacks = [p for p in allowed_parts if p != normalized_part]
        return normalized_part, normalized_fallbacks

    def _plan_affordance_action(self, image_paths: List[str], instruction: str, target_id: str) -> Dict[str, Any]:
        default_res = {"pick_part": "handle", "fallback_parts": ["body_outer"], "rationale": "default"}
        client = getattr(self.vlm_analyzer, "client", None)
        if not client or not image_paths:
            return default_res
        try:
            base64_images = [self.vlm_analyzer.encode_image_base64(p) for p in image_paths]
        except Exception:
            return default_res

        prompt = (
            "画像からマグの姿勢を把握し、次の2点を考慮して計画してください。\n"
            "(1) 取手が下にあるなど掴めない場合、どうすれば取手が見つかるか?\n"
            "(2) 取手以外で掴める可能性のある部位はどれか? (例:円柱側面)\n"
            "候補パーツ: handle, body_outer, body_inner, bottom。\n"
            "アフォーダンス: graspable, pourable, stable。\n"
            f"対象: {target_id}\n"
            f"指示: {instruction}\n"
            "出力JSON: {\"pick_part\": str, \"fallback_parts\": [str], \"rationale\": str}"
        )

        content = [{"type": "text", "text": prompt}]
        for b64 in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        try:
            response = client.chat.completions.create(
                model=self.vlm_analyzer.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            parsed = self.vlm_analyzer._safe_parse(response.choices[0].message.content, default_res) or default_res
            part = parsed.get("pick_part", "handle")
            fallbacks = parsed.get("fallback_parts", []) or []
            part, fallbacks = self._normalize_mug_pick_plan(target_id, part, fallbacks)
            if not fallbacks:
                fallbacks = ["body_inner"] if part == "body_outer" else ["body_outer"]
            return {"pick_part": part, "fallback_parts": fallbacks, "rationale": parsed.get("rationale", "")}
        except Exception:
            return default_res

    def _derive_vlm_tuning(self, target_id: str, pick_part: str, reason: str) -> Dict[str, float]:
        text = str(reason or "").lower()
        tuning = {"z_extra_drop": 0.0, "backoff_scale": 1.0, "quality_relax": 0.0}

        if pick_part == "body_inner":
            tuning["z_extra_drop"] -= 0.006
            tuning["quality_relax"] += 0.03
        elif pick_part == "body_outer":
            tuning["z_extra_drop"] -= 0.003
            tuning["quality_relax"] += 0.01
        elif pick_part == "handle":
            tuning["z_extra_drop"] += 0.004

        if any(k in text for k in ["横", "倒", "lying", "sideways", "fallen", "寝", "tilt"]):
            tuning["z_extra_drop"] -= 0.008
            tuning["backoff_scale"] *= 1.15
            tuning["quality_relax"] += 0.02

        if any(k in text for k in ["見え", "隠", "occlu", "blocked", "shadow"]):
            tuning["backoff_scale"] *= 1.12

        if str(target_id) == "mug_0":
            tuning["backoff_scale"] *= 1.08
        elif str(target_id) == "mug_1":
            tuning["z_extra_drop"] += 0.004
            tuning["backoff_scale"] *= 0.88

        tuning["z_extra_drop"] = float(np.clip(float(tuning["z_extra_drop"]), -0.03, 0.02))
        tuning["backoff_scale"] = float(np.clip(float(tuning["backoff_scale"]), 0.75, 1.35))
        tuning["quality_relax"] = float(np.clip(float(tuning["quality_relax"]), 0.0, 0.08))
        return tuning

    def _get_vlm_tuning_for_target(self, target_id: str) -> Dict[str, float]:
        aff = self._affordance_cache.get(str(target_id), {})
        tuning = aff.get("tuning", {}) if isinstance(aff, dict) else {}
        return tuning if isinstance(tuning, dict) else {}

    def _get_affordance_for_target(self, target_id: str) -> Dict[str, Any]:
        cached = self._affordance_cache.get(target_id)
        if cached:
            if "tuning" not in cached:
                cached = dict(cached)
                cached["tuning"] = self._derive_vlm_tuning(
                    target_id,
                    str(cached.get("part", "body_outer")),
                    str(cached.get("reason", "")),
                )
                self._affordance_cache[target_id] = cached
            return cached

        instruction = self._last_instruction or ""
        images: List[str] = []
        if self.world is not None:
            try:
                self._refresh_multiview_frames(steps=2)
                images = self._collect_latest_multiview_images()
                if images and not self.enable_multiview:
                    images = images[:1]
            except Exception:
                images = []

        plan = self._plan_affordance_action(images, instruction, target_id)
        normalized_part, normalized_fallbacks = self._normalize_mug_pick_plan(
            target_id,
            plan.get("pick_part", "handle"),
            plan.get("fallback_parts", []),
        )
        affordance = {
            "part": normalized_part,
            "affordance": "graspable",
            "fallback_parts": normalized_fallbacks,
            "reason": plan.get("rationale", ""),
        }
        affordance["tuning"] = self._derive_vlm_tuning(
            target_id,
            str(affordance.get("part", "body_outer")),
            str(affordance.get("reason", "")),
        )
        self._affordance_cache[target_id] = affordance
        self.metrics.history.append({"phase": "affordance_plan", "target": target_id, "affordance": affordance})
        return affordance

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        # wxyz x wxyz
        w1, x1, y1, z1 = [float(v) for v in q1]
        w2, x2, y2, z2 = [float(v) for v in q2]
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    def _quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
        return self._quat_mul(self._quat_mul(q, qv), self._quat_conj(q))[1:]

    def _load_isaac_grasp_file(self) -> None:
        self._isaac_grasp_entries = []
        if not self.isaac_grasp_file:
            return
        if yaml is None:
            logger.warning("Isaac grasp file skipped: PyYAML not available")
            return
        path = Path(self.isaac_grasp_file)
        if not path.exists():
            logger.warning("Isaac grasp file not found: %s", path)
            return
        try:
            data = yaml.safe_load(path.read_text()) or {}
            grasps = data.get("grasps", {}) if isinstance(data, dict) else {}
            entries = []
            for name, g in (grasps.items() if isinstance(grasps, dict) else []):
                if not isinstance(g, dict):
                    continue
                pos = g.get("position", [0.0, 0.0, 0.0])
                ori = g.get("orientation", {})
                w = float(ori.get("w", 1.0)) if isinstance(ori, dict) else 1.0
                xyz = ori.get("xyz", [0.0, 0.0, 0.0]) if isinstance(ori, dict) else [0.0, 0.0, 0.0]
                q = np.array([w, float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float32)
                n = float(np.linalg.norm(q))
                if n > 1e-8:
                    q = q / n
                entries.append({
                    "name": str(name),
                    "confidence": float(g.get("confidence", 0.0)),
                    "position": np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32),
                    "orientation": q,
                    "cspace_position": g.get("cspace_position", {}) if isinstance(g.get("cspace_position", {}), dict) else {},
                    "pregrasp_cspace_position": g.get("pregrasp_cspace_position", {}) if isinstance(g.get("pregrasp_cspace_position", {}), dict) else {},
                })
            entries.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
            self._isaac_grasp_entries = entries
            logger.info("[IsaacGrasp] loaded %d grasps from %s", len(entries), path)
        except Exception as exc:
            logger.warning("Isaac grasp load failed: %s", exc)

    def _select_isaac_grasp_for_target(self, target_path: str) -> Optional[Dict[str, Any]]:
        if not self._isaac_grasp_entries:
            return None
        target_id = str(target_path).split("/")[-1]
        selected = self._active_isaac_grasp_by_target.get(target_id)
        if selected is None:
            selected = self._isaac_grasp_entries[0]
            self._active_isaac_grasp_by_target[target_id] = selected
        return selected

    def _set_controller_gripper_target(self, width: float, hold: bool) -> None:
        if self.controller is None:
            return
        clamped = float(np.clip(width, 0.0, 0.08))
        try:
            if hold:
                self.controller.dynamic_params["grip_close"] = min(clamped, float(self.controller.dynamic_params.get("grip_open", 0.04)))
            else:
                self.controller.dynamic_params["grip_open"] = max(clamped, 0.01)
        except Exception:
            pass
        setter = getattr(self.controller, "_set_gripper_width_target", None)
        if callable(setter):
            try:
                setter(clamped, enable_hold=hold)
                return
            except Exception:
                pass
        try:
            if len(self.controller.current_pose) >= 9:
                self.controller.current_pose[7:] = clamped
        except Exception:
            pass

    def _ensure_transport_gripper_hold(self, target_path: str = "") -> None:
        if self.controller is None:
            return
        grasped = str(getattr(self.controller, "grasped_object", "") or "")
        requested = str(target_path or "")
        active = requested if requested else grasped
        if not active:
            return
        # Avoid applying transport hold when requested target is not actually grasped.
        # In no-attachment mode, rely on physical contact and keep closing during transport.
        if (not self.disable_attachment) and requested and grasped != requested:
            return
        close_width = float(self.controller.dynamic_params.get("grip_close", getattr(enum_eval_module, "GRIP_CLOSE_MUG", 0.005)))
        self._set_controller_gripper_target(close_width, hold=True)

    def _apply_isaac_grasp_cspace_for_target(self, target_id: str, use_pregrasp: bool) -> None:
        if self.controller is None:
            return
        g = self._active_isaac_grasp_by_target.get(target_id)
        if not g:
            return
        joint_map = g.get("pregrasp_cspace_position", {}) if use_pregrasp else g.get("cspace_position", {})
        if not isinstance(joint_map, dict) or not joint_map:
            return
        val = None
        for _, v in joint_map.items():
            try:
                val = float(v)
                break
            except Exception:
                continue
        if val is None:
            return
        if use_pregrasp:
            pregrasp_width = float(np.clip(val, 0.01, 0.08))
            self.controller.dynamic_params["grip_open"] = pregrasp_width
            self._set_controller_gripper_target(pregrasp_width, hold=False)
        else:
            close_width = float(np.clip(val, 0.0, 0.05))
            setattr(enum_eval_module, "GRIP_CLOSE_MUG", close_width)
            self.controller.dynamic_params["grip_close"] = close_width
            self._set_controller_gripper_target(close_width, hold=True)

    def _get_target_bbox_world(self, target_path: str):
        try:
            stage = get_current_stage()
            candidates = [f"{target_path}/Mesh", f"{target_path}/Body", target_path]
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            for cand in candidates:
                prim = stage.GetPrimAtPath(cand)
                if not prim.IsValid():
                    continue
                bbox = bbox_cache.ComputeWorldBound(prim).GetBox()
                mn = np.array([bbox.GetMin()[0], bbox.GetMin()[1], bbox.GetMin()[2]], dtype=np.float32)
                mx = np.array([bbox.GetMax()[0], bbox.GetMax()[1], bbox.GetMax()[2]], dtype=np.float32)
                if np.all(np.isfinite(mn)) and np.all(np.isfinite(mx)) and float(mx[2] - mn[2]) > 1e-4:
                    return mn, mx
        except Exception:
            pass
        return None, None

    def _get_prim_world_translation(self, prim_path: str) -> Optional[np.ndarray]:
        try:
            stage = get_current_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None
            xformable = UsdGeom.Xformable(prim)
            world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_tf.ExtractTranslation()
            pos = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
            if np.all(np.isfinite(pos)):
                return pos
        except Exception:
            pass
        return None

    def _get_basket_center_world(self) -> np.ndarray:
        default_center = np.array(
            BASKET_PLACE_SLOTS.get("PLACE_mug_2", np.array([0.0, 0.45, 0.18], dtype=np.float32)),
            dtype=np.float32,
        )
        try:
            basket_origin = self._get_prim_world_translation("/World/Basket")
            parts = [
                "/World/Basket/Bottom",
                "/World/Basket/Front",
                "/World/Basket/Back",
                "/World/Basket/Left",
                "/World/Basket/Right",
            ]
            mins: List[np.ndarray] = []
            maxs: List[np.ndarray] = []
            for part in parts:
                mn, mx = self._get_target_bbox_world(part)
                if mn is None or mx is None:
                    continue
                mins.append(np.array(mn, dtype=np.float32))
                maxs.append(np.array(mx, dtype=np.float32))

            center = np.array(default_center, dtype=np.float32)
            if mins and maxs:
                mn_all = np.min(np.stack(mins, axis=0), axis=0)
                mx_all = np.max(np.stack(maxs, axis=0), axis=0)
                bbox_center = 0.5 * (mn_all + mx_all)
                center[2] = float(max(default_center[2], bbox_center[2]))
                if basket_origin is None:
                    center[0] = float(bbox_center[0])
                    center[1] = float(bbox_center[1])
            if basket_origin is not None:
                center[0] = float(basket_origin[0])
                center[1] = float(basket_origin[1])

            logger.info(
                "[BasketCenter] origin=%s resolved=%s",
                None if basket_origin is None else np.array(basket_origin, dtype=np.float32).round(5).tolist(),
                np.array(center, dtype=np.float32).round(5).tolist(),
            )
            return center.astype(np.float32)
        except Exception:
            return default_center

    def _get_mug_body_center_world(self, target_path: str) -> Optional[np.ndarray]:
        if self.controller is None:
            return None
        body_path = f"{target_path}/Body"
        try:
            pos, _ = self.controller._get_safe_world_pose(body_path)
            pos = np.array(pos, dtype=np.float32)
            if np.all(np.isfinite(pos)):
                return pos
        except Exception:
            pass
        return None

    def _quat_from_axis_angle(self, axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.array(axis, dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        axis = axis / norm
        half = 0.5 * float(angle_rad)
        s = float(np.sin(half))
        return np.array([float(np.cos(half)), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)

    def _compute_com_angled_orientation(self, grasp_center: np.ndarray, obj_height: float, force_angle: bool = False) -> np.ndarray:
        base = np.array(self.controller.ee_orientation, dtype=np.float32)
        if not force_angle and obj_height >= 0.10:
            return base
        approach_xy = np.array([-float(grasp_center[0]), -float(grasp_center[1]), 0.0], dtype=np.float32)
        if float(np.linalg.norm(approach_xy[:2])) < 1e-6:
            approach_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        approach_xy = approach_xy / max(1e-6, float(np.linalg.norm(approach_xy)))
        tilt_axis = np.array([-approach_xy[1], approach_xy[0], 0.0], dtype=np.float32)
        tilt_q = self._quat_from_axis_angle(tilt_axis, np.deg2rad(0.0))
        q = self._quat_mul(tilt_q, base)
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-8 else base

    def _compute_pick_targets_with_affordance(self, target_path: str, pick_part: str) -> Dict[str, Any]:
        if self.controller is None:
            return {}
        if pick_part not in AFFORDANCE_SCHEMA:
            return self.controller.compute_pick_targets(target_path)

        self.controller.ensure_object_dynamic(target_path)
        safe_pos, safe_rot = self.controller._get_safe_world_pose(target_path)
        try:
            enum_eval_module.VisualHelper.create_or_move_proxy(pos=safe_pos)
        except Exception:
            pass

        isaac_grasp = self._select_isaac_grasp_for_target(target_path)
        if isaac_grasp is not None:
            rel_pos = np.array(isaac_grasp.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            rel_rot = np.array(isaac_grasp.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            world_grasp = np.array(safe_pos, dtype=np.float32) + self._quat_rotate(np.array(safe_rot, dtype=np.float32), rel_pos)
            bbox_min, bbox_max = self._get_target_bbox_world(target_path)
            if bbox_min is None or bbox_max is None:
                obj_min = np.array([float(safe_pos[0]) - 0.03, float(safe_pos[1]) - 0.03, float(safe_pos[2])], dtype=np.float32)
                obj_max = np.array([float(safe_pos[0]) + 0.03, float(safe_pos[1]) + 0.03, float(safe_pos[2]) + 0.15], dtype=np.float32)
            else:
                obj_min = np.array(bbox_min, dtype=np.float32)
                obj_max = np.array(bbox_max, dtype=np.float32)
            obj_min_z = float(obj_min[2])
            obj_max_z = float(obj_max[2])
            obj_center = 0.5 * (obj_min + obj_max)
            body_center_world = self._get_mug_body_center_world(target_path)
            if body_center_world is not None:
                obj_center[0] = float(body_center_world[0])
                obj_center[1] = float(body_center_world[1])
            obj_center_z = float(obj_center[2])
            obj_height = max(0.05, obj_max_z - obj_min_z)
            body_pick = pick_part in {"body_outer", "body_inner", "handle"}
            if body_pick:
                world_grasp[0] = float(obj_center[0])
                world_grasp[1] = float(obj_center[1])
                world_grasp[2] = float(np.clip(min(float(world_grasp[2]), obj_center_z + 0.010), obj_min_z + 0.004, obj_max_z + 0.03))
                # Keep grasp above a safe floor near observed object pose to avoid table penetration.
                world_grasp[2] = max(float(world_grasp[2]), float(safe_pos[2]) - 0.015)
                world_ori = self._compute_com_angled_orientation(world_grasp, obj_height, force_angle=True)
                side_sign = -1.0 if float(world_grasp[0]) >= 0.0 else 1.0
                pre_grasp = world_grasp + np.array([0.045 * side_sign, 0.0, 0.11], dtype=np.float32)
            else:
                world_ori = self._quat_mul(np.array(safe_rot, dtype=np.float32), rel_rot)
                pre_grasp = world_grasp + np.array([0.0, 0.0, 0.10], dtype=np.float32)
            retreat = world_grasp + np.array([0.0, 0.0, 0.18], dtype=np.float32)
            target_id = str(target_path).split("/")[-1]
            self._active_isaac_grasp_by_target[target_id] = isaac_grasp
            return {
                "pre_grasp": pre_grasp,
                "grasp": world_grasp,
                "retreat": retreat,
                "orientation": world_ori,
                "affordance_part": pick_part,
                "isaac_grasp": isaac_grasp.get("name", "grasp_0"),
            }

        z_adjust = float(self.controller.dynamic_params.get("approach_z_adjust", -0.02))
        reach_adjust = float(self.controller.dynamic_params.get("reach_adjust", 0.0))

        mug_name = str(target_path).split("/")[-1]
        angle_deg = float(self._mug_angles.get(mug_name, 0.0))
        rad = np.deg2rad(angle_deg)

        offset = np.zeros(3, dtype=np.float32)
        if pick_part == "handle":
            offset = np.array([0.07 * np.cos(rad), 0.07 * np.sin(rad), -0.03], dtype=np.float32)
        elif pick_part in {"body_outer", "body_inner"}:
            offset = np.array([0.03 * np.cos(rad), 0.03 * np.sin(rad), -0.02], dtype=np.float32)
        elif pick_part == "bottom":
            offset = np.array([0.0, 0.0, -0.02], dtype=np.float32)

        bbox_min, bbox_max = self._get_target_bbox_world(target_path)
        if bbox_min is None or bbox_max is None:
            obj_min = np.array([float(safe_pos[0]) - 0.03, float(safe_pos[1]) - 0.03, float(safe_pos[2])], dtype=np.float32)
            obj_max = np.array([float(safe_pos[0]) + 0.03, float(safe_pos[1]) + 0.03, float(safe_pos[2]) + 0.15], dtype=np.float32)
        else:
            obj_min = np.array(bbox_min, dtype=np.float32)
            obj_max = np.array(bbox_max, dtype=np.float32)
        obj_min_z = float(obj_min[2])
        obj_max_z = float(obj_max[2])
        obj_center = 0.5 * (obj_min + obj_max)
        body_center_world = self._get_mug_body_center_world(target_path)
        if body_center_world is not None:
            obj_center[0] = float(body_center_world[0])
            obj_center[1] = float(body_center_world[1])
        obj_center_z = float(obj_center[2])
        obj_height = max(0.05, obj_max_z - obj_min_z)

        handle_z_adjust = HANDLE_GRASP_Z_ADJUST if pick_part == "handle" else 0.0
        if pick_part == "handle":
            target_z = obj_center_z + z_adjust + handle_z_adjust
        elif pick_part in {"body_outer", "body_inner"}:
            target_z = obj_center_z + z_adjust - 0.01
        else:  # bottom
            target_z = obj_min_z + 0.03

        grasp_center = safe_pos + offset + self.controller.calib_offset + np.array([reach_adjust, 0.0, 0.0], dtype=np.float32)
        grasp_center[2] = float(target_z)
        body_pick = pick_part in {"body_outer", "body_inner"}
        force_angle = body_pick
        if body_pick or obj_height < 0.10:
            # Pin XY to mug body center so the gripper closes around the mug middle.
            grasp_center[0] = float(obj_center[0])
            grasp_center[1] = float(obj_center[1])
            logger.info("[WF-COM] target=%s part=%s center_xy=(%.3f, %.3f)", target_path.split('/')[-1], pick_part, grasp_center[0], grasp_center[1])

        # Keep grasp above the table but avoid over-lifting.
        if obj_height < 0.10:
            min_z = obj_min_z + 0.004
            grasp_center[2] = min(grasp_center[2], obj_center_z + 0.004)
        else:
            min_z = obj_min_z + 0.016
            if body_pick:
                grasp_center[2] = min(grasp_center[2], obj_center_z + 0.012)
        max_z = obj_max_z + 0.03
        grasp_center[2] = float(np.clip(grasp_center[2], min_z, max_z))
        # Keep grasp above a safe floor near observed object pose to avoid diving under mugs.
        grasp_center[2] = max(float(grasp_center[2]), float(safe_pos[2]) - 0.015)

        orientation = self._compute_com_angled_orientation(grasp_center, obj_height, force_angle=force_angle)
        pre_lift = max(0.10, min(0.16, 0.6 * obj_height))
        pregrasp_offset = np.array([0.0, 0.0, pre_lift], dtype=np.float32)
        if body_pick or obj_height < 0.10:
            approach_xy = np.array([-float(grasp_center[0]), -float(grasp_center[1])], dtype=np.float32)
            approach_norm = float(np.linalg.norm(approach_xy))
            if approach_norm < 1e-6:
                approach_xy = np.array([0.0, -1.0], dtype=np.float32)
                approach_norm = 1.0
            approach_xy = approach_xy / approach_norm
            backoff = 0.065
            target_id = str(target_path).split("/")[-1]
            if target_id == "mug_0":
                backoff = 0.085
            elif target_id == "mug_1":
                backoff = 0.045
            pregrasp_offset += np.array([approach_xy[0] * backoff, approach_xy[1] * backoff, 0.02], dtype=np.float32)
        pre_grasp = grasp_center + pregrasp_offset
        retreat = grasp_center + np.array([0.0, 0.0, max(0.16, pre_lift + 0.04)], dtype=np.float32)
        return {
            "pre_grasp": pre_grasp,
            "grasp": grasp_center,
            "retreat": retreat,
            "orientation": orientation,
            "affordance_part": pick_part,
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

    def _plan_rrt_like_waypoints(self, target_position: np.ndarray) -> List[np.ndarray]:
        if self.controller is None:
            return [np.array(target_position, dtype=np.float32)]
        try:
            start_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
        except Exception:
            return [np.array(target_position, dtype=np.float32)]

        start = np.array(start_pos, dtype=np.float32)
        goal = np.array(target_position, dtype=np.float32)
        dist = float(np.linalg.norm(goal - start))
        dz = abs(float(goal[2] - start[2]))

        # Shortest-path first: go straight when the motion is already safe-scale.
        if dist <= 0.45 and dz <= 0.08:
            return [goal]

        # Minimal detour: one vertical clearance point above goal XY, then descend.
        clearance = np.array([goal[0], goal[1], max(float(start[2]), float(goal[2])) + 0.06], dtype=np.float32)
        if float(np.linalg.norm(clearance - goal)) < 1e-3:
            return [goal]
        return [clearance, goal]

    def _extract_place_slot_from_action(self, action: str, target_id: str = "") -> str:
        _ = action
        _ = target_id
        # Force all destinations to basket center.
        return "PLACE_mug_2"

    def _plan_place_rrt_waypoints(self, target_position: np.ndarray, place_slot: str) -> List[np.ndarray]:
        _ = place_slot
        if self.controller is None:
            return [np.array(target_position, dtype=np.float32)]
        try:
            start_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
        except Exception:
            return [np.array(target_position, dtype=np.float32)]

        start = np.array(start_pos, dtype=np.float32)
        goal = np.array(target_position, dtype=np.float32)
        dist = float(np.linalg.norm(goal - start))
        dz = abs(float(goal[2] - start[2]))

        # Place motions are shortest-first as well.
        if dist <= 0.28 and dz <= 0.08:
            return [goal]

        # Use smaller, axis-wise transport waypoints so IK does not fail on one large diagonal jump.
        # If the hand is already high after grasp, do not lift further before the horizontal transfer.
        transit_z = max(float(goal[2]) + 0.06, 0.24)
        if float(start[2]) > transit_z:
            transit_z = float(start[2])
        else:
            transit_z = min(float(start[2]) + 0.03, transit_z)

        shift_y = np.array([start[0], goal[1], transit_z], dtype=np.float32)
        shift_x = np.array([goal[0], goal[1], transit_z], dtype=np.float32)
        pre_drop = np.array([goal[0], goal[1], max(float(goal[2]) + 0.04, min(transit_z, float(goal[2]) + 0.08))], dtype=np.float32)

        waypoints: List[np.ndarray] = []
        for wp in [shift_y, shift_x, pre_drop, goal]:
            if not waypoints or float(np.linalg.norm(np.array(wp, dtype=np.float32) - waypoints[-1])) > 1e-3:
                waypoints.append(np.array(wp, dtype=np.float32))
        return waypoints

    def _micro_retreat_after_ik_failure(self, base_position: np.ndarray, target_orientation: np.ndarray) -> bool:
        if self.world is None or self.controller is None:
            return False
        if self._grasp_emergency_stop:
            self.controller.last_error_message = self._grasp_emergency_reason or "grasp jump safety stop"
            return False
        try:
            retreat = np.array(base_position, dtype=np.float32) + np.array([0.0, 0.0, 0.03], dtype=np.float32)
            return self.controller.move_end_effector_to(
                retreat,
                target_orientation,
                self.world,
                steps=self._scale_motion_steps(35),
                position_tolerance=0.05,
                orientation_tolerance=0.6,
                stage_name="ik-fail-retreat",
            )
        except Exception:
            return False

    def _move_end_effector_with_ablation(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        steps: int,
        stage_name: str,
        route_hint: str = "",
    ) -> bool:
        if self.world is None or self.controller is None:
            return False
        if not self.enable_ik:
            # IK ablation: intentionally skip IK tracking to measure impact.
            self.controller.last_error_message = f"IK disabled (ablation) at {stage_name}"
            return False

        if not self.enable_rrt and (not self.enable_cumotion_style):
            ok = self.controller.move_end_effector_to(
                target_position,
                target_orientation,
                self.world,
                steps=self._scale_motion_steps(steps),
                stage_name=stage_name,
            )
            if not ok:
                self._micro_retreat_after_ik_failure(target_position, target_orientation)
            return ok

        # cuMotion-style conservative execution: always move through waypoints with slower segments.
        if route_hint.startswith("place:"):
            place_slot = route_hint.split(":", 1)[1] or "BASKET_HIGH"
            waypoints = self._plan_place_rrt_waypoints(np.array(target_position, dtype=np.float32), place_slot)
            self._publish_rrt_waypoint_markers(waypoints, prefix=f"{stage_name}_place")
        else:
            waypoints = self._plan_rrt_like_waypoints(np.array(target_position, dtype=np.float32))
        # Shortest-path bias: do not inflate with extra midpoints unless the planner already requested them.
        segment_steps = self._scale_motion_steps(max(18, int(steps / max(1, len(waypoints)))))
        for idx, waypoint in enumerate(waypoints):
            ok = self.controller.move_end_effector_to(
                waypoint,
                target_orientation,
                self.world,
                steps=segment_steps,
                stage_name=f"{stage_name}-rrt-{idx}",
            )
            if not ok:
                self._micro_retreat_after_ik_failure(waypoint, target_orientation)
                return False
        return True

    def _try_grasp_with_fallback(self, target_path: str) -> bool:
        if self.controller is None or self.world is None:
            return False
        if self.disable_attachment:
            return False

        close_width = float(self.controller.dynamic_params.get("grip_close", getattr(enum_eval_module, "GRIP_CLOSE_MUG", 0.005)))
        self._set_controller_gripper_target(close_width, hold=True)
        self.controller.close_gripper(self.world, target_path)
        if getattr(self.controller, "grasped_object", None) == target_path:
            return True

        # Suction-like fallback: force attach if finger closure could not latch.
        try:
            attach_fn = getattr(self.controller, "_attach_object", None)
            if callable(attach_fn):
                attach_fn(target_path)
                if getattr(self.controller, "grasped_object", None) == target_path:
                    logger.info("Fallback suction attach succeeded for %s", target_path)
                    return True
        except Exception as exc:
            logger.warning("Fallback suction attach failed for %s: %s", target_path, exc)

        return False

    def _force_place_in_basket(self, target_path: str, place_slot: str) -> bool:
        # Recovery place that still follows robot motion; no object teleport.
        if self.controller is None or self.world is None:
            return False
        try:
            place_targets = self.controller.compute_place_targets(place_slot)
            if str(place_slot).startswith("PLACE_mug_"):
                basket_center = self._get_basket_center_world()
                for key in ("pre_place", "place", "retreat"):
                    if key in place_targets:
                        pose = np.array(place_targets[key], dtype=np.float32)
                        pose[0] = float(basket_center[0])
                        pose[1] = float(basket_center[1])
                        place_targets[key] = pose

            if getattr(self.controller, "grasped_object", None) != target_path:
                # Try to re-grasp from current pose instead of warping object.
                close_width = float(self.controller.dynamic_params.get("grip_close", getattr(enum_eval_module, "GRIP_CLOSE_MUG", 0.005)))
                self._set_controller_gripper_target(close_width, hold=True)
                self.controller.close_gripper(self.world, target_path)
                if getattr(self.controller, "grasped_object", None) != target_path:
                    return False

            moved = self._move_end_effector_with_ablation(
                place_targets["pre_place"],
                place_targets["orientation"],
                steps=70,
                stage_name="recovery-pre-place",
                route_hint=f"place:{place_slot}",
            )
            if moved:
                moved = self._move_end_effector_with_ablation(
                    place_targets["place"],
                    place_targets["orientation"],
                    steps=50,
                    stage_name="recovery-place",
                    route_hint=f"place:{place_slot}",
                )
            if moved:
                moved = self._converge_to_place_center(place_slot, place_targets, stage_prefix="recovery-center-lock")
            if not moved:
                return False
            if not self._can_release_at_slot(place_slot):
                self._ensure_transport_gripper_hold(target_path)
                logger.warning("[ReleaseGuard] recovery place reached=false slot=%s (basket_zone=%s); keep closed", place_slot, self._is_hand_in_basket_release_zone())
                return False

            self._prepare_release_depenetration(target_path)
            self.controller.open_gripper(self.world)
            if self.enable_action_waits and self.verify_wait_sec > 0:
                wait_steps(self.world, self.verify_wait_sec, self.controller)
            return bool(self.controller.verify_placement(target_path))
        except Exception as exc:
            logger.warning("Emergency place failed for %s: %s", target_path, exc)
            return False

    def _is_hand_near_position(self, target_position: np.ndarray, tol_xy: float = 0.06, tol_z: float = 0.08) -> bool:
        if self.controller is None:
            return False
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            hand = np.array(hand_pos, dtype=np.float32)
            target = np.array(target_position, dtype=np.float32)
            dx = abs(float(hand[0] - target[0]))
            dy = abs(float(hand[1] - target[1]))
            dz = abs(float(hand[2] - target[2]))
            return bool(dx <= tol_xy and dy <= tol_xy and dz <= tol_z)
        except Exception:
            return False

    def _is_hand_in_basket_release_zone(self, margin_xy: float = 0.05, z_margin_low: float = 0.03, z_margin_high: float = 0.22) -> bool:
        if self.controller is None:
            return False
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            hand = np.array(hand_pos, dtype=np.float32)

            # Prefer actual basket geometry bounds.
            mn, mx = self._get_target_bbox_world("/World/Basket")
            if mn is None or mx is None:
                center = self._get_basket_center_world()
                half = np.array([0.22, 0.16, 0.10], dtype=np.float32)
                mn = center - half
                mx = center + half
            mn = np.array(mn, dtype=np.float32)
            mx = np.array(mx, dtype=np.float32)

            in_x = bool((float(mn[0]) - margin_xy) <= float(hand[0]) <= (float(mx[0]) + margin_xy))
            in_y = bool((float(mn[1]) - margin_xy) <= float(hand[1]) <= (float(mx[1]) + margin_xy))
            in_z = bool((float(mn[2]) - z_margin_low) <= float(hand[2]) <= (float(mx[2]) + z_margin_high))
            return bool(in_x and in_y and in_z)
        except Exception:
            return False

    def _is_hand_near_basket_center_relaxed(self, tol_xy: float = 0.22, tol_z: float = 0.28) -> bool:
        if self.controller is None:
            return False
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            hand = np.array(hand_pos, dtype=np.float32)
            center = self._get_basket_center_world()
            dx = abs(float(hand[0] - center[0]))
            dy = abs(float(hand[1] - center[1]))
            dz = abs(float(hand[2] - center[2]))
            return bool(dx <= tol_xy and dy <= tol_xy and dz <= tol_z)
        except Exception:
            return False

    def _can_release_at_slot(self, place_slot: str) -> bool:
        if self.controller is None:
            return False
        try:
            place_targets = self.controller.compute_place_targets(place_slot)
            target = np.array(place_targets["place"], dtype=np.float32)
            # Align release check with real basket center used in transfer path.
            basket_center = self._get_basket_center_world()
            target[0] = float(basket_center[0])
            target[1] = float(basket_center[1])
            near_slot = self._is_hand_near_position(target, tol_xy=0.10, tol_z=0.12)
            in_basket_zone = self._is_hand_in_basket_release_zone()
            near_center_relaxed = self._is_hand_near_basket_center_relaxed()
            return bool(near_slot or in_basket_zone or near_center_relaxed)
        except Exception:
            return False

    def _converge_to_place_center(self, place_slot: str, place_targets: Dict[str, np.ndarray], stage_prefix: str) -> bool:
        if self.controller is None:
            return False
        target_place = np.array(place_targets["place"], dtype=np.float32)
        if str(place_slot).startswith("PLACE_mug_"):
            basket_center = self._get_basket_center_world()
            target_place[0] = float(basket_center[0])
            target_place[1] = float(basket_center[1])
        orientation = np.array(place_targets["orientation"], dtype=np.float32)

        if self._can_release_at_slot(place_slot):
            return True

        retries: List[Tuple[int, float, float]] = [
            (70, 0.06, 0.08),
        ]
        for attempt_idx, (steps, tol_xy, tol_z) in enumerate(retries, start=1):
            moved = self._move_end_effector_with_ablation(
                target_place,
                orientation,
                steps=self._scale_motion_steps(steps),
                stage_name=f"{stage_prefix}-{attempt_idx}",
                route_hint=f"place:{place_slot}",
            )
            if (not moved) and self.enable_action_waits and self.place_settle_wait_sec > 0:
                wait_steps(self.world, min(0.2, self.place_settle_wait_sec), self.controller)
            if self._is_hand_near_position(target_place, tol_xy=tol_xy, tol_z=tol_z):
                logger.info(
                    "[PlaceCenterLock] slot=%s attempt=%d tol_xy=%.3f tol_z=%.3f",
                    place_slot,
                    attempt_idx,
                    tol_xy,
                    tol_z,
                )
                return True
        logger.warning("[PlaceCenterLock] slot=%s failed to converge to center after retries", place_slot)
        return False

    def _move_with_place_path_planning(
        self,
        place_slot: str,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        stage_name: str,
        steps: int,
        target_path: str = "",
    ) -> bool:
        if self.controller is None:
            return False
        target_position = np.array(target_position, dtype=np.float32)
        if str(place_slot).startswith("PLACE_mug_"):
            basket_center = self._get_basket_center_world()
            target_position[0] = float(basket_center[0])
            target_position[1] = float(basket_center[1])
        waypoints = self._plan_place_rrt_waypoints(np.array(target_position, dtype=np.float32), place_slot)
        if not waypoints:
            waypoints = [np.array(target_position, dtype=np.float32)]

        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            hand_log = np.array(hand_pos, dtype=np.float32).round(5).tolist()
        except Exception:
            hand_log = None
        logger.info(
            "[PlacePathPlan] stage=%s slot=%s hand=%s target=%s first_waypoint=%s",
            stage_name,
            place_slot,
            hand_log,
            np.array(target_position, dtype=np.float32).round(5).tolist(),
            np.array(waypoints[0], dtype=np.float32).round(5).tolist(),
        )

        self._publish_rrt_waypoint_markers(waypoints, prefix=f"{stage_name}_place")
        segment_steps = self._scale_motion_steps(max(16, int(steps / max(1, len(waypoints)))))
        for idx, waypoint in enumerate(waypoints):
            if target_path:
                self._ensure_transport_gripper_hold(target_path)
            stage_id = f"{stage_name}-path-{idx}"
            ok = self.controller.move_end_effector_to(
                np.array(waypoint, dtype=np.float32),
                np.array(target_orientation, dtype=np.float32),
                self.world,
                steps=segment_steps,
                stage_name=stage_id,
            )
            if not ok:
                try:
                    hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
                    hand_str = np.array(hand_pos, dtype=np.float32).round(5).tolist()
                except Exception:
                    hand_str = None
                logger.warning(
                    "[PlacePathFailed] stage=%s idx=%d/%d waypoint=%s hand=%s reason=%s",
                    stage_id,
                    idx,
                    len(waypoints),
                    np.array(waypoint, dtype=np.float32).round(5).tolist(),
                    hand_str,
                    getattr(self.controller, "last_error_message", ""),
                )
                return False
        return True

    def _transport_grasped_target_to_basket_center(self, place_slot: str, target_path: str, stage_prefix: str = "transport") -> bool:
        if self.controller is None:
            return False
        place_targets = self.controller.compute_place_targets(place_slot)
        basket_center = self._get_basket_center_world()
        target_place = np.array(place_targets["place"], dtype=np.float32)
        target_place[0] = float(basket_center[0])
        target_place[1] = float(basket_center[1])
        orientation = np.array(place_targets["orientation"], dtype=np.float32)

        # Keep transport single-purpose: maintain a closed gripper and follow the place RRT path only.
        self._ensure_transport_gripper_hold(target_path)
        moved = self._move_with_place_path_planning(
            place_slot,
            target_place,
            orientation,
            stage_name=f"{stage_prefix}-basket-center",
            steps=100,
            target_path=target_path,
        )
        if not moved:
            return False
        self._ensure_transport_gripper_hold(target_path)
        return True

    def _force_center_transfer_path(self, place_slot: str, target_path: str, stage_prefix: str = "fast") -> bool:
        if self.controller is None:
            return False
        place_targets = self.controller.compute_place_targets(place_slot)
        basket_center = self._get_basket_center_world()
        target_place = np.array(place_targets["place"], dtype=np.float32)
        orientation = np.array(place_targets["orientation"], dtype=np.float32)

        # Overwrite target XY using actual basket center from stage geometry.
        target_place[0] = float(basket_center[0])
        target_place[1] = float(basket_center[1])
        place_targets["place"] = target_place

        pre_place = np.array(place_targets.get("pre_place", target_place.copy()), dtype=np.float32)
        pre_place[0] = float(basket_center[0])
        pre_place[1] = float(basket_center[1])
        place_targets["pre_place"] = pre_place

        # Prefer one RRT-guided transport to basket center to avoid vertical oscillation.
        moved = self._transport_grasped_target_to_basket_center(
            place_slot,
            target_path,
            stage_prefix=f"{stage_prefix}-center",
        )

        # Conservative fallback: pre-place then place, still with the gripper held closed.
        if not moved:
            moved = self._move_with_place_path_planning(
                place_slot,
                pre_place,
                orientation,
                stage_name=f"{stage_prefix}-center-pre-place",
                steps=75,
                target_path=target_path,
            )
            if moved:
                moved = self._move_with_place_path_planning(
                    place_slot,
                    target_place,
                    orientation,
                    stage_name=f"{stage_prefix}-center-place",
                    steps=90,
                    target_path=target_path,
                )

        # Single center-lock attempt.
        if moved and (not self._can_release_at_slot(place_slot)):
            moved = self._converge_to_place_center(place_slot, place_targets, stage_prefix=f"{stage_prefix}-center-lock")
        return bool(moved)

    def _clamp_retreat_above_current_hand(self, pick_targets: Dict[str, Any], min_margin: float = 0.02) -> Dict[str, Any]:
        if self.controller is None:
            return pick_targets
        if not isinstance(pick_targets, dict) or ("retreat" not in pick_targets):
            return pick_targets
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            hand_z = float(np.array(hand_pos, dtype=np.float32)[2])
        except Exception:
            return pick_targets

        retreat = np.array(pick_targets.get("retreat", [0.0, 0.0, hand_z]), dtype=np.float32)
        grasp = np.array(pick_targets.get("grasp", retreat), dtype=np.float32)
        # Keep retreat target at or above current hand height so we never dip while carrying.
        retreat_floor = max(hand_z + float(min_margin), float(grasp[2]) + 0.02)
        if float(retreat[2]) < retreat_floor:
            retreat[2] = retreat_floor
            pick_targets["retreat"] = retreat
            logger.info("[RetreatClamp] raised retreat_z to %.3f (hand_z=%.3f)", float(retreat[2]), hand_z)
        return pick_targets

    def _turn_toward_basket_after_grasp(self, target_path: str) -> bool:
        if self.controller is None or self.world is None:
            return False
        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            hand = np.array(hand_pos, dtype=np.float32)
            place_targets = self.controller.compute_place_targets("PLACE_mug_2")
            basket_ori = np.array(place_targets["orientation"], dtype=np.float32)
            # Slight lift while rotating prevents scraping the mug on table right after grasp.
            turn_pos = np.array([float(hand[0]), float(hand[1]), float(hand[2]) + 0.02], dtype=np.float32)
            self._ensure_transport_gripper_hold(target_path)
            ok = self._move_end_effector_with_ablation(
                turn_pos,
                basket_ori,
                steps=45,
                stage_name="post-grasp-turn-to-basket",
            )
            if ok:
                logger.info("[PostGraspTurn] target=%s turned toward basket", target_path.split('/')[-1])
            return bool(ok)
        except Exception as exc:
            logger.warning("[PostGraspTurn] failed for %s: %s", target_path, exc)
            return False

    def _remove_followup_actions_for_target(self, action_queue: List[str], target_id: str) -> None:
        if not target_id:
            return
        removable_verbs = {"retreat", "pre_place", "place", "release", "home"}
        before = len(action_queue)
        kept: List[str] = []
        for action in action_queue:
            tokens = action.split()
            if len(tokens) >= 2 and tokens[0] in removable_verbs and tokens[1] == target_id:
                continue
            kept.append(action)
        if len(kept) != before:
            removed = before - len(kept)
            logger.info("[ActionPrune] target=%s removed_followups=%d", target_id, removed)
        action_queue[:] = kept

    def _is_target_grasped(self, target_path: str) -> bool:
        if self.controller is None:
            return False
        grasped = bool(str(getattr(self.controller, "grasped_object", "") or "") == str(target_path or ""))
        if grasped:
            return True
        if self.disable_attachment:
            return self._is_physical_grasp_likely(str(target_path or ""))
        return False

    def _stabilize_between_targets(self, current_target: str = "") -> None:
        if self.controller is None or self.world is None:
            return
        grasped_path = str(getattr(self.controller, "grasped_object", "") or "")
        if not grasped_path:
            return

        grasped_id = grasped_path.split("/")[-1]
        if current_target and grasped_id == current_target:
            return

        placed_like = False
        try:
            placed_like = bool(self.controller.verify_placement(grasped_path))
        except Exception:
            placed_like = False

        if current_target and (grasped_id != current_target):
            logger.warning("[CarryOver] grasped=%s while planning for %s; forcing detach cleanup", grasped_id, current_target)

        try:
            self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
            self.controller.open_gripper(self.world)
        except Exception:
            pass

        still_grasped = str(getattr(self.controller, "grasped_object", "") or "")
        if still_grasped:
            try:
                stage = get_current_stage()
                prim = stage.GetPrimAtPath(still_grasped)
                if prim.IsValid():
                    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
            except Exception:
                pass
            try:
                if hasattr(self.controller, "_set_collision_enabled"):
                    self.controller._set_collision_enabled(still_grasped, True)
                if hasattr(self.controller, "_set_gripper_collision_enabled"):
                    self.controller._set_gripper_collision_enabled(True)
            except Exception:
                pass
            try:
                self.controller.grasped_object = None
                self.controller.attach_locked_until_open = False
                self.controller.gripper_hold_enabled = False
            except Exception:
                pass

        try:
            if (current_target and grasped_id != current_target) or (not placed_like):
                self.controller.move_to_joint_pose("HOME", self.world, steps=self._scale_motion_steps(45))
        except Exception:
            pass

    def _collect_robot_motion_probe(self, action: str, target_id: str = "") -> Dict[str, Any]:
        probe: Dict[str, Any] = {
            "action": str(action),
            "target": str(target_id),
            "timestamp": time.time(),
            "sim_device": self.sim_device,
            "ik_method": self.ik_method,
        }

        try:
            probe["is_simulating"] = bool(SimulationManager.is_simulating())
        except Exception:
            probe["is_simulating"] = None

        if self.controller is None:
            return probe

        try:
            hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
            probe["eef_position"] = np.array(hand_pos, dtype=np.float32).round(5).tolist()
        except Exception:
            probe["eef_position"] = None

        target_path = f"/World/{target_id}" if target_id else ""
        if target_path:
            try:
                obj_pos, _ = self.controller._get_safe_world_pose(target_path)
                probe["target_position"] = np.array(obj_pos, dtype=np.float32).round(5).tolist()
            except Exception:
                probe["target_position"] = None
            try:
                probe["target_verified"] = bool(self.controller.verify_placement(target_path))
            except Exception:
                probe["target_verified"] = None

        probe["grasped_object"] = str(getattr(self.controller, "grasped_object", "") or "")
        return probe


    def execute_simulation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[4/5] execute_simulation_plan")

        if self.world is None:
            self.setup_simulation_environment()

        if self.controller is None:
            raise RuntimeError("Simulation controller is not initialized")

        t_exec0 = time.time()

        self._grasp_emergency_stop = False
        self._grasp_emergency_reason = ""
        self._grasp_prev_pos_by_target.clear()
        self._residual_events = []

        use_replan_grasp_offset = self._plan_uses_replan_grasp_offset(plan)

        execution = {
            "plan": plan,
            "executed_steps": [],
            "executed_length": 0,
            "collisions": 0,
            "status": "success",
            "failed_action": None,
            "failed_reason": None,
            "timestamp": time.time(),
            "motion_probe": [],
        }

        action_queue: List[str] = list(plan.get("steps", []))
        action_retry_counts: Dict[str, int] = {}
        last_success_action: Optional[str] = None

        while action_queue:
            action = action_queue.pop(0)
            try:
                action_target = action.split()[1] if len(action.split()) > 1 else None
            except Exception:
                action_target = None
            if action_target and action_target in self.failed_mugs:
                logger.info("Skipping %s because %s reached retry limit", action, action_target)
                continue
            if action_target and action_target in self.placed_mugs:
                logger.info("Skipping %s because %s is already placed", action, action_target)
                continue

            # State guards to prevent contradictory action loops.
            target_path_guard = f"/World/{action_target}" if action_target else ""
            target_grasped_guard = bool(target_path_guard and self._is_target_grasped(target_path_guard))
            target_placed_guard = bool(target_path_guard and self.controller and self.controller.verify_placement(target_path_guard))
            verb_guard = action.split()[0] if action else ""
            if verb_guard in {"pick", "open", "approach", "grasp", "close"} and target_grasped_guard:
                logger.info("[StateGuard] skip %s: %s already grasped", action, action_target)
                continue
            if verb_guard in {"retreat", "pre_place", "place", "release"} and (not target_grasped_guard) and (not target_placed_guard):
                logger.info("[StateGuard] skip %s: %s not grasped", action, action_target)
                continue

            step_ok = True
            if self.controller is not None:
                # Clear stale errors from previous actions so transfer steps are not skipped incorrectly.
                self.controller.last_error_message = ""
            probe_before = self._collect_robot_motion_probe(action, action_target or "")
            try:
                if action.startswith("pick"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    affordance = self._get_affordance_for_target(target_id)
                    pick_part = affordance.get("part", "body_outer")
                    pick_targets = self._compute_pick_targets_with_affordance(path, pick_part)
                    pick_targets = self._adjust_pick_targets_for_execution(path, pick_targets, use_replan_grasp_offset)
                    pick_targets = self._clamp_retreat_above_current_hand(pick_targets, min_margin=0.025)
                    print(f"[PICK] target={target_id} pre_grasp={pick_targets.get('pre_grasp')} grasp={pick_targets.get('grasp')}", flush=True)
                    self._prepare_release_depenetration(path)
                    self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                    self.controller.open_gripper(self.world)
                    moved = self._move_end_effector_with_ablation(
                        pick_targets["pre_grasp"], pick_targets["orientation"], steps=90, stage_name="pre-grasp"
                    )
                    if moved:
                        if self.enable_action_waits and self.pick_wait_sec > 0:
                            wait_steps(self.world, self.pick_wait_sec, self.controller)
                        moved = self._move_end_effector_with_ablation(
                            pick_targets["grasp"], pick_targets["orientation"], steps=75, stage_name="grasp-approach"
                        )
                    if target_id:
                        self.attempted_mugs.add(target_id)
                    if not moved:
                        step_ok = False

                elif action.startswith("open"):
                    target_id = action.split()[1]
                    self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                    self.controller.open_gripper(self.world)
                    step_ok = True

                elif action.startswith("approach"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    affordance = self._get_affordance_for_target(target_id)
                    pick_part = affordance.get("part", "handle")
                    pick_targets = self._compute_pick_targets_with_affordance(path, pick_part)
                    moved = self._move_end_effector_with_ablation(
                        pick_targets["pre_grasp"], pick_targets["orientation"], steps=90, stage_name="pre-grasp"
                    )
                    if not moved:
                        step_ok = False

                elif action.startswith("grasp"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    affordance = self._get_affordance_for_target(target_id)
                    pick_part = affordance.get("part", "handle")
                    pick_targets = self._compute_pick_targets_with_affordance(path, pick_part)
                    pick_targets = self._adjust_pick_targets_for_execution(path, pick_targets, use_replan_grasp_offset)
                    moved = self._move_end_effector_with_ablation(
                        pick_targets["grasp"], pick_targets["orientation"], steps=75, stage_name="grasp-approach"
                    )
                    if not moved:
                        step_ok = False

                elif action.startswith("close"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    # At grasp height: keep current opening and close directly.
                    self._apply_isaac_grasp_cspace_for_target(target_id, use_pregrasp=True)
                    self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                    if self.enable_action_waits and self.pre_close_wait_sec > 0:
                        wait_steps(self.world, self.pre_close_wait_sec, self.controller)
                    self._apply_isaac_grasp_cspace_for_target(target_id, use_pregrasp=False)
                    close_width = float(self.controller.dynamic_params.get("grip_close", getattr(enum_eval_module, "GRIP_CLOSE_MUG", 0.005)))
                    self._set_controller_gripper_target(close_width, hold=True)
                    self.controller.close_gripper(self.world, path)
                    score, reason = self._evaluate_grasp_quality(path)
                    quality_min = self._adaptive_grasp_quality_min(target_id)
                    self._update_grasp_feedback(path, score, quality_min)
                    logger.info("[GraspQuality] target=%s score=%.3f min=%.3f (%s)", target_id, score, quality_min, reason)
                    grasped = str(getattr(self.controller, "grasped_object", "") or "") == path
                    if self.disable_attachment and (not grasped):
                        grasped = self._is_physical_grasp_likely(path)
                    step_ok = bool((score >= quality_min) or grasped)
                    if (score < quality_min) and grasped:
                        logger.warning("[GraspQualityRelaxed] target=%s score=%.3f < %.3f but attached; continue", target_id, score, quality_min)
                    if not step_ok and self.controller is not None:
                        self.controller.last_error_message = f"grasp quality low ({score:.3f} < {quality_min:.3f})"

                elif action.startswith("retreat"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    self._ensure_transport_gripper_hold(path)
                    grasped = str(getattr(self.controller, "grasped_object", "") or "")
                    if grasped == path:
                        # Transport already handled by explicit place action / fast-place path.
                        # Skip retreat-to-pre-place to avoid redundant up/down oscillation.
                        moved = True
                        logger.info("[RetreatSkip] target=%s grasped; defer transfer to place action", target_id)
                    else:
                        affordance = self._get_affordance_for_target(target_id)
                        pick_part = affordance.get("part", "handle")
                        pick_targets = self._compute_pick_targets_with_affordance(path, pick_part)
                        pick_targets = self._clamp_retreat_above_current_hand(pick_targets, min_margin=0.025)
                        moved = self._move_end_effector_with_ablation(
                            pick_targets["retreat"], pick_targets["orientation"], steps=90, stage_name="post-grasp-retreat"
                        )
                    if not moved:
                        step_ok = False

                elif action.startswith("pre_place"):
                    target_id = action.split()[1]
                    target_path = f"/World/{target_id}"
                    if self._is_target_grasped(target_path):
                        # Place action computes and executes the center transfer path directly.
                        moved = True
                        logger.info("[PrePlaceSkip] target=%s grasped; handled inside place", target_id)
                    else:
                        place_slot = self._extract_place_slot_from_action(action, target_id)
                        place_targets = self.controller.compute_place_targets(place_slot)
                        self._ensure_transport_gripper_hold(target_path)
                        moved = self._move_end_effector_with_ablation(
                            place_targets["pre_place"], place_targets["orientation"], steps=90, stage_name="pre-place", route_hint=f"place:{place_slot}"
                        )
                    if not moved:
                        step_ok = False

                elif action.startswith("release"):
                    target_id = action.split()[1]
                    place_slot = self._extract_place_slot_from_action(action, target_id)
                    if not self._can_release_at_slot(place_slot):
                        step_ok = False
                        if self.controller is not None:
                            self.controller.last_error_message = f"release blocked: not at slot {place_slot}"
                    else:
                        self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                        self.controller.open_gripper(self.world)
                        if self.enable_action_waits and self.verify_wait_sec > 0:
                            wait_steps(self.world, self.verify_wait_sec, self.controller)
                    if target_id:
                        target_path = f"/World/{target_id}"
                        if self.controller.verify_placement(target_path):
                            self.placed_mugs.add(target_id)
                        else:
                            step_ok = False
                            execution["status"] = "partial"
                            if execution.get("failed_action") is None:
                                execution["failed_action"] = action
                                execution["failed_reason"] = f"release placement verify failed for {target_id}"
                            force_slot = "PLACE_mug_2"
                            forced = self._force_place_in_basket(target_path, force_slot)
                            if forced:
                                if self.enable_action_waits and self.verify_wait_sec > 0:
                                    wait_steps(self.world, self.verify_wait_sec, self.controller)
                                if self.controller.verify_placement(target_path):
                                    self.placed_mugs.add(target_id)
                                    step_ok = True
                        self.attempted_mugs.add(target_id)
                elif action.startswith("home"):
                    self.controller.move_to_joint_pose("HOME", self.world, steps=self._scale_motion_steps(60))
                    step_ok = True

                elif action.startswith("grab"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    affordance = self._get_affordance_for_target(target_id)
                    pick_part = affordance.get("part", "body_outer")
                    pick_targets = self._compute_pick_targets_with_affordance(path, pick_part)
                    pick_targets = self._adjust_pick_targets_for_execution(path, pick_targets, use_replan_grasp_offset)
                    # 1) Lower to grasp height.
                    approach_ok = self._move_end_effector_with_ablation(
                        pick_targets["grasp"], pick_targets["orientation"], steps=30, stage_name="final-grasp"
                    )
                    # 2) Wait briefly at grasp height, 3) close to mug grasp width.
                    self._apply_isaac_grasp_cspace_for_target(target_id, use_pregrasp=True)
                    self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                    if self.enable_action_waits and self.pre_close_wait_sec > 0:
                        wait_steps(self.world, self.pre_close_wait_sec, self.controller)
                    self.controller.close_gripper(self.world, path)
                    score, reason = self._evaluate_grasp_quality(path)
                    quality_min = self._adaptive_grasp_quality_min(target_id)
                    self._update_grasp_feedback(path, score, quality_min)
                    logger.info("[GraspQuality] target=%s score=%.3f min=%.3f (%s)", target_id, score, quality_min, reason)

                    grasped = str(getattr(self.controller, "grasped_object", "") or "") == path
                    if self.disable_attachment and (not grasped):
                        grasped = self._is_physical_grasp_likely(path)
                        if grasped:
                            logger.warning("[GraspQualityRelaxedNoAttach] target=%s score=%.3f accepted for transport", target_id, score)
                    if not grasped:
                        grasped = bool(self._try_grasp_with_fallback(path))
                        if grasped:
                            logger.warning("[GraspFallback] target=%s fallback attach succeeded; continue to transfer", target_id)

                    if (score < quality_min) and grasped:
                        logger.warning("[GraspQualityRelaxed] target=%s score=%.3f < %.3f but attached; forcing transfer", target_id, score, quality_min)
                    if (score < quality_min) and (not grasped) and self.controller is not None:
                        self.controller.last_error_message = f"grasp quality low ({score:.3f} < {quality_min:.3f})"

                    if grasped and self.disable_attachment:
                        self._lock_target_for_transport(path)

                    # Grab action success is judged by actual attachment, not by center transfer success.
                    if grasped:
                        self._ensure_transport_gripper_hold(path)
                        place_pose = "PLACE_mug_2"
                        logger.info("[TransportToBasket] target=%s selected=%s", target_id, place_pose)
                        transfer_ok = self._transport_grasped_target_to_basket_center(
                            place_pose,
                            path,
                            stage_prefix="grab",
                        )
                        release_ok = bool(transfer_ok and self._can_release_at_slot(place_pose))
                        if transfer_ok and release_ok:
                            self._prepare_release_depenetration(path)
                            self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                            self.controller.open_gripper(self.world)
                            if self.enable_action_waits and self.verify_wait_sec > 0:
                                wait_steps(self.world, self.verify_wait_sec, self.controller)
                            if self.controller.verify_placement(path):
                                self.placed_mugs.add(target_id)
                                self._remove_followup_actions_for_target(action_queue, target_id)
                            else:
                                forced = self._force_place_in_basket(path, place_pose)
                                if bool(forced and self.controller.verify_placement(path)):
                                    self.placed_mugs.add(target_id)
                                    self._remove_followup_actions_for_target(action_queue, target_id)
                        else:
                            # Once grasp succeeded, do not retry grab; defer transport/place as a separate phase.
                            step_ok = True
                            self._ensure_transport_gripper_hold(path)
                            hand_pos = None
                            try:
                                hand_pos, _ = self.controller._get_safe_world_pose("/World/Franka/panda_hand")
                                hand_pos = np.array(hand_pos, dtype=np.float32).round(5).tolist()
                            except Exception:
                                hand_pos = None
                            basket_zone = self._is_hand_in_basket_release_zone()
                            if not transfer_ok:
                                if self.controller is not None and not getattr(self.controller, "last_error_message", ""):
                                    self.controller.last_error_message = "transport to basket center failed"
                                logger.warning(
                                    "[TransportToBasketFailed] target=%s reason=transfer_ok_false hand=%s basket_zone=%s err=%s",
                                    target_id,
                                    hand_pos,
                                    basket_zone,
                                    getattr(self.controller, "last_error_message", ""),
                                )
                            else:
                                if self.controller is not None and not getattr(self.controller, "last_error_message", ""):
                                    self.controller.last_error_message = "release gate at basket center failed"
                                logger.warning(
                                    "[TransportToBasketFailed] target=%s reason=release_gate_false hand=%s basket_zone=%s err=%s",
                                    target_id,
                                    hand_pos,
                                    basket_zone,
                                    getattr(self.controller, "last_error_message", ""),
                                )
                            try:
                                target_idx = int(str(target_id).split("_")[-1])
                            except Exception:
                                target_idx = 0
                            deferred_place = f"place {target_id} target_{target_idx}"
                            if deferred_place not in action_queue:
                                action_queue.insert(0, deferred_place)
                                logger.info("[TransportDeferred] target=%s queued=%s", target_id, deferred_place)
                            if self.controller is not None:
                                # Do not let a transport warning turn into a grab retry.
                                self.controller.last_error_message = ""
                            execution["status"] = "partial"
                            execution["failed_action"] = deferred_place
                            execution["failed_reason"] = "deferred after successful grasp"

                    if self.enable_action_waits and self.grab_wait_sec > 0:
                        wait_steps(self.world, self.grab_wait_sec, self.controller)

                    # Do not fail grab just because center transfer failed; fail only when grasp itself failed.
                    if not grasped:
                        step_ok = False
                        if self.controller is not None and not getattr(self.controller, "last_error_message", ""):
                            self.controller.last_error_message = "grasp attach failed"

                elif action.startswith("place"):
                    target_id = action.split()[1] if len(action.split()) >= 2 else None
                    target_path = f"/World/{target_id}" if target_id else ""
                    # Skip meaningless place motion when target is not in gripper and not already placed.
                    if target_path and (not self._is_target_grasped(target_path)) and (not self.controller.verify_placement(target_path)):
                        # No-op success: nothing to place because object is not in gripper.
                        step_ok = True
                        if self.controller is not None:
                            self.controller.last_error_message = ""
                        logger.warning("[PlaceSkip] target=%s not grasped; skipping basket transfer motion", target_id)
                        moved = False
                    else:
                        images = capture_images_for_vlm(self.world)
                        suggested_pose = self.vlm_analyzer.find_empty_space(images) if self.vlm_analyzer else "BASKET_HIGH"
                        place_pose = "PLACE_mug_2"
                        logger.info("[PlaceSuggestion] target=%s vlm=%s selected=%s", target_id, suggested_pose, place_pose)
                        place_targets = self.controller.compute_place_targets(place_pose)
                        self._ensure_transport_gripper_hold(target_path)
                        moved = self._force_center_transfer_path(place_pose, target_path, stage_prefix="place")
                        if self.enable_action_waits and self.place_settle_wait_sec > 0:
                            wait_steps(self.world, self.place_settle_wait_sec, self.controller)

                        # In simple mode, prioritize deterministic release over strict guard.
                        can_release = bool(moved and self._can_release_at_slot(place_pose))
                        if self.simple_mode and moved and (not can_release):
                            logger.warning("[SimpleModeRelease] override release guard for target=%s slot=%s", target_id, place_pose)
                            can_release = True

                        if can_release and target_path:
                            self._prepare_release_depenetration(target_path)
                        if can_release:
                            self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
                            self.controller.open_gripper(self.world)
                        else:
                            self._ensure_transport_gripper_hold(target_path)
                            logger.warning("[ReleaseGuard] place reached=false slot=%s (basket_zone=%s); keep closed", place_pose, self._is_hand_in_basket_release_zone())
                            moved = False
                        placed_now = False
                        if moved:
                            retreat_ok = self._move_end_effector_with_ablation(place_targets["retreat"], place_targets["orientation"], steps=75, stage_name="post-place-retreat")
                            if not retreat_ok:
                                logger.warning("Retreat failed after place for %s", target_id)
                                moved = False

                        if target_id and moved:
                            if self.controller.verify_placement(target_path):
                                self.placed_mugs.add(target_id)
                                placed_now = True
                            else:
                                forced = self._force_place_in_basket(target_path, place_pose)
                                if forced:
                                    if self.enable_action_waits and self.verify_wait_sec > 0:
                                        wait_steps(self.world, self.verify_wait_sec, self.controller)
                                    if self.controller.verify_placement(target_path):
                                        self.placed_mugs.add(target_id)
                                        placed_now = True
                                    else:
                                        moved = False
                                else:
                                    moved = False

                        # Only return HOME after a confirmed place; otherwise keep current pose for retry.
                        if placed_now:
                            self.controller.move_to_joint_pose("HOME", self.world, steps=self._scale_motion_steps(60))
                            if self.enable_action_waits and self.verify_wait_sec > 0:
                                wait_steps(self.world, self.verify_wait_sec, self.controller)

                        if (not moved) or bool(getattr(self.controller, "last_error_message", "")):
                            step_ok = False
                else:
                    logger.warning("Unknown plan action %s", action)
                    step_ok = False

                probe_after = self._collect_robot_motion_probe(action, action_target or "")
                execution["motion_probe"].append({
                    "action": action,
                    "target": action_target,
                    "step_ok": bool(step_ok),
                    "pre": probe_before,
                    "post": probe_after,
                })

                if not step_ok:
                    execution["collisions"] += 1
                    execution["status"] = "partial"
                    reason = getattr(self.controller, "last_error_message", "") if self.controller else ""
                    if execution.get("failed_action") is None:
                        execution["failed_action"] = action
                        execution["failed_reason"] = reason or f"{action} failed"

                    retry_count = int(action_retry_counts.get(action, 0)) + 1
                    action_retry_counts[action] = retry_count
                    execution["executed_steps"].append(action)

                    if self._grasp_emergency_stop:
                        if execution.get("failed_action") is None:
                            execution["failed_action"] = action
                            execution["failed_reason"] = self._grasp_emergency_reason or "grasp jump safety stop"
                        break

                    max_retries_for_action = 2 if action.startswith("place") else 3
                    if retry_count < max_retries_for_action:
                        logger.warning(
                            "[ActionRetry] action=%s attempt=%d/%d failed; retrying same action",
                            action,
                            retry_count,
                            max_retries_for_action,
                        )
                        action_queue.insert(0, action)
                    else:
                        logger.error("[ActionRetry] action=%s exceeded max retries (%d)", action, max_retries_for_action)
                        break
                else:
                    action_retry_counts[action] = 0
                    last_success_action = action
                    execution["executed_steps"].append(action)

                if self.step_mode:
                    try:
                        input("Press Enter to continue...")
                    except EOFError:
                        pass
            except Exception as exc:
                logger.exception("Simulation execution failed on action %s", action)
                probe_after = self._collect_robot_motion_probe(action, action_target or "")
                execution["motion_probe"].append({
                    "action": action,
                    "target": action_target,
                    "step_ok": False,
                    "error": str(exc),
                    "pre": probe_before,
                    "post": probe_after,
                })
                execution["collisions"] += 1
                execution["status"] = "partial"
                if execution.get("failed_action") is None:
                    execution["failed_action"] = action
                    execution["failed_reason"] = str(exc)
                retry_count = int(action_retry_counts.get(action, 0)) + 1
                action_retry_counts[action] = retry_count
                execution["executed_steps"].append(action)
                max_retries_for_action = 2 if action.startswith("place") else 3
                if retry_count < max_retries_for_action:
                    logger.warning(
                        "[ActionRetry] exception on %s attempt=%d/%d; retrying same action",
                        action,
                        retry_count,
                        max_retries_for_action,
                    )
                    action_queue.insert(0, action)
                else:
                    logger.error("[ActionRetry] action=%s exceeded max retries (%d) after exception", action, max_retries_for_action)
                    break
                if self.step_mode:
                    try:
                        input("Press Enter to continue...")
                    except EOFError:
                        pass

        execution["executed_length"] = len(execution["executed_steps"])
        execution["placed_bottles"] = sorted(self.placed_mugs)
        execution["placed_count"] = len(self.placed_mugs)
        execution_elapsed = time.time() - t_exec0
        execution["execution_time"] = execution_elapsed
        execution["residual_event_count"] = len(self._residual_events)
        if self._residual_events:
            execution["residual_events_tail"] = self._residual_events[-5:]
        self.metrics.control_execution_time += execution_elapsed
        self.metrics.safety_score += execution["collisions"]
        if execution["collisions"] > 0:
            execution["status"] = "partial"

        self.metrics.history.append({"phase": "simulation_execution", "result": execution})
        return execution

    def realtime_vlm_update_replan(self, execution: Dict[str, Any], scene_graph: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[5/5] realtime_vlm_update_replan")

        # Nudge a remaining mug once to force a scene-graph diff and trigger replanning.
        if self.enable_scene_perturbation and (not self._scene_graph_perturbed) and self.world is not None:
            try:
                remaining = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT) if f"mug_{i}" not in self.placed_mugs and f"mug_{i}" not in self.failed_mugs]
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

        if self.enable_scene_perturbation and self._scene_graph_perturbed and self.metrics.replan_count == 0:
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



    def cleanup(self) -> None:
        for name in list(self._active_writers):
            writer = self._writers.get(name)
            if writer is None:
                continue
            try:
                writer.detach()
            except Exception:
                pass
            self._active_writers.discard(name)

        # Ask Replicator to stop orchestrating before tearing down render resources.
        try:
            import omni.replicator.core as rep
            rep.orchestrator.stop()
            wait_fn = getattr(rep.orchestrator, "wait_until_complete", None)
            if callable(wait_fn):
                wait_fn()
        except Exception:
            pass

        for rp in list(self._render_products.values()):
            try:
                destroy_fn = getattr(rp, "destroy", None)
                if callable(destroy_fn):
                    destroy_fn()
            except Exception:
                pass

        self._writers.clear()
        self._render_products.clear()

        try:
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass

        try:
            if self.world is not None:
                stop_fn = getattr(self.world, "stop", None)
                if callable(stop_fn):
                    stop_fn()
        except Exception:
            pass

        self.controller = None
        self.franka = None
        self.world = None

        try:
            if self._safety_ui_window is not None:
                self._safety_ui_window.visible = False
        except Exception:
            pass
        self._safety_ui_window = None
        self._safety_ui_models = {}

        try:
            usd_ctx = omni.usd.get_context()
            close_fn = getattr(usd_ctx, "close_stage", None)
            if callable(close_fn):
                close_fn()
            usd_ctx.new_stage()
        except Exception:
            pass

        # Flush one-shot teardown tasks queued in Kit before final app shutdown.
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            for _ in range(4):
                app.update()
        except Exception:
            pass

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

    def _extract_target_id_from_action(self, action: str) -> str:
        try:
            tokens = str(action).split()
        except Exception:
            return ""
        if len(tokens) < 2:
            return ""
        target = str(tokens[1]).strip()
        return target if target.startswith("mug_") else ""

    def _restrict_plan_to_target(self, plan: Optional[Dict[str, Any]], target_id: str, source: str) -> Dict[str, Any]:
        if not plan or not plan.get("steps"):
            return self._build_plan_for_targets([target_id], source=source)

        filtered_steps: List[str] = []
        for step in plan.get("steps", []):
            target = self._extract_target_id_from_action(step)
            if (not target) or target == target_id:
                filtered_steps.append(step)

        if not filtered_steps:
            return self._build_plan_for_targets([target_id], source=source)

        return {
            "steps": filtered_steps,
            "length": len(filtered_steps),
            "timestamp": time.time(),
            "source": source,
        }

    def _build_simple_plan_for_target(self, target_id: str, source: str = "simple") -> Dict[str, Any]:
        idx = 0
        try:
            idx = int(str(target_id).split("_")[-1])
        except Exception:
            idx = 0
        steps = [
            f"pick {target_id}",
            f"grab {target_id}",
        ]
        return {
            "steps": steps,
            "length": len(steps),
            "timestamp": time.time(),
            "source": source,
        }

    def _simple_mode_drop_to_basket_center(self, target_id: str, target_path: str) -> bool:
        if self.controller is None or self.world is None:
            return False
        place_pose = "PLACE_mug_2"
        place_targets = self.controller.compute_place_targets(place_pose)
        basket_center = self._get_basket_center_world()
        place_target = np.array(place_targets["place"], dtype=np.float32)
        place_target[0] = float(basket_center[0])
        place_target[1] = float(basket_center[1])

        ori = np.array(place_targets["orientation"], dtype=np.float32)
        self._ensure_transport_gripper_hold(target_path)

        moved = self._move_with_place_path_planning(
            place_pose,
            place_target,
            ori,
            stage_name="simple-place-rrt",
            steps=95,
            target_path=target_path,
        )
        if not moved:
            moved = self._move_end_effector_with_ablation(
                place_target,
                ori,
                steps=95,
                stage_name="simple-direct-place-fallback",
                route_hint=f"place:{place_pose}",
            )
        if not moved:
            return False

        try:
            self._prepare_release_depenetration(target_path)
        except Exception:
            pass
        self._set_controller_gripper_target(float(self.controller.dynamic_params.get("grip_open", 0.04)), hold=False)
        self.controller.open_gripper(self.world)
        if self.enable_action_waits and self.verify_wait_sec > 0:
            wait_steps(self.world, self.verify_wait_sec, self.controller)

        if self.controller.verify_placement(target_path):
            self.placed_mugs.add(target_id)
            return True

        forced = self._force_place_in_basket(target_path, place_pose)
        if bool(forced):
            if self.enable_action_waits and self.verify_wait_sec > 0:
                wait_steps(self.world, self.verify_wait_sec, self.controller)
            if self.controller.verify_placement(target_path):
                self.placed_mugs.add(target_id)
                return True
        return False

    def _run_simple_mode(self, human_instruction: str) -> Dict[str, Any]:
        max_retries_per_mug = 3

        self.metrics = WorkflowMetrics()
        self.metrics.start_time = time.time()
        self.placed_mugs = set()
        self.attempted_mugs = set()
        self.failed_mugs = set()
        self.mug_retry_counts = {}
        self._affordance_cache = {}
        self._last_instruction = human_instruction
        self._failure_replan_counts_by_target = {}

        self.setup_simulation_environment()
        hi = self.receive_human_instruction(human_instruction)
        _ = self.build_delta_scene_graph(hi)
        self.replan_on_failure = False

        target_order = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT)]
        for target_id in target_order:
            self._evaluate_basket_outcome()
            if target_id in self.placed_mugs:
                continue

            success = False
            for attempt in range(1, max_retries_per_mug + 1):
                self._stabilize_between_targets(target_id)
                plan = self._build_simple_plan_for_target(target_id, source=f"simple_{target_id}_attempt_{attempt}")
                execution = self.execute_simulation_plan(plan)
                self._evaluate_basket_outcome()
                self.mug_retry_counts[target_id] = attempt

                self.metrics.history.append({
                    "phase": "simple_target_attempt",
                    "target": target_id,
                    "attempt": attempt,
                    "execution": execution,
                })

                if target_id in self.placed_mugs:
                    success = True
                    self.failed_mugs.discard(target_id)
                    self.attempted_mugs.add(target_id)
                    break

            if not success:
                self.failed_mugs.add(target_id)
                self.attempted_mugs.add(target_id)
                self.metrics.history.append({
                    "phase": "simple_target_giveup",
                    "target": target_id,
                    "attempts": int(self.mug_retry_counts.get(target_id, 0)),
                })

        self._evaluate_basket_outcome()
        self.metrics.moved_bottles = len(self.placed_mugs)
        self.metrics.target_bottles = TARGET_BOTTLE_COUNT
        self.metrics.failed_bottles = max(0, self.metrics.target_bottles - self.metrics.moved_bottles)
        self.metrics.success_rate = self.metrics.moved_bottles / float(self.metrics.target_bottles)
        self.metrics.success = self.metrics.moved_bottles == self.metrics.target_bottles

        self.metrics.history.append({
            "phase": "simple_summary",
            "placed": sorted(self.placed_mugs),
            "failed": sorted(set(self.failed_mugs) - set(self.placed_mugs)),
            "retry_counts": dict(self.mug_retry_counts),
        })

        self.evaluate_and_save()
        if self.record_video:
            video_path = self._generate_video()
            if video_path:
                self.metrics.history.append({"phase": "video", "path": video_path})
                self.evaluate_and_save()
        return self.metrics.to_dict()

    def run(self, human_instruction: str, max_replans: int = MAX_CONSECUTIVE_FAILURES):
        _ = max_replans
        if self.simple_mode:
            logger.info("[SimpleMode] enabled: using deterministic pick->grab->place loop")
            return self._run_simple_mode(human_instruction)
        max_retries_per_mug = 3

        self.metrics = WorkflowMetrics()
        self.metrics.start_time = time.time()
        self.placed_mugs = set()
        self.attempted_mugs = set()
        self.failed_mugs = set()
        self.mug_retry_counts = {}
        self._affordance_cache = {}
        self._last_instruction = human_instruction
        self._failure_replan_counts_by_target: Dict[str, int] = {}

        self.setup_simulation_environment()

        hi = self.receive_human_instruction(human_instruction)
        sg_data = self.build_delta_scene_graph(hi)

        target_order = [f"mug_{i}" for i in range(TARGET_BOTTLE_COUNT)]

        while True:
            self._evaluate_basket_outcome()
            active_targets = [
                t for t in target_order
                if t not in self.placed_mugs and t not in self.failed_mugs
            ]
            if not active_targets:
                break

            current_target = min(
                active_targets,
                key=lambda t: (int(self.mug_retry_counts.get(t, 0)), target_order.index(t)),
            )
            attempt = int(self.mug_retry_counts.get(current_target, 0))
            current_plan = self._build_plan_for_targets([current_target], source=f"single_target_{current_target}_attempt_{attempt + 1}")
            self._stabilize_between_targets(current_target)

            while attempt < max_retries_per_mug and current_target not in self.placed_mugs:
                self._stabilize_between_targets(current_target)
                attempt += 1
                execution = self.execute_simulation_plan(current_plan)
                self._evaluate_basket_outcome()

                self.mug_retry_counts[current_target] = attempt
                failed_action = str(execution.get("failed_action") or "")
                failed_reason = str(execution.get("failed_reason") or "")

                self.metrics.history.append({
                    "phase": "single_target_attempt",
                    "target": current_target,
                    "attempt": attempt,
                    "execution": execution,
                })

                if current_target in self.placed_mugs:
                    self.mug_retry_counts[current_target] = 0
                    self.failed_mugs.discard(current_target)
                    self.attempted_mugs.add(current_target)
                    break

                if attempt >= max_retries_per_mug:
                    break

                replanned = None
                if self.replan_on_failure:
                    replanned = self._replan_after_failure(execution, sg_data["pddl"], target_id=current_target)

                current_plan = self._restrict_plan_to_target(
                    replanned,
                    current_target,
                    source=f"single_target_{current_target}_attempt_{attempt + 1}_replan",
                )

                self.metrics.history.append({
                    "phase": "single_target_replan_prepared",
                    "target": current_target,
                    "attempt": attempt,
                    "failed_action": failed_action,
                    "failed_reason": failed_reason,
                    "next_plan": current_plan,
                })

            if current_target not in self.placed_mugs:
                self.failed_mugs.add(current_target)
                self.attempted_mugs.add(current_target)
                self.metrics.history.append({
                    "phase": "single_target_giveup",
                    "target": current_target,
                    "attempts": int(self.mug_retry_counts.get(current_target, 0)),
                })

        self._evaluate_basket_outcome()
        self.metrics.moved_bottles = len(self.placed_mugs)
        self.metrics.target_bottles = TARGET_BOTTLE_COUNT
        self.metrics.failed_bottles = max(0, self.metrics.target_bottles - self.metrics.moved_bottles)
        self.metrics.success_rate = self.metrics.moved_bottles / float(self.metrics.target_bottles)
        self.metrics.success = self.metrics.moved_bottles == self.metrics.target_bottles

        self.metrics.history.append({
            "phase": "single_target_summary",
            "placed": sorted(self.placed_mugs),
            "failed": sorted(set(self.failed_mugs) - set(self.placed_mugs)),
            "retry_counts": dict(self.mug_retry_counts),
        })

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
    parser.add_argument("--step-mode", action="store_true", help="Execute pick/grab/place as micro-steps with pauses.")
    parser.add_argument("--simple-mode", action="store_true", help="Use deterministic pick->grab->place loop (pick_place.py-like).")
    parser.add_argument("--disable-physics-stabilization", action="store_true", help="Disable substeps/GPU/TGS physics stabilization.")
    parser.add_argument("--disable-contact-offsets", action="store_true", help="Disable PhysX contact/rest offset tuning.")
    parser.add_argument("--disable-solver-boost", action="store_true", help="Disable articulation solver iteration boost.")
    parser.add_argument("--disable-action-waits", action="store_true", help="Disable all action settle waits (pick/grab/place/verify).")
    parser.add_argument("--pick-wait-sec", type=float, default=0.2, help="Wait seconds between pre-grasp and grasp.")
    parser.add_argument("--grab-wait-sec", type=float, default=0.5, help="Wait seconds after closing gripper on grab.")
    parser.add_argument("--place-settle-wait-sec", type=float, default=0.5, help="Wait seconds after place approach before release.")
    parser.add_argument("--verify-wait-sec", type=float, default=1.0, help="Wait seconds before placement verification.")
    parser.add_argument("--pre-close-wait-sec", type=float, default=1.0, help="Wait seconds after opening at grasp height before closing gripper.")
    parser.add_argument("--disable-ccd", action="store_true", help="Disable continuous collision detection (CCD).")
    parser.add_argument("--disable-velocity-capping", action="store_true", help="Disable rigid-body max linear/angular velocity caps.")
    parser.add_argument("--max-linear-velocity", type=float, default=1.2, help="Rigid-body max linear velocity (m/s).")
    parser.add_argument("--max-angular-velocity", type=float, default=8.0, help="Rigid-body max angular velocity (rad/s).")
    parser.add_argument("--linear-damping", type=float, default=0.2, help="Rigid-body linear damping.")
    parser.add_argument("--angular-damping", type=float, default=0.35, help="Rigid-body angular damping.")
    parser.add_argument("--rigid-position-iterations", type=int, default=32, help="Rigid-body position solver iterations.")
    parser.add_argument("--rigid-velocity-iterations", type=int, default=8, help="Rigid-body velocity solver iterations.")
    parser.add_argument("--max-depenetration-velocity", type=float, default=1.0, help="Rigid-body max depenetration velocity (m/s).")
    parser.add_argument("--disable-grasp-jump-guard", action="store_true", help="Disable safety stop on sudden grasped-object position jumps.")
    parser.add_argument("--grasp-jump-stop-threshold", type=float, default=0.10, help="Safety stop threshold for sudden grasped-object jump per step (meters).")
    parser.add_argument("--enable-scene-perturbation", action="store_true", help="Enable debug scene perturbation for forced replan checks.")
    parser.add_argument("--disable-cumotion-style", action="store_true", help="Disable conservative cuMotion-style waypoint execution.")
    parser.add_argument("--gripper-static-friction", type=float, default=2.5, help="Static friction for gripper finger colliders.")
    parser.add_argument("--gripper-dynamic-friction", type=float, default=2.0, help="Dynamic friction for gripper finger colliders.")
    parser.add_argument("--mug-static-friction", type=float, default=1.8, help="Static friction for mug colliders.")
    parser.add_argument("--mug-dynamic-friction", type=float, default=1.4, help="Dynamic friction for mug colliders.")
    parser.add_argument("--render-decimation", type=int, default=3, help="Render every N physics steps (1 disables decimation).")
    parser.add_argument("--residual-warn-threshold", type=float, default=0.05, help="Warn threshold for grasp residual (m).")
    parser.add_argument("--residual-stop-threshold", type=float, default=0.12, help="Safety-stop threshold for grasp residual (m).")
    parser.add_argument("--grasp-quality-min", type=float, default=0.35, help="Minimum grasp quality score [0,1].")
    parser.add_argument("--isaac-grasp-file", type=str, default="", help="Path to isaac_grasp YAML file.")
    parser.add_argument("--grasp-z-offset", type=float, default=-0.015, help="Downward Z offset applied to all grasp poses (meters).")
    parser.add_argument("--replan-grasp-z-offset", type=float, default=-0.03, help="Extra downward Z offset applied to grasp poses during replans only (meters).")
    parser.add_argument("--attach-distance-threshold", type=float, default=0.255, help="Attachment distance threshold (meters). Larger values make attach easier.")
    parser.add_argument("--attach-distance-grace", type=float, default=0.045, help="Extra grace added to attachment distance threshold (meters).")
    parser.add_argument("--disable-attachment", action="store_true", help="Disable virtual attachment and rely on physical grasp/contact only.")
    parser.add_argument("--motion-step-scale", type=float, default=1.0, help="Scale all motion steps. Larger values make motions slower.")
    parser.add_argument("--replan-backend", type=str, default="auto", choices=["auto", "delta", "sayplan", "none"], help="External replanning backend to call on failure.")
    parser.add_argument("--replan-script", type=str, default="", help="Override replan script path (delta.py or sayplan.py).")
    parser.add_argument("--replan-actions-json", type=str, default="", help="Path to actions.json produced by external replanner.")
    parser.add_argument("--replan-python", type=str, default="", help="Python interpreter for external replanner.")
    parser.add_argument("--max-failure-replans", type=int, default=3, help="Max external replans per run (failure-triggered).")
    parser.add_argument("--no-replan-on-failure", action="store_true", help="Disable external replanning on action failure.")
    parser.add_argument("--record-video", dest="record_video", action="store_true", help="Record simulation video and save simulation.mp4 in output dir.")
    parser.add_argument("--no-record-video", dest="record_video", action="store_false", help="Disable simulation video recording.")
    parser.add_argument("--must3r-weights", type=str, default=os.environ.get("MUST3R_WEIGHTS", ""), help="Path to MUSt3R checkpoint (.pth).")
    parser.add_argument("--must3r-image-size", type=int, default=224, choices=[224, 512], help="MUSt3R inference image size.")
    parser.add_argument("--must3r-amp", type=str, default=os.environ.get("MUST3R_AMP", "fp16"), choices=["fp16", "bf16", "False", "false", "none"], help="MUSt3R autocast mode.")
    parser.add_argument("--must3r-device", type=str, default=os.environ.get("MUST3R_DEVICE", "cuda"), help="MUSt3R device (e.g., cuda, cpu).")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device (pick_place.py compatible).")
    parser.add_argument("--ik-method", type=str, choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"], default="damped-least-squares", help="Differential IK method label for motion investigation (pick_place.py compatible).")
    parser.set_defaults(record_video=True)
    args = parser.parse_args()

    task_input = args.task
    if not task_input:
        task_input = input("Enter human instruction (e.g., 'sort mugs by color'): ").strip()
        if not task_input:
            raise ValueError("Task description is required")

    workflow = None
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
                sim_device=args.device,
                ik_method=args.ik_method,
                step_mode=args.step_mode,
                replan_on_failure=not args.no_replan_on_failure,
                replan_backend=args.replan_backend,
                replan_script=args.replan_script or None,
                replan_actions_json=args.replan_actions_json or None,
                replan_python=args.replan_python or None,
                max_failure_replans=args.max_failure_replans,
                enable_physics_stabilization=not args.disable_physics_stabilization,
                enable_contact_offsets=not args.disable_contact_offsets,
                enable_solver_boost=not args.disable_solver_boost,
                enable_action_waits=not args.disable_action_waits,
                pick_wait_sec=args.pick_wait_sec,
                grab_wait_sec=args.grab_wait_sec,
                place_settle_wait_sec=args.place_settle_wait_sec,
                verify_wait_sec=args.verify_wait_sec,
                pre_close_wait_sec=args.pre_close_wait_sec,
                enable_ccd=not args.disable_ccd,
                enable_velocity_capping=not args.disable_velocity_capping,
                max_linear_velocity=args.max_linear_velocity,
                max_angular_velocity=args.max_angular_velocity,
                linear_damping=args.linear_damping,
                angular_damping=args.angular_damping,
                rigid_position_iterations=args.rigid_position_iterations,
                rigid_velocity_iterations=args.rigid_velocity_iterations,
                max_depenetration_velocity=args.max_depenetration_velocity,
                enable_grasp_jump_guard=not args.disable_grasp_jump_guard,
                grasp_jump_stop_threshold=args.grasp_jump_stop_threshold,
                enable_scene_perturbation=args.enable_scene_perturbation,
                enable_cumotion_style=not args.disable_cumotion_style,
                gripper_static_friction=args.gripper_static_friction,
                gripper_dynamic_friction=args.gripper_dynamic_friction,
                mug_static_friction=args.mug_static_friction,
                mug_dynamic_friction=args.mug_dynamic_friction,
                render_decimation=args.render_decimation,
                residual_warn_threshold=args.residual_warn_threshold,
                residual_stop_threshold=args.residual_stop_threshold,
                grasp_quality_min=args.grasp_quality_min,
                isaac_grasp_file=args.isaac_grasp_file or None,
                grasp_z_offset=args.grasp_z_offset,
                replan_grasp_z_offset=args.replan_grasp_z_offset,
                attach_distance_threshold=args.attach_distance_threshold,
                attach_distance_grace=args.attach_distance_grace,
                simple_mode=args.simple_mode,
                disable_attachment=args.disable_attachment,
                motion_step_scale=args.motion_step_scale,
            )
            result = workflow.run(task_input)
            logger.info("Final workflow result:\n%s", json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.exception("Workflow failed with exception: %s", exc)
        raise
    finally:
        if workflow is not None:
            try:
                workflow.cleanup()
            except Exception as exc:
                logger.warning("Workflow cleanup failed: %s", exc)
        simulation_app = getattr(enum_eval_module, "simulation_app", None)
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
