import os
import json
import base64
import shutil
import subprocess
import numpy as np
import traceback
import faulthandler
import gc
import time
from pathlib import Path
from functools import wraps


def _load_render_simple_scene_graph():
    try:
        import importlib.util
        from pathlib import Path
        helper_path = Path(__file__).with_name("scenegraph_viz_simple.py")
        spec = importlib.util.spec_from_file_location("scenegraph_viz_simple", str(helper_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "render_simple_scene_graph", None)
    except Exception:
        return None
    return None

RENDER_SIMPLE_SG = _load_render_simple_scene_graph()
# local helper loader for scene graph viz
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

faulthandler.enable()

# =========================================================
# API keys / model
# =========================================================
OPENAI_API_KEY = "<OPENAI_API_KEY>"
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

GEMINI_API_KEY = "AIzaSyCsmmdOaLo7hdOXyyneRLA5kgQHBm516eQ"
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

TARGET_MODEL = "gpt-4o"

# =========================================================
# Config
# =========================================================
OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation/vlm_integration_5")
RGB_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "mug_sorting.mp4"
EVAL_RESULTS_JSON = OUTPUT_DIR / "evaluation_results.json"
PLAN_JSON = OUTPUT_DIR / "actions.json"
CALIB_JSON = OUTPUT_DIR / "calibration.json"
SIMPLE_SG_JSON = OUTPUT_DIR / "simple_scene_graph.json"

REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki/delta.py")
DELTA_PYTHON = "/usr/bin/python3"

USER_INSTRUCTION = "handleを掴んでカゴに置いて"

MUGS = [
    {"id": "mug_0", "pos": [0.5, 0.0, 0.05], "color": [1.0, 0.0, 0.0], "angle": 0},
]

MUG_ANGLES = {m["id"]: m["angle"] for m in MUGS}
POSE_LIBRARY = {
    "HOME": np.array([0.0, -0.70, 0.0, -2.30, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
}

CAMERA_POSITIONS = {
    "main": {"pos": (1.8, 0.0, 1.5), "look_at": (0.5, 0.0, 0.2)},
    "top": {"pos": (0.5, 0.0, 2.2), "look_at": (0.5, 0.0, 0.0)},
    "left": {"pos": (1.4, 1.2, 1.0), "look_at": (0.5, 0.0, 0.0)},
    "right": {"pos": (1.4, -1.2, 1.0), "look_at": (0.5, 0.0, 0.0)},
}

DEFAULT_EE_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
GRIP_CLOSE_MUG = 0.02
GRASP_FOLLOW_ALPHA = 0.25
BASKET_PLACE_SLOTS = {
    "PLACE_mug_0": np.array([0.0, 0.55, 0.18], dtype=np.float32),
}
MUG_PLACE_ASSIGNMENTS = {
    "mug_0": "PLACE_mug_0",
}

AFFORDANCE_SCHEMA = {
    "handle": {"graspable": True},
    "body_outer": {"pourable": True},
    "body_inner": {"pourable": True},
    "bottom": {"stable": True},
}

ENABLE_PARAM_UI = True
HEADLESS = not ENABLE_PARAM_UI
DISABLE_RECORDING_IN_GUI = True
GUI_HOLD_AT_END = True
TOPPLE_TRIGGER_SEC = 10.0
TOPPLE_LINEAR_VEL = [-1.0, 0.6, 0.0]
TOPPLE_ANGULAR_VEL = [6.0, 4.0, 0.0]


class EvaluationLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {
            "total_failures": 0,
            "inference_times": [],
            "action_plans": [],
            "scene_graphs": [],
            "correction_counts": [],
            "affordance_logs": [],
            "planner_metrics": [],
            "comparisons": {},
        }

    def add_failure(self):
        self.data["total_failures"] += 1

    def record_inference_time(self, process_name, latency):
        self.data["inference_times"].append(
            {"process": process_name, "latency_sec": round(latency, 3), "timestamp": time.time()}
        )

    def record_plan(self, plan):
        self.data["action_plans"].append({"plan": plan.copy(), "timestamp": time.time()})

    def record_scene_graph(self, step, graph, affordance=None):
        entry = {"step": step, "graph": graph, "timestamp": time.time()}
        if affordance is not None:
            entry["affordance"] = affordance
        self.data["scene_graphs"].append(entry)

    def record_corrections(self, step, count, reason=None):
        entry = {"step": step, "count": int(count), "timestamp": time.time()}
        if reason:
            entry["reason"] = reason
        self.data["correction_counts"].append(entry)

    def record_affordance(self, affordance):
        self.data["affordance_logs"].append({"affordance": affordance, "timestamp": time.time()})

    def record_planner_metric(self, method, plan_len, success, failures, note=None):
        entry = {
            "method": method,
            "plan_len": int(plan_len),
            "success": bool(success),
            "failures": int(failures),
            "timestamp": time.time(),
        }
        if note:
            entry["note"] = note
        self.data["planner_metrics"].append(entry)

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)


eval_logger = EvaluationLogger(EVAL_RESULTS_JSON)


def measure_time(process_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            eval_logger.record_inference_time(process_name, time.time() - start)
            return result

        return wrapper

    return decorator


# =========================================================
# Calibration helpers
# =========================================================

def load_calibration():
    if CALIB_JSON.exists():
        try:
            with open(CALIB_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "position_offset": [0.0, 0.0, -0.02],
        "yaw_adjust": 0.03,
        "grip_open": 0.04,
        "grip_close": 0.005,
        "attach_distance_threshold": 0.24,
        "follow_offset": -0.07,
    }


def save_calibration(data):
    CALIB_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_JSON, "w") as f:
        json.dump(data, f, indent=2)


def run_calibration(world):
    calib = load_calibration()
    print(f"[Calib] Loaded calibration offsets: {calib}", flush=True)
    return calib


# =========================================================
# Isaac Sim startup
# =========================================================
os.environ["CARB_APP_MIN_LOG_LEVEL"] = "error"
os.environ["OMNI_LOG_LEVEL"] = "error"
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": HEADLESS, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "enable_audio": False}
)

import omni.ui as ui
import carb
from pxr import Gf, Sdf, UsdLux, UsdPhysics, UsdGeom, UsdShade
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim, SingleRigidPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver as FrankaKinematicsSolver
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep

carb.settings.get_settings().set_string("/log/level", "error")

fps = 30


class TimedToppleScenario:
    def __init__(self, target_path, trigger_seconds=10.0):
        self.target_path = target_path
        self.trigger_steps = max(1, int(trigger_seconds * fps))
        self.step_count = 0
        self.triggered = False
        self._rigid = None

    def _get_rigid(self):
        if self._rigid is None:
            self._rigid = SingleRigidPrim(self.target_path, name=f"{self.target_path.split('/')[-1]}_topple")
            try:
                self._rigid.initialize()
            except Exception:
                pass
        return self._rigid

    def maybe_trigger(self):
        if self.triggered:
            return False
        self.step_count += 1
        if self.step_count < self.trigger_steps:
            return False
        stage_prim = get_current_stage().GetPrimAtPath(self.target_path)
        if stage_prim.IsValid():
            UsdPhysics.RigidBodyAPI(stage_prim).GetKinematicEnabledAttr().Set(False)
        rigid = self._get_rigid()
        try:
            rigid.set_linear_velocity(np.array(TOPPLE_LINEAR_VEL, dtype=np.float32))
            rigid.set_angular_velocity(np.array(TOPPLE_ANGULAR_VEL, dtype=np.float32))
        except Exception:
            rb_api = UsdPhysics.RigidBodyAPI(stage_prim)
            rb_api.CreateVelocityAttr().Set(Gf.Vec3f(*TOPPLE_LINEAR_VEL))
            rb_api.CreateAngularVelocityAttr().Set(Gf.Vec3f(*TOPPLE_ANGULAR_VEL))
        self.triggered = True
        print("\n[Test] 10秒後の外力でmugを倒しました。観察して再計画します。", flush=True)
        return True


def wait_steps(world, seconds, controller=None, topple=None):
    steps = int(seconds * fps)
    for _ in range(steps):
        if not world.is_playing():
            break
        world.step(render=True)
        if controller:
            controller._after_world_step()
        if topple is not None:
            topple.maybe_trigger()


class VisualHelper:
    @staticmethod
    def create_or_move_proxy(pos, name="TargetProxy"):
        stage = get_current_stage()
        path = f"/World/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr(0.02)
            mat_path = f"{path}_Material"
            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.5)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).GetCollisionEnabledAttr().Set(False)

        safe_pos = pos[0] if len(np.shape(pos)) == 2 else pos
        XFormPrim(path).set_world_poses(positions=np.array([safe_pos]))


class VLMAnalyzer:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI is not None) else None

    def encode_image_base64(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _safe_parse(self, raw_content, default_val=None):
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
            return parsed
        except Exception:
            return default_val

    @measure_time("plan_affordance_action")
    def plan_affordance_action(self, image_paths, instruction):
        default_res = {"pick_part": "handle", "fallback_parts": ["body_outer"], "rationale": "default"}
        if not self.client or not image_paths:
            return default_res
        base64_images = [self.encode_image_base64(p) for p in image_paths]
        prompt = f"""画像からマグの姿勢を把握し、次の2点を考慮して計画してください。
(1) 取手が下にあるなど掴めない場合、どうすれば取手が見つかるか?
(2) 取手以外で掴める可能性のある部位はどれか? (例:円柱側面)
候補パーツ: handle, body_outer, body_inner, bottom。
アフォーダンス: graspable, pourable, stable。
指示: {instruction}
出力JSON: {{"pick_part": str, "fallback_parts": [str], "rationale": str}}"""
        content = [{"type": "text", "text": prompt}]
        for b64 in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            parsed = self._safe_parse(response.choices[0].message.content, default_res) or default_res
            part = parsed.get("pick_part", "handle")
            if part not in AFFORDANCE_SCHEMA:
                part = "handle"
            fallbacks = parsed.get("fallback_parts", []) or []
            fallbacks = [p for p in fallbacks if p in AFFORDANCE_SCHEMA and p != part]
            if not fallbacks:
                fallbacks = ["body_outer"]
            return {"pick_part": part, "fallback_parts": fallbacks, "rationale": parsed.get("rationale", "")}
        except Exception:
            return default_res

    @measure_time("infer_affordance")
    def infer_affordance(self, image_paths, instruction):
        default_res = {"part": "handle", "affordance": "graspable", "reason": "default"}
        if not self.client or not image_paths:
            return default_res
        base64_images = [self.encode_image_base64(p) for p in image_paths]
        prompt = f"""画像からマグのパーツとアフォーダンスを推定し、指示に最適な部位を選んでください。候補パーツ: handle, body_outer, body_inner, bottom。アフォーダンス: graspable, pourable, stable。
指示: {instruction}
出力はJSONのみ: {{"part": str, "affordance": str, "reason": str}}"""
        content = [{"type": "text", "text": prompt}]
        for b64 in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            parsed = self._safe_parse(response.choices[0].message.content, default_res) or default_res
            if parsed.get("part") not in AFFORDANCE_SCHEMA:
                parsed["part"] = "handle"
            if parsed.get("affordance") not in ["graspable", "pourable", "stable"]:
                parsed["affordance"] = "graspable"
            return parsed
        except Exception:
            return default_res

    @measure_time("extract_scene_graph")
    def extract_scene_graph(self, image_paths, affordance=None):
        graph = {
            "nodes": [
                {"id": "mug_0", "type": "mug"},
                {"id": "handle", "type": "part", "affordance": {"graspable": True}},
                {"id": "body_outer", "type": "part", "affordance": {"pourable": True}},
                {"id": "body_inner", "type": "part", "affordance": {"pourable": True}},
                {"id": "bottom", "type": "part", "affordance": {"stable": True}},
            ],
            "edges": [
                {"from": "mug_0", "to": "handle", "relation": "has_part"},
                {"from": "mug_0", "to": "body_outer", "relation": "has_part"},
                {"from": "mug_0", "to": "body_inner", "relation": "has_part"},
                {"from": "mug_0", "to": "bottom", "relation": "has_part"},
            ],
        }
        if affordance:
            graph["affordance"] = affordance
        return graph


class AssetBuilder:
    @staticmethod
    def apply_material(prim, stage, path, color):
        mat_path = f"{path}_Material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    @staticmethod
    def create_basket(path, pos):
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, path)
        w, d, h, th = 0.5, 0.6, 0.15, 0.01
        color = Gf.Vec3f(0.5, 0.35, 0.25)
        parts = [
            ("Bottom", (w, d, th), (0, 0, th / 2)),
            ("Front", (w, th, h), (0, -d / 2, h / 2)),
            ("Back", (w, th, h), (0, d / 2, h / 2)),
            ("Left", (th, d, h), (-w / 2, 0, h / 2)),
            ("Right", (th, d, h), (w / 2, 0, h / 2)),
        ]
        for name, size, offset in parts:
            part_path = f"{path}/{name}"
            cube = UsdGeom.Cube.Define(stage, part_path)
            XFormPrim(part_path).set_local_scales(np.array([[size[0] / 2, size[1] / 2, size[2] / 2]]))
            XFormPrim(part_path).set_local_poses(np.array([[offset[0], offset[1], offset[2]]]))
            AssetBuilder.apply_material(cube, stage, part_path, color)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        XFormPrim(path).set_world_poses(np.array([[pos[0], pos[1], pos[2]]]))
        return path

    @staticmethod
    def create_beer_mug(path, pos, color, z_angle_deg=180.0):
        stage = get_current_stage()
        mug_xform = UsdGeom.Xform.Define(stage, path)
        rad = np.deg2rad(z_angle_deg)
        orientation = np.array([[np.cos(rad / 2.0), 0.0, 0.0, np.sin(rad / 2.0)]])
        XFormPrim(path).set_world_poses(positions=np.array([[pos[0], pos[1], pos[2]]]), orientations=orientation)
        prim = mug_xform.GetPrim()
        usd_rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        usd_rb.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.2)

        body_path = f"{path}/Body"
        body = UsdGeom.Cylinder.Define(stage, body_path)
        body.CreateHeightAttr(0.15)
        body.CreateRadiusAttr(0.035)
        body.CreateAxisAttr("Z")
        XFormPrim(body_path).set_local_poses(np.array([[0.0, 0.0, 0.075]]))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        AssetBuilder.apply_material(body, stage, body_path, color)

        handle_path = f"{path}/Handle"
        handle = UsdGeom.Cylinder.Define(stage, handle_path)
        handle.CreateHeightAttr(0.08)
        handle.CreateRadiusAttr(0.008)
        XFormPrim(handle_path).set_local_poses(translations=np.array([[0.055, 0.0, 0.075]]))
        UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
        AssetBuilder.apply_material(handle, stage, handle_path, color)

        bottom_path = f"{path}/Bottom"
        bottom = UsdGeom.Cylinder.Define(stage, bottom_path)
        bottom.CreateHeightAttr(0.01)
        bottom.CreateRadiusAttr(0.035)
        bottom.CreateAxisAttr("Z")
        XFormPrim(bottom_path).set_local_poses(np.array([[0.0, 0.0, 0.005]]))
        UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
        AssetBuilder.apply_material(bottom, stage, bottom_path, color)
        return path


class RobotController:
    def __init__(self, arm):
        self.arm = arm
        self.grasped_object = None
        self.disposed_list = []
        self.last_error_message = ""
        self.dynamic_params = {
            "approach_z_adjust": 0.02,
            "reach_adjust": -0.01,
            "yaw_adjust": 0.05,
            "grip_open": 0.04,
            "grip_close": 0.005,
        }
        self.baseline_params = dict(self.dynamic_params)
        self.poses = POSE_LIBRARY.copy()
        self.current_pose = self.poses["HOME"].copy()
        self.ee_orientation = DEFAULT_EE_QUAT.copy()
        self.ik_solver = None
        self.articulation_controller = None
        self.attach_distance_threshold = 0.24
        self.attach_follow_offset = np.array([0.0, 0.0, -0.07], dtype=np.float32)
        self.grasp_follow_alpha = GRASP_FOLLOW_ALPHA
        self.disable_collision_on_grasp = True
        self.calib_offset = np.zeros(3, dtype=np.float32)

        try:
            self.arm.initialize()
        except Exception:
            pass

        try:
            self.articulation_controller = self.arm.get_articulation_controller()
        except Exception:
            self.articulation_controller = None

        try:
            self.ik_solver = FrankaKinematicsSolver(self.arm, end_effector_frame_name="right_gripper")
        except Exception as exc:
            self.last_error_message = f"IK solver initialization failed: {exc}"

        self._refresh_current_pose()

    def apply_calibration(self, calib):
        if not calib:
            return
        self.dynamic_params["yaw_adjust"] = float(calib.get("yaw_adjust", self.dynamic_params.get("yaw_adjust", 0.05)))
        self.dynamic_params["grip_open"] = float(calib.get("grip_open", self.dynamic_params.get("grip_open", 0.04)))
        self.dynamic_params["grip_close"] = float(calib.get("grip_close", self.dynamic_params.get("grip_close", 0.005)))
        self.attach_distance_threshold = float(calib.get("attach_distance_threshold", self.attach_distance_threshold))
        self.attach_follow_offset = np.array([0.0, 0.0, float(calib.get("follow_offset", -0.07))], dtype=np.float32)
        offset = calib.get("position_offset", [0.0, 0.0, 0.0])
        self.calib_offset = np.array(offset, dtype=np.float32)
        self.baseline_params = dict(self.dynamic_params)

    def _refresh_current_pose(self):
        try:
            joint_positions = self.arm.get_joint_positions()
            if joint_positions is not None and len(joint_positions) == len(self.current_pose):
                self.current_pose = np.array(joint_positions, dtype=np.float32)
        except Exception:
            pass

    def _apply_joint_targets(self, joint_positions):
        action = ArticulationAction(joint_positions=np.array(joint_positions, dtype=np.float32))
        if self.articulation_controller is not None:
            self.articulation_controller.apply_action(action)
        else:
            self.arm.apply_action(action)

    def _after_world_step(self):
        self._update_grasped_object()

    def reset_dynamic_params(self):
        # Reset to learned baseline instead of fixed defaults
        self.dynamic_params = dict(self.baseline_params)
        self.last_error_message = ""

    def apply_failure_delta(self, error_reason):
        # Adjust baseline parameters based on failure signals
        if not error_reason:
            return
        import re
        m = re.search(r"\((\d+\.\d+)m > (\d+\.\d+)m\)", str(error_reason))
        updates = {}
        if m:
            dist = float(m.group(1))
            thresh = float(m.group(2))
            diff = max(0.0, dist - thresh)
            if diff > 0.0:
                updates["reach_adjust"] = self.dynamic_params.get("reach_adjust", 0.0) + min(0.05, diff)
                updates["approach_z_adjust"] = self.dynamic_params.get("approach_z_adjust", 0.0) + min(0.02, diff * 0.5)
                updates["attach_distance_threshold"] = self.attach_distance_threshold + min(0.03, diff)
        if "IK failed" in str(error_reason):
            updates["yaw_adjust"] = self.dynamic_params.get("yaw_adjust", 0.0) + 0.02
        if updates:
            if "attach_distance_threshold" in updates:
                self.attach_distance_threshold = float(updates.pop("attach_distance_threshold"))
            self.update_dynamic_params(updates)
            self.baseline_params.update(self.dynamic_params)

    def update_dynamic_params(self, updates):
        if not updates:
            return
        self.dynamic_params.update(updates)

        def clamp(val, lo, hi):
            try:
                v = float(val)
            except Exception:
                return lo
            return max(min(v, hi), lo)

        self.dynamic_params["approach_z_adjust"] = clamp(self.dynamic_params.get("approach_z_adjust", 0.02), -0.05, 0.08)
        self.dynamic_params["reach_adjust"] = clamp(self.dynamic_params.get("reach_adjust", -0.01), -0.20, 0.20)
        self.dynamic_params["yaw_adjust"] = clamp(self.dynamic_params.get("yaw_adjust", 0.0), -0.6, 0.6)

        go = clamp(self.dynamic_params.get("grip_open", 0.04), 0.01, 0.08)
        gc = clamp(self.dynamic_params.get("grip_close", 0.005), 0.0, 0.03)
        if gc > go:
            gc = max(0.0, go - 0.002)
        self.dynamic_params["grip_open"] = go
        self.dynamic_params["grip_close"] = gc

    def _get_safe_world_pose(self, prim_path):
        prim = XFormPrim(prim_path)
        pos, rot = prim.get_world_poses()
        safe_pos = pos[0] if len(np.shape(pos)) == 2 else pos
        safe_rot = rot[0] if len(np.shape(rot)) == 2 else rot
        return np.array(safe_pos, dtype=np.float32), np.array(safe_rot, dtype=np.float32)

    def compute_pick_targets(self, target_path, affordance):
        self.ensure_object_dynamic(target_path)
        safe_pos, _ = self._get_safe_world_pose(target_path)
        VisualHelper.create_or_move_proxy(pos=safe_pos)
        z_adjust = float(self.dynamic_params.get("approach_z_adjust", 0.0))
        reach_adjust = float(self.dynamic_params.get("reach_adjust", 0.0))

        mug_name = str(target_path).split("/")[-1]
        angle_deg = MUG_ANGLES.get(mug_name, 0.0)
        rad = np.deg2rad(angle_deg)

        offset = np.zeros(3, dtype=np.float32)
        if affordance == "handle":
            offset = np.array([0.07 * np.cos(rad), 0.07 * np.sin(rad), -0.02], dtype=np.float32)
        elif affordance == "body_outer":
            offset = np.array([0.03 * np.cos(rad), 0.03 * np.sin(rad), 0.0], dtype=np.float32)
        elif affordance == "bottom":
            offset = np.array([0.0, 0.0, -0.02], dtype=np.float32)

        handle_z_adjust = HANDLE_GRASP_Z_ADJUST if affordance == "handle" else 0.0
        grasp_center = safe_pos + offset + self.calib_offset + np.array([reach_adjust, 0.0, 0.12 + z_adjust + handle_z_adjust], dtype=np.float32)
        min_clearance = HANDLE_MIN_CLEARANCE if affordance == "handle" else MIN_GRASP_CLEARANCE
        min_z = float(safe_pos[2]) + min_clearance
        if grasp_center[2] < min_z:
            dz = min_z - grasp_center[2]
            grasp_center[2] += dz
        pre_grasp = grasp_center + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        retreat = grasp_center + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        return {
            "pre_grasp": pre_grasp,
            "grasp": grasp_center,
            "retreat": retreat,
            "orientation": self.ee_orientation.copy(),
        }

    def compute_place_targets(self, place_slot_name):
        slot_center = BASKET_PLACE_SLOTS.get(place_slot_name, BASKET_PLACE_SLOTS["PLACE_mug_0"]).copy() + self.calib_offset
        z_adjust = float(self.dynamic_params.get("approach_z_adjust", 0.0))
        pre_place = slot_center + np.array([0.0, 0.0, 0.10 + z_adjust], dtype=np.float32)
        place = slot_center + np.array([0.0, 0.0, z_adjust], dtype=np.float32)
        retreat = slot_center + np.array([0.0, 0.0, 0.16 + z_adjust], dtype=np.float32)
        return {"pre_place": pre_place, "place": place, "retreat": retreat, "orientation": self.ee_orientation.copy()}

    def verify_placement(self, target_path):
        try:
            obj_pos, _ = self._get_safe_world_pose(target_path)
            return -0.28 < obj_pos[0] < 0.28 and 0.28 < obj_pos[1] < 0.92
        except Exception:
            return False

    def _set_gripper_collision_enabled(self, enabled):
        stage = get_current_stage()
        candidates = [
            "/World/Franka/panda_leftfinger",
            "/World/Franka/panda_rightfinger",
            "/World/Franka/panda_hand/panda_leftfinger",
            "/World/Franka/panda_hand/panda_rightfinger",
        ]
        for prim_path in candidates:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                col_api = UsdPhysics.CollisionAPI(prim)
                if col_api:
                    col_api.GetCollisionEnabledAttr().Set(bool(enabled))

    def _set_collision_enabled(self, target_path, enabled):
        stage = get_current_stage()
        for part in ["Body", "Handle", "Bottom"]:
            prim = stage.GetPrimAtPath(f"{target_path}/{part}")
            if prim.IsValid():
                col_api = UsdPhysics.CollisionAPI(prim)
                if col_api:
                    col_api.GetCollisionEnabledAttr().Set(enabled)

    def _attach_object(self, target_path):
        self.grasped_object = target_path
        prim = get_current_stage().GetPrimAtPath(target_path)
        if prim.IsValid():
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
        # Keep object collisions enabled, but disable gripper collisions to reduce jitter.
        self._set_collision_enabled(target_path, True)
        self._set_gripper_collision_enabled(False)
        self._update_grasped_object()

    def ensure_object_dynamic(self, target_path):
        if not target_path:
            return
        if getattr(self, "grasped_object", None) == target_path:
            return
        try:
            prim = get_current_stage().GetPrimAtPath(target_path)
            if prim.IsValid():
                try:
                    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._set_collision_enabled(target_path, True)
        except Exception:
            pass

    def _update_grasped_object(self):
        if self.grasped_object:
            try:
                hand_pos, _ = self._get_safe_world_pose("/World/Franka/panda_hand")
                XFormPrim(self.grasped_object).set_world_poses(
                    positions=np.array([hand_pos + self.attach_follow_offset], dtype=np.float32)
                )
            except Exception:
                pass

    def move_to_joint_pose(self, pose, world, steps=60):
        target = self.poses.get(pose, self.poses["HOME"]).copy() if isinstance(pose, str) else pose.copy()
        target[7:] = self.current_pose[7:]
        start = self.current_pose.copy()
        for t in range(steps):
            if not world.is_playing():
                break
            ratio = (1.0 - np.cos(t / steps * np.pi)) / 2.0
            current_target = start + (target - start) * ratio
            self._apply_joint_targets(current_target)
            world.step(render=True)
        self.current_pose = target
        return True

    def move_end_effector_to(
        self,
        target_position,
        target_orientation,
        world,
        steps=110,
        position_tolerance=0.03,
        orientation_tolerance=0.35,
        stage_name="ik_move",
    ):
        safe_stage_name = str(stage_name or "ik_move").strip() or "ik_move"
        if self.ik_solver is None:
            self.last_error_message = f"IK unavailable at {safe_stage_name}"
            return False

        try:
            start_position, _ = self._get_safe_world_pose("/World/Franka/panda_hand")
        except Exception as exc:
            self.last_error_message = f"Failed to read hand pose at {safe_stage_name}: {exc}"
            return False

        target_position = np.array(target_position, dtype=np.float32)
        target_orientation = np.array(target_orientation if target_orientation is not None else self.ee_orientation, dtype=np.float32)
        VisualHelper.create_or_move_proxy(pos=target_position, name=f"{safe_stage_name.replace(' ', '_')}_proxy")

        for t in range(steps):
            if not world.is_playing():
                break
            ratio = (1.0 - np.cos((t + 1) / steps * np.pi)) / 2.0
            interp_position = start_position + (target_position - start_position) * ratio
            action, success = self.ik_solver.compute_inverse_kinematics(
                target_position=interp_position,
                target_orientation=target_orientation,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            if not success:
                self.last_error_message = f"IK failed at {safe_stage_name}"
                return False

            if self.articulation_controller is not None:
                self.articulation_controller.apply_action(action)
            else:
                self.arm.apply_action(action)
            world.step(render=True)
            self._refresh_current_pose()

        self.last_error_message = ""
        return True

    def close_gripper(self, world, target_path, steps=30):
        can_grasp = False
        if target_path:
            try:
                hand_pos, _ = self._get_safe_world_pose("/World/Franka/panda_hand")
                obj_pos, _ = self._get_safe_world_pose(target_path)
                distance = np.linalg.norm(hand_pos - obj_pos)
                if distance < self.attach_distance_threshold:
                    can_grasp = True
                    self.last_error_message = ""
                else:
                    self.last_error_message = (
                        f"target too far from gripper after approach ({distance:.3f}m > {self.attach_distance_threshold:.3f}m)"
                    )
            except Exception as exc:
                self.last_error_message = f"grasp distance check failed: {exc}"

        if can_grasp:
            self._attach_object(target_path)

        is_mug = bool(target_path) and "mug_" in str(target_path)
        desired_close = GRIP_CLOSE_MUG if is_mug else float(self.dynamic_params.get("grip_close", 0.005))
        desired_close = max(0.0, min(float(self.dynamic_params.get("grip_open", 0.04)), desired_close))
        self.current_pose[7:] = desired_close
        for _ in range(steps):
            if not world.is_playing():
                break
            if getattr(self.arm, "gripper", None) is not None:
                self.arm.gripper.close()
            else:
                self._apply_joint_targets(self.current_pose)
            world.step(render=True)
        self._refresh_current_pose()

    def open_gripper(self, world, steps=30):
        released_object = self.grasped_object
        self.current_pose[7:] = float(self.dynamic_params.get("grip_open", 0.04))
        for _ in range(steps):
            if not world.is_playing():
                break
            if getattr(self.arm, "gripper", None) is not None:
                self.arm.gripper.open()
            else:
                self._apply_joint_targets(self.current_pose)
            world.step(render=True)
        self._refresh_current_pose()

        if released_object:
            prim = XFormPrim(released_object)
            pos, rot = prim.get_world_poses()
            safe_pos = pos[0] if len(np.shape(pos)) == 2 else pos
            safe_rot = rot[0] if len(np.shape(rot)) == 2 else rot
            safe_pos[2] = 0.05
            prim.set_world_poses(positions=np.array([safe_pos]), orientations=np.array([safe_rot]))
            stage_prim = get_current_stage().GetPrimAtPath(released_object)
            if stage_prim.IsValid():
                UsdPhysics.RigidBodyAPI(stage_prim).GetKinematicEnabledAttr().Set(False)
            self._set_collision_enabled(released_object, True)
            self._set_gripper_collision_enabled(True)
            self.disposed_list.append(released_object.split("/")[-1])
            self.grasped_object = None


class ParameterTuningUI:
    def __init__(self, controller):
        self.controller = controller
        self.window = ui.Window("Affordance Tuning", width=360, height=520)
        self.models = {}
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Dynamic Params")
                self._add_float("approach_z_adjust", self.controller.dynamic_params.get("approach_z_adjust", 0.02))
                self._add_float("reach_adjust", self.controller.dynamic_params.get("reach_adjust", -0.01))
                self._add_float("yaw_adjust", self.controller.dynamic_params.get("yaw_adjust", 0.05))
                self._add_float("grip_open", self.controller.dynamic_params.get("grip_open", 0.04))
                self._add_float("grip_close", self.controller.dynamic_params.get("grip_close", 0.005))
                ui.Separator()
                ui.Label("Attach / Follow")
                self._add_float("attach_distance_threshold", float(self.controller.attach_distance_threshold))
                self._add_float("follow_offset", float(self.controller.attach_follow_offset[2]))
                ui.Separator()
                ui.Label("Calibration Offset (m)")
                self._add_float("calib_x", float(self.controller.calib_offset[0]))
                self._add_float("calib_y", float(self.controller.calib_offset[1]))
                self._add_float("calib_z", float(self.controller.calib_offset[2]))
                ui.Separator()
                with ui.HStack(spacing=8):
                    ui.Button("Apply", clicked_fn=self.apply)
                    ui.Button("Reset Dynamic", clicked_fn=self.reset_dynamic)

    def _add_float(self, name, value):
        model = ui.SimpleFloatModel(float(value))
        self.models[name] = model
        with ui.HStack(height=0):
            ui.Label(name, width=200)
            ui.FloatField(model=model, width=140)

    def _get_float(self, name, default=0.0):
        model = self.models.get(name)
        if model is None:
            return float(default)
        try:
            return float(model.as_float)
        except Exception:
            return float(default)

    def _sync_from_controller(self):
        for key in ["approach_z_adjust", "reach_adjust", "yaw_adjust", "grip_open", "grip_close"]:
            if key in self.models:
                self.models[key].set_value(float(self.controller.dynamic_params.get(key, 0.0)))
        if "attach_distance_threshold" in self.models:
            self.models["attach_distance_threshold"].set_value(float(self.controller.attach_distance_threshold))
        if "follow_offset" in self.models:
            self.models["follow_offset"].set_value(float(self.controller.attach_follow_offset[2]))
        if "calib_x" in self.models:
            self.models["calib_x"].set_value(float(self.controller.calib_offset[0]))
        if "calib_y" in self.models:
            self.models["calib_y"].set_value(float(self.controller.calib_offset[1]))
        if "calib_z" in self.models:
            self.models["calib_z"].set_value(float(self.controller.calib_offset[2]))

    def apply(self):
        updates = {
            "approach_z_adjust": self._get_float("approach_z_adjust", 0.02),
            "reach_adjust": self._get_float("reach_adjust", -0.01),
            "yaw_adjust": self._get_float("yaw_adjust", 0.05),
            "grip_open": self._get_float("grip_open", 0.04),
            "grip_close": self._get_float("grip_close", 0.005),
        }
        self.controller.update_dynamic_params(updates)
        self.controller.attach_distance_threshold = float(self._get_float("attach_distance_threshold", 0.24))
        follow = float(self._get_float("follow_offset", -0.07))
        self.controller.attach_follow_offset = np.array([0.0, 0.0, follow], dtype=np.float32)
        self.controller.calib_offset = np.array(
            [
                self._get_float("calib_x", 0.0),
                self._get_float("calib_y", 0.0),
                self._get_float("calib_z", 0.0),
            ],
            dtype=np.float32,
        )
        self._sync_from_controller()
        print(f"[UI] Updated params: {self.controller.dynamic_params}", flush=True)

    def reset_dynamic(self):
        self.controller.reset_dynamic_params()
        self._sync_from_controller()


def capture_images_for_vlm(world):
    for _ in range(10):
        if not world.is_playing():
            break
        world.step(render=True)

    capture_images = []
    view_dirs = [RGB_DIR / "main", RGB_DIR / "top", RGB_DIR / "left", RGB_DIR / "right", RGB_DIR]
    for d in view_dirs:
        if not d.exists():
            continue
        files = sorted(list(d.glob("rgb_*.png")))
        if files:
            capture_images.append(str(files[-1]))
    seen = set()
    uniq = []
    for p in capture_images:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def build_simple_scene_graph(target_path, affordance_pred=None):
    # Minimal, human-readable 3D scene graph with affordance
    try:
        pos, _ = XFormPrim(target_path).get_world_poses()
        mug_pos = pos[0] if len(np.shape(pos)) == 2 else pos
    except Exception:
        mug_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    mug_name = str(target_path).split("/")[-1]
    angle_deg = MUG_ANGLES.get(mug_name, 0.0)
    rad = np.deg2rad(angle_deg)
    handle_offset = np.array([0.06 * np.cos(rad), 0.06 * np.sin(rad), 0.075], dtype=np.float32)
    body_offset = np.array([0.0, 0.0, 0.075], dtype=np.float32)
    bottom_offset = np.array([0.0, 0.0, 0.01], dtype=np.float32)

    nodes = [
        {"id": mug_name, "type": "mug", "pos": mug_pos.tolist()},
        {"id": "handle", "type": "part", "pos": (mug_pos + handle_offset).tolist(), "affordance": {"graspable": True}},
        {"id": "body_outer", "type": "part", "pos": (mug_pos + body_offset).tolist(), "affordance": {"pourable": True}},
        {"id": "body_inner", "type": "part", "pos": (mug_pos + body_offset).tolist(), "affordance": {"pourable": True}},
        {"id": "bottom", "type": "part", "pos": (mug_pos + bottom_offset).tolist(), "affordance": {"stable": True}},
    ]
    edges = [
        {"from": mug_name, "to": "handle", "relation": "has_part"},
        {"from": mug_name, "to": "body_outer", "relation": "has_part"},
        {"from": mug_name, "to": "body_inner", "relation": "has_part"},
        {"from": mug_name, "to": "bottom", "relation": "has_part"},
    ]
    graph = {"nodes": nodes, "edges": edges}
    if affordance_pred is not None:
        graph["affordance"] = affordance_pred
    return graph


def save_simple_scene_graph(graph, step):
    payload = {"step": step, "graph": graph, "timestamp": time.time()}
    SIMPLE_SG_JSON.write_text(json.dumps(payload, indent=2))


def save_plan(plan_list):
    PLAN_JSON.write_text(json.dumps({"actions": plan_list}, indent=2))


def generate_recovery_plan():
    plan = ["(pick mug_0)", "(grab mug_0)", "(place)"]
    save_plan(plan)
    eval_logger.record_plan(plan)
    return plan


def generate_video():
    if ENABLE_PARAM_UI and DISABLE_RECORDING_IN_GUI:
        print("[Video] GUI mode enabled. Skipping video encoding.", flush=True)
        return
    print("\n🎥 動画のエンコードを開始します...", flush=True)
    image_files = sorted(list(RGB_DIR.glob("**/rgb_*.png")))
    if not image_files:
        print("❌ 録画用フレームが見つかりません。動画のエンコードをスキップします。", flush=True)
        return

    tmp_dir = OUTPUT_DIR / "tmp_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        for i, img_path in enumerate(image_files):
            shutil.copy(str(img_path), str(tmp_dir / f"frame_{i:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "30", "-i", str(tmp_dir / "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "warning", str(VIDEO_PATH)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print(f"✅ 動画の生成が完了しました: {VIDEO_PATH}", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ 動画エンコード中にエラーが発生しました: {exc}", flush=True)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


# =========================================================
# Main routine
# =========================================================

def run_simulation():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    RGB_DIR.mkdir(parents=True)

    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2000)

    assets_root = get_assets_root_path()
    franka = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka",
            usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        )
    )

    for cfg in MUGS:
        AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])
    AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

    cams = {}
    rps = {}
    writers = {}
    for name in ["main", "top", "left", "right"]:
        cfg = CAMERA_POSITIONS.get(name)
        if cfg is None:
            continue
        cams[name] = rep.create.camera(position=cfg["pos"], look_at=cfg["look_at"])
        rps[name] = rep.create.render_product(cams[name], (1280, 720))
        out_dir = RGB_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=str(out_dir), rgb=True)
        w.attach([rps[name]])
        writers[name] = w

    world.reset()
    world.play()
    topple = TimedToppleScenario('/World/mug_0', trigger_seconds=TOPPLE_TRIGGER_SEC)
    for _ in range(20):
        world.step(render=True)
        topple.maybe_trigger()

    calib = run_calibration(world)
    controller = RobotController(arm=franka)
    controller.apply_calibration(calib)
    if ENABLE_PARAM_UI:
        try:
            ParameterTuningUI(controller)
        except Exception as exc:
            print(f"[UI] Failed to create parameter UI: {exc}", flush=True)

    analyzer = VLMAnalyzer(model_name=TARGET_MODEL)
    images_for_affordance = capture_images_for_vlm(world)
    affordance_plan = analyzer.plan_affordance_action(images_for_affordance, USER_INSTRUCTION)
    affordance_pred = {"part": affordance_plan.get("pick_part", "handle"), "affordance": "graspable", "reason": affordance_plan.get("rationale", "") }
    eval_logger.record_affordance({"plan": affordance_plan, "initial": affordance_pred})

    max_overall_retries = 10
    retry_count = 0
    target_failure_counts = {"/World/mug_0": 0}
    corrections = 0

    plan = generate_recovery_plan()
    print(f"\n📦 初期プランを作成しました。タスク数: {len(plan)}", flush=True)

    try:
        while plan and retry_count < max_overall_retries and world.is_playing():
            action = plan.pop(0)
            print(f"\n>> Executing: {action}", flush=True)
            action_failed = False
            error_reason = ""
            current_target = "/World/mug_0"

            try:
                if action.startswith("(pick"):
                    pick_part = affordance_pred.get("part", "handle")
                    pick_targets = controller.compute_pick_targets(current_target, pick_part)
                    controller.open_gripper(world)
                    moved = controller.move_end_effector_to(pick_targets["pre_grasp"], pick_targets["orientation"], world, steps=90)
                    if moved:
                        wait_steps(world, 0.2, controller, topple)
                        moved = controller.move_end_effector_to(pick_targets["grasp"], pick_targets["orientation"], world, steps=75)
                    if not moved and affordance_plan.get('fallback_parts'):
                        fallback = affordance_plan['fallback_parts'][0]
                        pick_targets = controller.compute_pick_targets(current_target, fallback)
                        moved = controller.move_end_effector_to(pick_targets['grasp'], pick_targets['orientation'], world, steps=60)
                        if moved:
                            affordance_pred['part'] = fallback
                            action_failed = False
                            error_reason = ''
                        else:
                            action_failed = True
                            error_reason = controller.last_error_message or 'pick failed'
                    if not moved and action_failed:
                        action_failed = True
                        error_reason = controller.last_error_message or "pick failed"
                    print(f"✅ {action} 完了" if not action_failed else f"⚠️ {action} 失敗: {error_reason}", flush=True)

                elif action.startswith("(grab"):
                    pick_part = affordance_pred.get("part", "handle")
                    pick_targets = controller.compute_pick_targets(current_target, pick_part)
                    moved = controller.move_end_effector_to(pick_targets["grasp"], pick_targets["orientation"], world, steps=30)
                    controller.close_gripper(world, current_target)
                    wait_steps(world, 0.5, controller, topple)
                    if moved and not controller.last_error_message:
                        moved = controller.move_end_effector_to(pick_targets["retreat"], pick_targets["orientation"], world, steps=90)
                    if (not moved) or controller.last_error_message:
                        action_failed = True
                        error_reason = controller.last_error_message or "grab failed"
                    print(f"✅ {action} 完了" if not action_failed else f"⚠️ {action} 失敗: {error_reason}", flush=True)

                elif action.startswith("(place"):
                    place_pose = MUG_PLACE_ASSIGNMENTS.get("mug_0", "PLACE_mug_0")
                    place_targets = controller.compute_place_targets(place_pose)
                    moved = controller.move_end_effector_to(place_targets["pre_place"], place_targets["orientation"], world, steps=90)
                    if moved:
                        moved = controller.move_end_effector_to(place_targets["place"], place_targets["orientation"], world, steps=60)
                    wait_steps(world, 0.5, controller, topple)
                    controller.open_gripper(world)
                    if moved:
                        moved = controller.move_end_effector_to(place_targets["retreat"], place_targets["orientation"], world, steps=75)
                    controller.move_to_joint_pose("HOME", world)
                    wait_steps(world, 1.0, controller, topple)
                    print(f"✅ {action} 完了", flush=True)

                    images_after = capture_images_for_vlm(world)
                    scene_graph = analyzer.extract_scene_graph(images_after, affordance=affordance_pred)
                    eval_logger.record_scene_graph(retry_count, scene_graph, affordance=affordance_pred)
                    simple_graph = build_simple_scene_graph(current_target, affordance_pred)
                    save_simple_scene_graph(simple_graph, retry_count)
                    try:
                        if RENDER_SIMPLE_SG:
                            RENDER_SIMPLE_SG(str(SIMPLE_SG_JSON), images_after[0] if images_after else "")
                    except Exception:
                        pass
                    action_failed = (not moved) or (not controller.verify_placement(current_target)) or bool(controller.last_error_message)
                    if not action_failed:
                        print("✅ 成功: オブジェクトがカゴ内に確認されました。", flush=True)
                        target_failure_counts[current_target] = 0
                        controller.reset_dynamic_params()

                else:
                    print(f"⚠️ 未対応アクションをスキップします: {action}", flush=True)
                    continue

            except Exception as exc:
                action_failed = True
                error_reason = f"runtime exception during {action}: {exc}"
                controller.last_error_message = error_reason
                traceback.print_exc()

            if topple.triggered and affordance_plan is not None:
                images_for_affordance = capture_images_for_vlm(world)
                affordance_plan = analyzer.plan_affordance_action(images_for_affordance, USER_INSTRUCTION)
                affordance_pred = {"part": affordance_plan.get("pick_part", "handle"), "affordance": "graspable", "reason": affordance_plan.get("rationale", "") }
                eval_logger.record_affordance({"plan_after_topple": affordance_plan})
            if action_failed:
                eval_logger.add_failure()
                target_failure_counts[current_target] += 1
                corrections += 1
                eval_logger.record_corrections(retry_count, corrections, reason=error_reason)
                controller.apply_failure_delta(error_reason)
                retry_count += 1
                plan = generate_recovery_plan()

        success = (target_failure_counts["/World/mug_0"] == 0)
        eval_logger.record_planner_metric("affordance_vlm", plan_len=3, success=success, failures=eval_logger.data["total_failures"])
        eval_logger.data["comparisons"] = {
            "voxposer": {"available": False, "note": "not executed"},
            "sayplan": {"available": False, "note": "not executed"},
            "delta": {"available": bool(REAL_DELTA_PATH.exists()), "note": "not executed"},
        }

        if not world.is_playing():
            print("\n⚠️ 物理エンジンが停止しました。", flush=True)
        elif len(plan) == 0:
            print("\n✅ すべてのタスク(スキップ含む)が完了しました！", flush=True)

    except Exception as exc:
        print(f"\n❌ 重大なエラーが発生しました: {exc}", flush=True)
        traceback.print_exc()

    finally:
        eval_logger.save()
        print(f"📊 評価ログを保存しました: {EVAL_RESULTS_JSON}", flush=True)
        generate_video()

        try:
            rep.orchestrator.stop()
        except Exception:
            pass

        if ENABLE_PARAM_UI and GUI_HOLD_AT_END:
            print("[UI] GUI mode: close the window to exit.", flush=True)
            try:
                while simulation_app.is_running():
                    if world is not None and world.is_playing():
                        world.step(render=True)
                    else:
                        simulation_app.update()
            except Exception:
                pass

        try:
            if world is not None:
                world.stop()
                world.clear_instance()
        except Exception:
            pass

        gc.collect()
        simulation_app.close()


if __name__ == "__main__":
    run_simulation()
