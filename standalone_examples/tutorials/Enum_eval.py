import os
import json
import base64
import shutil
import subprocess
import argparse
import numpy as np
import traceback
import faulthandler
import gc
import time
from pathlib import Path
from functools import wraps
from openai import OpenAI

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
BASE_OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation/vlm_delta_model")
DEFAULT_OUTPUT_TAG = "vlm_delta_run1"

# These will be configured after argument parsing.
OUTPUT_DIR = BASE_OUTPUT_DIR / DEFAULT_OUTPUT_TAG
RGB_DIR = OUTPUT_DIR / "frames"
VIDEO_PATH = OUTPUT_DIR / "mug_sorting.mp4"
EVAL_RESULTS_JSON = OUTPUT_DIR / "evaluation_results.json"
PLAN_JSON = OUTPUT_DIR / "actions.json"
CALIB_JSON = OUTPUT_DIR / "calibration.json"
SCENE_GRAPH_JSON_DIR = OUTPUT_DIR / "scene_graph_json"

DEFAULT_DELTA_IMPL = "delta"
DELTA_IMPL_TO_SCRIPT = {
    "delta_original": "delta_original.py",
    "original": "delta_original.py",
    "delta_orginal": "delta_original.py",
    "delta": "delta.py",
    "current": "delta.py",
}
REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki") / DELTA_IMPL_TO_SCRIPT[DEFAULT_DELTA_IMPL]
DELTA_PYTHON = "/usr/bin/python3"

MUGS = [
    {"id": "mug_0", "pos": [0.4, 0.5, 0.05], "color": [1.0, 0.0, 0.0], "angle": -120},
    {"id": "mug_1", "pos": [0.5, 0.25, 0.05], "color": [0.0, 1.0, 0.0], "angle": 180},
    {"id": "mug_2", "pos": [0.6, 0.0, 0.05], "color": [0.0, 0.0, 1.0], "angle": 120},
    {"id": "mug_3", "pos": [0.5, -0.25, 0.05], "color": [1.0, 1.0, 0.0], "angle": 60},
    {"id": "mug_4", "pos": [0.4, -0.5, 0.05], "color": [0.0, 1.0, 1.0], "angle": 0},
    {"id": "mug_5", "pos": [0.35, -0.62, 0.05], "color": [1.0, 0.0, 1.0], "angle": 20},
]

MUG_ANGLES = {m["id"]: m["angle"] for m in MUGS}
MUG_COLOR_TO_ID = {
    "red": "mug_0",
    "green": "mug_1",
    "blue": "mug_2",
    "yellow": "mug_3",
    "cyan": "mug_4",
    "magenta": "mug_5",
}
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
GRIP_CLOSE_MUG = 0.0015
GRASP_FOLLOW_ALPHA = 0.25
BASKET_PLACE_SLOTS = {
    "PLACE_mug_0": np.array([0.18, 0.45, 0.18], dtype=np.float32),
    "PLACE_mug_1": np.array([0.09, 0.57, 0.18], dtype=np.float32),
    "PLACE_mug_2": np.array([0.00, 0.45, 0.18], dtype=np.float32),
    "PLACE_mug_3": np.array([-0.09, 0.57, 0.18], dtype=np.float32),
    "PLACE_mug_4": np.array([-0.18, 0.45, 0.18], dtype=np.float32),
    "PLACE_mug_5": np.array([0.0, 0.62, 0.18], dtype=np.float32),
    "BASKET_HIGH": np.array([0.0, 0.55, 0.28], dtype=np.float32),
}
MUG_PLACE_ASSIGNMENTS = {
    "mug_0": "PLACE_mug_0",
    "mug_1": "PLACE_mug_1",
    "mug_2": "PLACE_mug_2",
    "mug_3": "PLACE_mug_3",
    "mug_4": "PLACE_mug_4",
    "mug_5": "PLACE_mug_5",
}

ENABLE_DISTURBANCE_TEST = False
DISTURBANCE_TRIGGER_SEC = 60.0
DISTURBANCE_TARGETS = ("mug_5",)
ALLOW_PENETRATION_GRASP = False
SAFE_ATTACH_DISTANCE_MIN = 0.10
SAFE_ATTACH_DISTANCE_MAX = 0.30
SAFE_ATTACH_DISTANCE_GRACE = 0.045
SAFE_FOLLOW_OFFSET_MIN = -0.12
SAFE_FOLLOW_OFFSET_MAX = -0.02
MIN_GRASP_CLEARANCE = 0.12
ENABLE_PARAM_UI = True
HEADLESS = not ENABLE_PARAM_UI
DISABLE_RECORDING_IN_GUI = True
GUI_HOLD_AT_END = True

# CLI options
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--output-tag', default=DEFAULT_OUTPUT_TAG, help='Output evaluation tag name')
parser.add_argument('--headless', action='store_true', help='Run without UI')
parser.add_argument('--no-window', action='store_true', help='Run without window display (alias for headless)')
parser.add_argument('--vlm_model', default=TARGET_MODEL, help='Vision-language model to use (gpt-4o etc.)')
cli_args, _ = parser.parse_known_args()

if cli_args.output_tag:
    DEFAULT_OUTPUT_TAG = cli_args.output_tag
if cli_args.headless or cli_args.no_window:
    ENABLE_PARAM_UI = False
    HEADLESS = True
else:
    ENABLE_PARAM_UI = True
    HEADLESS = False

TARGET_MODEL = cli_args.vlm_model


def _normalize_scene_graph_for_logging(graph):
    if not isinstance(graph, dict):
        return {"nodes": [], "edges": []}

    if "nodes" in graph and "edges" in graph:
        nodes = graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
        edges = graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
        clean_nodes = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = str(node.get("id", node.get("name", ""))).strip().lower().replace(" ", "_")
            if not nid:
                continue
            n = dict(node)
            n["id"] = nid
            n.setdefault("label", nid)
            clean_nodes.append(n)
        clean_edges = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            s = str(edge.get("source", edge.get("subject", ""))).strip().lower().replace(" ", "_")
            r = str(edge.get("relation", edge.get("predicate", ""))).strip().lower().replace(" ", "_")
            o = str(edge.get("target", edge.get("object", ""))).strip().lower().replace(" ", "_")
            if s and r and o:
                clean_edges.append({"source": s, "relation": r, "target": o})
        return {"nodes": clean_nodes, "edges": clean_edges}

    rooms = graph.get("rooms", {}) if isinstance(graph.get("rooms", {}), dict) else {}
    nodes = [{"id": "agent", "type": "agent", "label": "agent"}]
    seen_nodes = {"agent"}
    edges = []

    agent = graph.get("agent", {}) if isinstance(graph.get("agent", {}), dict) else {}
    agent_pos = str(agent.get("position", "workspace")).strip().lower().replace(" ", "_")
    if agent_pos:
        if agent_pos not in seen_nodes:
            nodes.append({"id": agent_pos, "type": "room", "label": agent_pos})
            seen_nodes.add(agent_pos)
        edges.append({"source": "agent", "relation": "in", "target": agent_pos})

    for room_name, room_info in rooms.items():
        room_id = str(room_name).strip().lower().replace(" ", "_")
        if room_id and room_id not in seen_nodes:
            nodes.append({"id": room_id, "type": "room", "label": room_id})
            seen_nodes.add(room_id)

        neighbors = room_info.get("neighbor", []) if isinstance(room_info, dict) else []
        for nb in neighbors if isinstance(neighbors, list) else []:
            nb_id = str(nb).strip().lower().replace(" ", "_")
            if nb_id:
                if nb_id not in seen_nodes:
                    nodes.append({"id": nb_id, "type": "room", "label": nb_id})
                    seen_nodes.add(nb_id)
                edges.append({"source": room_id, "relation": "neighbor", "target": nb_id})

        items = room_info.get("items", {}) if isinstance(room_info, dict) else {}
        for item_name, item_info in items.items() if isinstance(items, dict) else []:
            item_id = str(item_name).strip().lower().replace(" ", "_")
            info = item_info if isinstance(item_info, dict) else {}
            if item_id and item_id not in seen_nodes:
                nodes.append({
                    "id": item_id,
                    "type": "item",
                    "label": item_id,
                    "state": info.get("state", "free"),
                    "accessible": bool(info.get("accessible", True)),
                })
                seen_nodes.add(item_id)
            if room_id and item_id:
                edges.append({"source": item_id, "relation": "in_room", "target": room_id})

    dedup_edges = []
    seen_edges = set()
    for edge in edges:
        key = (edge["source"], edge["relation"], edge["target"])
        if key in seen_edges:
            continue
        seen_edges.add(key)
        dedup_edges.append(edge)

    return {"nodes": nodes, "edges": dedup_edges}


def _scene_graph_stats(graph):
    g = _normalize_scene_graph_for_logging(graph)
    return {
        "node_count": len(g.get("nodes", [])),
        "edge_count": len(g.get("edges", [])),
    }


class EvaluationLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {
            "total_failures": 0,
            "inference_times": [],
            "action_plans": [],
            "scene_graphs": [],
            "summary": {},
            "vlm_model": TARGET_MODEL,
        }

    def add_failure(self):
        self.data["total_failures"] += 1

    def record_inference_time(self, process_name, latency):
        self.data["inference_times"].append(
            {"process": process_name, "latency_sec": round(latency, 3), "timestamp": time.time()}
        )

    def record_plan(self, plan):
        self.data["action_plans"].append({"plan": plan.copy(), "length": len(plan), "timestamp": time.time()})

    def record_scene_graph(self, step, graph, affordance=None, diagnostics=None):
        norm = _normalize_scene_graph_for_logging(graph)
        entry = {
            "step": step,
            "graph": norm,
            "stats": _scene_graph_stats(norm),
            "timestamp": time.time(),
        }
        if affordance is not None:
            entry["affordance"] = affordance
        if diagnostics is not None:
            entry["diagnostics"] = diagnostics
        self.data["scene_graphs"].append(entry)

    def _finalize_summary(self):
        sg = self.data.get("scene_graphs", [])
        if not sg:
            self.data["summary"] = {"scene_graph_count": 0, "avg_nodes": 0, "avg_edges": 0}
            return
        node_counts = [int(x.get("stats", {}).get("node_count", 0)) for x in sg]
        edge_counts = [int(x.get("stats", {}).get("edge_count", 0)) for x in sg]
        self.data["summary"] = {
            "scene_graph_count": len(sg),
            "avg_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
            "avg_edges": float(np.mean(edge_counts)) if edge_counts else 0.0,
            "nonempty_scene_graphs": int(sum(1 for x in sg if x.get("stats", {}).get("node_count", 0) > 0 and x.get("stats", {}).get("edge_count", 0) > 0)),
        }

    def save(self):
        self._finalize_summary()
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)


eval_logger = None


def configure_output_paths(output_tag):
    global OUTPUT_DIR, RGB_DIR, VIDEO_PATH, EVAL_RESULTS_JSON, PLAN_JSON, CALIB_JSON, SCENE_GRAPH_JSON_DIR, eval_logger
    OUTPUT_DIR = BASE_OUTPUT_DIR / output_tag
    RGB_DIR = OUTPUT_DIR / "frames"
    VIDEO_PATH = OUTPUT_DIR / "mug_sorting.mp4"
    EVAL_RESULTS_JSON = OUTPUT_DIR / "evaluation_results.json"
    PLAN_JSON = OUTPUT_DIR / "actions.json"
    CALIB_JSON = OUTPUT_DIR / "calibration.json"
    SCENE_GRAPH_JSON_DIR = OUTPUT_DIR / "scene_graph_json"
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

def _sanitize_calibration(calib):
    if not isinstance(calib, dict):
        calib = {}
    pos = calib.get("position_offset", [0.0, 0.0, 0.0])
    try:
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    except Exception:
        px, py, pz = 0.0, 0.0, 0.0
    pz = max(-0.01, min(0.02, pz))
    calib["position_offset"] = [px, py, pz]
    try:
        go = float(calib.get("grip_open", 0.04))
    except Exception:
        go = 0.04
    try:
        gc = float(calib.get("grip_close", 0.005))
    except Exception:
        gc = 0.005
    go = max(0.01, min(0.08, go))
    gc = max(0.0, min(0.03, gc))
    if gc > go:
        gc = max(0.0, go - 0.002)
    calib["grip_open"] = go
    calib["grip_close"] = gc
    return calib

def load_calibration():
    if CALIB_JSON.exists():
        try:
            with open(CALIB_JSON, "r") as f:
                return _sanitize_calibration(json.load(f))
        except Exception:
            pass
    return _sanitize_calibration({
        "position_offset": [0.0, 0.0, 0.0],
        "yaw_adjust": 0.03,
        "grip_open": 0.04,
        "grip_close": 0.005,
        "attach_distance_threshold": 0.24,
        "follow_offset": -0.07,
    })

def save_calibration(data):
    CALIB_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_JSON, "w") as f:
        json.dump(data, f, indent=2)

def run_calibration(world):
    # Placeholder: integrate isaacsim.replicator.agent.camera_calibration when available
    # Currently just loads existing calibration if present
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
from pxr import Gf, Sdf, UsdLux, UsdPhysics, UsdGeom, UsdShade, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim, SingleRigidPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver as FrankaKinematicsSolver
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep

carb.settings.get_settings().set_string("/log/level", "error")

fps = 30


def wait_steps(world, seconds, controller=None, disturbance=None):
    steps = int(seconds * fps)
    for _ in range(steps):
        if not world.is_playing():
            break
        world.step(render=True)
        if controller:
            controller._after_world_step()
        elif disturbance is not None:
            disturbance.maybe_trigger()


class TimedDisturbanceScenario:
    def __init__(self, enabled=True, trigger_seconds=60.0):
        self.enabled = enabled
        self.trigger_steps = max(1, int(trigger_seconds * fps))
        self.step_count = 0
        self.triggered = False
        self._rigid_cache = {}

    def _get_rigid_prim(self, target_path):
        if target_path not in self._rigid_cache:
            rigid_prim = SingleRigidPrim(target_path, name=f"{target_path.split('/')[-1]}_disturbance")
            try:
                rigid_prim.initialize()
            except Exception:
                pass
            self._rigid_cache[target_path] = rigid_prim
        return self._rigid_cache[target_path]

    def _roll_target(self, target_path, linear_velocity, angular_velocity):
        rigid_prim = self._get_rigid_prim(target_path)
        stage_prim = get_current_stage().GetPrimAtPath(target_path)
        if stage_prim.IsValid():
            UsdPhysics.RigidBodyAPI(stage_prim).GetKinematicEnabledAttr().Set(False)
        try:
            rigid_prim.set_linear_velocity(np.array(linear_velocity, dtype=np.float32))
            rigid_prim.set_angular_velocity(np.array(angular_velocity, dtype=np.float32))
            return
        except Exception:
            pass

        rb_api = UsdPhysics.RigidBodyAPI(stage_prim)
        rb_api.CreateVelocityAttr().Set(Gf.Vec3f(*linear_velocity))
        rb_api.CreateAngularVelocityAttr().Set(Gf.Vec3f(*angular_velocity))

    def maybe_trigger(self, controller=None):
        self.step_count += 1
        if not self.enabled or self.triggered or self.step_count < self.trigger_steps:
            return False

        grasped = getattr(controller, "grasped_object", None) if controller is not None else None
        blocked_targets = {f"/World/{name}" for name in DISTURBANCE_TARGETS}
        if grasped in blocked_targets:
            return False

        self._roll_target("/World/mug_2", linear_velocity=[-1.00, 1.00, 0.0], angular_velocity=[5.5, 6.5, 0.0])
        self.triggered = True
        print("\n[Test] 1分後の外乱を発動: 青い mug_2 をアームが届く範囲へ向けて転がしました。再計画で最後まで継続します。", flush=True)
        return True


class VisualHelper:
    @staticmethod
    def _sanitize_proxy_name(name):
        raw_name = str(name or "").strip()
        safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name)
        safe_name = safe_name.strip("_")
        return safe_name or "TargetProxy"

    @staticmethod
    def create_or_move_proxy(pos, name="TargetProxy"):
        stage = get_current_stage()
        safe_name = VisualHelper._sanitize_proxy_name(name)
        path = f"/World/{safe_name}"
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
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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

    def _expand_object_names(self, detection_result):
        expanded = []
        if not isinstance(detection_result, dict):
            return expanded
        for obj in detection_result.get("objects", []):
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", "")).strip().lower().replace(" ", "_")
            if not name:
                continue
            try:
                count = max(1, int(obj.get("count", 1)))
            except Exception:
                count = 1
            for i in range(count):
                expanded.append(f"{name}_{i+1}" if count > 1 else name)
        return sorted(expanded)

    def _pairwise_jaccard(self, sets):
        scores = []
        n = len(sets)
        for i in range(n):
            for j in range(i + 1, n):
                a = sets[i]
                b = sets[j]
                union = a | b
                if not union:
                    scores.append(1.0)
                else:
                    scores.append(len(a & b) / len(union))
        return scores

    def _normalize_color_label(self, color):
        c = str(color or "").strip().lower().replace(" ", "_")
        aliases = {
            "赤": "red",
            "緑": "green",
            "青": "blue",
            "黄": "yellow",
            "シアン": "cyan",
            "水色": "cyan",
            "マゼンタ": "magenta",
            "紫": "magenta",
            "red": "red",
            "green": "green",
            "blue": "blue",
            "yellow": "yellow",
            "cyan": "cyan",
            "magenta": "magenta",
        }
        for key, value in aliases.items():
            if key in c:
                return value
        return "unknown"

    def _sanitize_name(self, name):
        return str(name or "").strip().lower().replace(" ", "_")

    def _canonicalize_object(self, raw_name, color):
        name = self._sanitize_name(raw_name)
        color_norm = self._normalize_color_label(color)
        if ("mug" in name or "cup" in name) and color_norm in MUG_COLOR_TO_ID:
            return MUG_COLOR_TO_ID[color_norm], color_norm
        return name, color_norm

    def _detect_objects_single_view(self, image_path):
        default_res = {"objects": [], "relations": [], "name_map": {}}
        if not self.client or not image_path:
            return default_res

        b64 = self.encode_image_base64(image_path)
        content = [
            {
                "type": "text",
                "text": (
                    "画像内の物体を検出し、JSONのみで返してください。"
                    "出力形式は {\"objects\": [{\"name\": str, \"count\": int, \"color\": str}], "
                    "\"relations\": [{\"subject\": str, \"relation\": str, \"object\": str}]}。"
                    "mug/cup には必ず color を付ける。"
                    "color は red/green/blue/yellow/cyan/magenta/unknown のいずれか。"
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_tokens=1200,
            )
            parsed = self._safe_parse(response.choices[0].message.content, default_res)
            objects = parsed.get("objects", []) if isinstance(parsed, dict) else []
            relations = parsed.get("relations", []) if isinstance(parsed, dict) else []

            clean_objects = []
            name_map = {}
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                raw_name = self._sanitize_name(obj.get("name", ""))
                if not raw_name:
                    continue
                try:
                    count = max(1, int(obj.get("count", 1)))
                except Exception:
                    count = 1
                canonical_name, color_norm = self._canonicalize_object(raw_name, obj.get("color", "unknown"))
                name_map[raw_name] = canonical_name
                clean_objects.append({
                    "name": canonical_name,
                    "raw_name": raw_name,
                    "color": color_norm,
                    "count": count,
                })

            clean_relations = []
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                s_raw = self._sanitize_name(rel.get("subject", ""))
                r = self._sanitize_name(rel.get("relation", ""))
                o_raw = self._sanitize_name(rel.get("object", ""))
                if not s_raw or not r or not o_raw:
                    continue
                s = name_map.get(s_raw, s_raw)
                o = name_map.get(o_raw, o_raw)
                clean_relations.append({"subject": s, "relation": r, "object": o})

            return {"objects": clean_objects, "relations": clean_relations, "name_map": name_map}
        except Exception:
            return default_res

    def _fuse_multiview_detections(self, view_detections):
        object_map = {}
        relation_set = set()
        per_view = []

        for view_idx, det in enumerate(view_detections):
            obj_names = []
            for obj in det.get("objects", []):
                if not isinstance(obj, dict):
                    continue
                name = self._sanitize_name(obj.get("name", ""))
                if not name:
                    continue
                try:
                    count = max(1, int(obj.get("count", 1)))
                except Exception:
                    count = 1
                color = self._normalize_color_label(obj.get("color", "unknown"))
                if name not in object_map:
                    object_map[name] = {"name": name, "count": count, "color": color, "votes": 1}
                else:
                    object_map[name]["count"] = max(object_map[name]["count"], count)
                    object_map[name]["votes"] += 1
                    if object_map[name].get("color", "unknown") == "unknown" and color != "unknown":
                        object_map[name]["color"] = color
                obj_names.append(name)

            for rel in det.get("relations", []):
                if not isinstance(rel, dict):
                    continue
                s = self._sanitize_name(rel.get("subject", ""))
                r = self._sanitize_name(rel.get("relation", ""))
                o = self._sanitize_name(rel.get("object", ""))
                if s and r and o:
                    relation_set.add((s, r, o))

            per_view.append({"view_index": view_idx, "objects": sorted(set(obj_names)), "object_count": len(set(obj_names))})

        objects = [{"name": v["name"], "count": v["count"], "color": v.get("color", "unknown")} for v in object_map.values()]
        objects.sort(key=lambda x: x["name"])
        relations = [{"subject": s, "relation": r, "object": o} for (s, r, o) in sorted(relation_set)]

        return {
            "objects": objects,
            "relations": relations,
            "multiview_fusion": {
                "views": per_view,
                "num_views": len(view_detections),
                "unique_objects": len(objects),
            },
        }

    @measure_time("assess_multiview_identity")
    def assess_multiview_identity(self, image_paths):
        diagnostics = {
            "per_view": [],
            "pairwise_jaccard": [],
            "avg_jaccard": 0.0,
            "min_jaccard": 0.0,
            "id_consistency": "unknown",
            "notes": "",
        }
        if not image_paths:
            diagnostics["id_consistency"] = "insufficient_views"
            diagnostics["notes"] = "no image paths"
            return diagnostics

        per_view_sets = []
        for path in image_paths:
            res = self.detect_objects_for_delta([path])
            expanded = self._expand_object_names(res)
            per_view_sets.append(set(expanded))
            diagnostics["per_view"].append({
                "image": str(path),
                "objects": expanded,
                "object_count": len(expanded),
            })

        if len(per_view_sets) < 2:
            diagnostics["id_consistency"] = "insufficient_views"
            diagnostics["notes"] = "need at least two views"
            return diagnostics

        scores = self._pairwise_jaccard(per_view_sets)
        diagnostics["pairwise_jaccard"] = [round(float(s), 4) for s in scores]
        diagnostics["avg_jaccard"] = round(float(np.mean(scores)), 4) if scores else 0.0
        diagnostics["min_jaccard"] = round(float(np.min(scores)), 4) if scores else 0.0

        if diagnostics["min_jaccard"] >= 0.7:
            diagnostics["id_consistency"] = "high"
            diagnostics["notes"] = "multi-view object identities look consistent"
        elif diagnostics["min_jaccard"] >= 0.4:
            diagnostics["id_consistency"] = "medium"
            diagnostics["notes"] = "partially consistent; possible alias/occlusion"
        else:
            diagnostics["id_consistency"] = "low"
            diagnostics["notes"] = "likely inconsistent IDs across views"

        return diagnostics


    @measure_time("find_empty_space")
    def find_empty_space(self, image_paths):
        if not self.client or not image_paths:
            return "BASKET_HIGH"
        base64_images = [self.encode_image_base64(p) for p in image_paths]
        content = [{
            "type": "text",
            "text": "カゴの中の画像を分析し、マグカップが置かれていない空きスペースを判定してください。出力はJSON形式で、キーを 'suggested_pose' とし、値は 'PLACE_mug_2'〜'PLACE_mug_4' のいずれかにしてください。",
        }]
        for b64 in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            return self._safe_parse(response.choices[0].message.content, {"suggested_pose": "BASKET_HIGH"}).get(
                "suggested_pose", "BASKET_HIGH"
            )
        except Exception:
            return "BASKET_HIGH"

    @measure_time("analyze_and_adjust_params")
    def analyze_and_adjust_params(self, error_reason, current_params, attempt, calib_delta=None):
        if not self.client:
            return current_params
        prompt = f"""
        失敗理由: {error_reason} (試行 {attempt}/3)
        現在パラメータ: {json.dumps(current_params)}
        キャリブレーション差分(推定): {json.dumps(calib_delta) if calib_delta else "null"}
        同じ値を繰り返さず、エラー理由に基づいてパラメータを微調整し、結果をJSONで出力してください。
        キーは "reach_adjust", "approach_z_adjust", "yaw_adjust", "grip_open", "grip_close" を含めること。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            adjusted = self._safe_parse(response.choices[0].message.content, current_params)
            return {k: v for k, v in adjusted.items() if k in current_params}
        except Exception:
            return current_params

    

    @measure_time("estimate_calibration_delta")
    def estimate_calibration_delta(self, image_paths, calib=None):
        default_res = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}
        if not self.client or not image_paths:
            return default_res
        base64_images = [self.encode_image_base64(p) for p in image_paths]
        content = [{
            "type": "text",
            "text": (
                "複数視点画像から、ロボット手先と対象物(マグ)の見た目のズレを推定し、"
                "キャリブレーション差分としてJSONで返してください。"
                "出力は {\"dx\": float, \"dy\": float, \"dz\": float, \"dyaw\": float} のみ。"
                "単位はメートル、dyaw はラジアン。値は小さめ(絶対値0.2以下)にしてください。"
                f"\n既知のcalibration.json(もしあれば): {json.dumps(calib) if calib else 'null'}"
            ),
        }]
        for b64 in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            parsed = self._safe_parse(response.choices[0].message.content, default_res) or default_res
            out = {}
            for k in ["dx", "dy", "dz", "dyaw"]:
                try:
                    out[k] = float(parsed.get(k, default_res[k]))
                except Exception:
                    out[k] = default_res[k]
            # clamp
            out["dx"] = max(min(out["dx"], 0.2), -0.2)
            out["dy"] = max(min(out["dy"], 0.2), -0.2)
            out["dz"] = max(min(out["dz"], 0.2), -0.2)
            out["dyaw"] = max(min(out["dyaw"], 0.6), -0.6)
            return out
        except Exception:
            return default_res
    @measure_time("process_and_analyze")
    def process_and_analyze(self, image_paths, instruction):
        default_res = {"is_failed": False, "reason": "APIエラー", "suggested_order": ""}
        if not self.client or not image_paths:
            return default_res

        from PIL import Image
        from io import BytesIO

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        combined = pil_images[0]
        buffered = BytesIO()
        combined.save(buffered, format="JPEG")
        base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        prompt = f"""指示「{instruction}」完了後の画像です。失敗がないか判定しJSONのみ返してください。{{"is_failed": boolean, "reason": "理由", "suggested_order": ""}}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    ],
                }],
                response_format={"type": "json_object"},
            )
            return self._safe_parse(response.choices[0].message.content, default_res)
        except Exception:
            return default_res

    @measure_time("extract_scene_graph")
    def extract_scene_graph(self, image_paths):
        if not self.client or not image_paths:
            return {"nodes": [], "edges": []}
        try:
            from PIL import Image
            from io import BytesIO

            pil_images = [Image.open(p).convert("RGB") for p in image_paths]
            combined = pil_images[0]
            buffered = BytesIO()
            combined.save(buffered, format="JPEG", quality=80)
            base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": '画像から3Dシーングラフを構築しJSON形式のみ出力してください。{"nodes": [], "edges": []}'},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "low"}},
                    ],
                }],
                response_format={"type": "json_object"},
                max_tokens=1500,
            )
            parsed = self._safe_parse(response.choices[0].message.content, {"nodes": [], "edges": []})
            normalized = _normalize_scene_graph_for_logging(parsed)
            if normalized.get("nodes") and normalized.get("edges"):
                return normalized
        except Exception:
            pass

        try:
            detection_result = self.detect_objects_for_delta(image_paths)
            graph = self.build_delta_scene_graph(detection_result)
            return _normalize_scene_graph_for_logging(graph)
        except Exception:
            return {"nodes": [], "edges": []}



    @measure_time("detect_objects_for_delta")
    def detect_objects_for_delta(self, image_paths):
        default_res = {"objects": [], "relations": [], "multiview_fusion": {"views": [], "num_views": 0, "unique_objects": 0}}
        if not self.client or not image_paths:
            return default_res

        view_detections = [self._detect_objects_single_view(path) for path in image_paths]
        fused = self._fuse_multiview_detections(view_detections)
        if not isinstance(fused, dict):
            return default_res

        fused.setdefault("objects", [])
        fused.setdefault("relations", [])
        fused.setdefault("multiview_fusion", {"views": [], "num_views": len(image_paths), "unique_objects": 0})
        return fused


    def build_delta_scene_graph(self, detection_result, scene_name="detected_workspace"):
        objects = detection_result.get("objects", []) if isinstance(detection_result, dict) else []
        relations = detection_result.get("relations", []) if isinstance(detection_result, dict) else []

        item_map = {}
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", "")).strip().lower().replace(" ", "_")
            if not name:
                continue
            try:
                count = max(1, int(obj.get("count", 1)))
            except Exception:
                count = 1
            for i in range(count):
                item_name = name if count == 1 else f"{name}_{i+1}"
                item_map[item_name] = {"accessible": True, "affordance": ["pick", "drop"], "state": "free"}

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subject = str(rel.get("subject", "")).strip().lower().replace(" ", "_")
            relation = str(rel.get("relation", "")).strip().lower().replace(" ", "_")
            if relation == "in_basket" and subject in item_map:
                item_map[subject]["state"] = "in_basket"

        return {
            "name": scene_name,
            "rooms": {
                "workspace": {
                    "items": item_map,
                    "neighbor": []
                }
            },
            "agent": {"position": "workspace", "state": "hand-free"}
        }

    def save_delta_scene_graph_json(self, image_paths, output_path):
        detection_result = self.detect_objects_for_delta(image_paths)
        graph = self.build_delta_scene_graph(detection_result)
        diagnostics = self.assess_multiview_identity(image_paths)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "detection": detection_result,
            "scene_graph": graph,
            "scene_graph_nodes_edges": _normalize_scene_graph_for_logging(graph),
            "multiview_identity": diagnostics,
            "created_at": time.time(),
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(output_path), graph, diagnostics
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
        # Slightly thicker basket walls to reduce high-speed tunneling.
        w, d, h, th = 0.5, 0.6, 0.15, 0.03
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
            try:
                c_api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                c_api.CreateContactOffsetAttr().Set(0.004)
                c_api.CreateRestOffsetAttr().Set(0.0)
            except Exception:
                pass
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
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(0.2)
        # Explicit COM/inertia for more stable grasp and release behavior.
        try:
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.075))
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(2.6e-4, 2.6e-4, 1.8e-4))
            mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        except Exception:
            pass

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
        return path


class RobotController:
    def __init__(self, arm, disturbance=None):
        self.arm = arm
        self.grasped_object = None
        self.disposed_list = []
        self.last_error_message = ""
        self.dynamic_params = {"approach_z_adjust": 0.02, "reach_adjust": -0.01, "yaw_adjust": 0.05, "grip_open": 0.04, "grip_close": 0.005}
        self.poses = POSE_LIBRARY.copy()
        self.current_pose = self.poses["HOME"].copy()
        self.ee_orientation = DEFAULT_EE_QUAT.copy()
        self.ik_solver = None
        self.articulation_controller = None
        self.attach_distance_threshold = 0.255
        self.attach_follow_offset = np.array([0.0, 0.0, -0.07], dtype=np.float32)
        self.attach_locked_until_open = False
        self.grasp_follow_alpha = GRASP_FOLLOW_ALPHA
        self.disable_collision_on_grasp = ALLOW_PENETRATION_GRASP
        self.calib_offset = np.zeros(3, dtype=np.float32)
        self.disturbance = disturbance
        self.gripper_hold_enabled = False
        self.desired_gripper_width = float(self.dynamic_params.get("grip_open", 0.04))
        self.gripper_joint_indices = [7, 8]

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

        self._resolve_gripper_joint_indices()
        self._refresh_current_pose()

    def _clamp_attach_distance(self, value):
        try:
            v = float(value)
        except Exception:
            v = SAFE_ATTACH_DISTANCE_MIN
        return max(SAFE_ATTACH_DISTANCE_MIN, min(SAFE_ATTACH_DISTANCE_MAX, v))

    def _clamp_follow_offset(self, value):
        try:
            v = float(value)
        except Exception:
            v = -0.07
        v = max(SAFE_FOLLOW_OFFSET_MIN, min(SAFE_FOLLOW_OFFSET_MAX, v))
        return np.array([0.0, 0.0, v], dtype=np.float32)

    def apply_calibration(self, calib):
        if not calib:
            return
        self.dynamic_params["yaw_adjust"] = float(calib.get("yaw_adjust", self.dynamic_params.get("yaw_adjust", 0.05)))
        self.dynamic_params["grip_open"] = float(calib.get("grip_open", self.dynamic_params.get("grip_open", 0.04)))
        self.dynamic_params["grip_close"] = float(calib.get("grip_close", self.dynamic_params.get("grip_close", 0.005)))
        self.attach_distance_threshold = self._clamp_attach_distance(
            calib.get("attach_distance_threshold", self.attach_distance_threshold)
        )
        self.attach_follow_offset = self._clamp_follow_offset(
            calib.get("follow_offset", -0.07)
        )
        offset = calib.get("position_offset", [0.0, 0.0, 0.0])
        self.calib_offset = np.array(offset, dtype=np.float32)

    def _refresh_current_pose(self):
        try:
            joint_positions = self.arm.get_joint_positions()
            if joint_positions is not None and len(joint_positions) == len(self.current_pose):
                self.current_pose = np.array(joint_positions, dtype=np.float32)
                if self.gripper_hold_enabled:
                    self._write_gripper_width_into_pose(self.current_pose, float(self.desired_gripper_width))
        except Exception:
            pass

    def _resolve_gripper_joint_indices(self):
        try:
            names = list(getattr(self.arm, "dof_names", []) or [])
            candidates = []
            for idx, name in enumerate(names):
                lname = str(name).lower()
                if "finger_joint" in lname or lname.endswith("finger_joint1") or lname.endswith("finger_joint2"):
                    candidates.append(int(idx))
            if len(candidates) >= 2:
                self.gripper_joint_indices = candidates[:2]
                return
        except Exception:
            pass
        try:
            n = len(self.current_pose)
            if n >= 9:
                self.gripper_joint_indices = [n - 2, n - 1]
        except Exception:
            self.gripper_joint_indices = [7, 8]

    def _write_gripper_width_into_pose(self, pose: np.ndarray, width: float) -> np.ndarray:
        if pose is None:
            return pose
        val = float(width)
        for idx in getattr(self, "gripper_joint_indices", [7, 8]):
            ii = int(idx)
            if 0 <= ii < len(pose):
                pose[ii] = val
        return pose

    def _apply_joint_targets(self, joint_positions):
        action = ArticulationAction(joint_positions=np.array(joint_positions, dtype=np.float32))
        if self.articulation_controller is not None:
            self.articulation_controller.apply_action(action)
        else:
            self.arm.apply_action(action)

    def _apply_gripper_width_direct(self, width: float) -> None:
        pose = self.current_pose.copy()
        self._write_gripper_width_into_pose(pose, float(width))
        self._apply_joint_targets(pose)
        self._write_gripper_width_into_pose(self.current_pose, float(width))

    def _set_gripper_width_target(self, width: float, enable_hold: bool) -> None:
        val = float(width)
        self.desired_gripper_width = val
        self.gripper_hold_enabled = bool(enable_hold)
        self._write_gripper_width_into_pose(self.current_pose, val)

    def _enforce_gripper_hold(self) -> None:
        if not self.gripper_hold_enabled:
            return
        self._apply_gripper_width_direct(float(self.desired_gripper_width))

    def _apply_arm_action_preserving_gripper(self, action):
        try:
            joint_positions = getattr(action, "joint_positions", None)
            if joint_positions is None:
                raise ValueError("missing joint positions")

            full_target = self.current_pose.copy()
            joint_positions = np.array(joint_positions, dtype=np.float32)
            joint_indices = getattr(action, "joint_indices", None)

            if joint_indices is not None and len(joint_indices) == len(joint_positions):
                for idx, value in zip(joint_indices, joint_positions):
                    ii = int(idx)
                    if 0 <= ii < len(full_target):
                        full_target[ii] = float(value)
            else:
                n = min(len(joint_positions), len(full_target))
                full_target[:n] = joint_positions[:n]

            if len(full_target) >= 2:
                if self.gripper_hold_enabled:
                    self._write_gripper_width_into_pose(full_target, float(self.desired_gripper_width))
                else:
                    for idx in getattr(self, "gripper_joint_indices", [7, 8]):
                        ii = int(idx)
                        if 0 <= ii < len(full_target) and 0 <= ii < len(self.current_pose):
                            full_target[ii] = self.current_pose[ii]
            self._apply_joint_targets(full_target)
            return
        except Exception:
            pass

        if self.articulation_controller is not None:
            self.articulation_controller.apply_action(action)
        else:
            self.arm.apply_action(action)

    def _after_world_step(self):
        self._enforce_gripper_hold()
        self._update_grasped_object()
        if self.disturbance is not None:
            self.disturbance.maybe_trigger(self)

    def reset_dynamic_params(self):
        self.dynamic_params = {"approach_z_adjust": 0.02, "reach_adjust": -0.01, "yaw_adjust": 0.05, "grip_open": 0.04, "grip_close": 0.005}
        self._set_gripper_width_target(float(self.dynamic_params.get("grip_open", 0.04)), enable_hold=False)
        self.last_error_message = ""


    def update_dynamic_params(self, updates):
        if not updates:
            return
        self.dynamic_params.update(updates)
        # Clamp to safe ranges (meters / radians)
        def clamp(val, lo, hi):
            try:
                v = float(val)
            except Exception:
                return lo
            return max(min(v, hi), lo)

        self.dynamic_params["approach_z_adjust"] = clamp(self.dynamic_params.get("approach_z_adjust", 0.02), -0.05, 0.08)
        self.dynamic_params["reach_adjust"] = clamp(self.dynamic_params.get("reach_adjust", -0.01), -0.20, 0.20)
        self.dynamic_params["yaw_adjust"] = clamp(self.dynamic_params.get("yaw_adjust", 0.0), -0.6, 0.6)

        # Franka gripper joint positions are typically ~0..0.05m
        go = clamp(self.dynamic_params.get("grip_open", 0.04), 0.01, 0.08)
        gc = clamp(self.dynamic_params.get("grip_close", 0.005), 0.0, 0.03)
        if gc > go:
            gc = max(0.0, go - 0.002)
        self.dynamic_params["grip_open"] = go
        self.dynamic_params["grip_close"] = gc

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = [float(v) for v in q1]
        w2, x2, y2, z2 = [float(v) for v in q2]
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)

    @staticmethod
    def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.array(axis, dtype=np.float32)
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        axis = axis / n
        half = 0.5 * float(angle_rad)
        s = float(np.sin(half))
        return np.array([float(np.cos(half)), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)

    def _get_safe_world_pose(self, prim_path):
        prim = XFormPrim(prim_path)
        pos, rot = prim.get_world_poses()
        safe_pos = pos[0] if len(np.shape(pos)) == 2 else pos
        safe_rot = rot[0] if len(np.shape(rot)) == 2 else rot
        return np.array(safe_pos, dtype=np.float32), np.array(safe_rot, dtype=np.float32)

    def _compute_angled_grasp_orientation(self, grasp_center: np.ndarray, obj_height: float) -> np.ndarray:
        base = np.array(self.ee_orientation, dtype=np.float32)
        if obj_height >= 0.10:
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

    def compute_pick_targets(self, target_path):
        self.ensure_object_dynamic(target_path)
        safe_pos, _ = self._get_safe_world_pose(target_path)
        VisualHelper.create_or_move_proxy(pos=safe_pos)
        z_adjust = float(self.dynamic_params.get("approach_z_adjust", 0.0))
        reach_adjust = float(self.dynamic_params.get("reach_adjust", 0.0))
        lateral_offset = np.array([reach_adjust, 0.0, 0.0], dtype=np.float32)
        # Use top-down center grasp to avoid side push and tipping.
        grasp_center = safe_pos + self.calib_offset + np.array([0.0, 0.0, 0.15 + z_adjust], dtype=np.float32) + lateral_offset
        try:
            stage = get_current_stage()
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox_prim = None
            for cand in [f"{target_path}/Body", f"{target_path}/Handle", target_path]:
                prim = stage.GetPrimAtPath(cand)
                if prim.IsValid():
                    bbox_prim = prim
                    break
            if bbox_prim is not None:
                box = bbox_cache.ComputeWorldBound(bbox_prim).GetBox()
                obj_min = np.array([float(box.GetMin()[0]), float(box.GetMin()[1]), float(box.GetMin()[2])], dtype=np.float32)
                obj_max = np.array([float(box.GetMax()[0]), float(box.GetMax()[1]), float(box.GetMax()[2])], dtype=np.float32)
            else:
                obj_min = np.array([float(safe_pos[0]) - 0.03, float(safe_pos[1]) - 0.03, float(safe_pos[2])], dtype=np.float32)
                obj_max = np.array([float(safe_pos[0]) + 0.03, float(safe_pos[1]) + 0.03, float(safe_pos[2]) + 0.15], dtype=np.float32)
        except Exception:
            obj_min = np.array([float(safe_pos[0]) - 0.03, float(safe_pos[1]) - 0.03, float(safe_pos[2])], dtype=np.float32)
            obj_max = np.array([float(safe_pos[0]) + 0.03, float(safe_pos[1]) + 0.03, float(safe_pos[2]) + 0.15], dtype=np.float32)
        obj_min_z = float(obj_min[2])
        obj_max_z = float(obj_max[2])
        obj_height = max(0.01, obj_max_z - obj_min_z)
        obj_center = 0.5 * (obj_min + obj_max)
        try:
            body_pos, _ = self._get_safe_world_pose(f"{target_path}/Body")
            body_pos = np.array(body_pos, dtype=np.float32)
            if np.all(np.isfinite(body_pos)):
                obj_center[0] = float(body_pos[0])
                obj_center[1] = float(body_pos[1])
        except Exception:
            pass
        # Force XY grasp target onto mug body center projection.
        grasp_center[0] = float(obj_center[0])
        grasp_center[1] = float(obj_center[1])
        if obj_height < 0.10:
            min_z = obj_min_z + 0.008
            grasp_center[2] = min(grasp_center[2], obj_center[2] + 0.012 + z_adjust)
        else:
            min_z = obj_min_z + MIN_GRASP_CLEARANCE
            grasp_center[2] = min(grasp_center[2], obj_center[2] + 0.02 + z_adjust)
        if grasp_center[2] < min_z:
            dz = min_z - grasp_center[2]
            grasp_center[2] += dz
        orientation = self._compute_angled_grasp_orientation(grasp_center, obj_height)
        pregrasp_offset = np.array([0.0, 0.0, 0.12], dtype=np.float32)
        if obj_height < 0.10:
            pregrasp_offset += np.array([-0.03 * float(np.sign(grasp_center[0]) or 1.0), 0.0, 0.02], dtype=np.float32)
        pre_grasp = grasp_center + pregrasp_offset
        retreat = grasp_center + np.array([0.0, 0.0, 0.2], dtype=np.float32)
        return {"pre_grasp": pre_grasp, "grasp": grasp_center, "retreat": retreat, "orientation": orientation, "affordance_offset": [0.0, 0.0, 0.0]}

    def get_place_slot_for_mug(self, target_path):
        mug_name = str(target_path).split("/")[-1]
        return MUG_PLACE_ASSIGNMENTS.get(mug_name, "BASKET_HIGH")

    def compute_place_targets(self, place_slot_name):
        slot_center = BASKET_PLACE_SLOTS.get(place_slot_name, BASKET_PLACE_SLOTS["BASKET_HIGH"]).copy() + getattr(self, "calib_offset", np.zeros(3, dtype=np.float32))
        z_adjust = float(self.dynamic_params.get("approach_z_adjust", 0.0))
        pre_place = slot_center + np.array([0.0, 0.0, 0.10 + z_adjust], dtype=np.float32)
        place = slot_center + np.array([0.0, 0.0, z_adjust], dtype=np.float32)
        # Do not allow placement target to drop below basket interior floor level.
        min_place_z = float(slot_center[2]) - 0.01
        place[2] = max(float(place[2]), min_place_z)
        pre_place[2] = max(float(pre_place[2]), float(place[2]) + 0.10)
        retreat = slot_center + np.array([0.0, 0.0, 0.16 + z_adjust], dtype=np.float32)
        retreat[2] = max(float(retreat[2]), float(place[2]) + 0.16)
        return {"pre_place": pre_place, "place": place, "retreat": retreat, "orientation": self.ee_orientation.copy()}

    def verify_placement(self, target_path):
        try:
            obj_pos, _ = self._get_safe_world_pose(target_path)
            in_xy = (-0.30 < obj_pos[0] < 0.30) and (0.30 < obj_pos[1] < 0.94)
            in_z = 0.02 < obj_pos[2] < 0.32
            return bool(in_xy and in_z)
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
        for part in ["Body", "Handle"]:
            prim = stage.GetPrimAtPath(f"{target_path}/{part}")
            if prim.IsValid():
                col_api = UsdPhysics.CollisionAPI(prim)
                if col_api:
                    col_api.GetCollisionEnabledAttr().Set(enabled)

    def _get_target_bbox_world(self, target_path):
        try:
            stage = get_current_stage()
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            mins = []
            maxs = []
            for cand in [f"{target_path}/Body", f"{target_path}/Handle", target_path]:
                prim = stage.GetPrimAtPath(cand)
                if not prim.IsValid():
                    continue
                box = bbox_cache.ComputeWorldBound(prim).GetBox()
                mn = np.array([box.GetMin()[0], box.GetMin()[1], box.GetMin()[2]], dtype=np.float32)
                mx = np.array([box.GetMax()[0], box.GetMax()[1], box.GetMax()[2]], dtype=np.float32)
                if np.all(np.isfinite(mn)) and np.all(np.isfinite(mx)):
                    mins.append(mn)
                    maxs.append(mx)
            if mins and maxs:
                return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)
        except Exception:
            pass
        return None, None

    def _compute_grasp_box_overlap(self, target_path):
        try:
            hand_pos, _ = self._get_safe_world_pose("/World/Franka/panda_hand")
        except Exception:
            return 0.0, "hand pose unavailable"
        bbox_min, bbox_max = self._get_target_bbox_world(target_path)
        if bbox_min is None or bbox_max is None:
            return 0.0, "target bbox unavailable"

        hand_pos = np.array(hand_pos, dtype=np.float32)
        box_center = hand_pos + np.array([0.0, 0.0, -0.02], dtype=np.float32)
        box_half = np.array([0.055, 0.045, 0.065], dtype=np.float32)
        grip_min = box_center - box_half
        grip_max = box_center + box_half

        inter_min = np.maximum(grip_min, bbox_min)
        inter_max = np.minimum(grip_max, bbox_max)
        inter_size = np.maximum(0.0, inter_max - inter_min)
        inter_vol = float(np.prod(inter_size))
        obj_size = np.maximum(1e-6, bbox_max - bbox_min)
        obj_vol = float(np.prod(obj_size))
        overlap = inter_vol / obj_vol if obj_vol > 1e-9 else 0.0

        obj_center = 0.5 * (bbox_min + bbox_max)
        center_inside = bool(np.all(obj_center >= grip_min) and np.all(obj_center <= grip_max))
        if center_inside:
            overlap = max(overlap, 0.25)
        return float(overlap), f"box_overlap={overlap:.3f} center_inside={center_inside}"

    def _attach_object(self, target_path):
        self.grasped_object = target_path
        self.attach_locked_until_open = True
        prim = get_current_stage().GetPrimAtPath(target_path)
        if prim.IsValid():
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
        # Keep object collisions enabled. Toggle finger collisions by policy.
        self._set_collision_enabled(target_path, True)
        self._set_gripper_collision_enabled(not self.disable_collision_on_grasp)
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
                follow = self._clamp_follow_offset(self.attach_follow_offset[2])
                target = hand_pos + follow
                prim = XFormPrim(self.grasped_object)
                current_pos, current_rot = prim.get_world_poses()
                safe_pos = current_pos[0] if len(np.shape(current_pos)) == 2 else current_pos
                alpha = float(self.grasp_follow_alpha)
                new_pos = safe_pos + (target - safe_pos) * alpha
                prim.set_world_poses(positions=np.array([new_pos], dtype=np.float32), orientations=np.array([current_rot[0] if len(np.shape(current_rot)) == 2 else current_rot]))
            except Exception:
                pass

    def move_to_joint_pose(self, pose, world, steps=60):
        target = self.poses.get(pose, self.poses["HOME"]).copy() if isinstance(pose, str) else pose.copy()
        if self.gripper_hold_enabled:
            target[7:] = float(self.desired_gripper_width)
        else:
            target[7:] = self.current_pose[7:]
        start = self.current_pose.copy()
        for t in range(steps):
            if not world.is_playing():
                break
            ratio = (1.0 - np.cos(t / steps * np.pi)) / 2.0
            current_target = start + (target - start) * ratio
            self._apply_joint_targets(current_target)
            world.step(render=True)
            self._after_world_step()
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

            self._apply_arm_action_preserving_gripper(action)
            world.step(render=True)
            self._after_world_step()
            self._refresh_current_pose()

        self.last_error_message = ""
        return True

    def close_gripper(self, world, target_path, steps=30):
        can_grasp = False
        if target_path:
            try:
                hand_pos, _ = self._get_safe_world_pose("/World/Franka/panda_hand")
                obj_pos, _ = self._get_safe_world_pose(target_path)
                distance = float(np.linalg.norm(hand_pos - obj_pos))
                threshold = self._clamp_attach_distance(self.attach_distance_threshold)
                grace = SAFE_ATTACH_DISTANCE_GRACE
                overlap, overlap_reason = self._compute_grasp_box_overlap(target_path)

                # Conservative attach gate to reduce false-positive grasp locks.
                # Even if UI threshold is large, keep weak-attach effective radius bounded.
                effective_threshold = min(float(threshold), 0.22)
                effective_grace = min(float(grace), 0.03)

                strong_overlap = overlap >= 0.10
                weak_overlap = overlap >= 0.06 and distance < (effective_threshold + effective_grace)
                if strong_overlap or weak_overlap:
                    can_grasp = True
                    self.last_error_message = "" if strong_overlap else (
                        f"grasp accepted by weak box overlap ({overlap_reason}, eff_th={effective_threshold:.3f}, eff_grace={effective_grace:.3f})"
                    )
                else:
                    self.last_error_message = (
                        f"grasp box miss ({overlap_reason}, dist={distance:.3f}, th={threshold:.3f}, grace={grace:.3f}, "
                        f"eff_th={effective_threshold:.3f}, eff_grace={effective_grace:.3f})"
                    )
            except Exception as exc:
                self.last_error_message = f"grasp box check failed: {exc}"

        if can_grasp:
            self._attach_object(target_path)

        is_mug = bool(target_path) and "mug_" in str(target_path)
        desired_close = float(self.dynamic_params.get("grip_close", 0.005))
        if is_mug:
            desired_close = min(desired_close, float(GRIP_CLOSE_MUG), 0.0015)
        desired_close = max(0.0, min(float(self.dynamic_params.get("grip_open", 0.04)), desired_close))
        self._set_gripper_width_target(desired_close, enable_hold=True)
        for _ in range(steps):
            if not world.is_playing():
                break
            self._apply_gripper_width_direct(desired_close)
            world.step(render=True)
            self._after_world_step()
        self._refresh_current_pose()

    def open_gripper(self, world, steps=30):
        released_object = self.grasped_object
        self.attach_locked_until_open = False
        target_open = float(self.dynamic_params.get("grip_open", 0.04))
        self._set_gripper_width_target(target_open, enable_hold=False)
        for _ in range(steps):
            if not world.is_playing():
                break
            self._apply_gripper_width_direct(target_open)
            world.step(render=True)
            self._after_world_step()
        self._refresh_current_pose()

        if released_object:
            prim = XFormPrim(released_object)
            pos, rot = prim.get_world_poses()
            safe_pos = pos[0] if len(np.shape(pos)) == 2 else pos
            safe_rot = rot[0] if len(np.shape(rot)) == 2 else rot
            in_basket_xy = (-0.32 < float(safe_pos[0]) < 0.32) and (0.25 < float(safe_pos[1]) < 1.00)
            if in_basket_xy:
                safe_pos[2] = max(float(safe_pos[2]), 0.16)
            else:
                safe_pos[2] = max(float(safe_pos[2]), 0.05)
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
        self.window = ui.Window("Grasp Tuning", width=380, height=620)
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
                self._add_bool("disable_collision_on_grasp", bool(self.controller.disable_collision_on_grasp))
                ui.Separator()
                ui.Label("Calibration Offset (m)")
                self._add_float("calib_x", float(self.controller.calib_offset[0]))
                self._add_float("calib_y", float(self.controller.calib_offset[1]))
                self._add_float("calib_z", float(self.controller.calib_offset[2]))
                ui.Separator()
                with ui.HStack(spacing=8):
                    ui.Button("Apply", clicked_fn=self.apply)
                    ui.Button("Reset Dynamic", clicked_fn=self.reset_dynamic)
                    ui.Button("Save Calib", clicked_fn=self.save_calibration)

    def _add_float(self, name, value):
        model = ui.SimpleFloatModel(float(value))
        self.models[name] = model
        with ui.HStack(height=0):
            ui.Label(name, width=200)
            ui.FloatField(model=model, width=140)

    def _add_bool(self, name, value):
        model = ui.SimpleBoolModel(bool(value))
        self.models[name] = model
        with ui.HStack(height=0):
            ui.Label(name, width=200)
            ui.CheckBox(model=model, width=140)

    def _get_float(self, name, default=0.0):
        model = self.models.get(name)
        if model is None:
            return float(default)
        try:
            return float(model.as_float)
        except Exception:
            return float(default)

    def _get_bool(self, name, default=False):
        model = self.models.get(name)
        if model is None:
            return bool(default)
        try:
            return bool(model.as_bool)
        except Exception:
            return bool(default)

    def _sync_from_controller(self):
        for key in ["approach_z_adjust", "reach_adjust", "yaw_adjust", "grip_open", "grip_close"]:
            if key in self.models:
                self.models[key].set_value(float(self.controller.dynamic_params.get(key, 0.0)))
        if "attach_distance_threshold" in self.models:
            self.models["attach_distance_threshold"].set_value(float(self.controller.attach_distance_threshold))
        if "follow_offset" in self.models:
            self.models["follow_offset"].set_value(float(self.controller.attach_follow_offset[2]))
        if "disable_collision_on_grasp" in self.models:
            self.models["disable_collision_on_grasp"].set_value(bool(self.controller.disable_collision_on_grasp))
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
        self.controller.attach_distance_threshold = self.controller._clamp_attach_distance(
            self._get_float("attach_distance_threshold", 0.24)
        )
        follow = float(self._get_float("follow_offset", -0.07))
        self.controller.attach_follow_offset = self.controller._clamp_follow_offset(follow)
        self.controller.disable_collision_on_grasp = bool(self._get_bool("disable_collision_on_grasp", False))
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

    def save_calibration(self):
        data = {
            "yaw_adjust": float(self.controller.dynamic_params.get("yaw_adjust", 0.05)),
            "grip_open": float(self.controller.dynamic_params.get("grip_open", 0.04)),
            "grip_close": float(self.controller.dynamic_params.get("grip_close", 0.005)),
            "attach_distance_threshold": float(self.controller.attach_distance_threshold),
            "follow_offset": float(self.controller.attach_follow_offset[2]),
            "position_offset": [
                float(self.controller.calib_offset[0]),
                float(self.controller.calib_offset[1]),
                float(self.controller.calib_offset[2]),
            ],
        }
        save_calibration(data)
        print(f"[UI] Saved calibration to {CALIB_JSON}", flush=True)


def capture_images_for_vlm(world=None):
    if world is not None:
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
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in capture_images:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def save_plan(plan_list):
    PLAN_JSON.write_text(json.dumps({"actions": plan_list}, indent=2))


def generate_recovery_plan(controller, ignore_list):
    new_plan = []
    for m in MUGS:
        mug_id = m["id"]
        path = f"/World/{mug_id}"
        if path in ignore_list:
            continue
        if not controller.verify_placement(path):
            new_plan.extend([f"(pick {mug_id})", f"(grab {mug_id})", "(place)"])
    save_plan(new_plan)
    eval_logger.record_plan(new_plan)
    return new_plan


@measure_time("run_replan")
def run_replan(failed_action, controller, analyzer=None, user_instruction=None, ignore_list=None):
    if ignore_list is None:
        ignore_list = []
    print(f"\n[Planner] Requesting LLM re-plan for: {failed_action}", flush=True)
    if REAL_DELTA_PATH.exists() and os.environ.get("OPENAI_API_KEY"):
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        if REAL_DELTA_PATH.name == "delta_original.py":
            cmd = [
                DELTA_PYTHON,
                str(REAL_DELTA_PATH),
                "--experiment", "all", "--episode", "1",
                "--domain", "laundry", "--domain-example", "laundry",
                "--scene", "allensville", "--scene-example", "office",
                "--print-plan",
            ]
        else:
            cmd = [
                DELTA_PYTHON,
                str(REAL_DELTA_PATH),
                "--domain", "pc",
                "--scene", "office",
                "--domain-example", "laundry",
                "--ref-pddl", "office_pc_domain.pddl",
            ]

        if user_instruction:
            cmd.extend(["--instruction", str(user_instruction)])

        if analyzer is not None:
            try:
                images_for_delta = capture_images_for_vlm(world=None)
                if images_for_delta:
                    SCENE_GRAPH_JSON_DIR.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time() * 1000)
                    graph_path = SCENE_GRAPH_JSON_DIR / f"detected_scene_graph_{ts}.json"
                    scene_graph_json_path, graph, diagnostics = analyzer.save_delta_scene_graph_json(images_for_delta, graph_path)
                    eval_logger.record_scene_graph(f"replan_{ts}", graph, affordance="delta_input", diagnostics=diagnostics)
                    if REAL_DELTA_PATH.name == "delta_original.py":
                        cmd.extend(["--scene-graph-json", scene_graph_json_path])
            except Exception as exc:
                print(f"[Planner] scene graph JSON export skipped: {exc}", flush=True)

        try:
            subprocess.run(cmd, cwd=REAL_DELTA_PATH.parent, env=env, check=True, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass
    return generate_recovery_plan(controller, ignore_list)

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


def handle_action_failure(action, current_target_bottle, controller, analyzer, ignore_list, target_failure_counts, retry_count, error_reason, calib_delta=None):
    eval_logger.add_failure()
    target_failure_counts[current_target_bottle] = target_failure_counts.get(current_target_bottle, 0) + 1
    if target_failure_counts[current_target_bottle] >= 3:
        print(f"❌ {current_target_bottle} の操作に3回失敗しました。スキップして次の動作へ進みます。", flush=True)
        ignore_list.append(current_target_bottle)
        controller.reset_dynamic_params()
    else:
        print(f"⚠️ 失敗: {current_target_bottle} (試行 {target_failure_counts[current_target_bottle]}/3)。LLMで原因を分析しパラメータを調整します。", flush=True)
        new_params = analyzer.analyze_and_adjust_params(error_reason or controller.last_error_message or "action failed", controller.dynamic_params, target_failure_counts[current_target_bottle], calib_delta=calib_delta)
        controller.update_dynamic_params(new_params)
        print(f"🔄 調整後のパラメータ: {controller.dynamic_params}", flush=True)
    retry_count += 1
    plan = run_replan(action, controller, analyzer, "失敗による再計画", ignore_list)
    return plan, retry_count


# =========================================================
# Main routine
# =========================================================
def run_simulation():
    delta_impl_raw = os.environ.get("DELTA_IMPL", DEFAULT_DELTA_IMPL).strip().lower()
    if delta_impl_raw in ("delta_original", "original", "delta_orginal", "orginal"):
        delta_impl = "delta_original"
    elif delta_impl_raw in ("delta", "current"):
        delta_impl = "delta"
    else:
        print("[Config] Unknown DELTA_IMPL={}. Falling back to {}.".format(delta_impl_raw, DEFAULT_DELTA_IMPL), flush=True)
        delta_impl = DEFAULT_DELTA_IMPL

    delta_script_override = os.environ.get("DELTA_SCRIPT", "").strip()
    global REAL_DELTA_PATH
    if delta_script_override:
        REAL_DELTA_PATH = Path(delta_script_override)
        output_tag = cli_args.output_tag or "vlm_delta_{}".format(REAL_DELTA_PATH.stem)
    else:
        REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki") / DELTA_IMPL_TO_SCRIPT[delta_impl]
        output_tag = cli_args.output_tag or "vlm_{}".format(delta_impl)

    configure_output_paths(output_tag)
    print(f"[Config] Delta script: {REAL_DELTA_PATH}", flush=True)
    print(f"[Config] Output dir: {OUTPUT_DIR}", flush=True)
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
    AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.0])    # Multi-view cameras for VLM/scene-graph
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
    disturbance = TimedDisturbanceScenario(enabled=ENABLE_DISTURBANCE_TEST, trigger_seconds=DISTURBANCE_TRIGGER_SEC)
    for _ in range(30):
        world.step(render=True)
        disturbance.maybe_trigger()

    calib = run_calibration(world)
    controller = RobotController(arm=franka, disturbance=disturbance)
    controller.apply_calibration(calib)
    param_ui = None
    if ENABLE_PARAM_UI:
        try:
            param_ui = ParameterTuningUI(controller)
        except Exception as exc:
            print(f"[UI] Failed to create parameter UI: {exc}", flush=True)
    analyzer = VLMAnalyzer(model_name=TARGET_MODEL)
    images_for_delta = capture_images_for_vlm(world)
    calib_delta = analyzer.estimate_calibration_delta(images_for_delta, calib=calib)
    print(f"[CalibDelta] VLM estimated delta: {calib_delta}", flush=True)

    max_overall_retries = 20
    retry_count = 0
    target_failure_counts = {}
    ignore_list = []
    current_target_bottle = None

    plan = generate_recovery_plan(controller, ignore_list)
    print(f"\n📦 初期プランを作成しました。タスク数: {len(plan)}", flush=True)

    try:
        while plan and retry_count < max_overall_retries and world.is_playing():
            action = plan.pop(0)
            print(f"\n>> Executing: {action}", flush=True)
            action_failed = False
            error_reason = ""

            try:
                if action.startswith("(pick"):
                    target_id = action.replace(")", "").replace("(", "").split()[1]
                    if target_id == "mug_5" and not get_current_stage().GetPrimAtPath("/World/mug_5").IsValid():
                        target_id = "mug_4"
                    current_target_bottle = f"/World/{target_id}"
                    pick_targets = controller.compute_pick_targets(current_target_bottle)
                    controller.open_gripper(world)
                    moved = controller.move_end_effector_to(pick_targets["pre_grasp"], pick_targets["orientation"], world, steps=90, stage_name="pre-grasp")
                    if moved:
                        wait_steps(world, 0.2, controller)
                        moved = controller.move_end_effector_to(pick_targets["grasp"], pick_targets["orientation"], world, steps=75, stage_name="grasp-approach")
                    if not moved:
                        action_failed = True
                        error_reason = controller.last_error_message or "pick failed"
                    print(f"✅ {action} 完了" if not action_failed else f"⚠️ {action} 失敗: {error_reason}", flush=True)

                elif action.startswith("(grab"):
                    target_id = action.replace(")", "").replace("(", "").split()[1]
                    if target_id == "mug_5" and not get_current_stage().GetPrimAtPath("/World/mug_5").IsValid():
                        target_id = "mug_4"
                    current_target_bottle = f"/World/{target_id}"
                    pick_targets = controller.compute_pick_targets(current_target_bottle)
                    moved = controller.move_end_effector_to(pick_targets["grasp"], pick_targets["orientation"], world, steps=30, stage_name="final-grasp")
                    controller.close_gripper(world, current_target_bottle)
                    wait_steps(world, 0.5, controller)
                    if moved and not controller.last_error_message:
                        moved = controller.move_end_effector_to(pick_targets["retreat"], pick_targets["orientation"], world, steps=90, stage_name="post-grasp-retreat")
                    if (not moved) or controller.last_error_message:
                        action_failed = True
                        error_reason = controller.last_error_message or "grab failed"
                    print(f"✅ {action} 完了" if not action_failed else f"⚠️ {action} 失敗: {error_reason}", flush=True)

                elif action.startswith("(place"):
                    images = capture_images_for_vlm(world)
                    suggested_pose = analyzer.find_empty_space(images)
                    place_pose = controller.get_place_slot_for_mug(current_target_bottle)
                    if place_pose == "BASKET_HIGH" and suggested_pose in BASKET_PLACE_SLOTS:
                        place_pose = suggested_pose
                    place_targets = controller.compute_place_targets(place_pose)
                    moved = controller.move_end_effector_to(place_targets["pre_place"], place_targets["orientation"], world, steps=90, stage_name="pre-place")
                    if moved:
                        moved = controller.move_end_effector_to(place_targets["place"], place_targets["orientation"], world, steps=60, stage_name="place")
                    wait_steps(world, 0.5, controller)
                    controller.open_gripper(world)
                    if moved:
                        moved = controller.move_end_effector_to(place_targets["retreat"], place_targets["orientation"], world, steps=75, stage_name="post-place-retreat")
                    controller.move_to_joint_pose("HOME", world)
                    wait_steps(world, 3.0, controller)
                    print(f"✅ {action} 完了", flush=True)

                    images_after = capture_images_for_vlm(world)
                    vlm_result = analyzer.process_and_analyze(images_after, action)
                    is_failed = vlm_result.get("is_failed", False)
                    error_reason = vlm_result.get("reason", "配置判定の失敗")
                    if controller.last_error_message:
                        error_reason += f" | {controller.last_error_message}"
                    scene_graph = analyzer.extract_scene_graph(images_after)
                    diagnostics = analyzer.assess_multiview_identity(images_after)
                    eval_logger.record_scene_graph(retry_count, scene_graph, affordance=None, diagnostics=diagnostics)
                    action_failed = is_failed or (not moved) or (not controller.verify_placement(current_target_bottle)) or bool(controller.last_error_message)
                    if not action_failed:
                        print("✅ 成功: オブジェクトがカゴ内に確認されました。", flush=True)
                        target_failure_counts[current_target_bottle] = 0
                        controller.reset_dynamic_params()

                else:
                    print(f"⚠️ 未対応アクションをスキップします: {action}", flush=True)
                    continue

            except Exception as exc:
                action_failed = True
                error_reason = f"runtime exception during {action}: {exc}"
                controller.last_error_message = error_reason
                traceback.print_exc()

            if action_failed and current_target_bottle is not None:
                plan, retry_count = handle_action_failure(action, current_target_bottle, controller, analyzer, ignore_list, target_failure_counts, retry_count, error_reason, calib_delta=calib_delta)
                continue

        if "recording_active" in locals() and recording_active:
            try:
                rep.orchestrator.stop()
            except Exception:
                pass
            recording_active = False

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
