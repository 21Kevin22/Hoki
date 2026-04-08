import os
import json
import base64
import requests
import shutil
import subprocess
import numpy as np
import traceback
import faulthandler
import gc
import time
import csv
import re
from pathlib import Path
from openai import OpenAI

faulthandler.enable()

# =========================================================
# 【重要】APIキーとモデルの設定
# =========================================================
OPENAI_API_KEY = "<OPENAI_API_KEY>" 
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

GEMINI_API_KEY = "AIzaSyCsmmdOaLo7hdOXyyneRLA5kgQHBm516eQ" 
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

TARGET_MODEL = "gpt-4o"

# =========================================================
# 0. 設定管理
# =========================================================
OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation/vlm_integration_4")
RGB_DIR = OUTPUT_DIR / "frames"
SCENE_GRAPH_IMG_DIR = OUTPUT_DIR / "scene_graph_images"
SCENE_GRAPH_TMP_DIR = OUTPUT_DIR / "tmp_capture"
VIDEO_PATH = OUTPUT_DIR / "mug_sorting.mp4"
PDDL_LOG_PATH = OUTPUT_DIR / "simulation_pddl_snapshots.log"

METRICS_CSV_PATH = OUTPUT_DIR / "vlm_metrics.csv"
ACCURACY_CSV_PATH = OUTPUT_DIR / "accuracy_metrics.csv"
PLAN_JSON = OUTPUT_DIR / "actions.json"

REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki/delta.py")
DELTA_PYTHON = "/usr/bin/python3"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

MUGS = [
    {"id": "mug_0", "pos": [0.4,  0.5,  0.05], "color": [1.0, 0.0, 0.0], "angle": -120, "pose": "L1_LOW"},
    {"id": "mug_1", "pos": [0.5,  0.25, 0.05], "color": [0.0, 1.0, 0.0], "angle": 180,  "pose": "L2_LOW"},
    {"id": "mug_2", "pos": [0.6,  0.0,  0.05], "color": [0.0, 0.0, 1.0], "angle": 120,  "pose": "CTR_LOW"},
    {"id": "mug_3", "pos": [0.5, -0.25, 0.05], "color": [1.0, 1.0, 0.0], "angle": 60,   "pose": "R2_LOW"},
    {"id": "mug_4", "pos": [0.4, -0.5,  0.05], "color": [0.0, 1.0, 1.0], "angle": 0,    "pose": "R1_LOW"},
]

POSE_LIBRARY = {
    "HOME":        np.array([0.0, -0.70, 0.0, -2.30, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
    "L1_LOW":      np.array([0.70, 0.45, 0.1, -1.8, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
    "L2_LOW":      np.array([0.35, 0.20, 0.1, -2.1, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
    "CTR_LOW":     np.array([0.0,  0.05, 0.1, -2.4, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
    "R2_LOW":      np.array([-0.35, -0.10, 0.1, -2.4, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
    "R1_LOW":      np.array([-0.75, -0.20, 0.0, -2.40, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
    "BASKET_HIGH": np.array([1.5, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
    "PLACE_mug_0": np.array([1.40, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
    "PLACE_mug_1": np.array([1.45, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
    "PLACE_mug_2": np.array([1.50, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
    "PLACE_mug_3": np.array([1.55, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
    "PLACE_mug_4": np.array([1.60, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
}

CAMERA_POSITIONS = {
    "top":   {"pos": (0.5, 0.0, 2.2),  "look_at": (0.5, 0.0, 0.0)},
    "left":  {"pos": (1.4, 1.2, 1.0),  "look_at": (0.5, 0.0, 0.0)},
    "right": {"pos": (1.4, -1.2, 1.0), "look_at": (0.5, 0.0, 0.0)},
    "main":  {"pos": (1.8, 0.0, 1.5),  "look_at": (0.5, 0.0, 0.2)}
}

class PlanningEvaluator:
    def __init__(self):
        self.metrics = []
    def record_event(self, action, instruction, latency, iterations, success):
        self.metrics.append([action, instruction, latency, iterations, success])

evaluator = PlanningEvaluator()

# =========================================================
# 1. Isaac Sim 起動
# =========================================================
os.environ["CARB_APP_MIN_LOG_LEVEL"] = "error"
os.environ["OMNI_LOG_LEVEL"] = "error"
from isaacsim import SimulationApp
app_config = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "enable_audio": False}
simulation_app = SimulationApp(app_config)

import carb
carb.settings.get_settings().set_string("/log/level", "error")

from pxr import Gf, Sdf, UsdLux, UsdPhysics, UsdGeom, UsdShade, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep

fps = 30

def wait_steps(world, seconds, controller=None):
    steps = int(seconds * fps)
    for _ in range(steps):
        world.step(render=True)
        if controller:
            controller._update_grasped_object()
        rep.orchestrator.step()

# =========================================================
# 2. ユーティリティ・VLMクラス
# =========================================================
class VLMAnalyzer:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def encode_image_base64(self, path):
        with open(path, "rb") as f: 
            return base64.b64encode(f.read()).decode('utf-8')

    def _safe_parse(self, raw_content):
        default_response = {"is_failed": False, "reason": "解析エラー", "suggested_order": ""}
        try:
            parsed_data = json.loads(raw_content)
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                parsed_data = parsed_data[0]
            if isinstance(parsed_data, dict):
                return parsed_data
            return default_response
        except Exception as error:
            print(f"⚠️ [VLM Parse Error] {error}", flush=True)
            return default_response

    def find_empty_space(self, image_paths):
        if not self.client: 
            return "BASKET_HIGH"
        
        base64_images = [self.encode_image_base64(p) for p in image_paths]
        content = [{"type": "text", "text": "カゴの中の画像を分析し、マグカップが置かれていない空きスペースを判定してください。出力はJSON形式で、キーを 'suggested_pose' とし、値は 'PLACE_mug_0', 'PLACE_mug_1', 'PLACE_mug_2', 'PLACE_mug_3', 'PLACE_mug_4' のいずれかから選んでください。"}]
        for b64 in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            return parsed.get("suggested_pose", "BASKET_HIGH")
        except Exception as e:
            print(f"⚠️ [VLM Space Error] {e}", flush=True)
            return "BASKET_HIGH"

    def analyze_and_adjust_params(self, error_reason, current_params):
        """失敗理由から物理パラメータの微調整値をLLMに提案させます。"""
        if not self.client: 
            return current_params
            
        prompt = f"""
        ロボットアームの把持・配置タスクに失敗しました。
        失敗の理由・状況: {error_reason}
        現在の動的パラメータ: {json.dumps(current_params)}
        
        物理的な失敗(空振り、浅すぎる、落とした等)を推測し、グリッパーの幅(gripper_width)やZ軸オフセット(grasp_offset_z)を微調整した新しいパラメータをJSONで出力してください。
        (例: 把持が浅い場合は grasp_offset_z を少し深くする等)
        キーは必ず "gripper_width" と "grasp_offset_z" を含めてください。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            adjusted_params = json.loads(response.choices[0].message.content)
            # 安全のため取得できたキーのみを返す
            return {k: v for k, v in adjusted_params.items() if k in current_params}
        except Exception as e:
            print(f"⚠️ [VLM Param Adjust Error] {e}", flush=True)
            return current_params

    def process_and_analyze(self, image_paths, instruction):
        if not self.client or len(image_paths) < 4: 
            return {"is_failed": False}
        
        from PIL import Image
        from io import BytesIO
        pil_images = [Image.open(p) for p in image_paths]
        width, height = pil_images[0].size
        combined = Image.new('RGB', (width * 2, height * 2))
        combined.paste(pil_images[0], (0, 0))
        combined.paste(pil_images[1], (width, 0))
        combined.paste(pil_images[2], (0, height))
        combined.paste(pil_images[3], (width, height))

        buffered = BytesIO()
        combined.save(buffered, format="JPEG")
        base64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = f"""
        指示「{instruction}」に基づきロボットが動作を完了しました。
        画像を確認し、対象物を落としてしまったなどの失敗がないか判定し、
        以下のJSONのみを必ず返してください。
        {{
            "is_failed": boolean, 
            "reason": "詳細な失敗の理由（LLMでのパラメータ調整に利用します）",
            "suggested_order": "リカバリが必要な場合の対象番号順（例: '1, 0, 2'）。不要なら空文字"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}],
                response_format={"type": "json_object"}
            )
            return self._safe_parse(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ [VLM API Error] {e}", flush=True)
            return {"is_failed": False, "reason": "API通信エラー", "suggested_order": ""}

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
            ("Right", (th, d, h), (w / 2, 0, h / 2))
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
    def create_torus_mesh(stage, path, major_r, minor_r, seg_major=32, seg_minor=12):
        mesh = UsdGeom.Mesh.Define(stage, path)
        verts, normals, faces, counts = [], [], [], []
        for i in range(seg_major):
            theta = 2.0 * np.pi * i / seg_major
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            center = np.array([0.0, major_r * cos_t, major_r * sin_t])
            for j in range(seg_minor):
                phi = 2.0 * np.pi * j / seg_minor
                cos_p, sin_p = np.cos(phi), np.sin(phi)
                normal = np.array([sin_p, cos_p * cos_t, cos_p * sin_t])
                point = center + minor_r * normal
                verts.append(Gf.Vec3f(*point))
                normals.append(Gf.Vec3f(*normal))
        for i in range(seg_major):
            i_next = (i + 1) % seg_major
            for j in range(seg_minor):
                j_next = (j + 1) % seg_minor
                v0, v1 = i * seg_minor + j, i_next * seg_minor + j
                v2, v3 = i_next * seg_minor + j_next, i * seg_minor + j_next
                faces.extend([v0, v1, v2, v0, v2, v3])
                counts.extend([3, 3])
        mesh.CreatePointsAttr(verts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation("vertex")
        return mesh

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
        
        mat_path = f"{path}/PhysicsMaterial"
        mat_prim = stage.DefinePrim(mat_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat.CreateStaticFrictionAttr(50.0)
        phys_mat.CreateDynamicFrictionAttr(50.0)
        
        body_path = f"{path}/Body"
        body = UsdGeom.Cylinder.Define(stage, body_path)
        body.CreateHeightAttr(0.12)
        body.CreateRadiusAttr(0.025) 
        body.CreateAxisAttr("Z")
        XFormPrim(body_path).set_local_poses(np.array([[0.0, 0.0, 0.06]]))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        AssetBuilder.apply_material(body, stage, body_path, color)
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(UsdShade.Material(mat_prim), "physics")
        
        handle_path = f"{path}/Handle"
        handle_mesh = AssetBuilder.create_torus_mesh(stage, handle_path, major_r=0.035, minor_r=0.006)
        XFormPrim(handle_path).set_local_poses(translations=np.array([[0.04, 0.0, 0.06]]), orientations=np.array([[0.7071, 0, 0, 0.7071]]))
        UsdPhysics.CollisionAPI.Apply(handle_mesh.GetPrim())
        
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(handle_mesh.GetPrim())
        mesh_col.CreateApproximationAttr("convexDecomposition")
        convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(handle_mesh.GetPrim())
        convex_api.CreateHullVertexLimitAttr(64)
        convex_api.CreateMaxConvexHullsAttr(32)
        
        AssetBuilder.apply_material(handle_mesh, stage, handle_path, color)
        UsdShade.MaterialBindingAPI.Apply(handle_mesh.GetPrim()).Bind(UsdShade.Material(mat_prim), "physics")
        return path

class RobotController:
    def __init__(self, arm: Articulation):
        self.arm = arm
        self.grasped_object = None
        self.disposed_list = []
        
        # LLMによって調整される動的パラメータをカプセル化
        self.dynamic_params = {
            "grasp_offset_z": -0.10,
            "gripper_width": 0.02
        }
        
        self.poses = POSE_LIBRARY.copy()
        for name in list(self.poses.keys()):
            if "_LOW" in name:
                hp = self.poses[name].copy()
                hp[1] -= 0.4 
                self.poses[name.replace("_LOW", "_HIGH")] = hp
        self.current_pose = self.poses["HOME"].copy()

    def reset_dynamic_params(self):
        """パラメータをデフォルトにリセットします。"""
        self.dynamic_params["grasp_offset_z"] = -0.10
        self.dynamic_params["gripper_width"] = 0.02

    def verify_placement(self, target_bottle_path):
        try:
            if not target_bottle_path:
                return False
            obj_prim = XFormPrim(target_bottle_path)
            obj_pos, _ = obj_prim.get_world_poses()
            
            is_in_x = -0.25 < obj_pos[0][0] < 0.25
            is_in_y = 0.30 < obj_pos[0][1] < 0.90
            
            return is_in_x and is_in_y
        except Exception as e:
            print(f"位置確認エラー: {e}", flush=True)
            return False

    def _set_collision_enabled(self, target_path, enabled):
        stage = get_current_stage()
        for part in ["Body", "Handle"]:
            prim = stage.GetPrimAtPath(f"{target_path}/{part}")
            if prim.IsValid():
                col_api = UsdPhysics.CollisionAPI(prim)
                if col_api: 
                    col_api.GetCollisionEnabledAttr().Set(enabled)

    def _update_grasped_object(self):
        if self.grasped_object:
            try:
                hand_prim = XFormPrim("/World/Franka/panda_hand")
                hp, _ = hand_prim.get_world_poses()
                # 動的パラメータによるオフセットの適用
                offset = np.array([[0.0, 0.0, self.dynamic_params["grasp_offset_z"]]])
                XFormPrim(self.grasped_object).set_world_poses(positions=hp + offset)
            except: 
                pass

    def move_to_pose(self, pose_name, world, steps=30):
        target = self.poses.get(pose_name, self.poses["HOME"]).copy()
        target[7:] = self.current_pose[7:]
        start = self.current_pose.copy()
        
        if not hasattr(self.arm, "_physics_view"): 
            self.arm._physics_view = None
        if hasattr(self.arm, "initialize") and not self.arm.is_physics_handle_valid(): 
            self.arm.initialize()
            
        for t in range(steps):
            r = (1.0 - np.cos(t / steps * np.pi)) / 2.0
            self.arm.set_joint_positions(start + (target - start) * r)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()
        self.current_pose = target

    def close_gripper(self, world, target_path, steps=15):
        can_grasp = False
        if target_path:
            try:
                hand_prim = XFormPrim("/World/Franka/panda_hand")
                hp, _ = hand_prim.get_world_poses()
                obj_prim = XFormPrim(target_path)
                op, _ = obj_prim.get_world_poses()
                
                distance = np.linalg.norm(hp[0] - op[0])
                if distance < 0.15:
                    can_grasp = True
                else:
                    print(f"⚠️ 対象物が遠すぎます。把持をスキップします (距離: {distance:.3f}m)", flush=True)
            except Exception as e:
                print(f"距離計算エラー: {e}", flush=True)

        if can_grasp:
            self.grasped_object = target_path
            prim = get_current_stage().GetPrimAtPath(target_path)
            if prim.IsValid(): 
                UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
            self._set_collision_enabled(target_path, False)

        # 動的パラメータによるグリッパー幅の適用
        self.current_pose[7:] = self.dynamic_params["gripper_width"]
        
        if not hasattr(self.arm, "_physics_view"): 
            self.arm._physics_view = None
        if hasattr(self.arm, "initialize") and not self.arm.is_physics_handle_valid(): 
            self.arm.initialize()
            
        for _ in range(steps):
            self.arm.set_joint_positions(self.current_pose)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()

    def open_gripper(self, world, steps=15):
        if not self.grasped_object: 
            return
            
        self.current_pose[7:] = 0.04
        
        if not hasattr(self.arm, "_physics_view"): 
            self.arm._physics_view = None
        if hasattr(self.arm, "initialize") and not self.arm.is_physics_handle_valid(): 
            self.arm.initialize()
            
        for _ in range(steps):
            self.arm.set_joint_positions(self.current_pose)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()

        obj_id = self.grasped_object.split("/")[-1]
        self.disposed_list.append(obj_id)
        prim = get_current_stage().GetPrimAtPath(self.grasped_object)
        if prim.IsValid(): 
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
        self._set_collision_enabled(self.grasped_object, True)
        self.grasped_object = None

def capture_images_for_vlm(world, view_configs, writer_vlm):
    if SCENE_GRAPH_TMP_DIR.exists(): 
        shutil.rmtree(SCENE_GRAPH_TMP_DIR)
    SCENE_GRAPH_TMP_DIR.mkdir(parents=True)
    
    writer_vlm.initialize(output_dir=str(SCENE_GRAPH_TMP_DIR), rgb=True)
    capture_images = []
    
    for rp, name in view_configs:
        writer_vlm.attach([rp])
        for _ in range(3):
            world.step(render=True)
            rep.orchestrator.step()
        writer_vlm.detach()
        saved_files = sorted(list(SCENE_GRAPH_TMP_DIR.glob("**/rgb_*.png")))
        if saved_files:
            capture_images.append(str(saved_files[-1]))
            
    return capture_images

def save_plan(plan_list): 
    PLAN_JSON.write_text(json.dumps({"actions": plan_list}, indent=2))

def generate_recovery_plan(controller, ignore_list):
    new_plan = []
    for m in MUGS:
        path = f"/World/{m['id']}"
        # 配置が済んでおらず、かつ3回連続で失敗したスキップ対象でないものを計画に追加
        if not controller.verify_placement(path) and path not in ignore_list:
            new_plan.extend([f"(pick {m['id']})", "(place)"])
            
    save_plan(new_plan)
    return new_plan

def run_replan(failed_action, controller, user_instruction=None, ignore_list=[]):
    start_time = time.time()
    print(f"\n[Planner] Requesting LLM re-plan for: {failed_action}", flush=True)
    success = False
    
    if REAL_DELTA_PATH.exists() and os.environ.get("OPENAI_API_KEY"):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        cmd = [DELTA_PYTHON, str(REAL_DELTA_PATH), "--experiment", "all", "--episode", "1", "--domain", "laundry", "--domain-example", "laundry", "--scene", "allensville", "--scene-example", "allensville", "--print-plan"]
        if user_instruction: 
            cmd.extend(["--instruction", str(user_instruction)])
        try:
            subprocess.run(cmd, cwd=REAL_DELTA_PATH.parent, env=env, check=True, text=True)
            success = True
        except subprocess.CalledProcessError: 
            pass
            
    plan = generate_recovery_plan(controller, ignore_list)
    evaluator.record_event(failed_action, user_instruction, time.time() - start_time, 1, success)
    return plan

def generate_video():
    print(f"\n🎥 動画のエンコードを開始します...", flush=True)
    image_files = sorted(list(RGB_DIR.glob("**/rgb_*.png")))
    
    if not image_files:
        print("⚠️ 録画用フレームが見つかりませんでした。", flush=True)
        return

    tmp_dir = OUTPUT_DIR / "tmp_frames"
    if tmp_dir.exists(): 
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    
    try:
        for i, img_path in enumerate(image_files): 
            shutil.copy(str(img_path), str(tmp_dir / f"frame_{i:04d}.png"))
            
        cmd = [
            "ffmpeg", "-y", "-framerate", "30", 
            "-i", str(tmp_dir / "frame_%04d.png"), 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", 
            "-loglevel", "warning", str(VIDEO_PATH)
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ 動画の生成が完了しました: {VIDEO_PATH}", flush=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 動画エンコード中にエラーが発生しました: {e}", flush=True)
    finally:
        if tmp_dir.exists(): 
            shutil.rmtree(tmp_dir)

# =========================================================
# 3. メインルーチン
# =========================================================
def run_simulation():
    if OUTPUT_DIR.exists(): 
        shutil.rmtree(OUTPUT_DIR)
    RGB_DIR.mkdir(parents=True)
    SCENE_GRAPH_IMG_DIR.mkdir(parents=True)

    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2000)
    
    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/World/Franka")
    franka = Articulation("/World/Franka", name="franka")
    world.scene.add(franka)

    mat_prim = stage.DefinePrim("/World/Franka/FingerPhysicsMaterial", "Material")
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateStaticFrictionAttr(50.0)
    phys_mat.CreateDynamicFrictionAttr(50.0)

    for f_path in ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger", "/World/Franka/panda_hand/panda_leftfinger", "/World/Franka/panda_hand/panda_rightfinger"]:
        prim = stage.GetPrimAtPath(f_path)
        if prim.IsValid():
            XFormPrim(f_path).set_local_scales(np.array([[1.5, 1.5, 2.5]]))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat_prim), "physics")

    for cfg in MUGS:
        AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])
    AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

    cam_top   = rep.create.camera(position=CAMERA_POSITIONS["top"]["pos"], look_at=CAMERA_POSITIONS["top"]["look_at"])
    cam_left  = rep.create.camera(position=CAMERA_POSITIONS["left"]["pos"], look_at=CAMERA_POSITIONS["left"]["look_at"])
    cam_right = rep.create.camera(position=CAMERA_POSITIONS["right"]["pos"], look_at=CAMERA_POSITIONS["right"]["look_at"])
    cam_wrist = rep.create.camera(position=(0.0, 0.0, 0.05), rotation=(0, 0, 0), parent="/World/Franka/panda_hand")
    cam_main  = rep.create.camera(position=CAMERA_POSITIONS["main"]["pos"], look_at=CAMERA_POSITIONS["main"]["look_at"])

    rp_top   = rep.create.render_product(cam_top, (512, 512))
    rp_left  = rep.create.render_product(cam_left, (512, 512))
    rp_right = rep.create.render_product(cam_right, (512, 512))
    rp_wrist = rep.create.render_product(cam_wrist, (512, 512))
    rp_main  = rep.create.render_product(cam_main, (1280, 720))

    world.reset()
    world.play()
    for _ in range(5): 
        world.step(render=True)

    controller = RobotController(franka)
    analyzer = VLMAnalyzer(model_name=TARGET_MODEL)
    writer_vlm = rep.WriterRegistry.get("BasicWriter")
    writer_main = rep.WriterRegistry.get("BasicWriter")
    writer_main.initialize(output_dir=str(RGB_DIR), rgb=True)
    writer_main.attach([rp_main])

    view_configs = [(rp_top, "top"), (rp_left, "left"), (rp_right, "right"), (rp_wrist, "wrist")]

    max_overall_retries = 20
    retry_count = 0
    target_failure_counts = {}
    ignore_list = []
    
    plan = generate_recovery_plan(controller, ignore_list)
    current_target_bottle = None

    try:
        while len(plan) > 0 and retry_count < max_overall_retries:
            action = plan.pop(0)
            print(f"\n>> Executing: {action}", flush=True)
            
            if "pick" in action:
                target_id = action.replace(")", "").split()[1]
                current_target_bottle = f"/World/{target_id}"
                cfg = next((m for m in MUGS if m["id"] == target_id), None)
                if not cfg: 
                    continue
                
                high_pose = cfg["pose"].replace("_LOW", "_HIGH")
                controller.open_gripper(world)
                controller.move_to_pose(high_pose, world)
                wait_steps(world, 0.2, controller)
                controller.move_to_pose(cfg["pose"], world)
                wait_steps(world, 0.5, controller)
                controller.close_gripper(world, current_target_bottle)
                wait_steps(world, 0.5, controller)
                controller.move_to_pose(high_pose, world)

            elif "place" in action:
                images = capture_images_for_vlm(world, view_configs, writer_vlm)
                place_pose = analyzer.find_empty_space(images)
                
                controller.move_to_pose(place_pose, world)
                wait_steps(world, 0.5, controller)
                controller.open_gripper(world)
                controller.move_to_pose("HOME", world)
                
                wait_steps(world, 3.0, controller)
                
                images_after = capture_images_for_vlm(world, view_configs, writer_vlm)
                vlm_result = analyzer.process_and_analyze(images_after, action)
                is_failed = vlm_result.get("is_failed", False)
                error_reason = vlm_result.get("reason", "配置判定の失敗")
                
                if is_failed or not controller.verify_placement(current_target_bottle):
                    # 失敗回数をインクリメント
                    target_failure_counts[current_target_bottle] = target_failure_counts.get(current_target_bottle, 0) + 1
                    
                    if target_failure_counts[current_target_bottle] >= 3:
                        print(f"❌ {current_target_bottle} の操作に3回失敗しました。スキップして次の動作へ進みます。", flush=True)
                        ignore_list.append(current_target_bottle)
                        controller.reset_dynamic_params()
                    else:
                        print(f"⚠️ 失敗: {current_target_bottle} (試行 {target_failure_counts[current_target_bottle]}/3)。LLMで原因を分析しパラメータを調整します。", flush=True)
                        new_params = analyzer.analyze_and_adjust_params(error_reason, controller.dynamic_params)
                        controller.dynamic_params.update(new_params)
                        print(f"🔄 調整後のパラメータ: {controller.dynamic_params}", flush=True)
                        
                    plan = run_replan(action, controller, "失敗による再計画", ignore_list)
                    retry_count += 1
                    continue
                else:
                    print("✅ 成功: オブジェクトがカゴ内に確認されました。", flush=True)
                    # 成功した場合は失敗カウントとパラメータをリセット
                    target_failure_counts[current_target_bottle] = 0
                    controller.reset_dynamic_params()

        if len(plan) == 0:
            print("✅ すべてのタスク(スキップ含む)が完了しました！", flush=True)

        rep.orchestrator.wait_until_complete()
        generate_video()
        
    except Exception:
        traceback.print_exc()
    finally:
        try:
            writer_vlm.detach()
            writer_main.detach()
            rep.orchestrator.stop()
            world.stop()
            world.clear_instance()
        except: 
            pass
        simulation_app.close()

if __name__ == "__main__":
    run_simulation()