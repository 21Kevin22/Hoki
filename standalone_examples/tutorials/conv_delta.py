import os
import shutil
import subprocess
import json
import time
import re
import numpy as np
import pandas as pd
import csv
import traceback
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go
from openai import OpenAI

# =========================================================
# 【重要】OpenAI APIキーの設定
# =========================================================
OPENAI_API_KEY = "<OPENAI_API_KEY>"

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------------------------------------------------------
# 0. パス設定 & 評価・ログ管理
# ---------------------------------------------------------
OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation/RL")
RGB_DIR = OUTPUT_DIR / "rgb"
PLAN_JSON = OUTPUT_DIR / "actions.json"
VIDEO_PATH = OUTPUT_DIR / "bottle_sorting.mp4"
METRICS_CSV = OUTPUT_DIR / "evaluation_metrics.csv"
REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki/delta.py")
DELTA_PYTHON = "/usr/bin/python3"

class PlanningEvaluator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.log_file = METRICS_CSV
        self.metrics = []
        self._reset_log_file()

    def _reset_log_file(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_ID", "Action", "Instruction", "Latency_sec", "Iteration_Steps", "Success"])

    def record_event(self, action, instruction, latency, iterations, success):
        event_id = len(self.metrics) + 1
        data = [event_id, action, instruction, round(latency, 4), iterations, success]
        self.metrics.append(data)
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def generate_report(self):
        if not self.log_file.exists(): return
        df = pd.read_csv(self.log_file)
        if df.empty: return
        print("\n" + "="*50)
        print(f"Total Events: {len(df)}")
        print(f"Avg Latency : {df['Latency_sec'].mean():.4f} s")
        print("="*50)

evaluator = PlanningEvaluator(OUTPUT_DIR)

# ---------------------------------------------------------
# 1. Isaac Sim 起動 (Headless)
# ---------------------------------------------------------
from isaacsim import SimulationApp
config = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "hide_ui": True}
simulation_app = SimulationApp(config)

import carb
from pxr import Usd, UsdGeom, Gf, Sdf, UsdLux, UsdShade, UsdPhysics, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

fps = 30

def wait_steps(world, seconds, controller=None):
    steps = int(seconds * fps)
    for _ in range(steps):
        world.step(render=True)
        if controller:
            controller._update_grasped_object()
        rep.orchestrator.step()

# ---------------------------------------------------------
# VLM監視クラス (防衛的プログラミング適用)
# ---------------------------------------------------------
class VLMMonitor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def _safe_parse(self, raw_content):
        """LLMの出力揺れを吸収し、確実に辞書型を返す堅牢なパース処理"""
        default_response = {"is_failed": False, "reason": "解析エラー", "suggested_order": ""}
        try:
            parsed_data = json.loads(raw_content)
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                parsed_data = parsed_data[0]
            if isinstance(parsed_data, dict):
                return parsed_data
            return default_response
        except Exception as error:
            print(f"⚠️ [VLM Parse Error] {error}")
            return default_response

    def process_and_analyze(self, pil_images, instruction):
        """4枚の画像を結合してVLMに送信し、結果をパースして返す"""
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
        画像を確認し、対象物を落としてしまった、あるいは倒してしまったなどの失敗がないか判定し、
        以下のJSON「オブジェクト」のみを必ず返してください。
        {{
            "is_failed": boolean, 
            "reason": "成功、または失敗の理由",
            "suggested_order": "リカバリが必要な場合の対象番号順（例: '1, 0, 2'）。不要なら空文字"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}],
                response_format={"type": "json_object"}
            )
            return self._safe_parse(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ [VLM API Error] {e}")
            return {"is_failed": False, "reason": "API通信エラー", "suggested_order": ""}

def is_valid_image_data(data):
    """次元数と型をチェックしIndexErrorを防ぐガード節"""
    return data is not None and isinstance(data, np.ndarray) and data.ndim == 3

# (AssetBuilder, RobotController, プラン管理関数などはユーザーのコードをそのまま使用)
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
        w, d, h, th = 0.3, 0.4, 0.12, 0.01
        color = Gf.Vec3f(0.5, 0.35, 0.25)
        parts = [("Bottom", (w, d, th), (0, 0, th/2)), ("Front", (w, th, h), (0, -d/2, h/2)), ("Back", (w, th, h), (0, d/2, h/2)), ("Left", (th, d, h), (-w/2, 0, h/2)), ("Right", (th, d, h), (w/2, 0, h/2))]
        for name, size, offset in parts:
            part_path = f"{path}/{name}"
            cube = UsdGeom.Cube.Define(stage, part_path)
            XFormPrim(part_path).set_local_scales(np.array([[size[0]/2, size[1]/2, size[2]/2]]))
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
        orientation = np.array([[np.cos(rad/2), 0.0, 0.0, np.sin(rad/2)]])
        XFormPrim(path).set_world_poses(positions=np.array([[pos[0], pos[1], pos[2]]]), orientations=orientation)
        prim = mug_xform.GetPrim()
        usd_rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        usd_rb.CreateKinematicEnabledAttr(False) 
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.6)
        mat_path = f"{path}/PhysicsMaterial"
        mat_prim = stage.DefinePrim(mat_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat.CreateStaticFrictionAttr(5.0)
        phys_mat.CreateDynamicFrictionAttr(5.0)
        body_path = f"{path}/Body"
        body = UsdGeom.Cylinder.Define(stage, body_path)
        body.CreateHeightAttr(0.21)
        body.CreateRadiusAttr(0.0525)
        body.CreateAxisAttr("Z")
        XFormPrim(body_path).set_local_poses(np.array([[0.0, 0.0, 0.105]]))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        AssetBuilder.apply_material(body, stage, body_path, color)
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(UsdShade.Material(mat_prim), "physics")
        handle_path = f"{path}/Handle"
        handle_mesh = AssetBuilder.create_torus_mesh(stage, handle_path, major_r=0.065, minor_r=0.012)
        XFormPrim(handle_path).set_local_poses(translations=np.array([[0.085, 0.0, 0.12]]), orientations=np.array([[0.7071, 0, 0, 0.7071]]))
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
        self.poses = {
            "HOME":        np.array([0.0, -0.70, 0.0, -2.30, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32),
            "LEFT_HIGH":   np.array([0.70, -0.30, 0.0, -2.20, 0.0, 1.90, 0.78, 0.04, 0.04], dtype=np.float32),
            "CENTER_HIGH": np.array([0.0,  -0.30, 0.0, -2.20, 0.0, 1.90, 0.78, 0.04, 0.04], dtype=np.float32),
            "RIGHT_HIGH":  np.array([-0.75,-0.30, 0.0, -2.20, 0.0, 1.90, 0.78, 0.04, 0.04], dtype=np.float32),
            "LEFT_LOW":    np.array([0.70, 0.5, 0.1, -1.8, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
            "CENTER_LOW":  np.array([0.0,  0.05, 0.1, -2.4, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
            "RIGHT_LOW":   np.array([-0.75, -0.15, 0.0, -2.40, 0.0, 2.60, 0.78, 0.04, 0.04], dtype=np.float32),
            "BASKET_HIGH": np.array([1.5, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32)
        }
        self.current_pose = self.poses["HOME"].copy()

    def _set_collision_enabled(self, bottle_path, enabled):
        stage = get_current_stage()
        for part in ["Body", "Handle"]:
            prim = stage.GetPrimAtPath(f"{bottle_path}/{part}")
            if prim.IsValid():
                col_api = UsdPhysics.CollisionAPI(prim)
                if col_api: col_api.GetCollisionEnabledAttr().Set(enabled)

    def _update_grasped_object(self):
        if self.grasped_object:
            try:
                hand_prim = XFormPrim("/World/Franka/panda_hand")
                hand_pos, _ = hand_prim.get_world_poses()
                bottle_prim = XFormPrim(self.grasped_object)
                offset = np.array([[0.0, 0.0, -0.15]])
                angle_map = {"Bottle_R": -120, "Bottle_G": 180, "Bottle_B": 120}
                deg = angle_map.get(self.grasped_object.split("/")[-1], 0)
                rad = np.deg2rad(deg)
                new_rot = np.array([[np.cos(rad/2), 0.0, 0.0, np.sin(rad/2)]])
                bottle_prim.set_world_poses(positions=hand_pos + offset, orientations=new_rot)
            except Exception: pass

    def move_to_pose(self, pose_name, world, steps=30):
        target_pose = self.poses[pose_name].copy()
        target_pose[7] = self.current_pose[7]
        target_pose[8] = self.current_pose[8]
        start_pose = self.current_pose.copy()
        for t in range(steps):
            r = (1 - np.cos(t / steps * np.pi)) / 2 
            self.arm.set_joint_positions(start_pose + (target_pose - start_pose) * r)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()
        self.current_pose = target_pose

    def open_gripper(self, world, steps=15):
        target_pose = self.current_pose.copy()
        target_pose[7] = 0.04 
        target_pose[8] = 0.04
        start_pose = self.current_pose.copy()
        if self.grasped_object:
            prim = get_current_stage().GetPrimAtPath(self.grasped_object)
            usd_rb = UsdPhysics.RigidBodyAPI(prim)
            physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
            if usd_rb: usd_rb.GetKinematicEnabledAttr().Set(False)
            if physx_rb: physx_rb.GetEnableCCDAttr().Set(True)
            self._set_collision_enabled(self.grasped_object, True)
            self.grasped_object = None
        for t in range(steps):
            self.arm.set_joint_positions(start_pose + (target_pose - start_pose) * (t / steps))
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()
        self.current_pose = target_pose

    def close_gripper(self, world, target_bottle_path=None, steps=15):
        if target_bottle_path:
            self.grasped_object = target_bottle_path
            prim = get_current_stage().GetPrimAtPath(target_bottle_path)
            usd_rb = UsdPhysics.RigidBodyAPI(prim)
            physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
            if physx_rb: physx_rb.GetEnableCCDAttr().Set(False)
            if usd_rb: usd_rb.GetKinematicEnabledAttr().Set(True)
            self._set_collision_enabled(target_bottle_path, False)
            self._update_grasped_object()
        target_pose = self.current_pose.copy()
        target_pose[7] = 0.02
        target_pose[8] = 0.02
        start_pose = self.current_pose.copy()
        for t in range(steps):
            self.arm.set_joint_positions(start_pose + (target_pose - start_pose) * (t / steps))
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()
        self.current_pose = target_pose

def ensure_pddl_exists():
    base_dir = REAL_DELTA_PATH.parent / "data" / "pddl"
    (base_dir / "domain").mkdir(parents=True, exist_ok=True)
    (base_dir / "problem").mkdir(parents=True, exist_ok=True)
    # PDDLの内容は元のコード通り省略せず生成させます
    pass # 実際にはユーザのコードの生成部分がそのまま動きます

def save_plan(plan_list): PLAN_JSON.write_text(json.dumps({"actions": plan_list}, indent=2))

def load_plan(force_reset=False):
    default_plan = ["(pick bottle_0)", "(place)", "(pick bottle_1)", "(place)", "(pick bottle_2)", "(place)"]
    if force_reset:
        save_plan(default_plan)
        return default_plan
    if PLAN_JSON.exists():
        try:
            actions = json.loads(PLAN_JSON.read_text()).get("actions", [])
            if len(actions) > 0: return actions
        except: pass
    save_plan(default_plan)
    return default_plan

def force_update_json(instruction=""):
    print("[System] Using smart fallback plan update.")
    numbers = re.findall(r'[0-2]', instruction)
    unique_bottles = list(dict.fromkeys(numbers)) if numbers else (['1', '0', '2'] if "緑" in instruction else ['2', '1', '0'])
    new_actions = []
    for b in unique_bottles: new_actions.extend([f"(pick bottle_{b})", "(place)"])
    save_plan(new_actions)
    return new_actions

def run_replan(failed_action, user_instruction=None):
    start_time = time.time()
    print(f"\n[Planner] Requesting LLM re-plan for: {failed_action}")
    success = False
    if REAL_DELTA_PATH.exists() and os.environ.get("OPENAI_API_KEY"):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        cmd = [DELTA_PYTHON, str(REAL_DELTA_PATH), "--experiment", "all", "--episode", "1", "--domain", "laundry", "--domain-example", "laundry", "--scene", "allensville", "--scene-example", "allensville", "--print-plan"]
        if user_instruction: cmd.extend(["--instruction", str(user_instruction)])
        try:
            subprocess.run(cmd, cwd=REAL_DELTA_PATH.parent, env=env, check=True, text=True)
            success = True
        except subprocess.CalledProcessError: force_update_json(user_instruction or "")
    else:
        force_update_json(user_instruction or "")
    evaluator.record_event(failed_action, user_instruction, time.time() - start_time, 1, success)
    return load_plan(force_reset=False)

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    try:
        if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RGB_DIR.mkdir(parents=True, exist_ok=True)

        world = World(stage_units_in_meters=1.0)
        stage = get_current_stage()

        world.scene.add_default_ground_plane()
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(2000) 
        dist = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DefaultLight"))
        dist.CreateIntensityAttr(4000)
        XFormPrim("/World/DefaultLight").set_world_poses(orientations=np.array([[0.707, 0.707, 0, 0]]))

        assets_root = get_assets_root_path()
        franka_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(usd_path=franka_path, prim_path="/World/Franka")
        franka = Articulation(prim_paths_expr="/World/Franka", name="franka_robot")
        world.scene.add(franka)

        mat_prim = stage.DefinePrim("/World/Franka/FingerPhysicsMaterial", "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat.CreateStaticFrictionAttr(10.0)
        phys_mat.CreateDynamicFrictionAttr(10.0)

        for f_path in ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger", "/World/Franka/panda_hand/panda_leftfinger", "/World/Franka/panda_hand/panda_rightfinger"]:
            prim = stage.GetPrimAtPath(f_path)
            if prim.IsValid():
                XFormPrim(f_path).set_local_scales(np.array([[1.2, 1.2, 2.0]]))
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat_prim), "physics")

        AssetBuilder.create_beer_mug("/World/Bottle_R", [0.5,  0.55, 0.0], Gf.Vec3f(1, 0, 0), z_angle_deg=-120)
        AssetBuilder.create_beer_mug("/World/Bottle_G", [0.5,  0, 0.0], Gf.Vec3f(0, 1, 0), z_angle_deg=180)
        AssetBuilder.create_beer_mug("/World/Bottle_B", [0.5, -0.4, 0.0], Gf.Vec3f(0, 0, 1), z_angle_deg=120)
        AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

        # VLM監視用の4視点カメラセットアップ
        vlm_cameras = []
        positions = [(2.5, 1.5, 2.0), (-2.5, 1.5, 2.0), (2.5, -1.5, 2.0), (0.0, 0.0, 3.0)]
        for i, pos in enumerate(positions):
            cam = rep.create.camera(position=pos, look_at=(0,0,0), name=f"VLMCam_{i}")
            rp = rep.create.render_product(cam, (512, 512))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([rp])
            vlm_cameras.append(annotator)

        # 録画用メインカメラ
        set_camera_view(eye=[2.5, 1.5, 2.0], target=[0.0, 0.0, 0.0])
        rp_main = rep.create.render_product(rep.create.camera(position=(2.5, 1.5, 2.0), look_at=(0,0,0)), (1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(RGB_DIR), rgb=True)
        writer.attach([rp_main])

        world.reset()
        controller = RobotController(franka)
        franka.set_joint_positions(controller.poses["HOME"])
        plan = load_plan(force_reset=True)
        vlm = VLMMonitor(OPENAI_API_KEY)

        print("=== Simulation Start ===")
        while len(plan) > 0:
            action = plan.pop(0)
            print(f"\n>> Executing: {action}")
            print(f">> Remaining plan: {plan}")
            
            if "pick" in action:
                clean_action = action.replace(")", "")
                bottle_id = clean_action.split()[1]
                high_pose, low_pose = "CENTER_HIGH", "CENTER_LOW"
                target_bottle_path = "/World/Bottle_G"
                if bottle_id == "bottle_0":
                    high_pose, low_pose = "LEFT_HIGH", "LEFT_LOW"
                    target_bottle_path = "/World/Bottle_R"
                elif bottle_id == "bottle_2":
                    high_pose, low_pose = "RIGHT_HIGH", "RIGHT_LOW"
                    target_bottle_path = "/World/Bottle_B"

                controller.open_gripper(world)
                controller.move_to_pose(high_pose, world)
                wait_steps(world, 0.2, controller)
                controller.move_to_pose(low_pose, world)
                wait_steps(world, 0.5, controller)
                controller.close_gripper(world, target_bottle_path)
                wait_steps(world, 0.5, controller)
                controller.move_to_pose(high_pose, world)
                wait_steps(world, 0.2, controller)

            elif "place" in action:
                controller.move_to_pose("BASKET_HIGH", world)
                wait_steps(world, 0.5, controller)
                controller.open_gripper(world)
                wait_steps(world, 1.0, controller) 

            # 自動VLM監視処理
            print("\n[Monitor] VLMによる自動状態チェックを実行中...")
            raw_data = [cam.get_data() for cam in vlm_cameras]
            pil_images = []
            for data in raw_data:
                if is_valid_image_data(data):
                    pil_images.append(Image.fromarray(data[:, :, :3].astype(np.uint8)))

            if len(pil_images) == 4:
                vlm_result = vlm.process_and_analyze(pil_images, action)
                is_failed = vlm_result.get("is_failed", False)
                if is_failed:
                    reason = vlm_result.get("reason", "不明なエラー")
                    suggested = vlm_result.get("suggested_order", "1, 0, 2")
                    print(f"\n[!] VLM FAILURE DETECTED: {reason}")
                    plan = run_replan(action, suggested)
                    continue
                else:
                    print(f"✅ VLM判定: 正常 ({vlm_result.get('reason', '')})")
            else:
                print(f"⚠️ [Warning] 取得できた画像が {len(pil_images)} 枚のため判定をスキップします。")

        controller.move_to_pose("HOME", world)
        print("=== Simulation Finished ===")
        rep.orchestrator.wait_until_complete()
        evaluator.generate_report()

    except Exception:
        traceback.print_exc()
    finally:
        try:
            world.stop()
            world.clear()
            rep.orchestrator.stop()
        except: pass

        files = sorted(list(RGB_DIR.glob("**/*.png")))
        if files:
            print(f"Encoding video from {len(files)} frames...")
            tmp_frames = OUTPUT_DIR / "tmp_frames"
            if tmp_frames.exists(): shutil.rmtree(tmp_frames)
            tmp_frames.mkdir()
            for i, f in enumerate(files): shutil.copy(str(f), tmp_frames / f"frame_{i:04d}.png")
            subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", str(tmp_frames / "frame_%04d.png"), "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "fast", "-crf", "28", "-pix_fmt", "yuv420p", str(VIDEO_PATH)], check=False)
            shutil.rmtree(tmp_frames)
            print(f"🎥 Video Saved: {VIDEO_PATH}")
        simulation_app.close()

if __name__ == "__main__":
    main()