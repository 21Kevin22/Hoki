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
import sys
from pathlib import Path
import plotly.graph_objects as go

# =========================================================
# 【重要】APIキーの設定 (Gemini / OpenAI 呼び出し用)
# =========================================================
OPENAI_API_KEY = "<OPENAI_API_KEY>"
GOOGLE_API_KEY = "AIzaSyCsmmdOaLo7hdOXyyneRLA5kgQHBm516eQ"  # ★必ず書き換えてください

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# =========================================================

OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation")
RGB_DIR = OUTPUT_DIR / "rgb"
PLAN_JSON = OUTPUT_DIR / "actions.json"
VIDEO_PATH = OUTPUT_DIR / "bottle_sorting.mp4"
METRICS_CSV = OUTPUT_DIR / "evaluation_metrics.csv"

REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki/delta.py")
# ★修正: Isaac SimのPythonではなく、AIライブラリが入っているあなたの環境のPythonを絶対パスで指定
DELTA_PYTHON = "/home/ubuntu/.pyenv/versions/3.8.5/bin/python"

# --- 評価・ログ管理 ---
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

# --- Isaac Sim 起動 ---
from isaacsim import SimulationApp
config = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "hide_ui": True}
simulation_app = SimulationApp(config)

from pxr import UsdGeom, Gf, Sdf, UsdLux, UsdShade, UsdPhysics, PhysxSchema
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
        if controller: controller._update_grasped_object()
        rep.orchestrator.step()

# --- アセット生成 ---
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
        parts = [
            ("Bottom", (w, d, th), (0, 0, th/2)), ("Front", (w, th, h), (0, -d/2, h/2)),
            ("Back", (w, th, h), (0, d/2, h/2)), ("Left", (th, d, h), (-w/2, 0, h/2)),
            ("Right", (th, d, h), (w/2, 0, h/2))
        ]
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
                verts.append(Gf.Vec3f(*(center + minor_r * normal)))
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
                hand_pos, _ = XFormPrim("/World/Franka/panda_hand").get_world_poses()
                bottle_prim = XFormPrim(self.grasped_object)
                offset = np.array([[0.0, 0.0, -0.15]])
                deg = {"Bottle_R": -120, "Bottle_G": 180, "Bottle_B": 120}.get(self.grasped_object.split("/")[-1], 0)
                rad = np.deg2rad(deg)
                new_rot = np.array([[np.cos(rad/2), 0.0, 0.0, np.sin(rad/2)]])
                bottle_prim.set_world_poses(positions=hand_pos + offset, orientations=new_rot)
            except: pass

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
            if UsdPhysics.RigidBodyAPI(prim): UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
            if PhysxSchema.PhysxRigidBodyAPI(prim): PhysxSchema.PhysxRigidBodyAPI(prim).GetEnableCCDAttr().Set(True)
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
            if PhysxSchema.PhysxRigidBodyAPI(prim): PhysxSchema.PhysxRigidBodyAPI(prim).GetEnableCCDAttr().Set(False)
            if UsdPhysics.RigidBodyAPI(prim): UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
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

# --- プラン管理 & LLM連携 ---
def save_plan(plan_list):
    PLAN_JSON.write_text(json.dumps({"actions": plan_list}, indent=2))

def load_plan(force_reset=False):
    default_plan = ["(pick bottle_0)", "(place)", "(pick bottle_1)", "(place)", "(pick bottle_2)", "(place)"]
    if force_reset:
        save_plan(default_plan)
        return default_plan
    if PLAN_JSON.exists():
        try: 
            actions = json.loads(PLAN_JSON.read_text()).get("actions", [])
            if actions: return actions
        except: pass
    save_plan(default_plan)
    return default_plan

def force_update_json(instruction=""):
    print("[System] Using basic fallback update.")
    numbers = re.findall(r'[0-2]', instruction)
    unique_bottles = list(dict.fromkeys(numbers)) if numbers else ['1', '0', '2']
    new_actions = []
    for b in unique_bottles:
        new_actions.extend([f"(pick bottle_{b})", "(place)"])
    save_plan(new_actions)
    return new_actions

def run_replan(failed_action, user_instruction=None):
    start_time = time.time()
    print(f"\n[Planner] Requesting Gemini 2.5 Flash re-plan for failure around: {failed_action}")
    success = False
    
    if REAL_DELTA_PATH.exists():
        env = os.environ.copy()
        # ★修正: Isaac Sim の環境変数を完全に剥がし、pyenv のクリーンな環境を用意する
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        
        cmd = [
            DELTA_PYTHON, str(REAL_DELTA_PATH), 
            "--model", "gemini-2.5-flash",
            "--domain", "laundry",
            "--scene", "allensville",
            "--ref-pddl", "data/pddl/problem/allensville_laundry_problem.pddl",
            "--print-plan"  # ★ 追加: 生成したプランをターミナルに出力させる
        ]
        if user_instruction: 
            cmd.extend(["--instruction", str(user_instruction)])
        
        try:
            print(f"[Planner] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=REAL_DELTA_PATH.parent, env=env, text=True, capture_output=True)
            
            # (中略：ログ表示の print 文などはそのまま)

            actions = []
            # ★ 修正: stdout と stderr の両方から pick/drop の文字を探す
            combined_output = result.stdout + "\n" + result.stderr

            for line in combined_output.split('\n'):
                line_lower = line.lower()
                if "pick" in line_lower and "bottle" in line_lower:
                    match = re.search(r'(bottle_[0-2])', line_lower)
                    if match: actions.append(f"(pick {match.group(1)})")
                elif "drop" in line_lower or "place" in line_lower:
                    actions.append("(place)")
            
            if actions:
                save_plan(actions)
                success = True
                print(f"[Planner] 成功! 抽出された新しいプラン: {actions}")
            else:
                print("[Planner] プランを抽出できませんでした。フォールバックを使用します。")
                force_update_json(user_instruction or "")
                
        except Exception as e:
            print(f"[Error] delta.py の実行プロセス自体に失敗しました。Error:\n{e}")
            force_update_json(user_instruction or "")
    else:
        print("[Warning] delta.py が見つかりません。")
        force_update_json(user_instruction or "")

    latency = time.time() - start_time
    evaluator.record_event(failed_action, user_instruction, latency, 1, success)
    return load_plan()

# --- メインループ ---
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
        
        franka_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(usd_path=franka_path, prim_path="/World/Franka")
        franka = Articulation(prim_paths_expr="/World/Franka", name="franka_robot")
        world.scene.add(franka)

        mat_prim = stage.DefinePrim("/World/Franka/FingerPhysicsMaterial", "Material")
        UsdPhysics.MaterialAPI.Apply(mat_prim).CreateStaticFrictionAttr(10.0)
        UsdPhysics.MaterialAPI.Apply(mat_prim).CreateDynamicFrictionAttr(10.0)

        for f_path in ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger",
                       "/World/Franka/panda_hand/panda_leftfinger", "/World/Franka/panda_hand/panda_rightfinger"]:
            prim = stage.GetPrimAtPath(f_path)
            if prim.IsValid():
                XFormPrim(f_path).set_local_scales(np.array([[1.2, 1.2, 2.0]]))
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat_prim), "physics")

        AssetBuilder.create_beer_mug("/World/Bottle_R", [0.5,  0.55, 0.0], Gf.Vec3f(1, 0, 0), z_angle_deg=-120)
        AssetBuilder.create_beer_mug("/World/Bottle_G", [0.5,  0, 0.0], Gf.Vec3f(0, 1, 0), z_angle_deg=180)
        AssetBuilder.create_beer_mug("/World/Bottle_B", [0.5, -0.4, 0.0], Gf.Vec3f(0, 0, 1), z_angle_deg=120)
        AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

        set_camera_view(eye=[2.5, 1.5, 2.0], target=[0.0, 0.0, 0.0])
        
        world.reset()
        controller = RobotController(franka)
        franka.set_joint_positions(controller.poses["HOME"])

        plan = load_plan(force_reset=True)
        print("=== Simulation Start ===")
        
        while len(plan) > 0:
            action = plan.pop(0)
            print(f"\n>> Executing: {action}")
            print(f">> Remaining plan: {plan}")
            
            if "pick" in action:
                bottle_id = action.replace(")", "").split()[1]
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
            
            print("\n[Monitor] Action completed.")
            print("Press [Enter] to continue, [f] to simulate Failure/Re-plan, [q] to quit.")
            monitor_in = input("Monitor >> ").strip().lower()
            
            if monitor_in in ['q', 'quit', 'exit']:
                break
            elif monitor_in == 'f':
                print("\n[!] 失敗を検知: ロボットがアイテムを落としたか、目標を見失いました。")
                user_in = input("LLMに状況/指示を入力してください >> ").strip()
                
                plan = run_replan(action, user_in)
                continue

        controller.move_to_pose("HOME", world)
        print("=== Simulation Finished ===")
        
        rep.orchestrator.wait_until_complete()
        evaluator.generate_report()

    except Exception:
        traceback.print_exc()
    finally:
        print("\n[System] クリーンアップ処理を開始します...")
        try:
            world.stop()
        except Exception as e:
            print(f"[System] 停止中に軽微なエラーをスキップしました: {e}")
            
        print("[System] シミュレーションアプリを安全に終了します。")
        simulation_app.close()

if __name__ == "__main__":
    main()