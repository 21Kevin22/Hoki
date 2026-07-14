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

faulthandler.enable()

# =========================================================
# 【重要】APIキーとモデルの設定
# =========================================================
OPENAI_API_KEY = "<OPENAI_API_KEY>" # ★ご自身のAPIキーに変更
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

TARGET_MODEL = "gpt-4o"

# =========================================================
# 0. 設定管理
# =========================================================
OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation/vlm_delta_final")
RGB_DIR = OUTPUT_DIR / "frames"
SCENE_GRAPH_IMG_DIR = OUTPUT_DIR / "scene_graph_images"
SCENE_GRAPH_TMP_DIR = OUTPUT_DIR / "tmp_capture"
VIDEO_PATH = OUTPUT_DIR / "mug_sorting.mp4"
PDDL_LOG_PATH = OUTPUT_DIR / "simulation_pddl_snapshots.log"

METRICS_CSV_PATH = OUTPUT_DIR / "vlm_metrics.csv"
ACCURACY_CSV_PATH = OUTPUT_DIR / "accuracy_metrics.csv"

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
    "BASKET_HIGH": np.array([1.5, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04], dtype=np.float32)
}

CAMERA_POSITIONS = {
    "top":   {"pos": (0.5, 0.0, 2.2),  "look_at": (0.5, 0.0, 0.0)},
    "left":  {"pos": (1.4, 1.2, 1.0),  "look_at": (0.5, 0.0, 0.0)},
    "right": {"pos": (1.4, -1.2, 1.0), "look_at": (0.5, 0.0, 0.0)},
    "main":  {"pos": (1.8, 0.0, 1.5),  "look_at": (0.5, 0.0, 0.2)}
}

# =========================================================
# 1. PDDL ドメイン定義 (DELTA論文準拠・構文エラー修正版)
# =========================================================
DOMAIN_PDDL_TEXT = """
(define (domain clean_mugs)
  (:requirements :strips :typing :negative-preconditions) 
  (:types agent item room)
  (:predicates 
    (agent_at ?a - agent ?r - room)
    (item_at ?i - item ?r - room)
    (agent_has_item ?a - agent ?i - item)
    (agent_loaded ?a - agent)
    (item_disposed ?i - item)
    (item_pickable ?i - item)
    (neighbor ?r1 ?r2 - room)
  )
  (:action move
    :parameters (?a - agent ?from ?to - room)
    :precondition (and (agent_at ?a ?from) (neighbor ?from ?to))
    :effect (and (not (agent_at ?a ?from)) (agent_at ?a ?to))
  )
  (:action pick
    :parameters (?a - agent ?i - item ?r - room)
    :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (item_pickable ?i) (not (agent_loaded ?a)))
    :effect (and (not (item_at ?i ?r)) (agent_has_item ?a ?i) (agent_loaded ?a))
  )
  (:action place
    :parameters (?a - agent ?i - item ?r - room ?dest - item)
    :precondition (and (agent_at ?a ?r) (agent_has_item ?a ?i) (item_at ?dest ?r))
    :effect (and (not (agent_has_item ?a ?i)) (not (agent_loaded ?a)) (item_at ?i ?r) (item_disposed ?i))
  )
)
"""

# =========================================================
# 2. 外部プランナー実行 (Isaac SimのPython環境干渉回避版)
# =========================================================
def run_pddl_planner(domain_text, problem_text, output_dir, timeout=60):
    domain_path = output_dir / "domain_temp.pddl"
    problem_path = output_dir / "problem_temp.pddl"
    with open(domain_path, "w") as f: f.write(domain_text)
    with open(problem_path, "w") as f: f.write(problem_text)
    
    # OS標準のPythonを使ってFast Downwardを呼び出す
    cmd = [
        "/usr/bin/python3", 
        "/home/ubuntu/downward/fast-downward.py",  # ★環境に合わせて確認
        "--alias", "seq-opt-lmcut", 
        str(domain_path), str(problem_path)
    ]
    
    # Isaac Simの環境変数をリセットして干渉を防ぐ
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    
    start_perf = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=clean_env)
        wall_time = time.perf_counter() - start_perf
        
        if result.returncode != 0:
            return {"success": False, "error": f"Exit code {result.returncode}", "details": result.stderr + "\n" + result.stdout}
            
        output = result.stdout
        search_time = float(re.search(r"Actual search time: ([\d.]+)s", output).group(1)) if "Actual search time" in output else 0.0
        expanded_nodes = int(re.search(r"Expanded (\d+) state", output).group(1)) if "Expanded" in output else 0
        plan_actions = re.findall(r"^(.+) \(\d+\)$", output, re.MULTILINE)
        
        return {
            "success": len(plan_actions) > 0,
            "search_time": search_time,
            "expanded_nodes": expanded_nodes,
            "plan_length": len(plan_actions),
            "plan_actions": plan_actions,
            "details": output
        }
    except Exception as e:
        return {"success": False, "error": "Exception", "details": str(e)}

# =========================================================
# 3. Isaac Sim 起動
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
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep

# ★ ロボット直接制御用ライブラリ
from omni.isaac.dynamic_control import _dynamic_control

# =========================================================
# 4. ユーティリティクラス群
# =========================================================
class VLMAnalyzer:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name

    def encode_image_base64(self, path):
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

    def get_scene_graph(self, image_paths, max_retries=3):
        prompt = """
        これはロボットアームがマグカップを片付けるタスクです。4枚の画像を分析し、3Dシーングラフを構築してください。
        【ID命名規則】mug_0, mug_1, mug_2, mug_3, mug_4, basket, table, panda
        【要件】
        1. nodes: id, category, affordance("item_has_handle", "item_containable", "pickable"等)
        2. edges: subject, predicate("on", "inside", "grasped_by", "near"), object
        純粋なJSONのみ出力してください。
        """
        for _ in range(max_retries):
            try:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
                content = [{"type": "text", "text": prompt}]
                for img_path in image_paths: 
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image_base64(img_path)}", "detail": "low"}})
                
                payload = {"model": self.model_name, "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": content}], "temperature": 0.0}
                resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                return json.loads(resp.json()["choices"][0]["message"]["content"])
            except Exception as e:
                print(f"[VLM Error] {e}")
                time.sleep(2)
        return {"nodes": [], "edges": []}

class PDDLStateGenerator:
    @staticmethod
    def get_snapshot(controller, items_config, step_name, vlm_graph_data=None):
        lines = [f"; --- PDDL Snapshot: {step_name} ---", "(define (problem clean_task)", "    (:domain clean_mugs)"]
        item_ids = " ".join([b["id"] for b in items_config])
        lines.append(f"    (:objects panda - agent table bin - room {item_ids} basket - item)")
        lines.append("    (:init")
        
        loc = "bin" if controller.grasped_object is None and "BASKET" in str(controller.current_pose) else "table"
        lines.append(f"        (agent_at panda {loc})")
        
        for b in items_config:
            b_id = b["id"]
            if controller.grasped_object == f"/World/{b_id}":
                lines.append(f"        (agent_has_item panda {b_id})")
                lines.append(f"        (agent_loaded panda)")
            elif b_id in controller.disposed_list: 
                lines.append(f"        (item_at {b_id} basket)")
                lines.append(f"        (item_disposed {b_id})")
            else:
                lines.append(f"        (item_at {b_id} table)")
                lines.append(f"        (item_pickable {b_id})")
        
        lines.append("        (item_at basket bin)")
        lines.append("        (neighbor table bin)\n        (neighbor bin table)\n    )")
        lines.append("    (:goal (and " + " ".join([f"(item_disposed {b['id']})" for b in items_config]) + "))")
        lines.append(")")
        
        content = "\n".join(lines)
        with open(PDDL_LOG_PATH, "a") as f: f.write(content + "\n\n")
        return content

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
    def create_beer_mug(path, pos, color, z_angle_deg):
        stage = get_current_stage()
        mug_xform = UsdGeom.Xform.Define(stage, path)
        rad = np.deg2rad(z_angle_deg)
        orientation = np.array([[np.cos(rad/2), 0.0, 0.0, np.sin(rad/2)]])
        XFormPrim(path).set_world_poses(positions=np.array([pos]), orientations=orientation)
        prim = mug_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.4)
        body_path = f"{path}/Body"
        body = UsdGeom.Cylinder.Define(stage, body_path)
        body.CreateHeightAttr(0.14)
        body.CreateRadiusAttr(0.03) 
        body.CreateAxisAttr("Z")
        XFormPrim(body_path).set_local_poses(np.array([[0.0, 0.0, 0.07]]))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        AssetBuilder.apply_material(body, stage, body_path, color)

    @staticmethod
    def create_basket(path, pos):
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, path)
        w, d, h, th = 0.3, 0.4, 0.12, 0.01
        parts = [("Bottom", (w, d, th), (0.0, 0.0, th / 2)), ("Front", (w, th, h), (0.0, -d / 2, h / 2)),
                 ("Back", (w, th, h), (0.0, d / 2, h / 2)), ("Left", (th, d, h), (-w / 2, 0.0, h / 2)), ("Right", (th, d, h), (w / 2, 0.0, h / 2))]
        for name, size, offset in parts:
            p_path = f"{path}/{name}"
            cube = UsdGeom.Cube.Define(stage, p_path)
            XFormPrim(p_path).set_local_scales(np.array([np.array(size) / 2.0]))
            XFormPrim(p_path).set_local_poses(np.array([offset]))
            AssetBuilder.apply_material(cube, stage, p_path, Gf.Vec3f(0.5, 0.35, 0.25))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        XFormPrim(path).set_world_poses(positions=np.array([pos]))

# =========================================================
# ★【安全版】Dynamic Control APIによる直接制御クラス
# =========================================================
class RobotController:
    def __init__(self, arm_path="/World/Franka"):
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.arm_path = arm_path
        self.art = self.dc.get_articulation(self.arm_path)
        
        self.grasped_object = None
        self.disposed_list = []
        self.grasp_offset = np.array([[0.0, 0.0, -0.10]])
        self.poses = POSE_LIBRARY.copy()
        for name in list(self.poses.keys()):
            if "_LOW" in name:
                hp = self.poses[name].copy()
                hp[1] -= 0.4 
                self.poses[name.replace("_LOW", "_HIGH")] = hp
        self.current_pose = self.poses["HOME"].copy()

    def _update_grasped_object(self):
        if self.grasped_object:
            try:
                hp, _ = XFormPrim("/World/Franka/panda_hand").get_world_poses()
                XFormPrim(self.grasped_object).set_world_poses(positions=hp + self.grasp_offset)
            except: pass

    def move_to_pose(self, pose_name, world, steps=30):
        target = self.poses[pose_name].copy()
        target[7:] = self.current_pose[7:]
        start = self.current_pose.copy()
        
        if self.art == _dynamic_control.INVALID_HANDLE:
            self.art = self.dc.get_articulation(self.arm_path)
            
        for t in range(steps):
            r = (1.0 - np.cos(t / steps * np.pi)) / 2.0
            current_target = start + (target - start) * r
            self.dc.set_articulation_dof_position_targets(self.art, current_target)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()
        self.current_pose = target

    def close_gripper(self, world, target_path, steps=15):
        if target_path:
            self.grasped_object = target_path
            try:
                hp, _ = XFormPrim("/World/Franka/panda_hand").get_world_poses()
                op, _ = XFormPrim(target_path).get_world_poses()
                self.grasp_offset = op - hp
            except:
                self.grasp_offset = np.array([[0, 0, -0.10]])
            prim = get_current_stage().GetPrimAtPath(target_path)
            if prim.IsValid(): UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)

        self.current_pose[7:] = 0.02
        if self.art == _dynamic_control.INVALID_HANDLE:
            self.art = self.dc.get_articulation(self.arm_path)
            
        for _ in range(steps):
            self.dc.set_articulation_dof_position_targets(self.art, self.current_pose)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()

    def open_gripper(self, world, steps=15):
        if not self.grasped_object: return
        self.current_pose[7:] = 0.04
        if self.art == _dynamic_control.INVALID_HANDLE:
            self.art = self.dc.get_articulation(self.arm_path)
            
        for _ in range(steps):
            self.dc.set_articulation_dof_position_targets(self.art, self.current_pose)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()

        obj_id = self.grasped_object.split("/")[-1]
        self.disposed_list.append(obj_id)
        prim = get_current_stage().GetPrimAtPath(self.grasped_object)
        if prim.IsValid(): UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
        self.grasped_object = None

def capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, event_name):
    start_time = time.time()
    step_dir = SCENE_GRAPH_IMG_DIR / event_name
    step_dir.mkdir(parents=True, exist_ok=True)
    if SCENE_GRAPH_TMP_DIR.exists(): shutil.rmtree(SCENE_GRAPH_TMP_DIR)
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
            new_path = step_dir / f"view_{name}.png"
            shutil.move(str(saved_files[-1]), str(new_path))
            capture_images.append(str(new_path))

    graph_result = analyzer.get_scene_graph(capture_images)
    vlm_processing_time = time.time() - start_time
    
    with open(METRICS_CSV_PATH, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow([analyzer.model_name, event_name, round(vlm_processing_time, 3), "", "", "", time.time()])
        
    return graph_result

def generate_video():
    image_files = sorted(list(RGB_DIR.glob("**/rgb_*.png")))
    if not image_files: return
    tmp_dir = OUTPUT_DIR / "tmp_frames"
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        for i, img_path in enumerate(image_files): shutil.copy(str(img_path), str(tmp_dir / f"frame_{i:04d}.png"))
        subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", str(tmp_dir / "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(VIDEO_PATH)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)

# =========================================================
# 5. メインルーチン (DELTAの統合フロー)
# =========================================================
def run_simulation():
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    RGB_DIR.mkdir(parents=True)
    SCENE_GRAPH_IMG_DIR.mkdir(parents=True)
    if PDDL_LOG_PATH.exists(): PDDL_LOG_PATH.unlink()
    
    with open(METRICS_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["model_name", "event_name", "vlm_processing_time_sec", "planning_time_sec", "expanded_nodes", "plan_length", "timestamp"])

    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2000)
    
    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/World/Franka")

    for cfg in MUGS:
        AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])
    AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

    cam_top   = rep.create.camera(position=CAMERA_POSITIONS["top"]["pos"], look_at=CAMERA_POSITIONS["top"]["look_at"])
    cam_left  = rep.create.camera(position=CAMERA_POSITIONS["left"]["pos"], look_at=CAMERA_POSITIONS["left"]["look_at"])
    cam_right = rep.create.camera(position=CAMERA_POSITIONS["right"]["pos"], look_at=CAMERA_POSITIONS["right"]["look_at"])
    cam_wrist = rep.create.camera(position=(0.0, 0.0, 0.05), rotation=(0, 0, 0), parent="/World/Franka/panda_hand")
    cam_main  = rep.create.camera(position=CAMERA_POSITIONS["main"]["pos"], look_at=CAMERA_POSITIONS["main"]["look_at"])

    rp_top   = rep.create.render_product(cam_top, (1280, 720))
    rp_left  = rep.create.render_product(cam_left, (1280, 720))
    rp_right = rep.create.render_product(cam_right, (1280, 720))
    rp_wrist = rep.create.render_product(cam_wrist, (1280, 720))
    rp_main  = rep.create.render_product(cam_main, (1280, 720))

    # ★変更点: 手動初期化チェックをせず、Worldのプレイに任せる
    world.reset()
    world.play()
    for _ in range(15): world.step(render=True)

    controller = RobotController(arm_path="/World/Franka")
    analyzer = VLMAnalyzer(model_name=TARGET_MODEL)
    writer_vlm = rep.WriterRegistry.get("BasicWriter")
    writer_main = rep.WriterRegistry.get("BasicWriter")
    writer_main.initialize(output_dir=str(RGB_DIR), rgb=True)
    writer_main.attach([rp_main])

    view_configs = [(rp_top, "top"), (rp_left, "left"), (rp_right, "right"), (rp_wrist, "wrist")]

    current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, "Step_0_Initial")
    current_pddl = PDDLStateGenerator.get_snapshot(controller, MUGS, "Initial State", vlm_graph_data=current_graph)

    step_count = 1
    max_steps = 5

    try:
        while step_count <= max_steps:
            print(f"\n=== Executing Step {step_count} (DELTA Planning) ===")
            
            planner_res = run_pddl_planner(DOMAIN_PDDL_TEXT, current_pddl, OUTPUT_DIR)
            
            if planner_res["success"]:
                with open(METRICS_CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([TARGET_MODEL, f"Step_{step_count}_Planning", "", planner_res["search_time"], planner_res["expanded_nodes"], planner_res["plan_length"], time.time()])
                
                action = planner_res["plan_actions"][0]
                match = re.search(r"mug_\d+", action)
                target_id = match.group() if match else None
                print(f"[Planner] Success! Nodes: {planner_res['expanded_nodes']}, Target: {target_id}")
            else:
                print("\n[System] All task completed or Planner failed.")
                print(f"[Debug] Error: {planner_res.get('error')}")
                print(f"[Debug] Details:\n{planner_res.get('details')}")
                break
                
            cfg = next((m for m in MUGS if m["id"] == target_id), None)
            if not cfg: break
            high_pose = cfg["pose"].replace("_LOW", "_HIGH")
            
            capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, f"Step_{step_count}_1_Before_Move")
            controller.move_to_pose(high_pose, world)
            controller.move_to_pose(cfg["pose"], world)
            
            capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, f"Step_{step_count}_2_Before_Grasp")
            controller.close_gripper(world, f"/World/{target_id}")
            controller.move_to_pose(high_pose, world)
            
            capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, f"Step_{step_count}_3_After_Grasp")
            controller.move_to_pose("BASKET_HIGH", world)
            
            capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, f"Step_{step_count}_4_Before_Drop")
            controller.open_gripper(world)
            controller.move_to_pose("HOME", world)
            
            current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, f"Step_{step_count}_5_After_Drop")
            
            current_pddl = PDDLStateGenerator.get_snapshot(controller, MUGS, f"After Step {step_count}", vlm_graph_data=current_graph)
            
            step_count += 1

        rep.orchestrator.wait_until_complete()
        generate_video()
    except Exception:
        traceback.print_exc()
    finally:
        try:
            if 'writer_vlm' in locals(): writer_vlm.detach()
            if 'writer_main' in locals(): writer_main.detach()
            rep.orchestrator.stop()
        except Exception: pass
        if 'world' in locals() and world is not None:
            world.stop()
            world.clear_instance()
        gc.collect()
        simulation_app.close()

if __name__ == "__main__":
    run_simulation()