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
OPENAI_API_KEY = "<OPENAI_API_KEY>" 
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

GEMINI_API_KEY = "AIzaSyCsmmdOaLo7hdOXyyneRLA5kgQHBm516eQ" 
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

TARGET_MODEL = "gpt-4o"  # "gemini-2.0-flash"  # "gpt-4o"  # "gemini-1.5-pro"  # "gpt-3.5-turbo"

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
REPLAN_LOG_PATH = OUTPUT_DIR / "replan_history.json"

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
# 1. 差分検知＆記録ロジック
# =========================================================
def extract_facts(pddl_text: str) -> set:
    facts = re.findall(r"\([a-z0-9_][a-z0-9_\s-]*\)", pddl_text.lower())
    reserved = {"and", "not", "or", "forall", "exists", "define", "problem", "domain", "init", "goal", "objects"}
    return {f for f in facts if f.strip("()").split()[0] not in reserved}

def get_pddl_diff_summary(ref_pddl: str, cur_pddl: str) -> str:
    ref_facts = extract_facts(ref_pddl)
    cur_facts = extract_facts(cur_pddl)
    missing = ref_facts - cur_facts
    extra = cur_facts - ref_facts
    
    summary = []
    if missing: summary.append(f"Missing facts: {', '.join(sorted(list(missing)))}")
    if extra: summary.append(f"Extra facts: {', '.join(sorted(list(extra)))}")
    return "\n".join(summary) if summary else "No differences detected."

TARGET_PDDL = """
(define (problem clean_task)
    (:domain clean_mugs)
    (:init
        (item_at mug_0 basket)
        (item_at mug_1 basket)
        (item_at mug_2 basket)
        (item_at mug_3 basket)
        (item_at mug_4 basket)
    )
)
"""

class ReplanLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        if not self.filepath.exists():
            with open(self.filepath, "w") as f: json.dump([], f)
                
    def log(self, step, diff_summary, raw_plan, parsed_target_ids, model_name):
        with open(self.filepath, "r+") as f:
            try: data = json.load(f)
            except: data = []
            data.append({
                "step": step,
                "model": model_name,
                "timestamp": time.time(),
                "diff_summary": diff_summary,
                "raw_vlm_output": raw_plan,
                "parsed_target_ids": parsed_target_ids
            })
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

# =========================================================
# 2. Isaac Sim 起動
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

# =========================================================
# 3. ユーティリティクラス
# =========================================================
class VLMAnalyzer:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def encode_image_base64(self, path):
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

    def _call_api(self, prompt, image_paths, max_retries=5):
        for attempt in range(max_retries):
            try:
                # --- OpenAI (GPT) ---
                if "gpt" in self.model_name.lower():
                    if not self.openai_api_key: return None
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.openai_api_key}"}
                    
                    content = [{"type": "text", "text": prompt}]
                    if image_paths:
                        for img_path in image_paths: 
                            content.append({
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/png;base64,{self.encode_image_base64(img_path)}",
                                    "detail": "low"  # 【追加】トークン消費と容量を抑えるため低解像度モードを指定
                                }
                            })
                    
                    payload = {
                        "model": self.model_name, 
                        "response_format": {"type": "json_object"}, 
                        "messages": [{"role": "user", "content": content}], 
                        "temperature": 0.0,
                        "max_tokens": 2048  # 【追加】出力が途切れないように最大トークンを明示
                    }
                    
                    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
                    resp.raise_for_status()
                    
                    # 堅牢なパースと原因究明用のエラー出力
                    resp_json = resp.json()
                    choices = resp_json.get("choices", [])
                    if not choices:
                        raise Exception(f"No choices returned. Response: {resp_json}")
                        
                    raw_text = choices[0].get("message", {}).get("content")
                    if not raw_text:
                        # 【修正】なぜNoneになったのか、APIからの生の返答（finish_reason等）を丸ごと表示する
                        raise Exception(f"GPT returned None. Reason: {choices[0].get('finish_reason')}. Full: {resp_json}")
                    
                    return json.loads(raw_text)

                # --- Google (Gemini) ---
                elif "gemini" in self.model_name.lower():
                    if not self.gemini_api_key: return None
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_api_key}"
                    headers = {'Content-Type': 'application/json'}
                    parts = [{"text": prompt}]
                    for img_path in image_paths: 
                        parts.append({"inline_data": {"mime_type": "image/png", "data": self.encode_image_base64(img_path)}})
                    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}}
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                    resp.raise_for_status()
                    
                    # 堅牢なパース処理
                    res_json = resp.json()
                    candidates = res_json.get("candidates", [])
                    if not candidates: 
                        raise Exception(f"No candidates returned. Response: {res_json}")
                    
                    content_dict = candidates[0].get("content", {})
                    content_parts = content_dict.get("parts", [])
                    if not content_parts or "text" not in content_parts[0]: 
                        raise Exception(f"No text content found. Finish Reason: {candidates[0].get('finishReason')}")
                        
                    raw_text = content_parts[0].get("text")
                    if not raw_text: 
                        raise Exception("Returned text is empty or None.")
                        
                    # Markdownタグ等の除去
                    raw_text = raw_text.strip()
                    if raw_text.startswith("```json"): raw_text = raw_text[7:-3].strip()
                    elif raw_text.startswith("```"): raw_text = raw_text[3:-3].strip()
                    
                    return json.loads(raw_text)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (attempt + 1) * 15
                    print(f"[Warning] API Rate Limit (429) Hit! Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else: 
                    print(f"[Warning] HTTP Error: {e.response.text}")
                    time.sleep(5)
            except Exception as e:
                print(f"[Warning] Request failed: {e}. Retrying...")
                time.sleep(5)
                
        return None

    def get_scene_graph(self, image_paths):
        prompt = """
        【前提条件（Context）】
        これは学術研究用の安全な3Dシミュレーション環境（Isaac Sim）の映像です。人間の顔やプライバシー、危険物は一切含まれていません。ロボットアームがマグカップを片付けるタスクを評価するためのものです。システムアシスタントとして、画像解析を安全に実行してください。

        あなたは高度な3D視覚推論システムです。提供された4枚の画像（俯瞰の3方向 ＋ ロボットアーム先端からの手元視点1方向）を統合して分析し、3Dシーングラフを構築してください。
        アームが物体を隠している場合でも、手元視点（wristカメラ）の画像を参考にして正確な関係性を推論してください。
        必ず以下の【厳密なJSONフォーマットとIDルール】に従って出力してください。

        【重要：IDの命名規則】
        画像内のオブジェクトは、色や種類に応じて以下の 'id' を必ず使用してください。
        - 赤色のマグカップ: "mug_0"
        - 緑色のマグカップ: "mug_1"
        - 青色のマグカップ: "mug_2"
        - 黄色のマグカップ: "mug_3"
        - 水色(シアン)のマグカップ: "mug_4"
        - カゴ: "basket"
        - 机: "table"
        - ロボットアーム: "panda"

        【要件】
        1. nodes: オブジェクトの属性。キーは "id", "category", "affordance" を持つこと。
           ※マグカップの場合、affordanceに "item_has_handle", "item_containable", "pickable" を含めること。
        2. edges: オブジェクト間の3D空間的関係。キーは必ず "subject", "predicate", "object" とすること。
           ※ predicate は "on", "inside", "grasped_by", "near" のいずれかを使用してください。

        出力は純粋なJSONのみとしてください。
        """
        try:
            res = self._call_api(prompt, image_paths)
            return res if res else {"nodes": [], "edges": []}
        except Exception as e: 
            print(f"[Graph Gen Error] {e}")
            return {"nodes": [], "edges": []}
            
    def replan_based_on_diff(self, current_pddl, target_pddl):
        diff_summary = get_pddl_diff_summary(target_pddl, current_pddl)
        if "Missing facts" not in diff_summary or "basket" not in diff_summary:
            return [], diff_summary, None
        prompt = f"Target state:\n{target_pddl}\nCurrent state:\n{current_pddl}\nDifferences:\n{diff_summary}\nOutput a JSON with a 'plan' key containing the list of target mug IDs (e.g., [\"mug_0\", \"mug_2\"]) that still need to be moved to the basket."
        start_time = time.time()
        result = self._call_api(prompt, [], max_retries=3)
        print(f"[Planner] Re-planning completed in {time.time() - start_time:.2f}s")
        plan = result.get("plan", []) if result else []
        return plan, diff_summary, result

class AccuracyEvaluator:
    @staticmethod
    def get_ground_truth(mugs, grasped_object, disposed_list):
        nodes = [{"id": "basket", "affordance": []}]
        edges = []
        for m in mugs:
            m_id = m["id"]
            nodes.append({"id": m_id, "affordance": ["pickable", "item_has_handle", "item_containable"]})
            if grasped_object == f"/World/{m_id}":
                 edges.append({"subject": m_id, "predicate": "grasped_by", "object": "panda"})
            elif m_id in disposed_list:
                 edges.append({"subject": m_id, "predicate": "inside", "object": "basket"})
            else:
                 edges.append({"subject": m_id, "predicate": "on", "object": "table"})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def evaluate(pred_graph, mugs, grasped_object, disposed_list):
        gt_graph = AccuracyEvaluator.get_ground_truth(mugs, grasped_object, disposed_list)
        
        gt_nodes = {n["id"]: set(n["affordance"]) for n in gt_graph["nodes"]}
        pred_nodes = {}
        for n in pred_graph.get("nodes", []):
            aff = n.get("affordance", [])
            if isinstance(aff, str): aff = [aff]
            pred_nodes[str(n.get("id", "")).lower()] = set([str(a).lower() for a in aff])
            
        node_matches = 0
        for nid, gt_aff in gt_nodes.items():
            if nid == "basket": continue
            p_aff = pred_nodes.get(nid, set())
            if not gt_aff and not p_aff: node_matches += 1.0
            elif gt_aff:
                score = sum(1 for g in gt_aff if any(g in p or p in g for p in p_aff))
                node_matches += score / len(gt_aff)
        node_acc = node_matches / len(mugs)
        
        gt_edges = {(e["subject"], e["predicate"], e["object"]) for e in gt_graph["edges"]}
        pred_edges_raw = pred_graph.get("edges", [])
        
        def norm_pred(p):
            p = str(p).lower()
            if "in" in p: return "inside"
            if "on" in p or "support" in p: return "on"
            if "grasp" in p or "hold" in p: return "grasped_by"
            return "near"
            
        pred_edges = {(str(e.get("subject", "")).lower(), norm_pred(e.get("predicate", "")), str(e.get("object", "")).lower()) for e in pred_edges_raw}
        
        edge_matches = 0
        for gt_sub, gt_pred, gt_obj in gt_edges:
            if any(gt_sub in p_sub and gt_pred == p_pred for p_sub, p_pred, p_obj in pred_edges): edge_matches += 1
        edge_acc = edge_matches / len(gt_edges) if gt_edges else 0.0
        
        return round(node_acc, 3), round(edge_acc, 3), gt_graph

class PDDLStateGenerator:
    @staticmethod
    def get_snapshot(controller, items_config, step_name, vlm_graph_data=None):
        lines = []
        lines.append(f"; --- PDDL Snapshot: {step_name} ---")
        lines.append("(define (problem clean_task)")
        lines.append("    (:domain clean_mugs)")
        item_ids = " ".join([b["id"] for b in items_config])
        lines.append(f"    (:objects panda - agent table bin - room {item_ids} basket - item)")
        lines.append("    (:init")
        loc = "bin" if controller.grasped_object is None and "BASKET" in str(controller.current_pose) else "table"
        lines.append(f"        (agent_at panda {loc})")
        
        vlm_nodes = {node.get("id", ""): node for node in vlm_graph_data.get("nodes", [])} if vlm_graph_data and "nodes" in vlm_graph_data else {}
        for b in items_config:
            b_id = b["id"]
            affordances = vlm_nodes.get(b_id, {}).get("affordance", ["pickable", "item_has_handle", "item_containable"]) 
            if controller.grasped_object == f"/World/{b_id}":
                lines.append(f"        (agent_has_item panda {b_id})")
                lines.append(f"        (agent_loaded panda)")
            elif b_id in controller.disposed_list: 
                lines.append(f"        (item_at {b_id} basket)")
                lines.append(f"        (item_disposed {b_id})")
            else:
                lines.append(f"        (item_at {b_id} table)")
                if isinstance(affordances, list):
                    if "pickable" in affordances: lines.append(f"        (item_pickable {b_id})")
                    if "item_has_handle" in affordances: lines.append(f"        (item_has_handle {b_id})")
                    if "item_containable" in affordances: lines.append(f"        (item_containable {b_id})")
        lines.append("        (item_at basket bin)")
        lines.append("        (neighbor table bin)")
        lines.append("        (neighbor bin table)")
        lines.append("    )")
        lines.append(")")
        content = "\n".join(lines)
        with open(PDDL_LOG_PATH, "a") as f:
            f.write(content + "\n\n")
            f.flush()
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
        handle_path = f"{path}/Handle"
        handle_mesh = AssetBuilder.create_torus_mesh(stage, handle_path, major_r=0.04, minor_r=0.008)
        XFormPrim(handle_path).set_local_poses(translations=np.array([[0.05, 0.0, 0.07]]), orientations=np.array([[0.7071, 0, 0, 0.7071]]))
        UsdPhysics.CollisionAPI.Apply(handle_mesh.GetPrim())
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(handle_mesh.GetPrim())
        mesh_col.CreateApproximationAttr("convexDecomposition")
        AssetBuilder.apply_material(handle_mesh, stage, handle_path, color)

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

class RobotController:
    def __init__(self, arm: Articulation):
        self.arm = arm
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

    def _set_collision_enabled(self, target_path, enabled):
        stage = get_current_stage()
        for part in ["Body", "Handle"]:
            prim = stage.GetPrimAtPath(f"{target_path}/{part}")
            if prim.IsValid():
                col_api = UsdPhysics.CollisionAPI(prim)
                if col_api: col_api.GetCollisionEnabledAttr().Set(enabled)

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
        
        if not hasattr(self.arm, "_physics_view"): self.arm._physics_view = None
        if hasattr(self.arm, "initialize") and not self.arm.is_physics_handle_valid(): self.arm.initialize()
            
        for t in range(steps):
            r = (1.0 - np.cos(t / steps * np.pi)) / 2.0
            self.arm.set_joint_positions(start + (target - start) * r)
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
            self._set_collision_enabled(target_path, False)

        self.current_pose[7:] = 0.02
        if not hasattr(self.arm, "_physics_view"): self.arm._physics_view = None
        if hasattr(self.arm, "initialize") and not self.arm.is_physics_handle_valid(): self.arm.initialize()
            
        for _ in range(steps):
            self.arm.set_joint_positions(self.current_pose)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()

    def open_gripper(self, world, steps=15):
        if not self.grasped_object: return
        self.current_pose[7:] = 0.04
        if not hasattr(self.arm, "_physics_view"): self.arm._physics_view = None
        if hasattr(self.arm, "initialize") and not self.arm.is_physics_handle_valid(): self.arm.initialize()
            
        for _ in range(steps):
            self.arm.set_joint_positions(self.current_pose)
            world.step(render=True)
            self._update_grasped_object()
            rep.orchestrator.step()

        obj_id = self.grasped_object.split("/")[-1]
        self.disposed_list.append(obj_id)
        prim = get_current_stage().GetPrimAtPath(self.grasped_object)
        if prim.IsValid(): UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
        self._set_collision_enabled(self.grasped_object, True)
        self.grasped_object = None

def capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, event_name, last_time=None):
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
    with open(step_dir / "scene_graph.json", "w") as f: json.dump(graph_result, f, indent=2)
        
    node_acc, edge_acc, gt_graph = AccuracyEvaluator.evaluate(graph_result, MUGS, controller.grasped_object, controller.disposed_list)
    with open(step_dir / "gt_scene_graph.json", "w") as f: json.dump(gt_graph, f, indent=2)
        
    end_time = time.time()
    elapsed_generation = end_time - start_time
    elapsed_since_last = (end_time - last_time) if last_time else 0.0
    
    with open(METRICS_CSV_PATH, "a", newline="") as csv_file:
        csv.writer(csv_file).writerow([analyzer.model_name, event_name, round(elapsed_generation, 3), round(elapsed_since_last, 3), end_time])
        
    with open(ACCURACY_CSV_PATH, "a", newline="") as acc_file:
        csv.writer(acc_file).writerow([analyzer.model_name, event_name, node_acc, edge_acc])
        
    return end_time, graph_result

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
# 4. メインルーチン 
# =========================================================
def run_simulation():
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    RGB_DIR.mkdir(parents=True)
    SCENE_GRAPH_IMG_DIR.mkdir(parents=True)
    if PDDL_LOG_PATH.exists(): PDDL_LOG_PATH.unlink()
    
    with open(METRICS_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["model_name", "event_name", "vlm_processing_time_sec", "interval_since_last_sec", "timestamp"])
        
    with open(ACCURACY_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["model_name", "event_name", "node_accuracy", "edge_accuracy"])

    replan_logger = ReplanLogger(REPLAN_LOG_PATH)

    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2000)
    
    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/World/Franka")
    
    franka = Articulation("/World/Franka", name="franka")
    world.scene.add(franka)

    for cfg in MUGS:
        AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])
    AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

    # 従来の3カメラ
    cam_top   = rep.create.camera(position=CAMERA_POSITIONS["top"]["pos"], look_at=CAMERA_POSITIONS["top"]["look_at"])
    cam_left  = rep.create.camera(position=CAMERA_POSITIONS["left"]["pos"], look_at=CAMERA_POSITIONS["left"]["look_at"])
    cam_right = rep.create.camera(position=CAMERA_POSITIONS["right"]["pos"], look_at=CAMERA_POSITIONS["right"]["look_at"])
    
    # 【追加】アーム先端(手元)カメラ (Eye-in-Hand)
    cam_wrist = rep.create.camera(
        position=(0.0, 0.0, 0.05), # フランジから少しオフセット
        rotation=(0, 0, 0),        # 正面を向く
        parent="/World/Franka/panda_hand" # アームの手に追従
    )

    cam_main  = rep.create.camera(position=CAMERA_POSITIONS["main"]["pos"], look_at=CAMERA_POSITIONS["main"]["look_at"])

    rp_top   = rep.create.render_product(cam_top, (1280, 720))
    rp_left  = rep.create.render_product(cam_left, (1280, 720))
    rp_right = rep.create.render_product(cam_right, (1280, 720))
    rp_wrist = rep.create.render_product(cam_wrist, (1280, 720)) # 手元カメラ用
    rp_main  = rep.create.render_product(cam_main, (1280, 720))

    world.reset()
    world.play()
    for _ in range(5): world.step(render=True)
    if hasattr(franka, "initialize") and not franka.is_physics_handle_valid(): franka.initialize()

    controller = RobotController(franka)
    analyzer = VLMAnalyzer(model_name=TARGET_MODEL)
    writer_vlm = rep.WriterRegistry.get("BasicWriter")
    writer_main = rep.WriterRegistry.get("BasicWriter")
    writer_main.initialize(output_dir=str(RGB_DIR), rgb=True)
    writer_main.attach([rp_main])

    # 【変更】4枚の画像をVLMに送る
    view_configs = [(rp_top, "top"), (rp_left, "left"), (rp_right, "right"), (rp_wrist, "wrist")]
    last_eval_time, current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, "Step_0_Initial")
    current_pddl = PDDLStateGenerator.get_snapshot(controller, MUGS, "Initial State", vlm_graph_data=current_graph)

    step_count = 1
    max_steps = 10 

    try:
        while step_count <= max_steps:
            plan_steps, diff_summary, raw_vlm_out = analyzer.replan_based_on_diff(current_pddl, TARGET_PDDL)
            target_ids = []
            for item in plan_steps:
                raw_id = str(item.get("id", list(item.values())[0]) if isinstance(item, dict) and item else item)
                if raw_id.strip():
                    match = re.search(r'\d+', raw_id)
                    if match: target_ids.append(f"mug_{match.group()}")
            
            unique_ids = list(dict.fromkeys(target_ids))
            
            if diff_summary and diff_summary != "No differences detected.":
                replan_logger.log(step_count, diff_summary, raw_vlm_out, unique_ids, analyzer.model_name)
            
            if not unique_ids:
                print("\n[System] All target facts achieved. Task Completed!")
                break
                
            target_id = unique_ids[0]
            print(f"\n=== Executing Step {step_count}: Target {target_id} ===")
            
            cfg = next((m for m in MUGS if m["id"] == target_id), None)
            if not cfg: continue
                
            high_pose = cfg["pose"].replace("_LOW", "_HIGH")
            
            last_eval_time, current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, f"Step_{step_count}_1_Before_Move", last_eval_time)
            controller.move_to_pose(high_pose, world)
            controller.move_to_pose(cfg["pose"], world)
            
            last_eval_time, current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, f"Step_{step_count}_2_Before_Grasp", last_eval_time)
            controller.close_gripper(world, f"/World/{target_id}")
            controller.move_to_pose(high_pose, world)
            
            last_eval_time, current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, f"Step_{step_count}_3_After_Grasp", last_eval_time)
            controller.move_to_pose("BASKET_HIGH", world)
            
            last_eval_time, current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, f"Step_{step_count}_4_Before_Drop", last_eval_time)
            controller.open_gripper(world)
            controller.move_to_pose("HOME", world)
            
            last_eval_time, current_graph = capture_and_analyze_scene(world, view_configs, writer_vlm, analyzer, controller, f"Step_{step_count}_5_After_Drop", last_eval_time)
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
            
        del rp_top, rp_left, rp_right, rp_wrist, rp_main
        del cam_top, cam_left, cam_right, cam_wrist, cam_main
        del controller, franka, world
        gc.collect()
        simulation_app.close()

if __name__ == "__main__":
    run_simulation()