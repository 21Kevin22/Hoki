import os
import sys
import numpy as np
import json
import gc

# 1. SimulationApp の初期化
from isaacsim import SimulationApp
CONFIG = {
    "headless": True, 
    "renderer": "RayTracedLighting",
    "width": 1024,
    "height": 1024
}
simulation_app = SimulationApp(launch_config=CONFIG)

# 2. コアライブラリのインポート
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Gf # 3D座標計算用

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.nucleus import get_assets_root_path

# 3. 保存先と設定
output_dir = os.path.join(os.getcwd(), "replicator_scenegraph_results")
os.makedirs(output_dir, exist_ok=True)

# --- アフォーダンス定義 (ナレッジベース) ---
# 物体のクラス名に対して「何ができるか」を定義します
AFFORDANCE_MAP = {
    "bottle": ["graspable", "liftable", "pourable", "containable"],
    "franka": ["manipulator", "fixed_base"],
    "unknown": []
}

def get_prim_world_position(prim_path):
    """
    USD APIを使用して、指定されたPrimの正確なワールド座標(x, y, z)を取得します。
    """
    if not prim_path or prim_path == "N/A":
        return None
    
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    
    # 変換行列(Transform Matrix)を取得
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    
    # Gf.Vec3d を list に変換して返す
    return [round(translation[0], 4), round(translation[1], 4), round(translation[2], 4)]

def setup_scene():
    print("Setting up the world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    assets_root_path = get_assets_root_path()

    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", 
        prim_path="/World/Arm"
    )

    y_offsets = np.linspace(-0.25, 0.25, 5)
    for i in range(5):
        b_path = f"/World/Bottle_{i}"
        create_prim(
            prim_path=b_path, 
            prim_type="Cylinder", 
            position=np.array([1.0, 1.0 + y_offsets[i], 0.4]), 
            scale=np.array([0.03, 0.03, 0.15])
        )
        rep.modify.semantics([('class', 'bottle')], rep.get.prim_at_path(b_path))
    
    return my_world

def main():
    my_world = setup_scene()

    # 4. Replicator 設定
    camera = rep.create.camera(position=(4.0, 0.0, 2.0), look_at=(1.0, 1.0, 0.4))
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    # BBox 2D Annotator
    bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    bbox_annotator.attach([render_product])
    
    # 【追加】3D BBox Annotator (補助的に使用)
    # これを使うと、primPathが取得しやすくなる場合があります
    bbox_3d_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
    bbox_3d_annotator.attach([render_product])

    print(f"=== Simulation Start ===")
    my_world.reset()
    rep.orchestrator.run()

    try:
        for i in range(20): # テストのためフレーム数を短縮
            my_world.step(render=True)
            rep.orchestrator.step()
            
            # 両方のアノテーターからデータを取得
            bbox_2d_data = bbox_annotator.get_data()
            bbox_3d_data = bbox_3d_annotator.get_data()
            
            if bbox_2d_data is None or "data" not in bbox_2d_data or len(bbox_2d_data["data"]) == 0:
                simulation_app.update()
                continue
            
            # シーングラフ構築用データリスト
            scene_graph_nodes = []
            
            id_to_labels = bbox_2d_data["info"]["idToLabels"]
            
            # データ解析
            for obj in bbox_2d_data["data"]:
                sem_id = str(obj["semanticId"])
                label_info = id_to_labels.get(sem_id, {})
                label = label_info.get("class", "unknown") if isinstance(label_info, dict) else "unknown"
                
                # Prim Path の取得 (N/A 対策)
                p_path = str(obj["primPath"]) if "primPath" in obj.dtype.names else "N/A"
                
                # 3D位置の取得 (USDから直接計算)
                # primPathが取れている場合はそれを使用、取れていない場合はスキップまたは推論
                world_pos = [0.0, 0.0, 0.0]
                if p_path != "N/A":
                    pos = get_prim_world_position(p_path)
                    if pos:
                        world_pos = pos
                
                # アフォーダンスの注入
                affordances = AFFORDANCE_MAP.get(label, [])

                # ノード情報の作成
                node = {
                    "id": p_path if p_path != "N/A" else f"unknown_{sem_id}",
                    "label": label,
                    "properties": {
                        "position_3d": world_pos, # [x, y, z]
                        "affordance": affordances, # ["graspable", ...]
                        "bbox_2d": [int(obj["x_min"]), int(obj["y_min"]), int(obj["x_max"]), int(obj["y_max"])]
                    }
                }
                scene_graph_nodes.append(node)

            # JSON保存 (フレーム情報を含むシーングラフ形式)
            graph_data = {
                "frame": i,
                "timestamp": i * (1.0/60.0), # 仮のタイムスタンプ
                "nodes": scene_graph_nodes
            }

            with open(os.path.join(output_dir, f"scene_graph_{i:04d}.json"), 'w') as f:
                json.dump(graph_data, f, indent=4)
            
            if i % 10 == 0:
                print(f"Processed frame {i}...")

    except Exception as e:
        print(f"Error during execution: {e}")

    finally:
        print("Starting shutdown sequence...")
        try:
            rep.orchestrator.stop()
        except Exception:
            pass

        for _ in range(5):
            simulation_app.update()

        print("Closing SimulationApp...")
        simulation_app.close()
        print(f"Finished. Results: {output_dir}")

if __name__ == "__main__":
    main()