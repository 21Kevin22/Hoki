import os
import sys
import numpy as np
import json
import time

# ---------------------------------------------------------
# 1. SimulationApp の初期化
# ---------------------------------------------------------
from isaacsim import SimulationApp
CONFIG = {
    "headless": True, 
    "renderer": "RayTracedLighting",
    "width": 1024,
    "height": 1024
}
simulation_app = SimulationApp(launch_config=CONFIG)

# ---------------------------------------------------------
# 2. ライブラリのインポート
# ---------------------------------------------------------
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import create_prim, get_prim_at_path
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.semantics import add_update_semantics

# ---------------------------------------------------------
# 3. 設定と定義
# ---------------------------------------------------------
output_dir = os.path.join(os.getcwd(), "scene_graph_final")
os.makedirs(output_dir, exist_ok=True)
print(f"=== 保存先ディレクトリ: {output_dir} ===")

# アフォーダンス定義
AFFORDANCE_MAP = {
    "bottle": ["graspable", "liftable", "pourable", "containable"],
    "shelf": ["support", "static", "storage"],
    "manipulator": ["active", "grasping", "moving"],
    "mobile_robot": ["locomotion", "transport", "active"],
    "unknown": []
}

# 監視対象のオブジェクトリスト（ここに登録したものの3D座標を追跡します）
TARGET_OBJECTS = {
    "/World/Arm": "manipulator",
    "/World/Car": "mobile_robot",
    "/World/Shelf/Board": "shelf",
    "/World/Bottle_0": "bottle",
    "/World/Bottle_1": "bottle",
    "/World/Bottle_2": "bottle",
    "/World/Bottle_3": "bottle",
    "/World/Bottle_4": "bottle",
}

def get_prim_world_position(prim_path):
    """USD APIを使用して、指定されたPrimの正確なワールド座標(x, y, z)を取得します。"""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    try:
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        return [round(translation[0], 4), round(translation[1], 4), round(translation[2], 4)]
    except:
        return None

# ---------------------------------------------------------
# 4. シーン構築関数
# ---------------------------------------------------------
def setup_scene():
    print("シーンを構築中...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    assets_root_path = get_assets_root_path()

    # --- Franka ---
    franka_path = "/World/Arm"
    add_reference_to_stage(usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", prim_path=franka_path)
    add_update_semantics(get_prim_at_path(franka_path), "manipulator")

    # --- Nova Carter ---
    carter_path = "/World/Car"
    add_reference_to_stage(usd_path=assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd", prim_path=carter_path)
    add_update_semantics(get_prim_at_path(carter_path), "mobile_robot")

    # --- 棚 ---
    shelf_pos = np.array([1.2, 0.0, 0.5])
    shelf_path = "/World/Shelf"
    board_path = shelf_path + "/Board"
    create_prim(shelf_path, "Xform", position=shelf_pos)
    create_prim(board_path, "Cube", scale=np.array([0.4, 0.8, 0.05]))
    add_update_semantics(get_prim_at_path(board_path), "shelf")

    # --- ボトル ---
    board_top_z = shelf_pos[2] + (0.05 / 2.0)
    bottle_h = 0.2
    bottle_z = board_top_z + (bottle_h / 2.0)
    y_offsets = np.linspace(-0.25, 0.25, 5)

    for i in range(5):
        b_path = f"/World/Bottle_{i}"
        create_prim(
            prim_path=b_path, 
            prim_type="Cylinder", 
            position=np.array([shelf_pos[0], shelf_pos[1] + y_offsets[i], bottle_z]), 
            scale=np.array([0.03, 0.03, bottle_h])
        )
        add_update_semantics(get_prim_at_path(b_path), "bottle")
    
    return my_world

# ---------------------------------------------------------
# 5. メイン実行ループ
# ---------------------------------------------------------
def main():
    try:
        my_world = setup_scene()
        my_world.reset()

        print("レンダラー待機中 (20フレーム)...")
        for _ in range(20):
            my_world.step(render=True)

        # --- Replicator 設定 ---
        # 俯瞰視点
        cam_prim = rep.create.camera(position=(3.5, 2.5, 1.5), look_at=(1.2, 0.0, 0.5))
        render_product = rep.create.render_product(cam_prim, resolution=(1024, 1024))

        # 2D BBoxのみを使用（primPathエラーを回避するため、これに依存しすぎない）
        bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        bbox_annotator.attach([render_product])

        print("=== シーングラフ生成開始 ===")
        rep.orchestrator.run()

        # 20フレーム実行
        for i in range(20):
            my_world.step(render=True)
            
            # 1. 画像認識データ (Replicator)
            bbox_data = bbox_annotator.get_data()
            detected_objects = []
            
            if bbox_data is not None and "data" in bbox_data:
                id_to_labels = bbox_data["info"]["idToLabels"]
                
                for obj in bbox_data["data"]:
                    # Semantic IDからラベル名を取得
                    sem_id = str(obj["semanticId"])
                    class_name = "unknown"
                    if sem_id in id_to_labels:
                        labels = id_to_labels[sem_id]
                        if isinstance(labels, dict) and 'class' in labels:
                            class_name = labels['class']
                    
                    # 2D BBox情報のみ保存 (パス取得はしない)
                    detected_objects.append({
                        "label": class_name,
                        "bbox_2d": [int(obj["x_min"]), int(obj["y_min"]), int(obj["x_max"]), int(obj["y_max"])],
                        "semantic_id": int(obj["semanticId"])
                    })

            # 2. ワールド正解データ (USD API)
            # ここで確実に3D座標を取得します
            world_state = []
            for path, label in TARGET_OBJECTS.items():
                pos = get_prim_world_position(path)
                if pos: # 存在する場合のみ
                    world_state.append({
                        "id": path,
                        "label": label,
                        "position_3d": pos,
                        "affordance": AFFORDANCE_MAP.get(label, [])
                    })

            # 3. データの統合と保存
            graph_data = {
                "frame_id": i,
                "timestamp": time.time(),
                "visual_detection": { # カメラで見えているもの
                    "count": len(detected_objects),
                    "objects": detected_objects
                },
                "world_knowledge": { # 実際に存在するものと座標
                    "count": len(world_state),
                    "objects": world_state
                }
            }

            file_name = f"scene_graph_{i:04d}.json"
            save_path = os.path.join(output_dir, file_name)
            
            with open(save_path, 'w') as f:
                json.dump(graph_data, f, indent=4)
            
            print(f"Frame {i}: Saved JSON with {len(world_state)} 3D objects.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            rep.orchestrator.stop()
        except:
            pass
        simulation_app.close()
        print("Simulation Finished.")

if __name__ == "__main__":
    main()