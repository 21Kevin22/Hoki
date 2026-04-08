import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# 1. Isaac Sim 起動
from isaacsim import SimulationApp
config = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "hide_ui": True}
simulation_app = SimulationApp(config)

from pxr import Usd, UsdGeom, Gf, UsdLux, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

# ---------------------------------------------------------
# シーン構築 (円柱ボトル)
# ---------------------------------------------------------
def setup_environment(world):
    stage = get_current_stage()
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2500)
    
    # ボトルの生成 (18cm x 3cm)
    bottle_configs = [
        ("/World/Bottle_R", [0.6,  0.2, 0.0], (1, 0, 0)),
        ("/World/Bottle_G", [0.6,  0.0, 0.0], (0, 1, 0)),
        ("/World/Bottle_B", [0.6, -0.2, 0.0], (0, 0, 1))
    ]
    
    for path, pos, color in bottle_configs:
        bottle_prim = UsdGeom.Cylinder.Define(stage, path)
        bottle_prim.CreateHeightAttr(0.18)
        bottle_prim.CreateRadiusAttr(0.03)
        XFormPrim(path).set_world_poses(positions=np.array([pos]))
        # 物理有効化
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))

    # ロボットアームのロード
    assets_root = get_assets_root_path()
    franka_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_path, prim_path="/World/Franka")
    franka = Articulation("/World/Franka", name="franka_robot")
    world.scene.add(franka)
    return franka

# ---------------------------------------------------------
# メイン
# ---------------------------------------------------------
def main():
    world = World(stage_units_in_meters=1.0)
    output_path = Path("/home/ubuntu/slocal/evaluation/RL/bottle_sorting_v12.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    franka = setup_environment(world)

    # カメラとアノテータの設定
    # 斜め上から全体を見渡す位置
    cam = rep.create.camera(position=(2.0, 1.2, 1.0), look_at=(0.5, 0.0, 0.1))
    rp = rep.create.render_product(cam, (1280, 720))
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_ann.attach([rp])

    # OpenCV VideoWriter の準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (1280, 720))

    world.reset()
    
    print(f"=== 録画開始: {output_path} ===")
    
    # 300フレーム（約10秒）録画する
    for i in range(300):
        # 1. 物理と描画を更新
        world.step(render=True)
        # 2. Replicatorのレンダリングパイプラインを同期
        rep.orchestrator.step()
        
        # 3. 画像データの取得
        rgb_data = rgb_ann.get_data()
        
        if rgb_data is not None and len(rgb_data) > 0:
            # RGBA -> BGR に変換して OpenCV で書き込み
            frame = cv2.resize(rgb_data[:, :, :3], (1280, 720))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            if i % 30 == 0:
                print(f"録画中... {i}/300 フレーム完了")
        else:
            print(f"⚠️ フレーム {i} の取得に失敗しました")

    # 後処理
    video_writer.release()
    print(f"=== 終了: 動画が正常に保存されました ===")
    print(f"ファイルサイズ: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    simulation_app.close()

if __name__ == "__main__":
    main()