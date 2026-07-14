import os
import sys

# ---------------------------------------------------------
# 1. SimulationAppの初期化 (Docker向けにHeadless=True)
# ---------------------------------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np
import subprocess
import gc
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

# ---------------------------------------------------------
# 動画保存用の設定（画像保存 + ffmpeg）
# ---------------------------------------------------------
output_dir = os.path.join(os.getcwd(), "robot_demo_output")
rgb_dir = os.path.join(output_dir, "rgb")
os.makedirs(rgb_dir, exist_ok=True)
video_path = os.path.join(output_dir, "robot_demo.mp4")

print(f"=== 保存先ディレクトリ: {output_dir} ===")

# ---------------------------------------------------------
# シーンの準備
# ---------------------------------------------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# カメラ位置の設定
set_camera_view(
    eye=[5.0, 0.0, 1.5], 
    target=[0.00, 0.00, 1.00], 
    camera_prim_path="/OmniverseKit_Persp"
)

# Franka (ロボットアーム) の追加
asset_path_franka = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path_franka, prim_path="/World/Arm")
arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")

# Nova Carter (移動ロボット) の追加
asset_path_carter = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
add_reference_to_stage(usd_path=asset_path_carter, prim_path="/World/Car")
car = Articulation(prim_paths_expr="/World/Car", name="my_car")

# 初期位置の設定（衝突回避）
arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
car.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

my_world.reset()

# ---------------------------------------------------------
# シミュレーションループ & 録画
# ---------------------------------------------------------
print("=== シミュレーション開始 ===")
viewport_api = get_active_viewport()
global_step_count = 0

# 合計フレーム数 (4サイクル * 100ステップ = 400フレーム)
total_cycles = 4
steps_per_cycle = 100

for i in range(total_cycles):
    print(f"Running cycle: {i}")
    
    # 動作指令
    if i == 1 or i == 3:
        print(" -> Moving")
        arm.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])
        car.set_joint_velocities([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    if i == 2:
        print(" -> Stopping")
        arm.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        car.set_joint_velocities([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # 物理ステップ & レンダリング
    for j in range(steps_per_cycle):
        my_world.step(render=True)
        simulation_app.update() # 描画更新

        # 画像保存
        file_name = f"rgb_{global_step_count:04d}.png"
        capture_viewport_to_file(viewport_api, os.path.join(rgb_dir, file_name))
        global_step_count += 1

        # カーターの状態表示（サイクル3のみ）
        if i == 3 and j % 50 == 0: # ログが多すぎないように間引く
            car_joint_positions = car.get_joint_positions()
            print(f"Car joints: {car_joint_positions[0][:3]}...") # 見やすく先頭だけ表示

print("=== シミュレーション終了 ===")

# ---------------------------------------------------------
# 後処理 & 動画生成
# ---------------------------------------------------------
# 環境を閉じる
try:
    if hasattr(my_world, "stop"): my_world.stop()
    del my_world
    del arm
    del car
    gc.collect()
except:
    pass

# ffmpegで動画化
if os.path.exists(rgb_dir) and len(os.listdir(rgb_dir)) > 0:
    print("動画を生成中...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "30",
        "-i", os.path.join(rgb_dir, "rgb_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"🎥 動画保存完了: {video_path}")
    except Exception as e:
        print(f"ffmpegエラー: {e}")

# Dockerフリーズ回避のための強制終了
print("Force exiting...")
os._exit(0)