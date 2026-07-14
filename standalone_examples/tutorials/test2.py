import os
import sys
import time
import numpy as np
import subprocess

# ---------------------------------------------------------
# 1. SimulationAppの初期化
# ---------------------------------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLighting"
})

# ---------------------------------------------------------
# 2. ライブラリのインポート
# ---------------------------------------------------------
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation  # 登録用にインポート
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.prims import create_prim

# 物理エンジン直接制御用
from omni.isaac.dynamic_control import _dynamic_control

import omni.usd
from pxr import UsdGeom, Gf, Vt
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

# ---------------------------------------------------------
# 3. 保存先の設定
# ---------------------------------------------------------
home_dir = os.path.expanduser("~")
output_dir = os.path.join(home_dir, "isaac_sim_output", "robot_fix_v2")
rgb_dir = os.path.join(output_dir, "rgb")
video_path = os.path.join(output_dir, "simulation_video.mp4")

os.makedirs(rgb_dir, exist_ok=True)
print(f"=== 保存先: {output_dir} ===")

# ---------------------------------------------------------
# 4. ヘルパー関数
# ---------------------------------------------------------
def set_prim_color(prim_path, color_rgb):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            color_vec = Gf.Vec3f(*color_rgb)
            gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([color_vec]))

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Error: Could not find assets root path")
    sys.exit()

# ---------------------------------------------------------
# 5. シーン構築
# ---------------------------------------------------------
print("シーンを構築中...")
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# カメラ位置
set_camera_view(eye=[3.0, 2.5, 1.5], target=[0.0, 0.0, 0.5])

# --- Franka (ロボットアーム) ---
franka_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=franka_path, prim_path="/World/Arm")
# 【重要】物理エンジンに登録するために Articulation クラスでラップする
# これをしておくと、Dynamic Controlが見つけやすくなります
franka = Articulation("/World/Arm", name="franka")
my_world.scene.add(franka)

# --- Nova Carter (移動ロボット) ---
carter_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
add_reference_to_stage(usd_path=carter_path, prim_path="/World/Car")
# 【重要】同様にCarも登録
car = Articulation("/World/Car", name="car")
my_world.scene.add(car)

# --- 棚とボトル ---
shelf_pos = np.array([1.2, 0.0, 0.5])
board_size = np.array([0.4, 0.8, 0.05])

create_prim("/World/Shelf", "Xform", position=shelf_pos)
create_prim("/World/Shelf/Board", "Cube", scale=board_size)
set_prim_color("/World/Shelf/Board", (0.3, 0.2, 0.1))

board_top_z = shelf_pos[2] + (board_size[2] / 2.0)
bottle_h = 0.2
bottle_z = board_top_z + (bottle_h / 2.0)
y_offsets = np.linspace(-0.25, 0.25, 5)
colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1)]

for i, y_off in enumerate(y_offsets):
    b_path = f"/World/Bottle_{i}"
    create_prim(b_path, "Cylinder", 
                position=np.array([shelf_pos[0], shelf_pos[1] + y_off, bottle_z]),
                scale=np.array([0.03, 0.03, bottle_h]))
    set_prim_color(b_path, colors[i])

# 環境リセット（ここでロボットが初期化されます）
my_world.reset()

# ---------------------------------------------------------
# 6. ウォーミングアップ
# ---------------------------------------------------------
print("レンダラー待機中 (ウォーミングアップ)...")
for _ in range(30):
    my_world.step(render=True)

# ---------------------------------------------------------
# 7. シミュレーション実行 (修正版)
# ---------------------------------------------------------
print("=== シミュレーション開始 ===")
viewport_api = get_active_viewport()

# Dynamic Control インターフェース取得
dc = _dynamic_control.acquire_dynamic_control_interface()

target_arm_pos = np.array([-1.5, 0.5, 0.0, -1.5, 0.0, 1.8, 0.4, 0.04, 0.04], dtype=np.float32)
initial_arm_pos = np.zeros(9, dtype=np.float32)
move_vel = np.array([1.0, 1.0], dtype=np.float32)
stop_vel = np.array([0.0, 0.0], dtype=np.float32)

# ハンドル変数の初期化
arm_handle = _dynamic_control.INVALID_HANDLE
car_handle = _dynamic_control.INVALID_HANDLE

for i in range(400):
    my_world.step(render=True)
    
    # ハンドルの再取得を試みる（もし見つかっていなければ）
    if arm_handle == _dynamic_control.INVALID_HANDLE:
        arm_handle = dc.get_articulation("/World/Arm")
    
    if car_handle == _dynamic_control.INVALID_HANDLE:
        car_handle = dc.get_articulation("/World/Car")

    # --- 制御ロジック ---
    # ハンドルが有効な場合のみ指令を送る
    if arm_handle != _dynamic_control.INVALID_HANDLE:
        if 100 <= i < 300:
            dc.set_articulation_dof_position_targets(arm_handle, target_arm_pos)
        else:
            dc.set_articulation_dof_position_targets(arm_handle, initial_arm_pos)
            
    if car_handle != _dynamic_control.INVALID_HANDLE:
        if 100 <= i < 300:
            dc.set_articulation_dof_velocity_targets(car_handle, move_vel)
        else:
            dc.set_articulation_dof_velocity_targets(car_handle, stop_vel)
    
    # --- 画像保存ロジック (最重要修正) ---
    # ロボットが見つかっても見つからなくても、必ず撮影する！
    if i % 2 == 0:
        file_name = f"rgb_{i//2:04d}.png"
        capture_viewport_to_file(viewport_api, os.path.join(rgb_dir, file_name))
        
        if i % 50 == 0:
            # 進捗を表示（ハンドルが見つかっているかも確認）
            arm_status = "OK" if arm_handle != _dynamic_control.INVALID_HANDLE else "None"
            car_status = "OK" if car_handle != _dynamic_control.INVALID_HANDLE else "None"
            print(f"Frame {i}/400: Arm={arm_status}, Car={car_status}")

print("=== シミュレーション終了 ===")

# ---------------------------------------------------------
# 8. 動画生成
# ---------------------------------------------------------
time.sleep(2.0)

if os.path.exists(rgb_dir) and len(os.listdir(rgb_dir)) > 0:
    print("動画生成中 (FFmpeg)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(rgb_dir, "rgb_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"\n🎉 動画作成成功！\n場所: {video_path}")
    except Exception as e:
        print(f"FFmpegエラー: {e}")
else:
    print("❌ 画像が保存されませんでした。")

simulation_app.close()