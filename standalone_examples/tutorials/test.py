import os
import sys
import time
import numpy as np
import subprocess
import gc

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
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.prims import create_prim

# 【重要】色設定のためにUSDの直接操作用ライブラリを追加
import omni.usd
from pxr import UsdGeom, Gf, Vt

# アセットパス取得
from omni.isaac.core.utils.nucleus import get_assets_root_path

# 画像保存用
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

# ---------------------------------------------------------
# 3. 保存先の設定
# ---------------------------------------------------------
output_dir = os.path.join(os.getcwd(), "franka_carter_shelf_fix4")
rgb_dir = os.path.join(output_dir, "rgb")
video_path = os.path.join(output_dir, "franka_carter_shelf_complete.mp4")
os.makedirs(rgb_dir, exist_ok=True)

print(f"=== 保存先: {output_dir} ===")

# ---------------------------------------------------------
# 4. ヘルパー関数
# ---------------------------------------------------------
def get_verified_assets_path():
    print("Connecting to Nucleus to find assets...")
    max_retries = 5
    for i in range(max_retries):
        path = get_assets_root_path()
        if path is not None:
            print(f"✅ Assets root found: {path}")
            return path
        print(f"Warning: Could not find assets root (Attempt {i+1}/{max_retries})...")
        time.sleep(2.0)
        simulation_app.update()
    return None

# 【修正】安全に色を設定する関数
def set_prim_color(prim_path, color_rgb):
    """
    create_primのattributes引数を使わず、USD API経由で確実に色を設定する
    color_rgb: (r, g, b) 0.0-1.0
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        # UsdGeom.Gprim として扱って displayColor を設定
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            # Gf.Vec3f に変換して配列としてセット
            color_vec = Gf.Vec3f(*color_rgb)
            gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([color_vec]))

assets_root_path = get_verified_assets_path()
if assets_root_path is None:
    print("【エラー】Isaac Simのアセットサーバーが見つかりません。")
    os._exit(1)

# ---------------------------------------------------------
# 5. シーン構築
# ---------------------------------------------------------
try:
    print("Setting up the world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    set_camera_view(
        eye=[5.0, 0.0, 1.5], 
        target=[0.00, 0.00, 1.00], 
        camera_prim_path="/OmniverseKit_Persp"
    )

    # --- Franka (Arm) ---
    franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    print(f"Loading Franka from: {franka_usd_path}")
    add_reference_to_stage(usd_path=franka_usd_path, prim_path="/World/Arm")
    arm = Articulation("/World/Arm", name="my_arm")

    # --- Nova Carter (Car) ---
    carter_usd_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
    print(f"Loading Carter from: {carter_usd_path}")
    add_reference_to_stage(usd_path=carter_usd_path, prim_path="/World/Car")
    car = Articulation("/World/Car", name="my_car")

    # 初期配置
    arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
    car.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

    # --- 棚とボトルの追加 ---
    print("Creating Shelf and Bottles...")
    
    # 1. 棚の親Xform作成
    create_prim(
        prim_path="/World/Shelf",
        prim_type="Xform",
        position=np.array([1.0, 1.0, 1.0]),
        scale=np.array([1.0, 1.0, 1.0])
    )

    # 2. 棚板 (Cube) 作成
    board_path = "/World/Shelf/Board"
    create_prim(
        prim_path=board_path,
        prim_type="Cube",
        position=np.array([0.0, 0.0, 0.0]),
        scale=np.array([0.3, 0.8, 0.05])
        # attributes={"displayColor": ...} はここで削除！
    )
 
    set_prim_color(board_path, (0.4, 0.2, 0.1))

    # 3. ボトル5本作成
    y_offsets = np.linspace(-0.25, 0.25, 5)
    bottle_colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1)]
    
    for i in range(5):
        b_path = f"/World/Bottle_{i}"
        b_pos = np.array([0.8, 1.0 + y_offsets[i], 0.325])
        
        create_prim(
            prim_path=b_path,
            prim_type="Cylinder",
            position=b_pos,
            scale=np.array([0.035, 0.035, 0.2])
            # attributes は削除
        )
        # 【修正】あとから色を設定
        set_prim_color(b_path, bottle_colors[i])

    # ワールドリセット
    simulation_app.update()
    my_world.reset()

except Exception as e:
    import traceback
    print("\n" + "!"*60)
    print("【シーン構築エラー】詳細:")
    traceback.print_exc()
    print("!"*60 + "\n")
    os._exit(1)

# ---------------------------------------------------------
# 6. シミュレーション実行 & 録画
# ---------------------------------------------------------
print("=== シミュレーション開始 ===")
viewport_api = get_active_viewport()
global_step = 0

try:
    for i in range(4):
        print(f"Running cycle: {i}")
        
        # 動作指令
        if i == 1 or i == 3:
            arm.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])
            car.set_joint_velocities([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        if i == 2:
            arm.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            car.set_joint_velocities([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        for j in range(100):
            my_world.step(render=True)
            simulation_app.update()

            file_name = f"rgb_{global_step:04d}.png"
            capture_viewport_to_file(viewport_api, os.path.join(rgb_dir, file_name))
            global_step += 1

    print("=== シミュレーション終了 ===")

except Exception as e:
    print(f"Runtime Error: {e}")

# ---------------------------------------------------------
# 7. 動画生成
# ---------------------------------------------------------
if os.path.exists(rgb_dir) and len(os.listdir(rgb_dir)) > 0:
    print("動画生成中 (FFmpeg)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", os.path.join(rgb_dir, "rgb_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"🎥 動画保存完了: {video_path}")
    except Exception as e:
        print(f"FFmpeg Error: {e}")
else:
    print("警告: 画像が保存されていません。")

print("Force exiting...")
sys.stdout.flush()
os._exit(0)