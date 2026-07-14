from isaacsim import SimulationApp
import os

# ヘッドレス・レンダリング有効設定
simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"}) 

import sys
import carb
import numpy as np
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# --- 出力先の設定 ---
output_folder = os.path.expanduser("~/slocal/out_video")
os.system(f"rm -rf {output_folder}/*")
os.makedirs(output_folder, exist_ok=True)

# --- Replicatorの設定 ---
render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
writer = rep.writers.get("BasicWriter")
writer.initialize(output_dir=output_folder, rgb=True)
writer.attach([render_product])

# --- シミュレーション準備 ---
assets_root_path = get_assets_root_path()
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
set_camera_view(eye=[5.0, 2.0, 1.5], target=[0.0, 0.0, 1.0])

# Frankaアーム追加
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")
arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")

# Carter台車追加
asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Car")
car = Articulation(prim_paths_expr="/World/Car", name="my_car")

arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
car.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

my_world.reset()

# 滑らかな動きのための目標位置
target_moving = np.array([-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04])
target_stopping = np.zeros(9)

print("Starting Simulation (Smooth Motion)...")
for i in range(4):
    print(f"running cycle: {i}")
    
    # ターゲットの決定
    if i == 1 or i == 3:
        target_pos = target_moving
        car_vel = np.array([1.0] * 7)
    else:
        target_pos = target_stopping
        car_vel = np.zeros(7)

    for j in range(100):
        # 【重要】瞬間移動(set_joint_positions)ではなく
        # 目標位置(set_joint_position_targets)を設定することで、物理的に滑らかに動かします
        arm.set_joint_position_targets(target_pos)
        car.set_joint_velocities([car_vel])
        
        my_world.step(render=True)
        rep.orchestrator.step() 

# --- 動画変換 ---
print("Wait for assets writing...")
rep.orchestrator.wait_until_complete()
writer.detach()

print("Encoding video with FFmpeg...")
os.system(f"cd {output_folder} && ffmpeg -y -framerate 30 -pattern_type glob -i 'rgb_*.png' -c:v libx264 -pix_fmt yuv420p robot_smooth_final.mp4")

print(f"\nSUCCESS! Video saved at: {output_folder}/robot_smooth_final.mp4")

# クラッシュを回避するため強制終了
os._exit(0)