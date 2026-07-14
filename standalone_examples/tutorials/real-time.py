# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

# 画像のような高品質な描画のため、レンダリング設定を指定して開始
simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"}) 

import sys
import carb
import numpy as np
import omni.kit.recorder
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdLux, Gf, Sdf, UsdGeom

# --- 動画保存用の設定 ---
recorder = omni.kit.recorder.get_instance()
output_path = "C:/tmp/isaac_sim_video.mp4" 
fps = 30
width = 1280
height = 720

def start_recording(path, fps, w, h):
    options = recorder.create_recorder_options()
    options.output_path = path
    options.fps = fps
    options.width = w
    options.height = h
    recorder.start_recorder(options)
    print(f"録画を開始しました: {path}")

# preparing the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
stage = get_current_stage()

# --- [MODIFIED] お手本の質感を再現する照明と地面の設定 ---

# 1. 地面をデフォルトから「暗い紺色」に変更して高級感を出す
my_world.scene.add_default_ground_plane()
ground_prim = UsdGeom.Mesh.Get(stage, "/World/defaultGroundPlane/Reference_0/Environment/GroundPlane")
if ground_prim:
    # 地面の色を深い紺色に（反射が映えるように）
    ground_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.02, 0.05, 0.1)])

# 2. シネマチックな「青白い」環境光 (DomeLight)
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/CinematicDomeLight"))
dome.CreateIntensityAttr(1000)
dome.CreateColorAttr(Gf.Vec3f(0.7, 0.8, 1.0)) # 涼しげな青白い光

# 3. 強い影を作る「斜め上からの」メインライト (DistantLight)
dist = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/MainLight"))
dist.CreateIntensityAttr(3000)
dist.CreateAngleAttr(0.2) # 影の輪郭をシャープに
# 斜め後ろから照らすことでエッジを際立たせる
dist_xform = XFormPrim("/World/MainLight")
dist_xform.set_world_poses(orientations=np.array([[0.5, 0.5, -0.5, 0.5]]))

# カメラビューの調整（少し斜めからのシネマチックなアングル）
set_camera_view(
    eye=[4.0, 2.0, 2.0], target=[0.0, 0.0, 0.5], camera_prim_path="/OmniverseKit_Persp"
)

# --- [ロボットの追加ロジックは変更なし] ---

# Add Franka
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")
arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")

# Add Carter
asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Car")
car = Articulation(prim_paths_expr="/World/Car", name="my_car")

arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
car.set_world_poses(positions=np.array([[0.0, -1.0, 0.0]]) / get_stage_units())

my_world.reset()

# --- 録画とシミュレーションループ ---
start_recording(output_path, fps, width, height)

for i in range(4):
    print("running cycle: ", i)
    if i == 1 or i == 3:
        print("moving")
        arm.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])
        car.set_joint_velocities([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    if i == 2:
        print("stopping")
        arm.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        car.set_joint_velocities([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    for j in range(100):
        my_world.step(render=True)
        if i == 3:
            car_joint_positions = car.get_joint_positions()
            print("car joint positions:", car_joint_positions)

recorder.stop_recorder()
print("録画が完了し、ファイルが保存されました。")

simulation_app.close()