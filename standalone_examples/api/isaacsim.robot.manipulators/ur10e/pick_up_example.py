from isaacsim import SimulationApp

# 1. Headlessモードで起動
simulation_app = SimulationApp({"headless": True})

import os
import numpy as np
import subprocess
import shutil
import sys

# Replicator
import omni.replicator.core as rep

# Isaac Sim Core
from isaacsim.core.api import World
from tasks.pick_place import PickPlace
from controller.pick_place import PickPlaceController

# -------------------------
# 2. Scene Setup
# -------------------------
my_world = World(stage_units_in_meters=1.0, physics_dt=1/200, rendering_dt=1/60)

target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0

my_task = PickPlace(
    name="ur10e_pick_place",
    target_position=target_position,
    cube_size=np.array([0.1, 0.0515, 0.1]),
)
my_world.add_task(my_task)

# Physics初期化プロセス (ここが重要)
print("=== Initializing Physics... ===")
my_world.reset()

# ★★★ GitHub Issue #1017 に基づく修正 ★★★
# Headlessモードでは reset() だけでは Physics View が作られないことがあるため、
# 明示的に play() を呼び出します。
print("=== Playing Simulation... ===")
my_world.play()

# -------------------------
# Controller Setup
# -------------------------
task_params = my_world.get_task("ur10e_pick_place").get_params()
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)

controller = PickPlaceController(
    name="controller",
    robot_articulation=my_ur10e,
    gripper=my_ur10e.gripper,
)
articulation_controller = my_ur10e.get_articulation_controller()

# -------------------------
# 3. Replicator Setup
# -------------------------
output_dir = os.path.join(os.getcwd(), "replicator_output")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

with rep.new_layer():
    camera = rep.create.camera(
        position=(-1.8, 0.5, 1.5),
        look_at=(-0.3, 0.6, 0.2)
    )
    render_product = rep.create.render_product(camera, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True)
    writer.attach([render_product])

rep.orchestrator.run()

# -------------------------
# 4. Simulation Loop
# -------------------------
max_steps = 600
print("=== Starting Simulation Loop ===")

# ★★★ ウォームアップ（重要）★★★
# 物理エンジンが完全に立ち上がるまで数ステップ空回しします。
# render=False にして高速に進めます。
for i in range(20):
    my_world.step(render=False)

# メインループ
for step in range(max_steps):
    # 物理とレンダリングを進める
    my_world.step(render=True)

    try:
        # ロボットの初期化チェック
        if not my_ur10e.handles_initialized:
            print(f"Step {step}: Waiting for robot initialization...")
            continue

        observations = my_world.get_observations()
        
        # 必要なキーがあるか確認
        if task_params["robot_name"]["value"] not in observations:
            continue

        current_joints = observations[task_params["robot_name"]["value"]]["joint_positions"]
        
        # データが None ならスキップ
        if current_joints is None:
            continue

        actions = controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0, 0, 0.20]),
        )
        articulation_controller.apply_action(actions)

        if controller.is_done():
            print(f"=== Task Completed at step {step} ===")
            break

    except Exception as e:
        # エラーが出ても止まらずログを出す
        if step % 50 == 0: # ログ抑制
            print(f"Warning at step {step}: {e}")

# -------------------------
# 5. Video Generation & FORCE EXIT
# -------------------------
print("=== Waiting for Replicator to finish writing... ===")
rep.orchestrator.wait_until_complete()

print("=== Generating Video ===")
rgb_dir = os.path.join(output_dir, "rgb")
video_path = os.path.join(output_dir, "ur10e_pick_place.mp4")

if os.path.exists(rgb_dir) and len(os.listdir(rgb_dir)) > 0:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", os.path.join(rgb_dir, "rgb_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"SUCCESS: Video saved to: {video_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")
else:
    print("No images found. Video generation skipped.")

print("=== Force Exiting to avoid Segmentation Fault ===")
os._exit(0)