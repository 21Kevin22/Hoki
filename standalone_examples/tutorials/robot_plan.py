import os
import shutil
import subprocess
import numpy as np
import traceback
import glob
import time
from pathlib import Path

# ---------------------------------------------------------
# 1. 環境設定 (最優先: Warp/UIの無効化)
# ---------------------------------------------------------
# 無効化対象リスト
DISABLE_EXTS = [
    "omni.warp",
    "omni.kit.window.extensions",
    "omni.kit.menu.utils",
    "omni.kit.browser.sample",
    "omni.kit.window.console",
    "omni.anim.navigation.schema",
    "omni.anim.graph.schema",
    "omni.kit.renderer.imgui" # 描画UIも抑制
]

# 環境変数での無効化
os.environ["OMNI_KIT_DISABLE_EXTENSIONS"] = ",".join(DISABLE_EXTS)

from isaacsim import SimulationApp

# SimulationAppへの引数作成
extra_args = ["--no-window"]
for ext in DISABLE_EXTS:
    # 拡張機能マネージャでの無効化
    extra_args.append("--disable-extension")
    extra_args.append(ext)
    # 設定システムレベルでの無効化 (より強力)
    extra_args.append(f"--/app/extensions/enabled/{ext}=false")

# Headless設定
config = {
    "headless": True,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
    "hide_ui": True,
    "extra_args": extra_args
}
simulation_app = SimulationApp(config)

# ---------------------------------------------------------
# 2. モジュールインポート
# ---------------------------------------------------------
# UsdLux を追加
from pxr import Usd, UsdGeom, Gf, Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.extensions import enable_extension
import omni.replicator.core as rep

# Replicator有効化
enable_extension("omni.replicator.core")

# ---------------------------------------------------------
# 3. 保存先設定
# ---------------------------------------------------------
OUTPUT_DIR = Path(os.getcwd()) / "bottle_sorting_output"
RGB_DIR = OUTPUT_DIR / "rgb"
VIDEO_PATH = OUTPUT_DIR / "bottle_sorting.mp4"

if RGB_DIR.exists(): shutil.rmtree(RGB_DIR)
os.makedirs(RGB_DIR, exist_ok=True)
print(f"=== 保存先: {OUTPUT_DIR} ===")

# ---------------------------------------------------------
# 4. アセット生成 (USD API)
# ---------------------------------------------------------
class AssetBuilder:
    @staticmethod
    def create_basket(path, pos):
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, path)
        
        w, d, h = 0.3, 0.4, 0.1
        th = 0.01 
        color = Gf.Vec3f(0.5, 0.35, 0.25) 

        parts = [
            ("Bottom", (w, d, th), (0, 0, th/2)),
            ("Front", (w, th, h), (0, -d/2, h/2)),
            ("Back", (w, th, h), (0, d/2, h/2)),
            ("Left", (th, d, h), (-w/2, 0, h/2)),
            ("Right", (th, d, h), (w/2, 0, h/2))
        ]

        for name, size, offset in parts:
            part_path = f"{path}/{name}"
            cube = UsdGeom.Cube.Define(stage, part_path)
            
            # 【修正】セットするスケールを (1, 3) の2次元配列にする
            scale_val = np.array([[size[0]/2, size[1]/2, size[2]/2]])
            XFormPrim(part_path).set_local_scales(scale_val)
            
            # offsetはタプルなので [] で囲むと (1, 3) になる
            XFormPrim(part_path).set_local_poses(np.array([offset]))
            cube.CreateDisplayColorAttr([color])

        XFormPrim(path).set_world_poses(np.array([pos]))
        return path

    @staticmethod
    def create_bottle(path, pos, color_vec):
        stage = get_current_stage()
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateHeightAttr(0.2)
        cyl.CreateRadiusAttr(0.04)
        cyl.CreateAxisAttr("Z")
        cyl.CreateDisplayColorAttr([color_vec])
        
        XFormPrim(path).set_world_poses(np.array([pos]))
        return path

# ---------------------------------------------------------
# 5. ロボット制御
# ---------------------------------------------------------
class RobotController:
    def __init__(self, arm: Articulation):
        self.arm = arm
        self.dof = arm.num_dof
        print(f"[System] Robot initialized with {self.dof} DOFs")
        
        self.poses = {
            "HOME":   [0.0, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7, 0.04, 0.04],
            "LEFT":   [0.8, -0.5, 0.0, -2.2, 0.0, 1.8, 0.7, 0.04, 0.04],
            "CENTER": [0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.7, 0.04, 0.04],
            "RIGHT":  [-0.8, -0.5, 0.0, -2.2, 0.0, 1.8, 0.7, 0.04, 0.04],
            "BASKET": [1.5, -0.2, 0.0, -2.0, 0.0, 2.0, 0.7, 0.04, 0.04]
        }

    def get_pose(self, name):
        base = np.array(self.poses[name], dtype=np.float32)
        target = np.zeros(self.dof, dtype=np.float32)
        # サイズ合わせ
        common_len = min(base.size, self.dof)
        target[:common_len] = base[:common_len]
        return target

    def interpolate(self, start_pose, end_pose, steps):
        for t in range(steps):
            r = t / float(steps)
            smooth_r = (1 - np.cos(r * np.pi)) / 2 
            yield start_pose + (end_pose - start_pose) * smooth_r

# ---------------------------------------------------------
# 6. Replicator設定
# ---------------------------------------------------------
def setup_replicator():
    cam = rep.create.camera(position=(2.0, 1.0, 1.5), look_at=(0, 0, 0.2))
    rp = rep.create.render_product(cam, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(RGB_DIR),
        rgb=True,
        bounding_box_2d_tight=False
    )
    writer.attach([rp])
    return writer

# ---------------------------------------------------------
# 7. メイン処理
# ---------------------------------------------------------
def main():
    try:
        world = World(stage_units_in_meters=1.0)
        GroundPlane(prim_path="/World/GroundPlane")
        
        stage = get_current_stage()
        # エラー修正: UsdLux をインポートしたので使用可能
        UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(900)

        usd_path = Path("./isaac_sim_grasping/grippers/franka_panda/franka_panda/franka_panda.usd")
        if not usd_path.exists():
            from isaacsim.storage.native import get_assets_root_path
            assets_root = get_assets_root_path()
            usd_path = Path(f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")

        print(f"[System] Loading Robot: {usd_path}")
        add_reference_to_stage(usd_path=str(usd_path), prim_path="/World/Arm")
        arm = Articulation("/World/Arm", name="my_arm")
        world.scene.add(arm)
        
        hand_prim = XFormPrim("/World/Arm/panda_hand")

        # アセット生成 (修正済みの create_basket を使用)
        bottle_paths = []
        bottle_paths.append(AssetBuilder.create_bottle("/World/Bottle_R", [0.4, 0.3, 0.1], Gf.Vec3f(1,0,0)))
        bottle_paths.append(AssetBuilder.create_bottle("/World/Bottle_G", [0.5, 0.0, 0.1], Gf.Vec3f(0,1,0)))
        bottle_paths.append(AssetBuilder.create_bottle("/World/Bottle_B", [0.4, -0.3, 0.1], Gf.Vec3f(0,0,1)))
        
        AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

        setup_replicator()

        world.reset()
        for _ in range(30): world.step(render=False)

        controller = RobotController(arm)
        
        raw_joints = arm.get_joint_positions()
        current_joints = np.array(raw_joints, dtype=np.float32).flatten()
        if current_joints.size != arm.num_dof:
            current_joints = np.zeros(arm.num_dof, dtype=np.float32)

        # --- grasp cylinder with replanning ---
        cyl_path = "/World/TargetCylinder"
        AssetBuilder.create_bottle(cyl_path, [0.2, 0.0, 0.1], Gf.Vec3f(1, 0, 1))

        def get_world_pos(prim_path):
            prim = XFormPrim(prim_path)
            pos, _ = prim.get_world_pose()
            return np.array(pos)

        def llm_replan(attempt, pos):
            print(f"[LLM] replanning attempt={attempt}")
            return pos + np.random.uniform(-0.05, 0.05, 3)

        attempts = 0
        picked = False
        while attempts < 3 and not picked:
            print(f"[Task] pick attempt {attempts + 1}")
            target_joints = controller.get_pose("CENTER")
            for next_j in controller.interpolate(current_joints, target_joints, 40):
                arm.set_joint_positions(next_j)
                world.step(render=False)
                rep.orchestrator.step()
            hand_pos, _ = hand_prim.get_world_pose()
            cyl_pos = get_world_pos(cyl_path)
            if np.linalg.norm(hand_pos - cyl_pos) < 0.1:
                print("[Task] grasp succeeded")
                picked = True
                XFormPrim(cyl_path).set_world_poses(positions=np.array([hand_pos]))
            else:
                attempts += 1
                print("[Task] grasp failed")
                new_pos = llm_replan(attempts, cyl_pos)
                XFormPrim(cyl_path).set_world_poses(np.array([new_pos]))
            current_joints = target_joints
        if not picked:
            print("[Task] giving up after 3 failures")

        # continue with original bottle actions
        actions = [
            ("LEFT", 40, 0),
            ("BASKET", 40, 0),
            ("HOME", 20, None),
            ("CENTER", 40, 1),
            ("BASKET", 40, 1),
            ("HOME", 20, None),
            ("RIGHT", 40, 2),
            ("BASKET", 40, 2),
            ("HOME", 20, None)
        ]

        print("=== Recording Start ===")
        
        for target_name, steps, bottle_idx in actions:
            target_joints = controller.get_pose(target_name)
            
            for next_j in controller.interpolate(current_joints, target_joints, steps):
                arm.set_joint_positions(next_j)
                
                if bottle_idx is not None and target_name == "BASKET":
                    pos, rot = hand_prim.get_world_pose()
                    b_prim = XFormPrim(bottle_paths[bottle_idx])
                    b_prim.set_world_poses(
                        positions=np.array([pos + np.array([0, 0, -0.15])]),
                        orientations=np.array([rot])
                    )

                world.step(render=False)
                rep.orchestrator.step()
            
            if bottle_idx is not None and target_name == "BASKET":
                b_prim = XFormPrim(bottle_paths[bottle_idx])
                drop_pos = np.array([0.0, 0.6, 0.12]) + np.random.uniform(-0.05, 0.05, 3)
                b_prim.set_world_poses(positions=np.array([drop_pos]))
                b_prim.set_world_orientations(np.array([[1, 0, 0, 0]]))

            current_joints = target_joints

        print("Waiting for writer...")
        rep.orchestrator.wait_until_complete()
        time.sleep(2.0)
        print("=== Recording Finished ===")

    except Exception:
        traceback.print_exc()
    finally:
        print("Processing video...")
        files = sorted(list(RGB_DIR.glob("**/*.png")))
        
        if not files:
            print("[Error] No PNG files found.")
        else:
            print(f"Found {len(files)} frames. Renaming...")
            
            frames_dir = OUTPUT_DIR / "frames"
            if frames_dir.exists(): shutil.rmtree(frames_dir)
            os.makedirs(frames_dir, exist_ok=True)

            for i, fpath in enumerate(files):
                shutil.copy(str(fpath), str(frames_dir / f"frame_{i:04d}.png"))
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(VIDEO_PATH)
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"🎥 Video Saved: {VIDEO_PATH}")
                shutil.rmtree(frames_dir)
            except Exception as e:
                print(f"[Error] ffmpeg failed: {e}")

        simulation_app.close()

if __name__ == "__main__":
    main()