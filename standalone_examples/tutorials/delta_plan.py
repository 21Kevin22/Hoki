import os
import shutil
import subprocess
import json
import time
import numpy as np
import traceback
from pathlib import Path

# ---------------------------------------------------------
# 0. テスト用スクリプトの自動生成 (ここを追加)
# ---------------------------------------------------------
# 実行時に test_delta.py がなければ自動で作ります
TEST_DELTA_CONTENT = """
import json
import os

# メインスクリプトと同じ場所の出力先を探す
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "bottle_sorting_output")
JSON_PATH = os.path.join(OUTPUT_DIR, "actions.json")

print(f"[MockDelta] Updating plan at: {JSON_PATH}")

# テスト用にプランを書き換える (青→緑→赤の順に変更)
new_plan = {
    "actions": [
        "(pick bottle_2)", "(place)",
        "(pick bottle_1)", "(place)",
        "(pick bottle_0)", "(place)"
    ]
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(JSON_PATH, "w") as f:
    json.dump(new_plan, f, indent=2)

print("[MockDelta] actions.json has been updated!")
"""

def create_mock_script():
    path = Path(os.getcwd()) / "test_delta.py"
    with open(path, "w") as f:
        f.write(TEST_DELTA_CONTENT)
    print(f"[System] Created mock script at: {path}")
    return path

# ---------------------------------------------------------
# 1. 環境設定 (Warp/UIの無効化)
# ---------------------------------------------------------
DISABLE_EXTS_LIST = [
    "omni.warp",
    "omni.kit.window.extensions",
    "omni.kit.menu.utils",
    "omni.kit.browser.sample",
    "omni.kit.window.console",
    "omni.anim.navigation.schema",
    "omni.anim.graph.schema",
    "omni.kit.renderer.imgui"
]
os.environ["OMNI_KIT_DISABLE_EXTENSIONS"] = ",".join(DISABLE_EXTS_LIST)

from isaacsim import SimulationApp

extra_args = ["--no-window"]
for ext in DISABLE_EXTS_LIST:
    extra_args.append("--disable-extension")
    extra_args.append(ext)
    extra_args.append(f"--/app/extensions/enabled/{ext}=false")

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
from pxr import Usd, UsdGeom, Gf, Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension
import omni.replicator.core as rep

enable_extension("omni.replicator.core")

# ---------------------------------------------------------
# 3. パス設定
# ---------------------------------------------------------
OUTPUT_DIR = Path(os.getcwd()) / "bottle_sorting_output"
RGB_DIR = OUTPUT_DIR / "rgb"
PLAN_JSON = OUTPUT_DIR / "actions.json"

# 自動生成したスクリプトのパスを使用
DELTA_PATH = create_mock_script()
DELTA_PYTHON = "/usr/bin/python3"

if RGB_DIR.exists(): shutil.rmtree(RGB_DIR)
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初期プラン
DEFAULT_PLAN = ["(pick bottle_0)", "(place)"]

def save_plan(plan_list):
    PLAN_JSON.write_text(json.dumps({"actions": plan_list}, indent=2))

def load_plan():
    if PLAN_JSON.exists():
        try:
            content = PLAN_JSON.read_text()
            print(f"[System] Current actions.json content: {content}")
            return json.loads(content).get("actions", [])
        except: return []
    save_plan(DEFAULT_PLAN)
    return DEFAULT_PLAN

# ---------------------------------------------------------
# 4. アセット & ロボット設定
# ---------------------------------------------------------
class AssetBuilder:
    @staticmethod
    def create_basket(path, pos):
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, path)
        w, d, h, th = 0.3, 0.4, 0.12, 0.01
        color = Gf.Vec3f(0.5, 0.35, 0.25)
        parts = [("Bottom", (w, d, th), (0, 0, th/2)),("Front", (w, th, h), (0, -d/2, h/2)),("Back", (w, th, h), (0, d/2, h/2)),("Left", (th, d, h), (-w/2, 0, h/2)),("Right", (th, d, h), (w/2, 0, h/2))]
        for name, size, offset in parts:
            part = f"{path}/{name}"
            cube = UsdGeom.Cube.Define(stage, part)
            XFormPrim(part).set_local_scales(np.array([[size[0]/2, size[1]/2, size[2]/2]]))
            XFormPrim(part).set_local_poses(np.array([offset]))
            cube.CreateDisplayColorAttr([color])
        XFormPrim(path).set_world_poses(np.array([pos]))
        return path
    @staticmethod
    def create_bottle(path, pos, color):
        stage = get_current_stage()
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateHeightAttr(0.2)
        cyl.CreateRadiusAttr(0.04)
        cyl.CreateAxisAttr("Z")
        cyl.CreateDisplayColorAttr([color])
        XFormPrim(path).set_world_poses(np.array([pos]))
        return path

class RobotController:
    def __init__(self, arm: Articulation):
        self.arm = arm
        self.dof = arm.num_dof
        # 修正版の姿勢 (めり込み防止・ワープ防止済み)
        raw_poses = {
            "HOME":        [0.0, -0.70, 0.0, -2.30, 0.0, 1.57, 0.78, 0.04, 0.04],
            "LEFT_HIGH":   [0.8, -0.30, 0.0, -2.20, 0.0, 1.80, 0.78, 0.04, 0.04],
            "LEFT_LOW":    [0.75, 0.40, 0.0, -2.40, 0.0, 2.90, 0.78, 0.04, 0.04],
            "CENTER_HIGH": [0.0, -0.30, 0.0, -2.00, 0.0, 1.80, 0.78, 0.04, 0.04],
            "CENTER_LOW":  [0.05, 0.40, 0.0, -2.40, 0.0, 2.90, 0.78, 0.04, 0.04],
            "RIGHT_HIGH":  [-0.8, -0.30, 0.0, -2.20, 0.0, 1.80, 0.78, 0.04, 0.04],
            "RIGHT_LOW":   [-0.75, 0.40, 0.0, -2.40, 0.0, 2.90, 0.78, 0.04, 0.04],
            "BASKET_HIGH": [1.5, -0.20, 0.0, -1.80, 0.0, 1.57, 0.78, 0.04, 0.04],
            "BASKET_LOW":  [1.5,  0.10, 0.0, -2.20, 0.0, 2.80, 0.78, 0.04, 0.04]
        }
        self.poses = {}
        for k, v in raw_poses.items():
            arr = np.zeros(self.dof, dtype=np.float32)
            src = np.array(v)
            arr[:min(self.dof, src.size)] = src[:min(self.dof, src.size)]
            self.poses[k] = arr
    def get_pose(self, name): return self.poses.get(name, self.poses["HOME"])
    def interpolate(self, start, end, steps):
        for t in range(steps):
            r = (1 - np.cos(t / steps * np.pi)) / 2
            yield start + (end - start) * r

def setup_replicator():
    cam = rep.create.camera(position=(2.5, 1.5, 2.0), look_at=(0, 0, 0.0))
    rp = rep.create.render_product(cam, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(RGB_DIR), rgb=True, bounding_box_2d_tight=False)
    writer.attach([rp])
    return writer

# ---------------------------------------------------------
# 5. 再計画ロジック (ファイル書き換え確認用)
# ---------------------------------------------------------
def run_replan(failed_action):
    print(f"\n[Planner] Action '{failed_action}' failed. Triggering Replan...")
    print(f"[Planner] BEFORE Replan: {load_plan()}")
    
    # 外部スクリプト実行 (Kitの影響を排除)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    
    cmd = [DELTA_PYTHON, str(DELTA_PATH)]
    try:
        subprocess.run(cmd, env=env, check=True)
        new_plan = load_plan()
        print(f"[Planner] AFTER Replan: {new_plan}\n")
        return new_plan
    except Exception as e:
        print(f"[Planner] Replan failed: {e}")
        return []

# ---------------------------------------------------------
# 6. メイン処理
# ---------------------------------------------------------
def main():
    try:
        world = World(stage_units_in_meters=1.0)
        assets_root = get_assets_root_path()
        if assets_root:
            env_path = assets_root + "/Isaac/Environments/Grid/default_environment.usd"
            add_reference_to_stage(usd_path=env_path, prim_path="/World/Env")
        else: GroundPlane(prim_path="/World/GroundPlane")

        stage = get_current_stage()
        UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(1500)

        if not assets_root: usd_path = Path("./isaac_sim_grasping/grippers/franka_panda/franka_panda/franka_panda.usd")
        else: usd_path = Path(assets_root) / "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(usd_path=str(usd_path), prim_path="/World/Arm")
        arm = Articulation("/World/Arm", name="my_arm")
        world.scene.add(arm)
        hand_prim = XFormPrim("/World/Arm/panda_hand")

        bottle_paths = {
            "bottle_0": AssetBuilder.create_bottle("/World/Bottle_R", [0.4, 0.3, 0.1], Gf.Vec3f(1,0,0)),
            "bottle_1": AssetBuilder.create_bottle("/World/Bottle_G", [0.5, 0.0, 0.1], Gf.Vec3f(0,1,0)),
            "bottle_2": AssetBuilder.create_bottle("/World/Bottle_B", [0.4, -0.3, 0.1], Gf.Vec3f(0,0,1)),
        }
        BASKET_POS = np.array([0.0, 0.6, 0.05])
        AssetBuilder.create_basket("/World/Basket", BASKET_POS)

        setup_replicator()
        world.reset()
        for _ in range(30): world.step(render=False)
        controller = RobotController(arm)
        
        # 初期プラン
        save_plan(["(pick bottle_0)", "(place)"])
        plan = load_plan()

        force_fail = True 

        print("=== Simulation Start ===")
        while plan:
            action = plan.pop(0)
            print(f">> Executing: {action}")
            
            # テスト: 最初のplaceアクションで失敗させて再計画を確認
            if "place" in action and force_fail:
                print(f"[Test] Forcing failure for testing replan...")
                force_fail = False 
                new_plan = run_replan(action)
                if new_plan:
                    print("!!! PLAN UPDATED SUCCESSFULLY !!!")
                    plan = new_plan 
                    continue
                else:
                    print("!!! PLAN UPDATE FAILED !!!")
                    break
            
            # 通常シミュレーション (短縮版)
            for _ in range(10): world.step(render=False) 
            print(f"[Success] {action} finished.")

        print("=== Finished ===")
        
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()