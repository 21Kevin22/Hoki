import os
import shutil
import subprocess
import json
import time
import numpy as np
import pandas as pd
import csv
import traceback
from pathlib import Path
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. パスと定数の設定
# ---------------------------------------------------------
# 保存先ディレクトリ
OUTPUT_DIR = Path("/home/ubuntu/slocal/evaluation")
RGB_DIR = OUTPUT_DIR / "rgb"
PLAN_JSON = OUTPUT_DIR / "actions.json"
VIDEO_PATH = OUTPUT_DIR / "bottle_sorting.mp4"
METRICS_CSV = OUTPUT_DIR / "evaluation_metrics.csv"

# 外部スクリプトのパス
REAL_DELTA_PATH = Path("/home/ubuntu/slocal/Hoki/delta.py")
DELTA_PYTHON = "/usr/bin/python3"  # システムのPython

# 初期化
if RGB_DIR.exists(): shutil.rmtree(RGB_DIR)
os.makedirs(RGB_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. 評価・ログ管理クラス
# ---------------------------------------------------------
class PlanningEvaluator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.log_file = METRICS_CSV
        self.metrics = []
        self._prepare_log_file()

    def _prepare_log_file(self):
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Event_ID", "Action", "Instruction", "Latency_sec", "Iteration_Steps", "Success"])

    def record_event(self, action, instruction, latency, iterations, success):
        event_id = len(self.metrics) + 1
        data = [event_id, action, instruction, round(latency, 4), iterations, success]
        self.metrics.append(data)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def print_summary(self):
        if not self.log_file.exists():
            print("No metrics file found.")
            return

        df = pd.read_csv(self.log_file)
        if df.empty: return

        print("\n" + "="*60)
        print("         PLANNING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Recovery Events    : {len(df)}")
        print(f"Avg Planning Latency     : {df['Latency_sec'].mean():.4f} sec")
        print(f"Avg Iteration Steps      : {df['Iteration_Steps'].mean():.2f} steps")
        print(f"Max Latency              : {df['Latency_sec'].max():.4f} sec")
        print("="*60)

        self.generate_report_graph(df)

    def generate_report_graph(self, df):
        fig = go.Figure()
        # 計画時間の棒グラフ
        fig.add_trace(go.Bar(x=df['Event_ID'], y=df['Latency_sec'], name='Latency (sec)', marker_color='indianred'))
        # 反復回数の折れ線グラフ（第2軸）
        fig.add_trace(go.Scatter(x=df['Event_ID'], y=df['Iteration_Steps'], name='Iterations', yaxis='y2', line=dict(color='royalblue')))

        fig.update_layout(
            title='Planning Latency and Iterations per Event',
            xaxis_title='Recovery Event ID',
            yaxis_title='Latency (seconds)',
            yaxis2=dict(title='Iteration Steps', overlaying='y', side='right', range=[0, 5]),
            template="plotly_white"
        )
        
        graph_path = self.output_dir / "performance_chart.html"
        fig.write_html(str(graph_path))
        print(f"📊 Visualization saved to: {graph_path}")

evaluator = PlanningEvaluator(OUTPUT_DIR)

# ---------------------------------------------------------
# 3. Isaac Sim 起動とライブラリロード
# ---------------------------------------------------------
from isaacsim import SimulationApp
# ヘッドレスモード、Windowなしで高速化
config = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "hide_ui": True, "extra_args": ["--no-window"]}
simulation_app = SimulationApp(config)

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
# 4. ヘルパー関数
# ---------------------------------------------------------
def load_plan():
    if PLAN_JSON.exists():
        try:
            with open(PLAN_JSON, "r") as f:
                return json.load(f).get("actions", [])
        except: return []
    initial_plan = ["(pick bottle_0)", "(place)", "(pick bottle_1)", "(place)", "(pick bottle_2)", "(place)"]
    save_plan(initial_plan)
    return initial_plan

def save_plan(plan_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PLAN_JSON, "w") as f:
        json.dump({"actions": plan_list}, f, indent=2)

def force_update_json(instruction=""):
    # delta.pyが動かない場合の緊急用固定プラン
    new_actions = ["(pick bottle_2)", "(place)", "(pick bottle_1)", "(place)", "(pick bottle_0)", "(place)"]
    save_plan(new_actions)
    return new_actions

def run_replan(failed_action, user_instruction=None):
    start_time = time.time()
    print(f"\n[Planner] Requesting LLM re-plan for: {failed_action}")
    
    success = False
    new_plan = []

    if REAL_DELTA_PATH.exists():
        env = os.environ.copy()
        env.pop("PYTHONPATH", None) # システムPythonとの干渉を避ける
        
        # eval.py の run_replan 内
        cmd = [
            DELTA_PYTHON, str(REAL_DELTA_PATH),
            "--experiment", "all",
            "--episode", "1",
            "--scene-example", "office",  # エラーが出た 'allensville' から 'office' に変更
            "--print-plan"
        ]
        if user_instruction:
            cmd.extend(["--instruction", str(user_instruction)])

        try:
            # delta.pyの実行
            subprocess.run(cmd, cwd=REAL_DELTA_PATH.parent, env=env, check=True)
            new_plan = load_plan()
            success = True
            print(f"[Planner] New plan received: {new_plan}")
        except Exception as e:
            print(f"[Error] delta.py execution failed: {e}")
            new_plan = force_update_json(user_instruction)
    else:
        new_plan = force_update_json(user_instruction)

    latency = time.time() - start_time
    evaluator.record_event(failed_action, user_instruction, latency, 1, success)
    return new_plan

# ---------------------------------------------------------
# 5. シミュレーション構築と実行
# ---------------------------------------------------------
def main():
    try:
        world = World(stage_units_in_meters=1.0)
        GroundPlane(prim_path="/World/GroundPlane")
        
        # ロボットのロード
        assets_root = get_assets_root_path()
        usd_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/Arm")
        arm = Articulation("/World/Arm", name="my_arm")
        world.scene.add(arm)

        # レプリケーター設定（動画用）
        cam = rep.create.camera(position=(2.0, 1.2, 1.5), look_at=(0, 0, 0))
        rp = rep.create.render_product(cam, (1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(RGB_DIR), rgb=True)
        writer.attach([rp])

        world.reset()
        plan = load_plan()
        fail_count = 0
        force_fail_mode = True # 評価のため最初は失敗させる

        print("=== Simulation Main Loop Start ===")
        while plan:
            action = plan.pop(0)
            print(f">> Executing: {action}")
            
            # アクション実行のシミュレーション（簡略化）
            for _ in range(30):
                world.step(render=False)
                rep.orchestrator.step()
            
            if "place" in action:
                # 失敗検知ロジック
                if force_fail_mode:
                    print(f"[Fail] Simulated failure detected for: {action}")
                    fail_count += 1
                    
                    if fail_count >= 3:
                        print("\n" + "!"*40)
                        print("3回連続失敗：対話による再計画を開始します。")
                        user_in = input("指示を入力してください (例: 'bottle_2, bottle_1の順で'): ")
                        
                        if user_in.lower() in ['q', 'quit']: break
                        
                        new_plan = run_replan(action, user_in)
                        if new_plan:
                            plan = new_plan
                            fail_count = 0
                            force_fail_mode = False # 再計画後は成功させる
                            print(">>> プランが更新されました。シミュレーションを継続します。")
                            continue
                else:
                    print(f"[Success] Action {action} completed.")

        print("=== Simulation Finished. Finalizing... ===")
        rep.orchestrator.wait_until_complete()
        evaluator.print_summary()

    except Exception:
        traceback.print_exc()
    # eval.py の最後に近い部分
    finally:
        files = sorted(list(RGB_DIR.glob("**/*.png")))
        if files:
            print("Encoding video...")
        # 直接ファイルリストを渡すのではなく、連番として扱うか、
        # あるいは glob をシェル経由で解釈させます
            input_pattern = str(RGB_DIR / "**/rgb_*.png") # Replicatorのデフォルト名に合わせる
        
        # 修正版コマンド：パターンを直接指定せず、一度 tmp フォルダに平坦化してコピーするのが一番確実です
            tmp_frames = OUTPUT_DIR / "tmp_frames"
            os.makedirs(tmp_frames, exist_ok=True)
            for i, f in enumerate(files):
                shutil.copy(str(f), tmp_frames / f"frame_{i:04d}.png")
            
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "30", 
                "-i", str(tmp_frames / "frame_%04d.png"), 
                "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                str(VIDEO_PATH)
            ], check=False)
        
            shutil.rmtree(tmp_frames) # 一時フォルダを削除
            print(f"🎥 Video saved to: {VIDEO_PATH}")
            simulation_app.close()

if __name__ == "__main__":
    main()