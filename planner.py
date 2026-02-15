import os
import subprocess
import sys

# あなたが見つけた Fast Downward の絶対パス
PLANNER_BIN = "/home/ubuntu/slocal1/DELTA/pddlgym_planners/pddlgym_planners/FD/fast-downward.py"

def query(domain_path, problem_path, plan_filename="sas_plan", max_time=120):
    """
    PDDLGymのパーサーを使わず、Fast Downwardを直接呼び出す
    """
    plan_file = os.path.join(os.getcwd(), plan_filename)
    if os.path.exists(plan_file): os.remove(plan_file)

    # --alias を使わず、直接 A* + LM-cut 検索を指定する
    command = [
        "python3", PLANNER_BIN,
        "--plan-file", plan_file,
        domain_path,
        problem_path,
        "--search", "astar(lmcut())"  # ← ここを修正
    ]

    print(f"\n🚀 プランナー実行中...\n{' '.join(command)}")

    try:
        # stdout/stderrをキャプチャ
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=max_time)

        if "Solution found" in stdout:
            cost = "N/A"
            plan_time = "N/A"
            for line in stdout.split("\n"):
                if "Plan cost:" in line: cost = line.split(":")[-1].strip()
                if "Planner time:" in line: plan_time = line.split(":")[-1].strip()

            if os.path.exists(plan_file):
                with open(plan_file, "r") as f:
                    plan = f.read()
                return plan, cost, plan_time, 1, "" # Success
            
        return None, 0, 0, 2, stdout + stderr # Failure

    except subprocess.TimeoutExpired:
        process.kill()
        return None, 0, 0, 3, "Timeout"
    except Exception as e:
        return None, 0, 0, 2, str(e)