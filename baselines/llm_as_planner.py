import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
import time
import os
import sys
import copy
import re
import matplotlib.pyplot as plt

# ターミナル色付け用
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

# パス設定
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from data.scene_graph import load_scene_graph, prune_sg_with_item
from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p

from adaptation_manager import AdaptationManager

# --- Parameters ---
DEFAULT_LLM = "gpt-4o"
TEMPERATURE = 0.
TOP_P = 1.
EPISODE = 5

# --- Examples & Queries ---
DOMAIN_EXAMPLE = "laundry"
SCENE_EXAMPLE = example.get_scenes(DOMAIN_EXAMPLE)[0]
DOMAIN_QUERY = "pc"
SCENE_QUERY = example.get_scenes(DOMAIN_QUERY)[0]

# --- Helper Functions ---
def SRC_DOMAIN_PATH(d):
    return f"data/pddl/domain/{d}_domain.pddl"

def SRC_PROBLEM_PATH(s, d):
    return f"data/pddl/problem/{s}_{d}_problem.pddl"

def LOG_PATH(t):
    return Path(f"result/{t}")

def read_domain_schema(domain_path):
    """ドメイン定義ファイルを読み込む"""
    try:
        full_path = parent_dir / domain_path
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not read domain file at {domain_path}: {e}{Colors.RESET}")
        return ""

def extract_map_connections(problem_path):
    """
    PDDL問題ファイルを解析して部屋の接続情報を抽出する（修正版）
    """
    connections = []
    try:
        full_path = parent_dir / problem_path
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # PDDLのコメント(;)を除去
        clean_content = re.sub(r';.*$', '', content, flags=re.MULTILINE)
        
        # (connected A B) のパターン抽出
        matches = re.findall(r'\(\s*connected\s+([\w-]+)\s+([\w-]+)\s*\)', clean_content, re.IGNORECASE)
        
        if matches:
            print(f"{Colors.GREEN}>>> Map Extracted: Found {len(matches)} connections.{Colors.RESET}")
            connections = [f"{m[0]} <-> {m[1]}" for m in matches]
        else:
            print(f"{Colors.RED}>>> Map Extraction Failed: No 'connected' predicates found via regex.{Colors.RESET}")
            
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not extract map info: {e}{Colors.RESET}")

    # ★修正: Allensvilleの正確なマップ定義
    if not connections and "allensville" in str(problem_path):
        print(f"{Colors.YELLOW}>>> Using Corrected Fallback Map for Allensville{Colors.RESET}")
        connections = [
            "corridor_1 <-> lobby", 
            "corridor_1 <-> corridor_3", 
            "corridor_3 <-> living_room", 
            # 修正: bedroom_1 は corridor_3 ではなく corridor_2 につながっているはず
            # "corridor_3 <-> bedroom_1", (削除: これが間違いでした)
            "corridor_3 <-> bedroom_2", 
            "corridor_3 <-> bathroom_2", 
            "corridor_3 <-> corridor_2",
            
            # 修正: ここを追加
            "corridor_2 <-> bedroom_1", 
            "corridor_2 <-> bathroom_1", 
            
            "living_room <-> dining_room",
            "living_room <-> kitchen"
        ]

    if not connections:
        return ""

    unique_connections = sorted(list(set(connections)))
    hint = "\n\n[MAP TOPOLOGY - NAVIGATION RULES]\n"
    hint += "The robot can ONLY move between these directly connected rooms:\n"
    hint += "\n".join(unique_connections)
    hint += "\n(WARNING: Do NOT assume connections. You must pathfind. E.g., to go from Corridor_3 to Bedroom_1, you must go: Corridor_3 -> Corridor_2 -> Bedroom_1)"
    
    return hint

def simulate_environment_change(base_graph, episode_idx, current_goal):
    """環境変化のシミュレーション"""
    new_graph = copy.deepcopy(base_graph)
    new_goal = current_goal

    if episode_idx == 2:
        print(f"\n{Colors.YELLOW}>>> [SIMULATION] TRIGGERING ENVIRONMENT CHANGE!{Colors.RESET}")
        new_goal = "all parts are on the table"
        print(f">>> [SIMULATION] Goal changed to: {new_goal}")
    
    if episode_idx == 3:
        print(f"\n{Colors.YELLOW}>>> [SIMULATION] Environment restored to original.{Colors.RESET}")

    return new_graph, new_goal

def visualize_results(data_list, log_base_path):
    """結果をグラフ化して保存する"""
    if not data_list:
        return

    episodes = [d[0] + 1 for d in data_list]
    successes = [d[2] for d in data_list] # 0 or 1
    plan_costs = [d[10] for d in data_list]
    gt_costs = [d[11] for d in data_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(episodes, successes, marker='o', linestyle='-', color='b')
    ax1.set_title('Success per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success (1=Yes, 0=No)')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Fail', 'Success'])
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(episodes, plan_costs, marker='o', label='Plan Cost', color='orange')
    ax2.plot(episodes, gt_costs, linestyle='--', label='GT Cost', color='gray')
    ax2.set_title('Plan Cost vs Ground Truth')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cost (Steps)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    save_path = log_base_path / "result_visualization.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n{Colors.GREEN}>>> Visualization saved to: {save_path}{Colors.RESET}")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_LLM)
    parser.add_argument("-t", "--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("-e", "--episode", type=int, default=EPISODE)
    parser.add_argument("-p", "--print-prompt", action="store_true", default=True)
    parser.add_argument("-r", "--print-response", action="store_true", default=True)
    parser.add_argument("--print-plan", action="store_true", default=False)
    parser.add_argument("-d", "--domain", type=str, default=DOMAIN_QUERY)
    parser.add_argument("--domain-example", type=str, default=DOMAIN_EXAMPLE)
    parser.add_argument("-s", "--scene", type=str, default=SCENE_QUERY)
    parser.add_argument("--scene-example", type=str, default=SCENE_EXAMPLE)
    
    args = parser.parse_args()

    print(f"Using model {args.model}")
    
    exp = example.get_example(args.domain_example)
    qry = example.get_example(args.domain)
    
    add_obj_exp = exp["add_obj"]
    add_act_exp = exp["add_act"]
    goal_exp = exp["goal"]
    items_keep_exp = exp["item_keep"]

    add_obj_qry = qry["add_obj"]
    add_act_qry = qry["add_act"]
    
    initial_goal_qry = qry["goal"]
    items_keep_qry = qry["item_keep"]
    gt_cost = qry["gt_cost"][args.scene]

    scene_exp = load_scene_graph(args.scene_example)
    scene_exp = prune_sg_with_item(scene_exp, items_keep_exp)
    
    base_scene_qry = load_scene_graph(args.scene)
    base_scene_qry = prune_sg_with_item(base_scene_qry, items_keep_qry)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base_path = LOG_PATH(curr_time)
    
    success = 0
    data_list = []
    
    model = llm.load_llm(args.model, args.temperature, args.top_p)
    adapter = AdaptationManager()

    for e in range(args.episode):
        model.reset()
        log_path = log_base_path / f"e_{e:03}"
        log_path.mkdir(parents=True, exist_ok=True)
        
        plan_cost = 0
        exit_code = 0

        # 1. 環境変化シミュレーション
        current_scene_qry, current_goal_qry = simulate_environment_change(base_scene_qry, e, initial_goal_qry)

        # 2. 適応プロセス
        adaptation_advice = adapter.process(e, current_goal_qry, current_scene_qry, model, str(log_path))

        # 3. プロンプト作成
        content, prompt = p.sg_2_plan(scene_exp, current_scene_qry, goal_exp, current_goal_qry,
                                      add_obj_exp, add_obj_qry, add_act_exp, add_act_qry)
        
        domain_text = read_domain_schema(SRC_DOMAIN_PATH(args.domain))
        if domain_text:
            schema_hint = f"\n\n[CRITICAL RULE]: Here is the exact PDDL domain definition.\nYou MUST follow preconditions strictly.\n```pddl\n{domain_text}\n```"
            prompt += schema_hint

        map_hint = extract_map_connections(SRC_PROBLEM_PATH(args.scene, args.domain))
        if map_hint:
            prompt += map_hint
        else:
            print(f"{Colors.RED}>>> CRITICAL: Map hint is empty!{Colors.RESET}")

        if adaptation_advice:
            prompt += adaptation_advice

        print(f"Tokens for Prompt: {model.count_tokens(prompt)}")
        if args.print_prompt:
            model.log(content + prompt, log_path / f"llmasplanner_{args.domain}_{args.scene}.prompt")

        # 4. LLM実行
        start = time.time()
        output_tar = model.query(content, prompt)
        llm_time = time.time() - start

        if args.print_response:
            model.log(output_tar, log_path / f"llmasplanner_{args.domain}_{args.scene}.response")
        print(f"Response time: {llm_time:.2f}s")

        # 5. プラン保存・検証
        plan_file = log_path / f"llmasplanner_{args.domain}_{args.scene}.plan"
        try:
            plan_list, plan_cost = llm_utils.export_sayplan_plan(output_tar, str(plan_file))
            print(f"Plan cost: {plan_cost}, GT cost: {gt_cost}")
        except Exception as err:
            print(f"{Colors.RED}Error in exporting plan: {err}{Colors.RESET}")
        
        is_valid, val_info = planner.validate(
            str(parent_dir / SRC_DOMAIN_PATH(args.domain)), 
            str(parent_dir / SRC_PROBLEM_PATH(args.scene, args.domain)), 
            str(plan_file)
        )

        if is_valid:
            success += 1
            exit_code = 1
            result_msg = f"{Colors.GREEN}SUCCESS{Colors.RESET}"
        else:
            result_msg = f"{Colors.RED}FAILURE{Colors.RESET}"
            feedback, exit_code = planner.val_feedback(val_info)
            print(f"Feedback: {feedback}")

        print(f"==== Episode {e + 1}/{args.episode}, Result: {result_msg} ====")
        
        data_list.append([
            e, exit_code, success, args.model, args.temperature,
            args.domain_example, args.scene_example, args.domain, args.scene,
            llm_time, plan_cost, gt_cost
        ])

    df = pd.DataFrame(data_list, columns=[
        "Episode", "Exit Code", "Success", "LLM", "Temp",
        "Domain Exp", "Scene Exp", "Domain Qry", "Scene Qry",
        "LLM Time", "Plan Cost", "GT Cost"
    ])
    df.to_csv(log_base_path / "log.csv")
    print(f"Success rate: {success / args.episode * 100:.2f}%")
    
    visualize_results(data_list, log_base_path)