import argparse
import json
import copy
import os
import time
import sys
import random
import pandas as pd
from datetime import datetime
from pathlib import Path
import openai 
import ast
import re

# --- 既存のモジュール群 ---
# (環境に合わせてパスが通っていることを前提とします)
from data.scene_graph import load_scene_graph, prune_sg_with_item, extract_accessible_items_from_sg
from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p

try:
    from scene_graph_diff import compare_scene_graphs
except ImportError:
    print("Warning: scene_graph_diff.py not found.")
    def compare_scene_graphs(ideal, current): return {}

# --- Constants ---
DEFAULT_LLM = "gpt-4o-mini"
TEMPERATURE = 0.
TOP_P = 1.
EPISODE = 50
MAX_TIME = 60
INITIAL_WAIT_SECONDS = 5
MAX_WAIT_SECONDS = 60
MAX_RETRIES = 10

# Examples
DOMAIN_EXAMPLE = "laundry"
SCENE_EXAMPLE = example.get_scenes(DOMAIN_EXAMPLE)[0]
# 修正: デフォルトのドメインを 'office' に統一
DOMAIN_QUERY = "office" 
SCENE_QUERY = example.get_scenes(DOMAIN_QUERY)[0] if example.get_scenes(DOMAIN_QUERY) else "office_scene"

# Paths
def SRC_DOMAIN_PATH(d): return "data/pddl/domain/{}_domain.pddl".format(d)
def SRC_PROBLEM_PATH(s, d): return "data/pddl/problem/{}_{}_problem.pddl".format(s, d)
def LOG_PATH(t): return "result/{}".format(t)
MODIFIED_DATASET_PATH = "modified_scene_graphs.json"

# Template Domain (完全版)
BASE_OFFICE_DOMAIN = """
(define (domain office)
    (:requirements :strips :typing :adl)
    (:types agent room item)
    (:predicates
        (neighbor ?r1 - room ?r2 - room)
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (item_pickable ?i - item)
        (item_loadable ?i - item)
        (item_accessible ?i - item)
        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
        (item_in ?i1 - item ?i2 - item)
        (item_empty ?i - item)
        
        ; Added Predicates
        (item_dirty ?i - item)
        (item_clean ?i - item)
        (item_closed ?i - item)
        (item_open ?i - item)
        (item_is_sink ?i - item)
    )

    (:action goto
        :parameters (?a - agent ?r1 - room ?r2 - room)
        :precondition (and (agent_at ?a ?r1) (neighbor ?r1 ?r2))
        :effect (and (not(agent_at ?a ?r1)) (agent_at ?a ?r2))
    )

    (:action pick
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (item_accessible ?i) (item_pickable ?i) (not(agent_loaded ?a)) (not(agent_has_item ?a ?i)))
        :effect (and (agent_at ?a ?r) (not(item_at ?i ?r)) (agent_loaded ?a) (agent_has_item ?a ?i))
    )

    (:action drop
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (not(item_at ?i ?r)) (item_accessible ?i) (item_pickable ?i) (agent_loaded ?a) (agent_has_item ?a ?i))
        :effect (and (agent_at ?a ?r) (item_at ?i ?r) (not(agent_loaded ?a)) (not(agent_has_item ?a ?i)))
    )

    (:action pick_loadable
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (item_accessible ?i) (item_loadable ?i) (item_empty ?i) (not(agent_loaded ?a)) (not(agent_has_item ?a ?i)))
        :effect (and (agent_at ?a ?r) (not(item_at ?i ?r)) (agent_loaded ?a) (agent_has_item ?a ?i))
    )

    (:action drop_loadable
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (not(item_at ?i ?r)) (item_accessible ?i) (item_loadable ?i) (agent_loaded ?a) (agent_has_item ?a ?i))
        :effect (and (agent_at ?a ?r) (item_at ?i ?r) (not(agent_loaded ?a)) (not(agent_has_item ?a ?i)))
    )

    (:action load
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?i1 ?r) (item_loadable ?i1) (item_empty ?i1) (agent_loaded ?a) (agent_has_item ?a ?i2) (not(item_in ?i2 ?i1)))
        :effect (and (item_in ?i2 ?i1) (not(item_at ?i2 ?r)) (not(agent_loaded ?a)) (not(agent_has_item ?a ?i2)) (not(item_empty ?i1)))
    )

    (:action unload
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?i1 ?r) (item_loadable ?i1) (not(item_empty ?i1)) (not(agent_loaded ?a)) (not(agent_has_item ?a ?i2)) (item_in ?i2 ?i1))
        :effect (and (not(item_in ?i2 ?i1)) (item_at ?i2 ?r) (item_empty ?i1))
    )

    (:action wash
        :parameters (?a - agent ?i - item ?s - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?s ?r) (item_is_sink ?s) (agent_loaded ?a) (agent_has_item ?a ?i) (item_dirty ?i))
        :effect (and (not (item_dirty ?i)) (item_clean ?i))
    )

    (:action open
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (item_closed ?i) (item_accessible ?i))
        :effect (and (not (item_closed ?i)) (item_open ?i))
    )

    (:action close
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (item_open ?i) (item_accessible ?i))
        :effect (and (not (item_open ?i)) (item_closed ?i))
    )
)
"""

# --- API Retry Logic ---
def query_llm_with_retry(model_instance, max_retries=MAX_RETRIES):
    wait_time = INITIAL_WAIT_SECONDS
    for attempt in range(max_retries):
        try:
            return model_instance.query_msg_chain()
        except openai.RateLimitError as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "insufficient" in error_msg:
                print(f"\n[Fatal Error] API Quota Exceeded (残高不足): {e}")
                sys.exit(1)
            print(f"\n[Warning] Rate Limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, MAX_WAIT_SECONDS)
        except openai.APIConnectionError as e:
            print(f"\n[Error] Connection Error: {e}. Retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"\n[Error] Unhandled API Error: {e}")
            raise e 
    raise Exception(f"Max retries ({max_retries}) exceeded.")

def parse_llm_list(text):
    try:
        text = re.sub(r'^```[a-zA-Z]*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'```$', '', text, flags=re.MULTILINE)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match: return ast.literal_eval(match.group(0))
        return []
    except: return []

def apply_perturbations_to_sg(sg, modifications):
    new_sg = copy.deepcopy(sg)
    obj_ref_map = {} 
    def traverse_and_map(container):
        if "items" in container:
            for k, v in container["items"].items(): obj_ref_map[k] = {"data": v}
        if "assets" in container:
            for k, v in container["assets"].items():
                obj_ref_map[k] = {"data": v}
                traverse_and_map(v)
        if "rooms" in container:
             for k, v in container["rooms"].items(): traverse_and_map(v)
    traverse_and_map(new_sg)
    for k, info in obj_ref_map.items():
        if "sink" in k.lower(): info["data"]["accessible"] = True
    print(f"Applying {len(modifications)} modifications...")
    for mod in modifications:
        target_label = mod.get("label")
        target_state = mod.get("state")
        for ref_key, ref_val in obj_ref_map.items():
            if ref_key.startswith(target_label):
                if target_state: ref_val["data"]["state"] = target_state
                break
    return new_sg

# --- Domain Patching ---
def patch_domain_pddl(pddl_text):
    # すでにBASE_OFFICE_DOMAINで完結している場合はそのまま返す
    # 万が一不足があればここで補完
    if "(:action wash" not in pddl_text:
        # 簡易的な追記ロジック
        pddl_text = pddl_text.strip()
        if pddl_text.endswith(")"): pddl_text = pddl_text[:-1]
        # (必要なアクション定義をここに追加...今回は省略可)
        pddl_text += "\n)"
    return pddl_text

# --- Helper for Goal ---
def find_balanced_block(text, start_index):
    count = 0
    for i in range(start_index, len(text)):
        if text[i] == '(': count += 1
        elif text[i] == ')':
            count -= 1
            if count == 0: return i + 1
    return -1

def parse_pddl_typed_objects(pddl_text):
    """
    PDDLの (:objects ...) ブロックを解析し、型ごとのリストを作成します。
    正規表現よりも厳密に 'name - type' 構造を読み取ります。
    """
    objects_by_type = {}
    
    # (:objects ... ) ブロックを探す
    match = re.search(r'\(:objects([\s\S]*?)\)', pddl_text)
    if not match:
        return {}

    content = match.group(1)
    # コメント除去 (;以降を行末まで削除)
    content = re.sub(r';.*', '', content)
    # 改行やタブをスペースに変換して分割
    tokens = content.split()

    current_objs = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # '-' が来たら、その次は型名
        if token == '-':
            if i + 1 < len(tokens):
                obj_type = tokens[i+1]
                
                if obj_type not in objects_by_type:
                    objects_by_type[obj_type] = []
                
                # 蓄積していたオブジェクト名をこの型に登録
                objects_by_type[obj_type].extend(current_objs)
                current_objs = [] # リセット
                i += 2 # '-' と 'type' をスキップ
            else:
                break
        else:
            current_objs.append(token)
            i += 1
            
    return objects_by_type

# --- Improved Function: Final Robust Patching ---
def patch_problem_pddl(pddl_text, diff_report):
    
    # 1. オブジェクトの厳密な把握
    parsed_objs = parse_pddl_typed_objects(pddl_text)
    
    # 既存のオブジェクトリストを取得（なければ空リスト）
    rooms = parsed_objs.get("room", [])
    items = parsed_objs.get("item", [])
    agents = parsed_objs.get("agent", [])

    # フォールバック：部屋が認識できなかった場合
    if len(rooms) < 2:
        rooms = ["kitchen", "office", "corridor"]
        # PDDLテキストにも強制追記
        if "(:objects" in pddl_text:
            pddl_text = pddl_text.replace("(:objects", "(:objects\n        " + " ".join(rooms) + " - room")

    # 2. Diff情報の取得と整理
    changes = diff_report.get("state_changes", []) + diff_report.get("new_objects", []) if diff_report else []
    
    dirty_items = []
    closed_items = []
    items_inside = {} # {item: container}
    
    for c in changes:
        obj = c.get("object", c.get("object"))
        st = c.get("actual", c.get("state", ""))
        pert = c.get("perturbation_type", "")
        loc = c.get("location", "")

        if "dirty" in st or "dirty" in pert: dirty_items.append(obj)
        
        # コンテナ判定 (inside/closed)
        is_container = False
        if "closed" in st: 
            closed_items.append(obj)
            is_container = True
        
        if loc and "hidden" in pert:
            # 隠れている場所が部屋名でなければコンテナとみなす
            if loc not in rooms:
                items_inside[obj] = loc
                if loc not in closed_items: closed_items.append(loc)
                is_container = True
    
    # 3. 未定義アイテムの自動登録
    # Diffに出てくるが定義されていないアイテム/コンテナを追加
    all_known_items = set(items)
    items_to_add = []
    
    # コンテナの追加チェック
    for container in items_inside.values():
        if container not in all_known_items:
            items_to_add.append(container)
            all_known_items.add(container)
    
    # Initで使用されているが定義されていないアイテムのチェック (例: bed)
    # 簡易的に item_at ?i ?r の ?i をスキャン
    init_item_matches = re.findall(r'\(item_at\s+([\w-]+)\s+[\w-]+\)', pddl_text)
    for i_name in init_item_matches:
        if i_name not in all_known_items and i_name not in agents: # エージェントは除外
            items_to_add.append(i_name)
            all_known_items.add(i_name)

    if items_to_add:
        # PDDLに追記
        print(f"  [Patch] Adding missing items to :objects -> {items_to_add}")
        if "(:objects" in pddl_text:
            pddl_text = pddl_text.replace("(:objects", f"(:objects\n        {' '.join(items_to_add)} - item")
        items.extend(items_to_add)

    # 4. 初期状態(:init)の再構築
    # 既存の neighbor 定義などは壊れている可能性が高いため、
    # 信頼できる情報だけで init ブロックの一部を「追記」する形をとります。
    
    insertion = ""
    
    # 属性付与
    for i in dirty_items: insertion += f" (item_dirty {i})"
    for i in closed_items: insertion += f" (item_closed {i})"
    
    # コンテナ論理
    for item, container in items_inside.items():
        insertion += f" (item_in {item} {container})"
        insertion += f" (item_loadable {container})"
        insertion += f" (not (item_empty {container}))"
        # コンテナがどこにあるか不明なら、とりあえず最初の部屋へ
        if container not in init_item_matches:
            insertion += f" (item_at {container} {rooms[0] if rooms else 'office'})"

    # シンク
    insertion += " (item_is_sink sink_1) (item_accessible sink_1)"
    if "sink_1" not in all_known_items and "(:objects" in pddl_text:
        pddl_text = pddl_text.replace("(:objects", "(:objects\n        sink_1 - item")
    
    # --- 重要: 部屋の接続 (Neighbor) ---
    # エージェントを含めないよう、厳密に rooms リストだけを使用
    connections = []
    for r1 in rooms:
        for r2 in rooms:
            if r1 != r2:
                connections.append(f"(neighbor {r1} {r2})")
    
    # 既存の neighbor 定義が「エージェントを含む」など汚染されている場合、
    # それらが邪魔をするので、既存の neighbor を無効化できればベストですが、
    # 簡易的に「正しい定義を追加」して、プランナーが正しいパスを見つけられるようにします。
    # (seq-opt-lmcut なら不要なneighborがあっても、有効なパスがあれば解けます)
    insertion += " " + " ".join(connections)
    
    # エージェント配置
    if "agent_at" not in pddl_text:
        start_room = rooms[0] if rooms else "office"
        insertion += f" (agent_at agent1 {start_room})"

    pddl_text = pddl_text.replace("(:init", "(:init" + insertion)

    # 5. ゴールの再生成
    goal_predicates = []
    
    for c in changes:
        obj = c.get("object")
        expected_loc = c.get("expected_location", "table")
        perturbation = c.get("perturbation_type", "")
        
        # アイテムが items リストに含まれているか確認（エージェントをゴールにしない）
        if obj not in items: continue

        if "misplaced" in perturbation:
            target_room = expected_loc if expected_loc in rooms else (rooms[0] if rooms else "office")
            goal_predicates.append(f"(item_at {obj} {target_room})")
        
        if "dirty" in perturbation:
            goal_predicates.append(f"(item_clean {obj})")
        
        if "hidden" in perturbation:
            goal_predicates.append(f"(item_at {obj} {rooms[0] if rooms else 'table'})")

    # ゴールがない場合の保険
    if not goal_predicates and items:
        target = items[0]
        dest = rooms[0] if rooms else "office"
        goal_predicates.append(f"(item_at {target} {dest})")

    new_goal_str = f"(:goal (and {' '.join(goal_predicates)}))"

    # ゴールブロック置換
    goal_start = pddl_text.find("(:goal")
    if goal_start != -1:
        # バランスの取れた括弧を探す
        count = 0
        goal_end = -1
        for i in range(goal_start, len(pddl_text)):
            if pddl_text[i] == '(': count += 1
            elif pddl_text[i] == ')':
                count -= 1
                if count == 0:
                    goal_end = i + 1
                    break
        
        if goal_end != -1:
            pddl_text = pddl_text[:goal_start] + new_goal_str + pddl_text[goal_end:]
        else:
            pddl_text += "\n" + new_goal_str
    else:
        last_paren = pddl_text.rfind(")")
        pddl_text = pddl_text[:last_paren] + "\n" + new_goal_str + "\n)"

    print(f"  [Patch] Generated {len(goal_predicates)} recovery goals.")
    return pddl_text

# --- Main Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", nargs="+")
    parser.add_argument("-m", "--model", default=DEFAULT_LLM)
    parser.add_argument("-t", "--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("-e", "--episode", type=int, default=EPISODE)
    parser.add_argument("-p", "--print-prompt", action="store_true", default=True)
    parser.add_argument("-r", "--print-response", action="store_true", default=True)
    parser.add_argument("--print-plan", action="store_true", default=False)
    # ここでデフォルトを DOMAIN_QUERY ("office") に設定
    parser.add_argument("-d", "--domain", default=DOMAIN_QUERY)
    parser.add_argument("--domain-example", default=DOMAIN_EXAMPLE)
    parser.add_argument("-s", "--scene", default=SCENE_QUERY)
    parser.add_argument("--scene-example", default=SCENE_EXAMPLE)
    parser.add_argument("--no-plan", action="store_true", default=False)
    parser.add_argument("--max-time", type=float, default=MAX_TIME)
    parser.add_argument("--use-diff", action="store_true", default=False)
    parser.add_argument("--problem-id", default="problem_1")
    args = parser.parse_args()

    # Load Data
    modified_data = {}
    if args.use_diff and os.path.isfile(MODIFIED_DATASET_PATH):
        with open(MODIFIED_DATASET_PATH, "r") as f: modified_data = json.load(f).get("problems", {})
        print(f"Loaded modified dataset: {len(modified_data)} problems.")

    # Load Examples
    try:
        with open(SRC_DOMAIN_PATH(args.domain_example), "r") as f: domain_exp = f.read()
        with open(SRC_PROBLEM_PATH(args.scene_example, args.domain_example), "r") as f: problem_exp = f.read()
    except FileNotFoundError:
        # ファイルがない場合のダミー
        domain_exp = ""
        problem_exp = ""

    exp = example.get_example(args.domain_example)
    qry = example.get_example(args.domain)
    
    if args.use_diff:
        domain_exp_lite = ""
        problem_exp_lite = ""
    else:
        domain_exp_lite = domain_exp
        problem_exp_lite = problem_exp

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S/")
    
    # --- 修正: data_list の初期化をここに配置 ---
    data_list = [] 
    
    model = llm.load_llm(args.model, args.temperature, args.top_p)

    for e in range(args.episode):
        model.reset()
        log_path = os.path.join(LOG_PATH(curr_time), "e_{:03}/".format(e))
        Path(log_path).mkdir(parents=True, exist_ok=True)

        try:
            scene_exp = load_scene_graph(args.scene_example)
            scene_ideal = load_scene_graph(args.scene)
        except:
            # シーングラフ読み込み失敗時のフォールバック（空辞書）
            scene_exp = {}
            scene_ideal = {}

        scene_current = scene_ideal 
        diff_report = {}

        if args.use_diff and args.problem_id in modified_data:
            print(f"\n--- [Episode {e+1}] Generating Current State for {args.problem_id} ---")
            scene_current = apply_perturbations_to_sg(scene_ideal, modified_data[args.problem_id].get("current_scene_graph", []))
            diff_report = compare_scene_graphs(scene_ideal, scene_current)
            with open(os.path.join(log_path, "diff_report.json"), "w") as f: json.dump(diff_report, f, indent=2)
            scene_qry = scene_current
        else:
            scene_qry = scene_ideal

        d_tar_file = os.path.join(log_path, "{}_domain.pddl".format(args.domain))
        p_tar_file = os.path.join(log_path, "{}_{}_problem.pddl".format(args.scene, args.domain))
        
        # --- Stage 1: Domain ---
        if args.use_diff:
            print("  [Diff Mode] Using Template for Domain (Skipping LLM)...")
            domain_tar = BASE_OFFICE_DOMAIN
            # BASE_OFFICE_DOMAINの定義が(domain office)であることを確認
            # args.domainがofficeであればそのまま使用可能
            if args.domain != "office":
                # ドメイン名が違う場合は置換する
                domain_tar = domain_tar.replace("(domain office)", f"(domain {args.domain})")
            
            domain_tar = patch_domain_pddl(domain_tar)
        else:
            content_d, prompt_d = p.nl_2_pddl_domain(domain_exp, args.domain, exp.get("add_obj",[]), qry.get("add_obj",[]), exp.get("add_act",[]), qry.get("add_act",[]))
            model.init_prompt_chain(content_d, prompt_d)
            domain_tar = query_llm_with_retry(model)
            model.update_prompt_chain_w_response(domain_tar)

        llm_utils.export_result(domain_tar, d_tar_file)

        # --- Stage 2: Pruning (Skipped in Diff Mode) ---
        if not args.use_diff:
            items_exp = extract_accessible_items_from_sg(scene_exp)
            items_qry = extract_accessible_items_from_sg(scene_qry) 
            content_pr, prompt_pr = p.nl_prune_item(items_exp, items_qry, exp.get("goal",""), qry.get("goal",""), exp.get("item_keep",[]), domain_exp, domain_tar)
            model.update_prompt_chain(content_pr, prompt_pr)
            prune_tar = query_llm_with_retry(model)
            item_keep = parse_llm_list(prune_tar)
            model.update_prompt_chain_w_response(prune_tar)
            scene_exp = prune_sg_with_item(scene_exp, exp.get("item_keep",[]))
            scene_qry = prune_sg_with_item(scene_qry, item_keep)

        # --- Stage 3: Problem ---
        content_p, prompt_p = p.sg_2_pddl_problem(
            args.domain_example, domain_exp_lite, problem_exp_lite, 
            scene_exp, scene_qry, exp.get("goal",""), qry.get("goal",""), domain_tar, args.domain
        )
        
        if args.use_diff and diff_report:
            diff_str = json.dumps(diff_report, indent=2)[:500] 
            prompt_p += f"\n\n[UPDATE] State changes:\n{diff_str}\nPlan recovery actions."

        if args.use_diff:
            model.init_prompt_chain(content_p, prompt_p)
        else:
            model.update_prompt_chain(content_p, prompt_p)
            
        problem_tar = query_llm_with_retry(model)
        
        if args.use_diff:
            print("  [Diff Mode] Patching Problem PDDL...")
            problem_tar = patch_problem_pddl(problem_tar, diff_report)
            
            # ドメイン名の一致確認
            if f"(:domain {args.domain})" not in problem_tar:
                # (:domain something) を (:domain args.domain) に置換
                problem_tar = re.sub(r'\(:domain\s+\w+\)', f'(:domain {args.domain})', problem_tar)

        llm_utils.export_result(problem_tar, p_tar_file)
        
        if args.no_plan: continue

        # --- Planner ---
        if not os.path.isfile(d_tar_file): d_tar_file = SRC_DOMAIN_PATH(args.domain)
        
        # PDDLGymへの登録と実行
        # 注意: pddlgym.registerは名前重複エラーが出やすいため、try-catchやチェックが必要だが
        # ここではplannerモジュール側で制御されていると仮定
        planner.export_domain_to_pddlgym(args.domain, d_tar_file)
        planner.export_problem_to_pddlgym(args.domain, p_tar_file, p_idx="00", clear_dir=True)
        planner.register_new_pddlgym_env(args.domain)

        plan, plan_time, node, cost, exit_code = planner.query_pddlgym(args.domain, max_time=args.max_time)
        
        is_valid = False
        if exit_code == 1: # 成功時
            plan_file = os.path.join(log_path, "{}_{}.plan".format(args.domain, args.scene))
            with open(plan_file, "w") as pf: pf.write("\n".join(plan))
            is_valid = True

        print(f"Episode {e+1}/{args.episode}, Valid Plan Found: {is_valid}")
        # --- 修正: 定義済みの data_list に追加 ---
        data_list.append([e, is_valid, args.scene, args.use_diff])

    # ログ出力
    df = pd.DataFrame(data_list, columns=["Episode", "PlanFound", "Scene", "UseDiff"])
    df.to_csv(os.path.join(LOG_PATH(curr_time), "log.csv"))
    print("Done.")