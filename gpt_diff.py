import os
import json
import math
import sys

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from planner import query
except ImportError as e:
    print(f"❌ Error: planner.py が見つかりません。")
    sys.exit(1)

PDDL_DIR = os.path.join(BASE_DIR, "pddl_generated")
os.makedirs(PDDL_DIR, exist_ok=True)
IDEAL_JSON = os.path.join(BASE_DIR, "ideal_scenegraph.json")
REAL_JSON = os.path.join(BASE_DIR, "real_scenegraph.json")

# ---------------------------------------------------------
# ドメイン定義 (修正版)
# ---------------------------------------------------------
DOMAIN_PDDL_CONTENT = """(define (domain office)
  (:requirements :strips :typing :negative-preconditions)
  (:types agent room item)
  (:predicates
    (neighbor ?r1 - room ?r2 - room)
    (agent_at ?a - agent ?r - room)
    (item_at ?i - item ?r - room)
    (agent_loaded ?a - agent)
    (agent_has_item ?a - agent ?i - item)
    (can_graspable ?i - item)
    (can_liftable ?i - item)
    (can_accessible ?i - item)
  )

  (:action goto
    :parameters (?a - agent ?r1 - room ?r2 - room)
    :precondition (and (agent_at ?a ?r1) (neighbor ?r1 ?r2))
    :effect (and (not (agent_at ?a ?r1)) (agent_at ?a ?r2))
  )

  (:action pick
    :parameters (?a - agent ?i - item ?r - room)
    :precondition (and (agent_at ?a ?r) (item_at ?i ?r) (can_accessible ?i) (can_graspable ?i) (can_liftable ?i) (not (agent_loaded ?a)))
    :effect (and (not (item_at ?i ?r)) (agent_loaded ?a) (agent_has_item ?a ?i))
  )

  (:action drop
    :parameters (?a - agent ?i - item ?r - room)
    :precondition (and (agent_at ?a ?r) (agent_has_item ?a ?i))
    :effect (and (item_at ?i ?r) (not (agent_loaded ?a)) (not (agent_has_item ?a ?i)))
  )
)"""

def ensure_domain_file():
    d_path = os.path.join(PDDL_DIR, "office.pddl")
    with open(d_path, "w") as f:
        f.write(DOMAIN_PDDL_CONTENT.strip())
    return d_path

def detect_json_diff_and_update_pddl():
    if not os.path.exists(IDEAL_JSON) or not os.path.exists(REAL_JSON):
        return None, "❌ JSONファイルが見つかりません。"

    with open(IDEAL_JSON, 'r') as f: ideal = json.load(f)
    with open(REAL_JSON, 'r') as f: real = json.load(f)

    move_targets, affordance_facts = [], []
    real_nodes = real.get("nodes", [])
    ideal_nodes = ideal.get("nodes", [])

    for i, real_obj in enumerate(real_nodes):
        uid = f"{real_obj.get('label', 'bottle')}_{i}"
        affs = real_obj.get("properties", {}).get("affordance", [])
        
        # 検出された物体は基本的に「アクセス可能」とする
        affordance_facts.append(f"(can_accessible {uid})")
        
        for aff in affs:
            if f"can_{aff}" in ["can_graspable", "can_liftable"]:
                affordance_facts.append(f"(can_{aff} {uid})")
        
        curr_pos = real_obj["properties"]["position_3d"]
        if i < len(ideal_nodes):
            ideal_pos = ideal_nodes[i]["properties"]["position_3d"]
            dist = math.sqrt(sum((i_p - r_p)**2 for i_p, r_p in zip(ideal_pos, curr_pos)))
            if dist > 0.05:
                move_targets.append({"id": uid, "dist": round(dist, 3)})

    if not move_targets:
        return None, "✅ すべての位置が正しいです。"

    target_ids = [t["id"] for t in move_targets]
    init_at = "\n    ".join([f"(item_at {tid} room_initial)" for tid in target_ids])
    init_aff = "\n    ".join(affordance_facts)
    goal_at = "\n    ".join([f"(item_at {tid} room_target)" for tid in target_ids])

    # 修正: (empty-hand) を削除し、(agent_loaded) も書かないことで「持っていない」状態にする
    new_pddl = f"""(define (problem fix_scene)
  (:domain office)
  (:objects
    agent1 - agent
    room_initial room_target - room
    {" ".join(target_ids)} - item
  )
  (:init
    (agent_at agent1 room_initial)
    (neighbor room_initial room_target)
    (neighbor room_target room_initial)
    {init_at}
    {init_aff}
  )
  (:goal (and {goal_at}))
)"""
    summary = ", ".join([f"{t['id']}({t['dist']}m)" for t in move_targets])
    return new_pddl, f"🔍 ズレ検知: {summary}"

def run_planning_process():
    domain_file = ensure_domain_file()
    problem_file = os.path.join(PDDL_DIR, "problem0.pddl")
    
    # 実行前に問題ファイルが存在するか確認
    if not os.path.exists(problem_file):
        return "❌ problem0.pddl がありません。先に 'compare' を実行してください。"

    print(f"⚙️ プランナー起動中...")
    plan_str, cost, time, code, err = query(domain_file, problem_file)

    if code == 1 and plan_str:
        res = "✅ プラン生成成功:\n"
        for i, line in enumerate(plan_str.strip().split('\n')):
            if line.startswith('('):
                res += f"  {i+1}. {line}\n"
        return res + f"\n(Time: {time}s, Cost: {cost})"
    else:
        # 詳細なエラーを出力
        return f"❌ プランニング失敗。PDDLの整合性を確認してください:\n{err}"

if __name__ == "__main__":
    ensure_domain_file()
    print("\n🤖 PDDL アフォーダンス プランナー (直接実行版)")
    while True:
        try:
            cmd = input("\n👤 指示を待機中 (compare/plan/exit): ").strip().lower()
            if cmd == "compare":
                new_pddl, msg = detect_json_diff_and_update_pddl()
                print(msg)
                if new_pddl:
                    with open(os.path.join(PDDL_DIR, "problem0.pddl"), "w") as f: f.write(new_pddl)
                    print("📝 problem0.pddl を更新しました。")
            elif cmd == "plan":
                print(run_planning_process())
            elif cmd == "exit": break
        except KeyboardInterrupt: break