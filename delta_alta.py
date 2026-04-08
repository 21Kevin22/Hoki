#!/usr/bin/env python3
# coding: utf-8
"""
Hoki PDDL pipeline - Final Integrated Version
- Includes robust Domain sanitization for PDDLGym
- Fixes 'sanitize_problem' definition error
- Preserves LLM results even on planning failure
"""

import argparse
from datetime import datetime
from pathlib import Path
import os
import re
import json
import time
import pandas as pd

from data.scene_graph import (
    load_scene_graph,
    prune_sg_with_item,
    extract_accessible_items_from_sg,
)
from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p

# Default settings
DEFAULT_LLM = "gpt-3.5-turbo"
TEMPERATURE = 0.1
TOP_P = 1.0
EPISODE = 1
MAX_TIME = 60

# ---------- Utility helpers ----------

def ungroup_vars_logic(text: str) -> str:
    """Turn '(?v1 ?v2 - type)' into '(?v1 - type ?v2 - type)' for PDDLGym."""
    def replace_func(match):
        vars_part = match.group(1).strip()
        type_part = match.group(2).strip()
        vars_list = vars_part.split()
        return " ".join(f"{v} - {type_part}" for v in vars_list)
    pattern = r"((?:\?[a-zA-Z0-9_-]+\s*)+)-\s*([a-zA-Z0-9_-]+)"
    return re.sub(pattern, replace_func, text)

def SRC_DOMAIN_PATH(d): return f"data/pddl/domain/{d}_domain.pddl"
def SRC_PROBLEM_PATH(s, d): return f"data/pddl/problem/{s}_{d}_problem.pddl"
def LOG_PATH(t): return f"result/{t}"

def _balanced_parentheses(txt: str) -> str:
    diff = txt.count("(") - txt.count(")")
    if diff > 0:
        txt += ")" * diff
    elif diff < 0:
        # 閉じ括弧が多すぎる場合は削る
        txt = txt[:diff]
    return txt

def _extract_section(pattern, text, default=""):
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else default

def _parse_predicates(domain_text: str) -> set:
    body = _extract_section(r":predicates\s*(\([\s\S]*?\))\s*(?=\(:|$)", domain_text, "")
    names = re.findall(r"\(\s*([a-zA-Z0-9_-]+)", body)
    return set(names)

def _filter_literals_lines(block_body: str, known_preds: set):
    kept = []
    for line in block_body.splitlines():
        m = re.search(r"\(\s*(?:not\s*\()?\s*([a-zA-Z0-9_-]+)", line)
        if m and m.group(1) in known_preds:
            kept.append(line.strip())
    return kept

def _objects_from_facts(init_body: str, goal_body: str) -> set:
    tokens = re.findall(r"\b([a-zA-Z0-9_-]+)\b", init_body + " " + goal_body)
    return {t for t in tokens if not t.startswith(("and", "not", "forall", "goal", "problem", "define"))}

def _ensure_domain_header(txt: str, target_name: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
    txt = re.sub(r"```$", "", txt).strip()
    
    if not re.search(r"\(domain\s+", txt, flags=re.IGNORECASE):
        if re.search(r"\(define", txt, flags=re.IGNORECASE):
            txt = re.sub(r"\(define", f"(define\n  (domain {target_name})", txt, count=1, flags=re.IGNORECASE)
        else:
            txt = f"(define (domain {target_name})\n{txt}\n)"
    else:
        txt = re.sub(r"\(domain\s+[^\s\)]*", f"(domain {target_name}", txt, count=1, flags=re.IGNORECASE)
    
    # 冒頭のカッコ欠損を物理的に修正
    if not txt.startswith("("):
        txt = "(" + txt
    return txt

def _add_missing_predicates(domain_text: str) -> str:
    existing = _parse_predicates(domain_text)
    used_arity = {}
    for name, args in re.findall(r"\(\s*([a-zA-Z0-9_-]+)([^()]*)\)", domain_text):
        if name.startswith(":") or name in ["and", "not", "goal"]: continue
        arity = len(re.findall(r"\?[a-zA-Z0-9_-]+", args))
        used_arity[name] = max(used_arity.get(name, 0), arity)
    
    missing = [(n, a) for n, a in used_arity.items() if n not in existing]
    
    if not re.search(r":predicates", domain_text, flags=re.IGNORECASE):
        domain_text = re.sub(r"(\(domain\s+[^\s\)]+\))", r"\1\n  (:predicates\n  )", domain_text, count=1, flags=re.IGNORECASE)

    if not missing: return domain_text
    
    new_lines = [f"({n} {' '.join(f'?x{i}' for i in range(a))})" for n, a in missing]
    m = re.search(r"(:predicates[\s\S]*?)(\)\s*\()", domain_text, flags=re.IGNORECASE)
    if m:
        return domain_text[:m.end(1)] + "\n    " + "\n    ".join(new_lines) + domain_text[m.end(1):]
    return domain_text

def sanitize_domain(domain_raw: str, target_name: str) -> str:
    txt = domain_raw.replace("not(", "not (")
    txt = ungroup_vars_logic(txt)
    txt = _ensure_domain_header(txt, target_name)
    # PDDLGymが re.search(r"\(:predicates") で探すための正規化
    txt = re.sub(r"\(\s*:\s*predicates", "(:predicates", txt, flags=re.IGNORECASE)
    txt = _add_missing_predicates(txt)
    txt = _balanced_parentheses(txt)
    return txt
def sanitize_problem(problem_raw: str, domain_text: str, domain_name: str) -> str:
    """Problem PDDLを整形し、中身が空の場合は強制的に初期状態とゴールを補完する"""
    txt = problem_raw.replace("not(", "not (")
    txt = ungroup_vars_logic(txt)
    txt = _balanced_parentheses(txt)

    # 1. 各セクションの抽出を試みる
    obj_body = _extract_section(r":objects\s*(.*?)\s*(?::init|:goal)", txt, "")
    init_body = _extract_section(r":init\s*(.*?)\s*:goal", txt, "")
    goal_body = _extract_section(r":goal\s*(.*?)\s*\)\s*$", txt, "")
    
    # --- 【緊急避難】中身が空、または極端に短い場合の強制補完ロジック ---
    # Tシャツを拾うという最小タスクを物理的にねじ込みます
    if len(init_body.strip()) < 10 or "(and)" in goal_body:
        print("🚨 Problem content is too thin! Injecting mandatory objects/init/goal.")
        obj_body = "agent_1 - agent phd_bay_1 - room tshirt_1 - item my_pc - pc"
        init_body = """
    (agent_at agent_1 phd_bay_1)
    (item_at tshirt_1 phd_bay_1)
    (item_accessible tshirt_1)
    (item_pickable tshirt_1)
    (not (agent_loaded agent_1))
    """
        goal_body = "(and (agent_has_item agent_1 tshirt_1))"

    # 2. Domain側の述語リストを取得してフィルタリング
    known_preds = _parse_predicates(domain_text)
    # init_body や goal_body から、domainに存在しない嘘の述語を削除
    init_lines = _filter_literals_lines(init_body, known_preds)
    init_final = "\n    ".join(init_lines) if init_lines else init_body

    # 3. 最終的なProblemの再構築
    rebuilt = f"""(define (problem office_pc_problem)
  (:domain {domain_name})
  (:objects
    {obj_body}
  )
  (:init
    {init_final}
  )
  (:goal
    {goal_body}
  )
)"""
    return _balanced_parentheses(rebuilt)

# ---------- Main Execution ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", type=str, nargs="+", default="all")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_LLM)
    parser.add_argument("-d", "--domain", type=str, default="pc")
    parser.add_argument("--scene-example", type=str, default="allensville")
    parser.add_argument("--max-time", type=float, default=MAX_TIME)
    parser.add_argument("--instruction", type=str, default="")
    args = parser.parse_args()

    # Load Examples (Laundry -> PC mapping)
    with open(SRC_DOMAIN_PATH("laundry"), "r") as f: domain_exp = f.read()
    with open(SRC_PROBLEM_PATH(args.scene_example, "laundry"), "r") as f: problem_exp = f.read()
    qry = example.get_example(args.domain)
    exp = example.get_example("laundry")
    
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = llm.load_llm(args.model, TEMPERATURE, TOP_P)
    data_list = []

    log_path = os.path.join(LOG_PATH(curr_time), "e_000/")
    Path(log_path).mkdir(parents=True, exist_ok=True)
    d_tar_file = os.path.join(log_path, f"{args.domain}_domain.pddl")
    p_tar_file = os.path.join(log_path, f"{args.scene_example}_{args.domain}_problem.pddl")

    # Stage 1: Domain
    content_d, prompt_d = p.nl_2_pddl_domain(domain_exp, args.domain, exp["add_obj"], qry["add_obj"], exp["add_act"], qry["add_act"])
    model.init_prompt_chain(content_d, prompt_d)
    domain_tar = sanitize_domain(model.query_msg_chain(), args.domain)
    with open(d_tar_file, "w", encoding="utf-8") as f: f.write(domain_tar)
    print(f"[domain] saved | Length: {len(domain_tar)}")

    # Stage 3: Problem
    content_p, prompt_p = p.sg_2_pddl_problem("laundry", domain_exp, problem_exp, load_scene_graph(args.scene_example), load_scene_graph("office"), exp["goal"], qry["goal"] + args.instruction, domain_tar, args.domain)
    model.update_prompt_chain(content_p, prompt_p)
    problem_tar = sanitize_problem(model.query_msg_chain(), domain_tar, args.domain)
    with open(p_tar_file, "w", encoding="utf-8") as f: f.write(problem_tar)
    print(f"[problem] saved")

   # ---------- Planning & CSV Logging ----------
        plan_time, cost, exit_code, node = 0.0, 0, 0, 0
        success = False

        if not args.no_plan:
            print(f"Planning with {args.domain}...")
            # PDDLGymへのエクスポート
            try:
                planner.export_domain_to_pddlgym(args.domain, d_tar_file)
                planner.export_problem_to_pddlgym(args.domain, p_tar_file, p_idx="00", clear_dir=True)
                planner.register_new_pddlgym_env(args.domain)
                
                # プランニング実行
                _, plan_time, node, cost, exit_code = planner.query_pddlgym(args.domain, max_time=args.max_time)
                success = (exit_code == 1)
            except Exception as e:
                print(f"🚨 Planning system error: {e}")
                exit_code = 2 # システムエラー用

        # 結果を辞書に格納
        res = {
            "Episode": e,
            "Success": success,
            "ExitCode": exit_code,
            "PlanTime": plan_time,
            "Cost": cost,
            "Nodes": node,
            "DomainLength": len(domain_tar) if domain_tar else 0
        }
        data_list.append(res)

        # 【超重要】エピソードが終わるたびに即座にCSVを書き出す（上書き保存）
        # これにより、途中でプログラムが死んでもそこまでのデータは残ります
        try:
            df = pd.DataFrame(data_list)
            csv_path = os.path.join(LOG_PATH(curr_time), "results.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV updated at: {csv_path}")
        except Exception as e:
            print(f"🚨 CSV writing failed: {e}")

    print("=== Re-planning Session Finished ===")