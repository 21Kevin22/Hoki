#!/usr/bin/env python3
# coding: utf-8
"""
Hoki PDDL Pipeline: Modular, Robust, and Evaluation-Ready.
Supported Models: GPT-3.5-Turbo, GPT-4o, etc.
"""

import argparse
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple, List, Dict, Any

from data.scene_graph import load_scene_graph
from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p

# ---------- 定数・パス設定 ----------
DEFAULT_LLM = "gpt-4o"
TEMPERATURE = 0.1

def SRC_DOMAIN_PATH(d): return f"data/pddl/domain/{d}_domain.pddl"
def SRC_PROBLEM_PATH(s, d): return f"data/pddl/problem/{s}_{d}_problem.pddl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ---------- PDDL 整形ロジック (Sanitizer) ----------

def ungroup_vars_logic(text: str) -> str:
    """(?v1 ?v2 - type) を展開してパースエラーを防ぐ"""
    def replace_func(match):
        vars_list = match.group(1).strip().split()
        type_part = match.group(2).strip()
        return " ".join(f"{v} - {type_part}" for v in vars_list)
    pattern = r"((?:\?[a-zA-Z0-9_-]+\s*)+)-\s*([a-zA-Z0-9_-]+)"
    return re.sub(pattern, replace_func, text)

def sanitize_pddl_optimized(raw_text: str, domain_pddl: str, domain_name: str) -> Tuple[str, int]:
    """
    GPT-4o等の出力を整形し、修正回数をカウントする。
    Returns: (整形済みテキスト, 修正回数)
    """
    correction_count = 0
    
    # 1. Markdownタグと余計な説明文の除去
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw_text)
    text = text.replace("```", "").strip()
    if text != raw_text.strip(): correction_count += 1

    # 2. 構文の正規化
    text = text.replace("not(", "not (")
    text = text.replace("( )", "") # 孤立した括弧の除去
    text = ungroup_vars_logic(text).lower()

    # 3. 述語リストの取得（オブジェクト混入防止用）
    predicates = re.findall(r"\(\s*([a-z0-9_-]+)", domain_pddl.lower())
    forbidden = set(predicates) | {
        "and", "not", "or", "forall", "exists", "define", "domain", 
        "problem", "objects", "init", "goal", "begin", "task", "item_in", "item_at"
    }

    # 4. セクション抽出 (堅牢な正規表現)
    init_match = re.search(r":init\s*(.*?)\s*(?=:goal|\))", text, flags=re.DOTALL)
    goal_match = re.search(r":goal\s*(.*?)\s*(?=\)\s*\)|$)", text, flags=re.DOTALL)
    
    init_body = init_match.group(1).strip() if init_match else ""
    goal_body = goal_match.group(1).strip() if goal_match else "(and)"

    # 5. オブジェクトの抽出とフィルタリング
    tokens = re.findall(r"\b[a-z][a-z0-9_]*\b", init_body + " " + goal_body)
    valid_objs = sorted({t for t in tokens if t not in forbidden and len(t) > 1})

    # 6. 論理的補完 (経路とアフォーダンス)
    if "neighbor" not in init_body:
        init_body += "\n  (neighbor kitchen table) (neighbor table kitchen) (neighbor table basket) (neighbor basket table)"
        correction_count += 1

    # 7. 再構成
    final_pddl = f"""(define (problem task_replanned)
  (:domain {domain_name})
  (:objects
  {" ".join(valid_objs)}
  )
  (:init
  {init_body}
  )
  (:goal
  (and {goal_body.replace('(and', '').replace(')', '').strip()})
  )
)"""
    return final_pddl, correction_count

# ---------- 差分検知ロジック ----------

def extract_facts(pddl_text: str) -> Set[str]:
    facts = re.findall(r"\([a-z0-9_][a-z0-9_\s-]*\)", pddl_text.lower())
    reserved = {"and", "not", "or", "forall", "exists"}
    return {f for f in facts if f.strip("()").split()[0] not in reserved}

def get_pddl_diff_summary(ref_pddl: str, cur_pddl: str) -> str:
    ref_facts = extract_facts(ref_pddl)
    cur_facts = extract_facts(cur_pddl)
    missing = ref_facts - cur_facts
    extra = cur_facts - ref_facts
    
    summary = []
    if missing: summary.append(f"Missing facts: {', '.join(list(missing)[:5])}")
    if extra: summary.append(f"Extra facts: {', '.join(list(extra)[:5])}")
    return "\n".join(summary) if summary else "No differences detected."

# ---------- パイプライン・ステージ ----------

def run_diff_replan_stage(args, model, domain_pddl, current_problem, ref_pddl_path) -> str:
    """差分に基づき再計画し、定量的データを記録する"""
    start_time_total = time.time()
    
    # 正解データの読み込み
    if not os.path.exists(ref_pddl_path):
        logging.warning(f"Reference file {ref_pddl_path} not found.")
        return current_problem

    with open(ref_pddl_path, "r") as f:
        ref_pddl = f.read()

    diff_summary = get_pddl_diff_summary(ref_pddl, current_problem)
    logging.info(f"[Diff Detected]\n{diff_summary}")

    # LLM推論
    content = f"Target state: {ref_pddl}\nCurrent state: {current_problem}\nDifferences: {diff_summary}"
    prompt = (
    f"Fix the current PDDL to achieve target. {args.instruction}\n"
    f"Ensure objects are filtered.\n"
    f"CRITICAL INSTRUCTION: Output ONLY the raw PDDL string. "
    f"Do NOT include markdown formatting like ```pddl or ```. "
    f"Do NOT output any conversational text, explanations, or greetings. "
    f"Just the code."
    )
    
    infer_start = time.time()
    # modelの状態を確認して初期化または更新
    if not hasattr(model, 'prompt_chain') or not model.prompt_chain:
        model.init_prompt_chain(content, prompt)
    else:
        try:
            model.update_prompt_chain(content, prompt)
        except:
            model.init_prompt_chain(content, prompt)
            
    raw_output = model.query_msg_chain()
    latency_api = time.time() - infer_start

    # 後処理（修正回数カウント付き）
    final_pddl, corrections = sanitize_pddl_optimized(raw_output, domain_pddl, args.domain)
    
    # 定量評価ログ
    logging.info(f"--- Evaluation [{args.model}] ---")
    logging.info(f"Latency (API): {latency_api:.2f}s")
    logging.info(f"Auto-Corrections: {corrections}")
    logging.info(f"Total Stage Time: {time.time() - start_time_total:.2f}s")
    
    return final_pddl

# ---------- メイン処理 ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_LLM)
    parser.add_argument("-d", "--domain", type=str, default="pc")
    parser.add_argument("-s", "--scene", type=str, default="office")
    parser.add_argument("--domain-example", type=str, default="laundry")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--ref-pddl", type=str, default="office_pc_domain.pddl")
    args = parser.parse_args()

    # ログ・出力先準備
    log_dir = os.path.join("result", datetime.now().strftime("%Y%m%d_%H%M%S"), "e_000")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ロード
    model = llm.load_llm(args.model, TEMPERATURE, 1.0)
    
    # ドメインファイル存在チェック
    domain_path = SRC_DOMAIN_PATH(args.domain)
    if not os.path.exists(domain_path):
        logging.error(f"Missing domain file: {domain_path}")
        exit(1)
    with open(domain_path, "r") as f:
        domain_pddl = f.read()

    # 現状（Problem）の取得
    prob_path = SRC_PROBLEM_PATH(args.scene, args.domain)
    if os.path.exists(prob_path):
        with open(prob_path, "r") as f:
            initial_problem = f.read()
    else:
        # なければ最小限のテンプレートから開始
        initial_problem = f"(define (problem task) (:domain {args.domain}) (:objects ) (:init ) (:goal (and )))"

    # 再計画実行
    try:
        final_problem = run_diff_replan_stage(args, model, domain_pddl, initial_problem, args.ref_pddl)
        
        # 結果保存
        final_path = os.path.join(log_dir, "final_replanned_problem.pddl")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_problem)
        
        logging.info(f"SUCCESS: Result saved to {final_path}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")