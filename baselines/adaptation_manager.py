import os
import re
from dataclasses import dataclass, field
from typing import Set, Tuple, Optional, Any

@dataclass
class TaskState:
    """タスクの状態（ゴールとシーングラフの要素）を保持するデータクラス"""
    goal: str
    scene_items: Set[str] = field(default_factory=set)

    def __eq__(self, other):
        if not isinstance(other, TaskState):
            return False
        return self.goal == other.goal and self.scene_items == other.scene_items

class AdaptationManager:
    """環境変化の検知と適応戦略の生成を管理するクラス"""

    def __init__(self):
        self.previous_state: Optional[TaskState] = None

    def _get_current_state(self, goal_str: str, scene_graph: Any) -> TaskState:
        """シーングラフオブジェクトから状態データを抽出"""
        items = set()
        if hasattr(scene_graph, "nodes"):
            for node in scene_graph.nodes:
                name = node if isinstance(node, str) else getattr(node, "name", str(node))
                items.add(name)
        return TaskState(goal=goal_str, scene_items=items)

    def _detect_diff(self, current: TaskState, previous: TaskState) -> str:
        """差分をテキスト形式で返す"""
        if current == previous:
            return ""
        
        diff_msg = []
        if current.goal != previous.goal:
            diff_msg.append(f"- Goal Changed: '{previous.goal}' -> '{current.goal}'")
        
        added = current.scene_items - previous.scene_items
        removed = previous.scene_items - current.scene_items
        
        if added: diff_msg.append(f"- Objects Added: {list(added)[:5]}...")
        if removed: diff_msg.append(f"- Objects Removed: {list(removed)[:5]}...")
        
        return "\n".join(diff_msg)

    def _query_llm_for_strategy(self, model: Any, diff_text: str, current_goal: str) -> Tuple[str, str]:
        """LLMに問い合わせて戦略を作成"""
        
        system_content = "You are an expert robot planner developer specialized in adaptive planning."
        
        # トリプルクォートを使わず、改行コード(\n)で連結する方法に変更
        user_prompt = (
            "The environment or goal has changed compared to the previous episode.\n"
            "I need you to generate a Python strategy code snippet to handle this change, and a brief advice for the planner.\n\n"
            "Differences detected:\n" + diff_text + "\n\n"
            "Current Goal: " + current_goal + "\n\n"
            "Please provide:\n"
            "1. \"Explanation\": A short advice on how to adjust the plan.\n"
            "2. \"Code\": A python code snippet that could theoretically preprocess the data.\n\n"
            "Output format:\n"
            "Explanation: [Your advice here]\n"
            "Code:\n"
            "```python\n"
            "[Your code here]\n"
            "```"
        )

        # モデルの仕様に合わせて呼び出し（__code__を使用）
        if hasattr(model, "query") and hasattr(model.query, "__code__") and model.query.__code__.co_argcount >= 3:
            response = model.query(system_content, user_prompt)
        else:
            response = model.query(user_prompt)

        explanation = ""
        code = ""
        
        # 正規表現による抽出
        expl_match = re.search(r"Explanation:\s*(.*?)\s*(?:Code:|$)", response, re.DOTALL | re.IGNORECASE)
        if expl_match:
            explanation = expl_match.group(1).strip()
        else:
            explanation = response[:200].replace("\n", " ") + "..."

        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if not code_match:
            code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            
        if code_match:
            code = code_match.group(1).strip()
            
        # ★修正箇所: このreturnのインデントをメソッド内に収めました
        return explanation, code

    def process(self, episode_idx: int, goal_str: str, scene_graph: Any, model: Any, log_path: str) -> str:
        """メインから呼ばれる処理メソッド"""
        current_state = self._get_current_state(goal_str, scene_graph)
        advice_str = ""

        if self.previous_state is not None:
            diff_text = self._detect_diff(current_state, self.previous_state)
            
            if diff_text:
                print(f"\n[AdaptationManager] Change Detected in Episode {episode_idx}!")
                print(diff_text)
                
                explanation, code = self._query_llm_for_strategy(model, diff_text, goal_str)
                
                strategy_file = os.path.join(log_path, f"adaptation_strategy_e{episode_idx:03}.py")
                try:
                    with open(strategy_file, "w", encoding="utf-8") as f:
                        f.write(code)
                except Exception as e:
                    print(f"[Error] Failed to save strategy file: {e}")
                
                advice_str = f"\n\nIMPORTANT NOTE: The environment has changed. {explanation}"
                print(f"[AdaptationManager] Strategy: {explanation}")
            else:
                print(f"[AdaptationManager] No changes detected in Episode {episode_idx}.")

        self.previous_state = current_state
        return advice_str