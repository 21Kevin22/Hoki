"""vlm_delta workflow skeleton implementation.

5-step architecture:
 1. receive human intent
 2. generate delta scene graph / PDDL style goal representation
 3. query GPT-based planner for action plan
 4. execute in simulation + logging
 5. realtime VLM scene graph update + replan based on discrepancy

Example usage: python3 standalone_examples/tutorials/workflow.py --task "sort mugs by color"
"""

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from standalone_examples.tutorials.Enum_eval import (
    VLMAnalyzer,
    capture_images_for_vlm,
    RobotController,
    AssetBuilder,
    run_calibration,
    load_calibration,
)
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import Gf, UsdLux


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

@dataclass
class WorkflowMetrics:
    success: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_planning_time: float = 0.0
    replan_count: int = 0
    dynamic_scene_graph_ratio: float = 0.0
    trajectory_length: float = 0.0
    safety_score: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return {
            "success": self.success,
            "duration": self.end_time - self.start_time if self.end_time else 0.0,
            "total_planning_time": self.total_planning_time,
            "replan_count": self.replan_count,
            "dynamic_scene_graph_ratio": self.dynamic_scene_graph_ratio,
            "trajectory_length": self.trajectory_length,
            "safety_score": self.safety_score,
            "history": self.history,
        }


class VLMDeltaWorkflow:
    def __init__(self, output_dir: str = "/home/ubuntu/slocal/evaluation/vlm_delta_workflow"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = WorkflowMetrics()

        self.vlm_analyzer = VLMAnalyzer(model_name=os.environ.get("TARGET_MODEL", "gpt-4o"))
        self.world: Optional[World] = None
        self.controller: Optional[RobotController] = None
        self.franka: Optional[Franka] = None
        self.plan_logger = []

    def setup_simulation_environment(self):
        if self.world is not None:
            return

        self.world = World(stage_units_in_meters=1.0)
        stage = get_current_stage()
        self.world.scene.add_default_ground_plane()

        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(2000)

        assets_root = get_assets_root_path()
        self.franka = self.world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            )
        )

        # place mugs in initial scene.
        for cfg in [
            {"id": "mug_0", "pos": [0.4, 0.5, 0.05], "color": [1.0, 0.0, 0.0], "angle": -120},
            {"id": "mug_1", "pos": [0.5, 0.25, 0.05], "color": [0.0, 1.0, 0.0], "angle": 180},
            {"id": "mug_2", "pos": [0.6, 0.0, 0.05], "color": [0.0, 0.0, 1.0], "angle": 120},
            {"id": "mug_3", "pos": [0.5, -0.25, 0.05], "color": [1.0, 1.0, 0.0], "angle": 60},
            {"id": "mug_4", "pos": [0.4, -0.5, 0.05], "color": [0.0, 1.0, 1.0], "angle": 0},
            {"id": "mug_5", "pos": [0.35, -0.62, 0.05], "color": [1.0, 0.0, 1.0], "angle": 20},
        ]:
            AssetBuilder.create_beer_mug(f"/World/{cfg['id']}", cfg["pos"], Gf.Vec3f(*cfg["color"]), cfg["angle"])

        AssetBuilder.create_basket("/World/Basket", [0.0, 0.6, 0.05])

        self.world.reset()
        self.world.play()

        calib = run_calibration(self.world)
        self.controller = RobotController(arm=self.franka)
        self.controller.apply_calibration(calib)

        logger.info("Simulation environment ready")

    def receive_human_instruction(self, instruction_text: str) -> Dict[str, Any]:
        logger.info("[1/5] receive_human_instruction")
        human_intent = {
            "task": instruction_text,
            "timestamp": time.time(),
        }
        self.metrics.history.append({"phase": "human_instruction", "intent": instruction_text})
        return human_intent

    def build_delta_scene_graph(self, human_intent: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[2/5] build_delta_scene_graph")

        # (1) Capture environment imagery if simulation is running
        detection_result = None
        if self.world is not None:
            images = capture_images_for_vlm(self.world)
            detection_result = self.vlm_analyzer.detect_objects_for_delta(images)
        
        if detection_result is not None and detection_result.get("objects"):
            scene_graph = self.vlm_analyzer.build_delta_scene_graph(detection_result)
        else:
            scene_graph = {
                "nodes": ["agent", "workspace"],
                "edges": [["agent", "in", "workspace"]],
                "goal": f"execute({human_intent['task']})",
                "metadata": {"source": "delta"},
            }

        pddl_goal = {
            "domain": "mug_sorting",
            "problem": "task0",
            "goal": scene_graph.get("goal", f"execute({human_intent['task']})"),
            "human_intent": human_intent,
        }

        self.metrics.history.append({"phase": "delta_scene_graph", "scene_graph": scene_graph, "pddl": pddl_goal})
        return {"scene_graph": scene_graph, "pddl": pddl_goal}

    def gpt_plan_action(self, pddl_goal: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[3/5] gpt_plan_action")
        t0 = time.time()

        plan_template = {
            "steps": [],
            "length": 0,
            "timestamp": time.time(),
        }

        if self.vlm_analyzer and getattr(self.vlm_analyzer, 'client', None):
            prompt = (
                "以下のPDDL目標を満たす順序付きアクション列をJSONで出力してください。" 
                "結果は {\"steps\": [\"pick mug_0\", ...], \"length\": N} を想定します。"
                f"\nGoal: {json.dumps(pddl_goal, ensure_ascii=False)}"
            )
            try:
                resp = self.vlm_analyzer.client.chat.completions.create(
                    model=self.vlm_analyzer.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                )
                parsed = self.vlm_analyzer._safe_parse(resp.choices[0].message.content, plan_template)
                if isinstance(parsed, dict) and parsed.get("steps"):
                    plan = {
                        "steps": parsed.get("steps", []),
                        "length": len(parsed.get("steps", [])),
                        "timestamp": time.time(),
                    }
                else:
                    raise ValueError("GPT response is not valid plan")
            except Exception as exc:
                logger.warning("gpt_plan_action: GPT call failed %s, fallback plan", exc)
                plan = {
                    "steps": ["pick mug_0", "place mug_0 target_0", "pick mug_1", "place mug_1 target_1"],
                    "length": 4,
                    "timestamp": time.time(),
                }
        else:
            plan = {
                "steps": ["pick mug_0", "place mug_0 target_0", "pick mug_1", "place mug_1 target_1"],
                "length": 4,
                "timestamp": time.time(),
            }

        self.metrics.total_planning_time += time.time() - t0
        self.metrics.trajectory_length = plan["length"]
        self.metrics.history.append({"phase": "gpt_plan", "plan": plan})
        return plan

    def execute_simulation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[4/5] execute_simulation_plan")

        if self.world is None:
            self.setup_simulation_environment()

        if self.controller is None:
            raise RuntimeError("Simulation controller is not initialized")

        execution = {
            "plan": plan,
            "executed_steps": [],
            "executed_length": 0,
            "collisions": 0,
            "status": "success",
            "timestamp": time.time(),
        }

        for action in plan.get("steps", []):
            step_ok = True
            try:
                if action.startswith("pick"):
                    target_id = action.split()[1]
                    path = f"/World/{target_id}"
                    pick_targets = self.controller.compute_pick_targets(path)
                    self.controller.open_gripper(self.world)
                    moved = self.controller.move_end_effector_to(pick_targets["pre_grasp"], pick_targets["orientation"], self.world, steps=90, stage_name="pre-grasp")
                    if moved:
                        moved = self.controller.move_end_effector_to(pick_targets["grasp"], pick_targets["orientation"], self.world, steps=75, stage_name="grasp-approach")
                    if not moved:
                        step_ok = False
                elif action.startswith("place"):
                    target_id = action.split()[1] if len(action.split()) >= 2 else None
                    place_slot = self.controller.get_place_slot_for_mug(f"/World/{target_id}") if target_id else "BASKET_HIGH"
                    place_targets = self.controller.compute_place_targets(place_slot)
                    moved = self.controller.move_end_effector_to(place_targets["pre_place"], place_targets["orientation"], self.world, steps=90, stage_name="pre-place")
                    if moved:
                        moved = self.controller.move_end_effector_to(place_targets["place"], place_targets["orientation"], self.world, steps=60, stage_name="place")
                    self.controller.open_gripper(self.world)
                    if moved:
                        moved = self.controller.move_end_effector_to(place_targets["retreat"], place_targets["orientation"], self.world, steps=75, stage_name="post-place-retreat")
                    self.controller.move_to_joint_pose("HOME", self.world)
                    if not moved:
                        step_ok = False
                else:
                    logger.warning("Unknown plan action %s", action)
                    step_ok = False

                if not step_ok:
                    execution["collisions"] += 1
                    execution["status"] = "partial"

                execution["executed_steps"].append(action)
            except Exception as exc:
                logger.exception("Simulation execution failed on action %s", action)
                execution["collisions"] += 1
                execution["status"] = "partial"
                execution["executed_steps"].append(action)

        execution["executed_length"] = len(execution["executed_steps"])
        self.metrics.safety_score += execution["collisions"]
        if execution["collisions"] > 0:
            execution["status"] = "partial"

        self.metrics.history.append({"phase": "simulation_execution", "result": execution})
        return execution

    def realtime_vlm_update_replan(self, execution: Dict[str, Any], scene_graph: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[5/5] realtime_vlm_update_replan")

        current_scene_graph = {"nodes": [], "edges": []}
        if self.world is not None:
            images = capture_images_for_vlm(self.world)
            if images:
                current_scene_graph = self.vlm_analyzer.extract_scene_graph(images)

        target_nodes = set(scene_graph.get("nodes", []))
        current_nodes = set(current_scene_graph.get("nodes", []))
        if target_nodes:
            overlap = len(target_nodes.intersection(current_nodes))
            diff_ratio = overlap / len(target_nodes)
        else:
            diff_ratio = 1.0

        # リアルタイム差分が大きい or 部分成功なら再計画
        should_replan = diff_ratio < 0.95 or execution.get("status") != "success"

        if should_replan:
            self.metrics.replan_count += 1
            self.metrics.dynamic_scene_graph_ratio = diff_ratio
            self.metrics.adaptivity_score = self.metrics.replan_count / (self.metrics.replan_count + 1)
            new_plan = self.gpt_plan_action(goal)
            self.metrics.history.append({"phase": "replan", "diff_ratio": diff_ratio, "new_plan": new_plan})
            return new_plan

        self.metrics.dynamic_scene_graph_ratio = diff_ratio
        self.metrics.history.append({"phase": "realtime_vlm", "diff_ratio": diff_ratio, "status": "no_replan"})
        return execution.get("plan", {})



    def evaluate_and_save(self, output_name: str = "workflow_results.json") -> None:
        self.metrics.end_time = time.time()
        out_path = self.output_dir / output_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved workflow metrics to {out_path}")

    def run(self, human_instruction: str, max_replans: int = 3):
        hi = self.receive_human_instruction(human_instruction)
        sg_data = self.build_delta_scene_graph(hi)
        plan = self.gpt_plan_action(sg_data["pddl"])
        execution = self.execute_simulation_plan(plan)

        for _ in range(max_replans):
            plan = self.realtime_vlm_update_replan(execution, sg_data["scene_graph"], sg_data["pddl"])
            execution = self.execute_simulation_plan(plan)
            if execution.get("status") == "success":
                break

        self.metrics.success = execution.get("status") == "success"
        self.evaluate_and_save()
        return self.metrics.to_dict()


def main():
    parser = argparse.ArgumentParser(description="VLM-Delta 5-step workflow demo")
    parser.add_argument("--task", type=str, default=None, help="Task description from human")
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/slocal/evaluation/vlm_delta_workflow", help="Output directory")
    args = parser.parse_args()

    task_input = args.task
    if not task_input:
        task_input = input("Enter human instruction (e.g., 'sort mugs by color'): ").strip()
        if not task_input:
            raise ValueError("Task description is required")

    workflow = VLMDeltaWorkflow(output_dir=args.output_dir)
    result = workflow.run(task_input)

    logger.info("Final workflow result:\n%s", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
