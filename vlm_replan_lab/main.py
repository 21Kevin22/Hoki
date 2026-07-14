import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})


from simulation.sim_manager import SimulationManager
from simulation.world_builder import WorldBuilder
from simulation.camera_manager import CameraManager
from robot.franka_robot import FrankaRobot
from perception.detector import Detector
from scene_graph.graph_builder import SceneGraphBuilder
from scene_graph.scene_memory import SceneMemory
from planning.task_planner import TaskPlanner
from planning.motion_planner import MotionPlanner
from planning.vlm_planner import VLMPlanner
from control.executor import Executor
from control.failure_detector import FailureDetector
from vlm_replan_lab.utils.video_recorder import VideoRecorder


def main():
    warmup_steps = 10
    max_steps = 1500
    min_steps = 200
    done_streak_threshold = 50

    sim = SimulationManager()

    world_builder = WorldBuilder(sim.world)
    world_builder.build()

    robot = FrankaRobot(sim.world)
    robot.spawn()

    camera = CameraManager(sim.world)
    detector = Detector()
    graph_builder = SceneGraphBuilder()
    memory = SceneMemory()
    vlm_planner = VLMPlanner()
    task_planner = TaskPlanner()
    motion_planner = MotionPlanner(robot, world_builder)
    executor = Executor(robot, sim.world)
    failure_detector = FailureDetector()
    recorder = VideoRecorder(
        output_path="/home/ubuntu/slocal/evaluation/vlm_gpt/mug_replan_run.mp4",
        fps=20,
    )

    sim.initialize()
    robot.initialize()
    camera.initialize()


    for _ in range(warmup_steps):
        sim.step()
        camera.capture()

    instruction = "put all mugs into basket upright"

    symbolic_plan = []
    low_level_queue = []
    current_symbolic_index = 0

    step_count = 0
    done_streak = 0

    try:
        while step_count < max_steps:
            step_count += 1
            sim.step()

            multi_view = camera.capture_all()
            frame = multi_view.get("main", None)

            objects = detector.detect(world_builder.mug_names, world_builder.basket_name)
            print("objects:", [obj["name"] for obj in objects])

            graph = graph_builder.build(objects)
            memory.update(graph)

            debug_views = camera.render_debug_views(objects, graph)
            composite_frame = camera.compose_views(multi_view, debug_views)

            print("views:", sorted(multi_view.keys()))
            print("frame:", frame is not None)
            print("composite:", composite_frame is not None)

            if composite_frame is not None:
                print("composite shape:", composite_frame.shape)
                recorder.capture(composite_frame)
            elif frame is not None:
                print("frame shape:", frame.shape)
                recorder.capture(frame)

            print("graph keys:", list(graph.keys()))
            print("all_done:", task_planner.all_done(graph))
            print("STEP:", step_count)
            if task_planner.all_done(graph):
                done_streak += 1
            else:
                done_streak = 0

            if step_count >= min_steps and done_streak >= done_streak_threshold:
                print("[Done stable]")
                break

            busy = len(low_level_queue) > 0

            failure = failure_detector.check(
                graph,
                memory,
                busy=busy,
                ignored_targets=task_planner.ignored_mugs,
            )

            if failure is not None:
                failure_item = failure[0] if isinstance(failure, list) else failure
                print(f"[Failure] {failure_item}")
                symbolic_plan = task_planner.replan_for_failure(graph, failure_item)
                low_level_queue = []
                current_symbolic_index = 0

            if not symbolic_plan and not low_level_queue:
                high_level_hint = vlm_planner.infer(graph, instruction, multi_view)
                symbolic_plan = task_planner.plan(graph, instruction, high_level_hint)
                current_symbolic_index = 0
                if symbolic_plan:
                    print("[Plan]", symbolic_plan)

            if not low_level_queue and current_symbolic_index < len(symbolic_plan):
                action = symbolic_plan[current_symbolic_index]
                low_level_queue = motion_planner.compile(action)
                current_symbolic_index += 1
                print("[LowLevelQueue]", low_level_queue)

            if low_level_queue:
                cmd = low_level_queue[0]
                done = executor.execute(cmd)
                if done:
                    low_level_queue.pop(0)

    except Exception as exc:
        import traceback
        print("[Main Exception]", repr(exc))
        traceback.print_exc()
        raise
    finally:
        recorder.save()
        sim.close()
        simulation_app.close()


if __name__ == "__main__":
    main()