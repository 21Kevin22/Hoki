# ==========================================================
# IsaacSim 4.x Stable Research Template
# VLM + SceneGraph + Replan
# ==========================================================

import os
import numpy as np
import random

os.environ["CARB_APP_MIN_LOG_LEVEL"] = "error"

# ----------------------------------------------------------
# SimulationApp
# ----------------------------------------------------------

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True
})

# ----------------------------------------------------------
# Imports
# ----------------------------------------------------------

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

from omni.isaac.sensor import Camera
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.utils.types import ArticulationAction

import omni.usd


# ==========================================================
# Simulation
# ==========================================================

class Simulation:

    def __init__(self):

        self.world = World(stage_units_in_meters=1.0)

        GroundPlane("/World/GroundPlane")

    def step(self):

        self.world.step(render=True)

    def close(self):

        simulation_app.close()


# ==========================================================
# Robot
# ==========================================================

class Robot:

    def __init__(self, world):

        assets_root = get_assets_root_path()

        franka_usd = assets_root + "/Isaac/Robots/Franka/franka.usd"

        print("Loading Franka:", franka_usd)

        add_reference_to_stage(
            franka_usd,
            "/World/Franka"
        )

        self.world = world
        self.robot = None

    def initialize(self):

        # --------------------------------------------------
        # Prim生成待機
        # --------------------------------------------------

        stage = omni.usd.get_context().get_stage()

        prim = None

        while prim is None:

            self.world.step(render=True)

            prim = stage.GetPrimAtPath("/World/Franka")

        print("Franka prim ready")

        # articulation root
        self.robot = Articulation("/World/Franka")

        self.world.scene.add(self.robot)

        self.robot.initialize()

        print("Robot initialized")

    def apply(self, joints):

        action = ArticulationAction(
            joint_positions=joints
        )

        self.robot.apply_action(action)


# ==========================================================
# Camera
# ==========================================================

class CameraSensor:

    def __init__(self):

        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([1.5,1.5,1.5]),
            frequency=10,
            resolution=(320,240)
        )

    def initialize(self):

        self.camera.initialize()

    def capture(self):

        frame = self.camera.get_current_frame()

        if frame is None:
            return None

        if "rgba" not in frame:
            return None

        return frame["rgba"]


# ==========================================================
# Detector
# ==========================================================

class Detector:

    def detect(self, image):

        return [
            {"name":"cup","pos":[0.5,0.1,0.4]},
            {"name":"table","pos":[0,0,0]}
        ]


# ==========================================================
# SceneGraph
# ==========================================================

class SceneGraph:

    def __init__(self):

        self.nodes = {}
        self.edges = []

    def build(self, objects):

        self.nodes.clear()
        self.edges.clear()

        for obj in objects:

            self.nodes[obj["name"]] = obj["pos"]

        if "cup" in self.nodes:

            self.edges.append(("cup","on","table"))

        return self


# ==========================================================
# VLM
# ==========================================================

class VLMReasoner:

    def infer(self, graph, instruction):

        if "cup" in graph.nodes and "pick" in instruction:

            return {"action":"pick","target":"cup"}

        return None


# ==========================================================
# Planner
# ==========================================================

class TaskPlanner:

    def plan(self, vlm_output):

        if vlm_output is None:

            return []

        return ["move","grasp","lift"]


# ==========================================================
# Motion
# ==========================================================

class MotionPlanner:

    def compute(self, step):

        if step == "move":

            return np.zeros(7)

        if step == "grasp":

            return np.ones(7)*0.3

        if step == "lift":

            return np.ones(7)*0.6

        return np.zeros(7)


# ==========================================================
# Executor
# ==========================================================

class Executor:

    def __init__(self, robot):

        self.robot = robot

    def execute(self, joints):

        self.robot.apply(joints)


# ==========================================================
# Failure
# ==========================================================

class FailureDetector:

    def check(self):

        return random.random() < 0.02


# ==========================================================
# MAIN
# ==========================================================

def main():

    sim = Simulation()

    robot = Robot(sim.world)

    camera = CameraSensor()

    detector = Detector()

    graph = SceneGraph()

    vlm = VLMReasoner()

    planner = TaskPlanner()

    motion = MotionPlanner()

    executor = Executor(robot)

    failure = FailureDetector()

    # physics start
    sim.world.reset()

    # warmup
    for _ in range(50):

        sim.world.step(render=True)

    # initialize
    robot.initialize()

    camera.initialize()

    print("System ready")

    instruction = "pick the cup"

    plan = None
    step_index = 0

    while simulation_app.is_running():

        sim.step()

        image = camera.capture()

        objects = detector.detect(image)

        scene = graph.build(objects)

        if plan is None or len(plan)==0:

            result = vlm.infer(scene, instruction)

            plan = planner.plan(result)

            step_index = 0

            if len(plan)==0:
                continue

            print("New plan:", plan)

        if step_index < len(plan):

            step = plan[step_index]

            joints = motion.compute(step)

            executor.execute(joints)

            step_index += 1

        if failure.check():

            print("Failure detected -> replanning")

            plan = None

    sim.close()


# ==========================================================

if __name__ == "__main__":

    main()