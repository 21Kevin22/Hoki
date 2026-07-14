import os
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": True
})

# ------------------------------------------------
# Imports
# ------------------------------------------------

import omni
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

# ------------------------------------------------
# Asset paths (your environment)
# ------------------------------------------------

URDF_PATH = "/home/ubuntu/slocal/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf"

ROBOT_DESC_PATH = "/home/ubuntu/slocal/extsDeprecated/omni.isaac.examples/omni/isaac/examples/path_planning/path_planning_example_assets/franka_conservative_spheres_robot_description.yaml"

USD_PATH = "/home/ubuntu/slocal/isaac_sim_grasping/grippers/franka_panda/franka_panda/franka_panda.usd"

print("URDF:", URDF_PATH)
print("RobotDesc:", ROBOT_DESC_PATH)
print("USD:", USD_PATH)

# ------------------------------------------------
# Create world
# ------------------------------------------------

print("Creating world...")

world = World(stage_units_in_meters=1.0)

# Ground

GroundPlane(
    prim_path="/World/GroundPlane",
    size=10
)

# ------------------------------------------------
# Spawn Franka
# ------------------------------------------------

print("Spawning Franka robot...")

add_reference_to_stage(
    usd_path=USD_PATH,
    prim_path="/World/Franka"
)

world.reset()

# ------------------------------------------------
# Articulation interface
# ------------------------------------------------

robot = Articulation("/World/Franka")

world.scene.add(robot)

robot.initialize()

print("Robot DOF:", robot.num_dof)

# ------------------------------------------------
# IK Solver
# ------------------------------------------------

print("Initializing IK solver...")

ik_solver = LulaKinematicsSolver(
    robot_description_path=ROBOT_DESC_PATH,
    urdf_path=URDF_PATH
)

print("IK solver initialized")

# ------------------------------------------------
# Target pose
# ------------------------------------------------

target_position = np.array([0.5, 0.0, 0.5])
target_orientation = np.array([1,0,0,0])

# ------------------------------------------------
# Simple joint control
# ------------------------------------------------

def move_robot():

    current_positions = robot.get_joint_positions()

    action = ArticulationAction(
        joint_positions=current_positions
    )

    robot.apply_action(action)

# ------------------------------------------------
# IK test
# ------------------------------------------------

def ik_test():

    try:

        joint_positions = ik_solver.compute_inverse_kinematics(
            target_position,
            target_orientation
        )

        action = ArticulationAction(
            joint_positions=joint_positions
        )

        robot.apply_action(action)

    except Exception as e:

        print("IK error:", e)

# ------------------------------------------------
# Camera sensor (for dataset)
# ------------------------------------------------

from omni.isaac.sensor import Camera

camera = Camera(
    prim_path="/World/camera",
    position=np.array([1,1,1]),
    frequency=20,
    resolution=(640,480)
)

world.scene.add(camera)

# ------------------------------------------------
# Simulation loop
# ------------------------------------------------

print("Starting simulation...")

frame = 0

while simulation_app.is_running():

    world.step(render=False)

    if frame == 100:

        print("Running IK test")

        ik_test()

    if frame % 50 == 0:

        print("frame:", frame)

    frame += 1

simulation_app.close()