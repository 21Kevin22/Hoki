from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction

class Robot:

    def __init__(self, world, prim_path):

        self.robot = Articulation(prim_path)

        world.scene.add(self.robot)
        robot.initialize()
        sim.initialize()
        robot.initialize()
        camera.initialize()
        for _ in range(60):
            sim.step()

    def get_state(self):

        return self.robot.get_joint_positions()

    def apply(self, joints):

        action = ArticulationAction(
            joint_positions=joints
        )

        self.robot.apply_action(action)