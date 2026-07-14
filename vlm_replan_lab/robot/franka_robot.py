from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np


class FrankaRobot:
    def __init__(self, world):
        self.world = world
        self.robot = None
        self._initialized = False

    def spawn(self):
        if self.robot is None:
            self.robot = self.world.scene.add(
                Franka(
                    prim_path="/World/Franka",
                    name="franka",
                )
            )

    def initialize(self):
        # ここでは physics が立ち上がった後に呼ぶ
        if self.robot is None:
            raise RuntimeError("Call spawn() before initialize().")

        self.robot.initialize()
        self._initialized = True

    def get_joint_positions(self):
        if self.robot is None:
            return None
        return self.robot.get_joint_positions()

    def apply_action(self, joint_positions):
        if not self._initialized:
            return

        q = np.array(joint_positions, dtype=float).reshape(-1)
        action = ArticulationAction(joint_positions=q)
        self.robot.apply_action(action)