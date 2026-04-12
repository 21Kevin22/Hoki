import numpy as np


class MotionPlanner:
    def __init__(self, robot, world_builder):
        self.robot = robot
        self.world_builder = world_builder

        self.home = np.array([0.0, -0.8, 0.0, -2.2, 0.0, 1.6, 0.8, 0.04, 0.04], dtype=float)
        self.pregrasp = np.array([0.2, -0.3, 0.1, -1.8, 0.1, 2.0, 0.8, 0.04, 0.04], dtype=float)
        self.grasp = np.array([0.2, -0.1, 0.1, -1.7, 0.1, 2.1, 0.8, 0.00, 0.00], dtype=float)
        self.lift = np.array([0.6, -0.1, 0.0, -1.6, 0.0, 1.8, 0.8, 0.00, 0.00], dtype=float)
        self.place = np.array([1.0, -0.3, 0.0, -1.4, 0.0, 1.6, 0.8, 0.04, 0.04], dtype=float)

    def compile(self, action):
        t = action["type"]
        target = action["target"]

        if t == "pick_place":
            return [
                {"kind": "joint_target", "q": self.home.copy()},
                {"kind": "joint_target", "q": self.pregrasp.copy()},
                {"kind": "joint_target", "q": self.grasp.copy()},
                {"kind": "joint_target", "q": self.lift.copy()},
                {"kind": "joint_target", "q": self.place.copy()},
                {"kind": "place_in_basket", "target": target},
                {"kind": "joint_target", "q": self.home.copy()},
            ]

        if t == "recover_upright":
            q1 = self.pregrasp.copy()
            q2 = self.grasp.copy()
            q3 = self.lift.copy()
            q1[0] += 0.1
            q2[1] += 0.2
            q3[0] += 0.2
            return [
                {"kind": "joint_target", "q": self.home.copy()},
                {"kind": "joint_target", "q": q1},
                {"kind": "joint_target", "q": q2},
                {"kind": "force_upright", "target": target},
                {"kind": "joint_target", "q": q3},
                {"kind": "joint_target", "q": self.home.copy()},
            ]

        if t == "recover_back":
            q1 = self.home.copy()
            q2 = self.pregrasp.copy()
            q3 = self.lift.copy()

            q2[0] -= 0.3
            q3[0] -= 0.5

            return [
                {"kind": "joint_target", "q": q1},
                {"kind": "joint_target", "q": q2},
                {"kind": "joint_target", "q": q3},
                {"kind": "joint_target", "q": self.home.copy()},
            ]

        return []
