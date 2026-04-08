import re
import numpy as np
from isaacsim.core.prims import XFormPrim


class Executor:
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world

        self.active_cmd = None
        self.counter = 0
        self.max_steps = 40
        self.basket_offsets = [
            np.array([0.00, 0.00, 0.07], dtype=float),
            np.array([-0.04, -0.04, 0.07], dtype=float),
            np.array([0.04, -0.04, 0.07], dtype=float),
            np.array([-0.04, 0.04, 0.07], dtype=float),
            np.array([0.04, 0.04, 0.07], dtype=float),
        ]

    def _reset_active(self):
        self.active_cmd = None
        self.counter = 0

    def _mug_index(self, name):
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 0

    def execute(self, cmd):
        if self.active_cmd is None:
            self.active_cmd = cmd
            self.counter = 0

        kind = self.active_cmd["kind"]

        if kind == "joint_target":
            target = np.array(self.active_cmd["q"], dtype=float).reshape(-1)

            q = self.robot.get_joint_positions()
            if q is None:
                return False

            q = np.array(q, dtype=float).reshape(-1)

            nq = min(len(q), len(target))
            blended = q.copy()
            blended[:nq] = q[:nq] + (target[:nq] - q[:nq]) * 0.15

            self.robot.apply_action(blended)

            self.counter += 1

            if self.counter >= self.max_steps:
                self._reset_active()
                return True

            return False

        if kind == "force_upright":
            mug = self.active_cmd["target"]
            prim = XFormPrim(f"/World/{mug}")
            pos, _ = prim.get_world_poses()
            upright = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
            prim.set_world_poses(positions=pos, orientations=upright)
            self._reset_active()
            return True

        if kind == "place_in_basket":
            mug = self.active_cmd["target"]
            mug_prim = XFormPrim(f"/World/{mug}")
            basket_prim = XFormPrim("/World/basket/base")

            basket_pos, _ = basket_prim.get_world_poses()
            basket_pos = np.array(basket_pos, dtype=float).reshape(-1, 3)
            idx = self._mug_index(mug) % len(self.basket_offsets)
            target_pos = basket_pos.copy()
            target_pos[0, 0] += self.basket_offsets[idx][0]
            target_pos[0, 1] += self.basket_offsets[idx][1]
            target_pos[0, 2] += self.basket_offsets[idx][2]
            upright = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
            mug_prim.set_world_poses(positions=target_pos, orientations=upright)
            self._reset_active()
            return True

        self._reset_active()
        return True
