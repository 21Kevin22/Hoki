import numpy as np


class SceneGraph3D:

    def __init__(self):
        self.table_z = 0.0
        self.threshold_touch = 0.08
        self.threshold_stack = 0.1

    def build(self, objects):
        graph = {}

        # -------------------------
        # node
        # -------------------------
        for obj in objects:
            name = obj["name"]
            pos = np.array(obj["position"])
            quat = obj["orientation"]

            up = self._quat_to_up(quat)

            graph[name] = {
                "pos": pos,
                "upright": up[2] > 0.7,
                "fallen": up[2] < 0.3,
                "height": pos[2],
                "on_table": pos[2] < self.table_z + 0.2,
                "relations": {}
            }

        # -------------------------
        # relation
        # -------------------------
        for a in graph:
            for b in graph:
                if a == b:
                    continue

                pa = graph[a]["pos"]
                pb = graph[b]["pos"]

                dx = pa[0] - pb[0]
                dy = pa[1] - pb[1]
                dz = pa[2] - pb[2]

                rel = []

                # 左右
                if dx > 0.05:
                    rel.append("right_of")
                elif dx < -0.05:
                    rel.append("left_of")

                # 前後
                if dy > 0.05:
                    rel.append("front_of")
                elif dy < -0.05:
                    rel.append("behind")

                # 接触
                if abs(dx) < self.threshold_touch and abs(dy) < self.threshold_touch:
                    rel.append("touching")

                # スタック
                if dz > self.threshold_stack:
                    rel.append("stacked_on")

                graph[a]["relations"][b] = rel

        return graph

    def _quat_to_up(self, q):
        try:
            # IsaacSim Quatd対応
            return np.array([
                q.GetImaginary()[0],
                q.GetImaginary()[1],
                q.GetImaginary()[2]
            ])
        except:
            return np.array([0, 0, 1])