from scene_graph.relation_engine import RelationEngine
import numpy as np


class SceneGraphBuilder:
    def __init__(self):
        self.relation_engine = RelationEngine()

    def _normalize_orientation(self, orientation):
        if orientation is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        if hasattr(orientation, "GetReal") and hasattr(orientation, "GetImaginary"):
            imag = orientation.GetImaginary()
            return np.array([
                float(orientation.GetReal()),
                float(imag[0]),
                float(imag[1]),
                float(imag[2]),
            ], dtype=float)

        return np.array(orientation, dtype=float)

    def build(self, objects):
        graph = {}

        basket = None
        mugs = []

        # ---------------------------
        # ① ノード生成
        # ---------------------------
        for obj in objects:
            graph[obj["name"]] = {
                "type": obj["type"],
                "position": np.array(obj["position"], dtype=float),
                "orientation": self._normalize_orientation(obj.get("orientation")),
                "relations": {},
            }

            if obj["type"] == "basket":
                basket = obj
            elif obj["type"] == "mug":
                mugs.append(obj)

        if basket is None:
            return graph

        basket_pos = np.array(basket["position"], dtype=float)

        # ---------------------------
        # ② 状態推定（ここが重要）
        # ---------------------------
        for mug in mugs:
            name = mug["name"]

            pos = graph[name]["position"]
            ori = graph[name]["orientation"]

            # ---- inside判定（厳しくする）----
            dx = abs(pos[0] - basket_pos[0])
            dy = abs(pos[1] - basket_pos[1])
            dz = pos[2] - basket_pos[2]

            inside = (
                dx < 0.08 and   # ←ここ重要（以前より厳しく）
                dy < 0.08 and
                0.0 < dz < 0.15
            )

            # ---- upright判定 ----
            z_axis = np.array([0, 0, 1])
            # 簡易：orientationのz方向成分で判断
            upright = abs(ori[2]) < 0.3

            # ---- fallen ----
            fallen = (not upright) and (not inside)

            # ---- touching basket wall ----
            touching_wall = inside and (
                dx > 0.06 or dy > 0.06
            )

            # ---- outside table ----
            outside_table = pos[2] < -0.05

            graph[name]["inside_basket"] = inside
            graph[name]["upright"] = upright
            graph[name]["fallen"] = fallen
            graph[name]["touching_basket_wall"] = touching_wall
            graph[name]["outside_table"] = outside_table
            graph[name]["stacked_unstable"] = False

        # ---------------------------
        # ③ 関係推定（3D relation）
        # ---------------------------
        for i in range(len(mugs)):
            for j in range(i + 1, len(mugs)):
                a = mugs[i]
                b = mugs[j]

                pa = graph[a["name"]]["position"]
                pb = graph[b["name"]]["position"]

                rel_a = []
                rel_b = []

                if self.relation_engine.left_of(pa, pb):
                    rel_a.append("left_of")
                    rel_b.append("right_of")

                if self.relation_engine.behind(pa, pb):
                    rel_a.append("behind")

                if self.relation_engine.behind(pb, pa):
                    rel_b.append("behind")

                if self.relation_engine.touching(pa, pb):
                    rel_a.append("touching")
                    rel_b.append("touching")

                if self.relation_engine.stacked(pa, pb):
                    rel_a.append("stacked_on")
                    if graph[a["name"]]["fallen"]:
                        graph[a["name"]]["stacked_unstable"] = True

                elif self.relation_engine.stacked(pb, pa):
                    rel_b.append("stacked_on")
                    if graph[b["name"]]["fallen"]:
                        graph[b["name"]]["stacked_unstable"] = True

                graph[a["name"]]["relations"][b["name"]] = rel_a
                graph[b["name"]]["relations"][a["name"]] = rel_b

        return graph