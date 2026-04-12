class TaskPlanner:
    def __init__(self):
        self.recovery_counts = {}
        self.max_recovery_per_mug = 3
        self.ignored_mugs = set()

    def plan(self, graph, instruction, vlm_hint=None):
        actions = []

        for name, node in graph.items():
            if "mug" not in name:
                continue

            if name in self.ignored_mugs:
                continue

            if node.get("fallen", False):
                actions.append({"type": "recover_upright", "target": name})
                actions.append({"type": "pick_place", "target": name})
                continue

            if node.get("touching_basket_wall", False):
                actions.append({"type": "move_away_from_wall", "target": name})
                continue

            if node.get("stacked_unstable", False):
                actions.append({"type": "separate_stack", "target": name})
                continue

            if node.get("outside_table", False):
                actions.append({"type": "return_to_table", "target": name})
                continue

            if not node.get("inside_basket", False):
                actions.append({"type": "pick_place", "target": name})

        return actions

    def replan_for_failure(self, graph, failure):
        mug = failure["target"]
        ftype = failure["type"]

        self.recovery_counts[mug] = self.recovery_counts.get(mug, 0) + 1

        if self.recovery_counts[mug] > self.max_recovery_per_mug:
            print(f"[Planner] giving up on {mug}")
            self.ignored_mugs.add(mug)
            return []

        if ftype == "fallen":
            return [
                {"type": "recover_upright", "target": mug},
                {"type": "pick_place", "target": mug},
            ]

        if ftype == "touching_basket_wall":
            return [
                {"type": "move_away_from_wall", "target": mug},
                {"type": "pick_place", "target": mug},
            ]

        if ftype == "stacked_unstable":
            return [
                {"type": "separate_stack", "target": mug},
                {"type": "pick_place", "target": mug},
            ]

        if ftype == "outside_table":
            return [
                {"type": "return_to_table", "target": mug},
                {"type": "pick_place", "target": mug},
            ]

        return self.plan(graph, "put all mugs into basket upright")

    def all_done(self, graph):
        total = 0
        done = 0

        for name, node in graph.items():
            if "mug" not in name:
                continue
            total += 1

            if (
                node.get("inside_basket", False)
                and node.get("upright", True)
                and not node.get("fallen", False)
            ):
                done += 1

        print("[all_done]", done, "/", total)
        return done == total and total > 0