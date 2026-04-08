class SceneMemory:
    def __init__(self):
        self.history = []
        self.max_history = 120

        self.unstable_counts = {}
        self.last_failure = None

    def update(self, graph):
        self.history.append(graph)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        for name, node in graph.items():
            if node.get("type") != "mug":
                continue

            unstable = (
                node.get("fallen", False)
                or node.get("touching_basket_wall", False)
                or node.get("stacked_unstable", False)
                or node.get("outside_table", False)
            )

            if unstable:
                self.unstable_counts[name] = self.unstable_counts.get(name, 0) + 1
            else:
                self.unstable_counts[name] = 0

    def consecutive_unstable(self, mug_name):
        return self.unstable_counts.get(mug_name, 0)