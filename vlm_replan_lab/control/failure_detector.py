class FailureDetector:

    def check(self, graph, memory, busy=False, ignored_targets=None):
        failures = []

        for name, node in graph.items():

            if ignored_targets and name in ignored_targets:
                continue

            if node.get("type") != "mug":
                continue

            # 倒れている
            if node.get("fallen", False):
                failures.append({"type": "fallen", "target": name})

            # テーブル外
            if node.get("outside_table", False):
                failures.append({"type": "outside_table", "target": name})

            # 接触問題
            for other, rel in node.get("relations", {}).items():
                if "touching" in rel:
                    failures.append({
                        "type": "touching_basket_wall",
                        "target": name
                    })

        return failures if failures else None