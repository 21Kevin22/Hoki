class VLMPlanner:
    def infer(self, graph, instruction, multi_view_images):
        # ここは GPT / PaLM-E / LLaVA への差し替え口
        # 今は簡易ヒントのみ返す
        fallen = [n for n, d in graph["nodes"].items() if d.get("fallen", False)]
        if fallen:
            return {"priority": "recover_fallen_first", "targets": fallen}
        return {"priority": "fill_basket"}