class VLMInterface:

    def infer(self, graph, instruction):

        if "cup" in graph.nodes and "pick" in instruction:

            return {
                "action":"pick",
                "target":"cup"
            }

        return None