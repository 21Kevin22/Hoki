class SceneGraph:

    def __init__(self):

        self.nodes = {}
        self.edges = []

    def build(self, objects):

        self.nodes.clear()
        self.edges.clear()

        for obj in objects:

            self.nodes[obj["name"]] = obj["position"]

        if "cup" in self.nodes:

            self.edges.append(("cup","on","table"))

        return self