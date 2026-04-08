class SceneMemory:

    def __init__(self):
        self.failure_count = {}

    def update_failure(self, failure):
        t = failure["target"]
        self.failure_count[t] = self.failure_count.get(t, 0) + 1

    def is_hopeless(self, target):
        return self.failure_count.get(target, 0) > 5