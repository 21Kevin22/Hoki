from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane


class SimulationManager:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)

    def initialize(self):
        GroundPlane("/World/GroundPlane", size=10.0)

        # 重要: physics を立ち上げる
        self.world.reset()

        # warmup
        for _ in range(60):
            self.world.step(render=True)

    def step(self):
        self.world.step(render=True)

    def close(self):
        try:
            self.world.stop()
        except Exception:
            pass