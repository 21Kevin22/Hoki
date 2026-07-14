from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane

class Simulation:

    def __init__(self):

        self.app = SimulationApp({"headless": True})

        self.world = World(stage_units_in_meters=1.0)

        GroundPlane("/World/GroundPlane")

    def step(self):

        self.world.step(render=False)

    def running(self):

        return self.app.is_running()

    def close(self):

        self.app.close()