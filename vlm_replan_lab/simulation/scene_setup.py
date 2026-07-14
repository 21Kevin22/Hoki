import numpy as np

from isaacsim.core.api.objects import GroundPlane
from omni.isaac.core.objects import DynamicCuboid


class SceneSetup:

    def __init__(self, world):

        self.world = world

    def build_scene(self):

        GroundPlane("/World/ground")

        self._create_table()

        self._create_basket()

        self._create_mugs()

    def _create_table(self):

        DynamicCuboid(
            prim_path="/World/table",
            position=np.array([0,0,0.5]),
            scale=np.array([1.5,1,0.1]),
            color=np.array([0.5,0.3,0.2])
        )

    def _create_basket(self):

        DynamicCuboid(
            prim_path="/World/basket",
            position=np.array([0.6,0,0.6]),
            scale=np.array([0.25,0.25,0.2]),
            color=np.array([0.2,0.6,0.2])
        )

    def _create_mugs(self):

        for i in range(5):

            DynamicCuboid(
                prim_path=f"/World/mug_{i}",
                position=np.array([0.2 + 0.1*i,0,0.65]),
                scale=np.array([0.05,0.05,0.1]),
                color=np.array([0.8,0.8,0.9])
            )