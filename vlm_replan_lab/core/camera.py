import numpy as np
from omni.isaac.sensor import Camera

class CameraSensor:

    def __init__(self):

        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([1,1,1]),
            frequency=10,
            resolution=(320,240)
        )

    def capture(self):

        return self.camera.get_rgba()