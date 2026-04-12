import numpy as np


class RelationEngine:

    def left_of(self, a, b):
        return a[0] < b[0] - 0.05

    def right_of(self, a, b):
        return a[0] > b[0] + 0.05

    def behind(self, a, b):
        return a[1] > b[1] + 0.05

    def touching(self, a, b):
        return np.linalg.norm(a - b) < 0.08

    def stacked(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dz = a[2] - b[2]

        return dx < 0.03 and dy < 0.03 and 0.03 < dz < 0.12

    def outside_table(self, pos):
        return pos[2] < -0.05