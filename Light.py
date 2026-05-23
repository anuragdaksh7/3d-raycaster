import numpy as np


class Light:
    def __init__(self, x, y, z, intensity, color):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.intensity = float(intensity)
        self.c = np.array([self.x, self.y, self.z], dtype=float)
        self.color = np.array(color, dtype=float)

    def getColor(self):
        return self.color