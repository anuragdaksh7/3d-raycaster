import numpy as np

class Light:
    def __init__(self, x, y, z, intensity, color):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
        self.c = np.array([self.x, self.y, self.z])
        self.color = np.array(color)
