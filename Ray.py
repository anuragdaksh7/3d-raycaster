import numpy as np


class Ray:
    def __init__(self, x, y, z, l, m, n):
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.m = m
        self.n = n
        self.p = np.array([self.x, self.y, self.z ])
        self.d = np.array([self.l, self.m, self.n ])
        self.mag = np.linalg.norm(self.d)
        self.d = self.d / self.mag
        