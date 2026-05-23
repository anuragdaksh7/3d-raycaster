import numpy as np


class Ray:
    def __init__(self, x, y, z, dir_coeff, color=(1.0, 1.0, 1.0), strength=4):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.color = np.array(color, dtype=float)
        self.strength = strength
        self.p = np.array([self.x, self.y, self.z], dtype=float)
        self.d = np.array(dir_coeff, dtype=float)
        norm = np.linalg.norm(self.d)
        if norm == 0:
            raise ValueError("Ray direction cannot be the zero vector")
        self.d = self.d / norm

    def colorMerger(self, color2):
        self.color = np.clip(self.color * np.array(color2, dtype=float), 0.0, 1.0)
        return self.color


if __name__ == "__main__":
    ray = Ray(1, 1, 1, (1, 0, 0), (1, 1, 1))
    print(ray.colorMerger((0, 1, 0)))