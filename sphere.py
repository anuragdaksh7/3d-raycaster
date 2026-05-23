import math
import numpy as np


EPSILON = 1e-4


def normalize(vector):
    vector = np.array(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length


class Sphere:
    def __init__(
        self,
        x,
        y,
        z,
        radius,
        color,
        roughness=0.0,
        reflectivity=0.15,
        specular=64,
        metallic=0.0,
        emissive=0.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radius = float(radius)
        self.color = np.array(color, dtype=float)
        self.roughness = float(roughness)
        self.reflectivity = float(reflectivity)
        self.specular = float(specular)
        self.metallic = float(metallic)
        self.emissive = float(emissive)
        self.c = np.array([self.x, self.y, self.z], dtype=float)

    def pointOfIntersection(self, ray):
        oc = ray.p - self.c
        b = np.dot(oc, ray.d)
        c = np.dot(oc, oc) - self.radius * self.radius

        if c > 0 and b > 0:
            return None

        discriminant = b * b - c
        if discriminant < 0:
            return None

        t = -b - math.sqrt(discriminant)
        if t < EPSILON:
            t = -b + math.sqrt(discriminant)
        if t < EPSILON:
            return None
        return t, ray.p + t * ray.d

    def intersect(self, ray):
        hit = self.pointOfIntersection(ray)
        if hit is None:
            return None
        return hit[0]

    def normal_at(self, point):
        return normalize(point - self.c)

    def color_at(self, point):
        return self.color


class Plane:
    def __init__(
        self,
        point,
        normal,
        color=(0.85, 0.85, 0.88),
        secondary_color=(0.18, 0.18, 0.2),
        checker_size=1.0,
        roughness=0.0,
        reflectivity=0.0,
        specular=8,
        metallic=0.0,
        emissive=0.0,
    ):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(normal)
        self.color = np.array(color, dtype=float)
        self.secondary_color = np.array(secondary_color, dtype=float)
        self.checker_size = float(checker_size)
        self.roughness = float(roughness)
        self.reflectivity = float(reflectivity)
        self.specular = float(specular)
        self.metallic = float(metallic)
        self.emissive = float(emissive)
        self.c = self.point

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.d)
        if abs(denom) < EPSILON:
            return None
        t = np.dot(self.point - ray.p, self.normal) / denom
        if t < EPSILON:
            return None
        return t

    def normal_at(self, point):
        return self.normal

    def color_at(self, point):
        pattern = int(math.floor(point[0] / self.checker_size) + math.floor(point[2] / self.checker_size))
        return self.color if pattern % 2 == 0 else self.secondary_color


if __name__ == "__main__":
    from Ray import Ray

    s = Sphere(0, 0, 0, 1, (1, 1, 1))
    r = Ray(10, 0, 0, (-1, 0, 0))
    print(s.pointOfIntersection(r))