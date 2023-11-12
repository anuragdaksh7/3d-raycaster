import numpy as np
import math

class Sphere:
    def __init__(self, x, y, z, radius, roughness = 0):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.roughness = roughness
        self.c = np.array([self.x, self.y, self.z])

    def pointOfIntersection(self, ray):
        # a ray is a 3d vector which will collide with the sphere and we will return the reflected ray
        # ray reper = np.array([[ x, y, z],[l,m,n]])
        m = ray.p - self.c
        b = np.dot(m, ray.d)
        c = np.dot(m, m) - self.radius * self.radius

        if c> 0 and b> 0:
            return 0

        discr = b*b - c
        if discr < 0:
            return 0

        t = -b - math.sqrt(discr)
        t = max(t, 0)
        return (1, ray.p + t*ray.d)
    
    def reflectedRay(self, ray):
        result = self.pointOfIntersection(ray)
        if result == 0:
            return
        _, p = result
        