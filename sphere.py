import random
import numpy as np
import math

class Sphere:
    def __init__(self, x, y, z, radius, color, roughness = 0):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.color = color
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
    
    def normalize(self, c1, c2):
        c1 = np.array(c1)
        c2 = np.array(c2)
        d = c1-c2
        d = d / np.linalg.norm(d)
        return d

    def getLightIntensity(self, point, light, surface_normal):
        # print(point,light.c, light.color, light.getColor())
        light.color = np.array([1.0,1.0,1.0])
        distance = np.linalg.norm(light.c - point)
        intensity = light.color
        # print(light.color)
        diffuse_coefficient = np.dot(surface_normal, self.normalize(point , light.c))
        
        # print(intensity)
        # print(diffuse_coefficient)
        intensity *= max(0,diffuse_coefficient)
        # print(intensity)
        return intensity

    def reflectedRay(self, ray, light):
        # print(light.color)
        result = self.pointOfIntersection(ray)
        if result == 0:
            return
        # print("fdf")
        _, p = result
        normal = (p - self.c)
        normal = normal / np.linalg.norm(normal)
        reflected = ray.d - 2 * np.dot(ray.d, normal) * normal
        x = random.randint(-int(self.roughness*180), int(self.roughness*180))
        y = random.randint(-int(self.roughness*180), int(self.roughness*180))
        z = random.randint(-int(self.roughness*180), int(self.roughness*180))
        randomDir = np.array([x,y,z])
        randomDirUnitVector = randomDir/np.linalg.norm(randomDir)
        if (np.dot(randomDirUnitVector,normal)< 0):
            randomDirUnitVector*= -1
        ray.colorMerger(self.color)
        ray.colorMerger(self.getLightIntensity(p,light,reflected))
        # print(ray.color)
    
if __name__ == '__main__':
    from Ray import Ray
    s = Sphere(0, 0, 0, 1, (1,1,1))
    r = Ray(10, 0, 0, -1, 0, 0)
    print(s.pointOfIntersection(r))
    