import numpy as np
from Ray import Ray


def getLine(p1,p2):
    return [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]

class Camera:
    def __init__(self, x, y, z, angle):
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle

    def emitRays(self, w, h, q_factor = 1): # q_factor = no of rays per pixel
        # pixels = [list(range(w)) for _ in range(h)]
        self.rays = [[Ray(self.x,self.y,self.z,getLine([self.x, self.y, self.z], [j, i, 0]),(1.0,1.0,1.0),)for j in range(w)] for i in range(h)]
        # print((rays))




if __name__ == '__main__':
    camera = Camera(0, 0, 0, 0)
    # camera.makeImage(10, 10)