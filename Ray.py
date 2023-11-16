import numpy as np


class Ray:
    def __init__(self, x, y, z, dir_coeff, color, strength = 4):
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.p = np.array([self.x, self.y, self.z ])
        self.d = np.array([dir_coeff ])
        self.d = self.d/np.linalg.norm(self.d)
    
    def colorMerger(self, color2):
        self.color = [color2[i]*self.color[i] for i in range(3)]
        return self.color

if __name__ == "__main__":
    a = Ray(1,1,1,1,1,1,(1,1,1))
    print(a.colorMerger((0,1,0)))