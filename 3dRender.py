import itertools
import math
import pygame
import numpy as np
import random

from Cam import Camera
from Light import Light
from Ray import Ray
from sphere import Sphere


pygame.init()

SCREEN_SIZE = 700
RENDER_DISTANCE = 100
ANGLE_INCREAMENT = 0.1


run = True
# x,y,z = [random.randint(-1000,1000) for i]
camera = Camera(100, 100, -1000, (0,0,0))
camera.emitRays(700,700,1)
sphere = Sphere(350,350,10, 120, (0.2,0.4,0.6), 0)
light = Light(100,10,10, 100, (1.0,1.0,1.0))

for i in camera.rays:
    for j in i:
        sphere.reflectedRay(j, light)

colors = []
# for i in camera.rays:
#     a = j.color
#     a = [z*3 for z in a]
#     tmp = [j.color for j in i]
#     print(tmp)
#     colors.append(tmp)

for i in camera.rays:
    tmp = []
    for j in i:
        a = j.color
        a = [z*255 for z in a]
        tmp.append(a)
    colors.append(tmp)
# print(colors)
win = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    win.fill((255, 255, 255))
    for i, j in itertools.product(range(SCREEN_SIZE), range(SCREEN_SIZE)):
        win.set_at((i, j), (int(colors[i][j][0]), int(colors[i][j][1]), int(colors[i][j][2])))
    pygame.display.update()
pygame.quit()