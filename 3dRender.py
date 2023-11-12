import math
import pygame
import numpy as np
import random

pygame.init()

SCREEN_SIZE = 700
RENDER_DISTANCE = 100
ANGLE_INCREAMENT = 0.1

class Light:
    def __init__(self, x, y, z, power):
        self.x = x
        self.y = y
        self.power = power

