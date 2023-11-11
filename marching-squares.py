import itertools
import pygame
import math

pygame.init()

INFLUENCE_RADIUS = 100

def smootingFunction(center, point):
    d = math.sqrt((center[0]-point[0])**2+(center[1]-point[1])**2)
    r = 1-d/INFLUENCE_RADIUS
    if r<0:
        return 0
    # print(r)
    r = int(r*255)
    return r

def anotherSmoothingFunction(influence):
    c = 10/21
    return 0 if influence< c else -3086*(c-influence)*(c+influence)

def distance(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)

win = pygame.display.set_mode((700,700))
centers = [[340,340],[80,340],[340,80],list(pygame.mouse.get_pos())]
run = True
chunkBuffer = []
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    win.fill((0,0,0))
    centers[-1] = list(pygame.mouse.get_pos())
    m = 0
    for cc in centers:
        for i, j in itertools.product(range(cc[0]-70,cc[0]+70), range(cc[1]-70,cc[1]+70)):
            influence = sum(
                (10/(distance(center, (i, j))+1))
                for center in centers
            )
            if influence>m:
                m = influence
            if influence> 10/21:
                win.set_at((i, j), (0, 0, max(0,min(255,int(anotherSmoothingFunction(influence))))))

    centers[1][0] += 1
    centers[2][1] += 1
    # pygame.draw.circle(win, (255,0,0), (40,40), 20, 0)
    pygame.display.update()
pygame.quit()
print("\n")
print(m)