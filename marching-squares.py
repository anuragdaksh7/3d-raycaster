import itertools
import pygame
import math

pygame.init()

INFLUENCE_RADIUS = 20
CHUNK_SIZE = 40

def drawChunkBorder(width = 700,chunkSize = CHUNK_SIZE):
    for i in range(700//chunkSize):
        for j in range(700//chunkSize):
            pygame.draw.rect(win, (0,255,0), pygame.Rect(i*chunkSize,j*chunkSize,chunkSize,chunkSize),1)

def chunkFromCoord(cx,cy):
    return int(cx//CHUNK_SIZE+(cy//CHUNK_SIZE)*(700//CHUNK_SIZE))

def chunkToCoord(chunk, chunkSize = CHUNK_SIZE):
    c = 700//chunkSize
    x = chunk%c
    y = chunk//c
    return (x*chunkSize,y*chunkSize)

def renderActiveChunk(chunk, chunkSize = CHUNK_SIZE):
    x,y = chunkToCoord(chunk, chunkSize)
    pygame.draw.rect(win, (255,255,255), pygame.Rect(x,y,CHUNK_SIZE,CHUNK_SIZE))

def smootingFunction(center, point):
    d = math.sqrt((center[0]-point[0])**2+(center[1]-point[1])**2)
    r = 1-d/INFLUENCE_RADIUS
    if r<0:
        return 0
    # print(r)
    r = int(r*255)
    return r

def chunkDistance(c1,c2):
    a = abs(c1%(700//CHUNK_SIZE)-c2%(700//CHUNK_SIZE))
    b = abs(c1//(700//CHUNK_SIZE)-c2//(700//CHUNK_SIZE))
    return a+b

def anotherSmoothingFunction(influence):
    c = 10/(INFLUENCE_RADIUS+1)
    k = -1*(influence - c)*(influence - 22 + c)*(248/60)
    return 0 if k< 40 else k

def distance(c1,c2):
    return math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)

win = pygame.display.set_mode((700,700))
centers = [[340,340],[80,340],[340,80]]
run = True
# 50x50 pixels chunks

# def addValidChunk(chunk):
clock = pygame.time.Clock()

chunkBuffer = []
while run:
    chunkBuffer = set()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    win.fill((0,0,0))
    # centers[-1] = list(pygame.mouse.get_pos())
    m = 0
    fps = int(clock.get_fps())
    pygame.display.set_caption(str(fps))
    for cc in centers:
        cx, cy = cc
        tmp = int(cx//CHUNK_SIZE+(cy//CHUNK_SIZE)*(700//CHUNK_SIZE))
        chunkBuffer.add(tmp)
        chunkBuffer.add(tmp-700//CHUNK_SIZE-1)
        chunkBuffer.add(tmp-700//CHUNK_SIZE  )
        chunkBuffer.add(tmp-700//CHUNK_SIZE+1)
        chunkBuffer.add(tmp - 1      )
        chunkBuffer.add(tmp + 1      )
        chunkBuffer.add(tmp+700//CHUNK_SIZE-1)
        chunkBuffer.add(tmp+700//CHUNK_SIZE  )
        chunkBuffer.add(tmp+700//CHUNK_SIZE+1)
    # print(chunkBuffer)
    l = 0
    for chunk in list(chunkBuffer):
        # renderActiveChunk(chunk)
        effectiveCenters = []
        for center in centers:
            chk = chunkFromCoord(center[0],center[1])
            tmpp = chunk-700//CHUNK_SIZE
            tmpp_ = chunk+700//CHUNK_SIZE
            
            if chunkDistance(chk,chunk)<=3:
                effectiveCenters.append(center)
        cx,cy = chunkToCoord(chunk)
        for i, j in itertools.product(range(cx,cx+CHUNK_SIZE), range(cy,cy+CHUNK_SIZE)):
            l+=1
            influence = sum(
                (10/(distance(center, (i, j))+1))
                for center in effectiveCenters
            )

            if influence>m:
                m = influence
            if influence> 10/(INFLUENCE_RADIUS+1)+0.1:
                win.set_at((i, j), (0, 0, max(0,min(255,int(anotherSmoothingFunction(influence))))))

    # for cc in centers:
    #     for i, j in itertools.product(range(cc[0]-70,cc[0]+70), range(cc[1]-70,cc[1]+70)):
    #         influence = sum(
    #             (10/(distance(center, (i, j))+1))
    #             for center in centers
    #         )
    #         if influence>m:
    #             m = influence
    #         if influence> 10/(INFLUENCE_RADIUS+1):
    #             win.set_at((i, j), (0, 0, max(0,min(255,int(anotherSmoothingFunction(influence))))))

    centers[1][0] += 1
    centers[2][1] += 1
    # pygame.draw.circle(win, (255,0,0), (40,40), 20, 0)
    drawChunkBorder()
    pygame.display.update()
    clock.tick(60)  # Adjust the value to set the desired frame rate
    print(l)
pygame.quit()
print("\n")
print(m)