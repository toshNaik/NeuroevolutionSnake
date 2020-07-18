import pygame
from settings import ROWS, WIDTH, SQRT2

class Cube(object):
    global ROWS, WIDTH, SQRT2
    def __init__(self, start, dirx=1, diry=0, color=(255,0,0)):
        self.pos = start
        self.dirx = 1
        self.diry = 0
        self.color = color

    def isHorizontal(self, cube, distances = True):
        '''
        Returns horizontal distance between 2 cubes. (If their vertical distances are same). else returns 0
        '''
        if self.pos[1] == cube.pos[1]:
            distance = self.pos[0] - cube.pos[0]
            if distances:
                return distance
            else:
                if distance > 0: return 1
                else: return -1
        return 0
    
    def isVertical(self, cube, distances = True):
        '''
        Returns vertical distance between 2 cubes. (If their horizontal distances are same). else returns 0
        '''
        if self.pos[0] == cube.pos[0]:
            distance = self.pos[1] - cube.pos[1]
            if distances:
                return distance
            else:
                if distance > 0: return 1
                else: return -1
        return 0

    def is135or315(self, cube, distances = True):
        '''
        Returns distance between 2 cubes. (If they lie on the line x + y = 0). else returns 0
        '''
        if self.pos[0] - cube.pos[0] == self.pos[1] - cube.pos[1]:
            distance = SQRT2 * (self.pos[0] - cube.pos[0])
            if distances:
                return distance
            else:
                if distance > 0: return 1
                else: return -1
        return 0

    def is45or225(self, cube, distances = True):
        '''
        Returns distance between 2 cubes. (If they lie on the line x - y = 0). else returns 0
        '''
        if self.pos[0] - cube.pos[0] == cube.pos[1] - self.pos[1]:
            distance = SQRT2 * (self.pos[0] - cube.pos[0])
            if distances:
                return distance
            else:
                if distance > 0: return 1
                else: return -1
        return 0

    def move(self, dirx, diry):
        '''
        Changes direction of motion of cube
        '''
        self.dirx = dirx
        self.diry = diry
        self.pos = (self.pos[0]+self.dirx, self.pos[1]+self.diry)

    def draw(self, window):
        '''
        Draws cube on window
        '''
        dis = WIDTH // ROWS
        i = self.pos[0]
        j = self.pos[1]
        pygame.draw.rect(window, self.color, (i*dis, j*dis, dis, dis))
