import nn
import pygame
import random
from cube import Cube
from settings import SIZE_POP, ROWS, WIDTH, SQRT2, UP, DOWN, LEFT, RIGHT

class Snake(object):
    
    def __init__(self, color, pos, brain = None):   
        x = random.randint(-1,1)
        if x != 0: y = 0
        else: y = random.choice([-1,1])
        self.head = Cube(pos, dirx=x ,diry=y, color=(100,0,0))
        self.body = []
        self.body.append(self.head)
        self.addCube()
        self.addCube()
        self.turns = {}
        self.dirx = x
        self.diry = y
        if brain == None:
            self.brain = nn.NeuralNetwork([24, 16, 10, 4])
        else:
            self.brain = brain
        self.score = 0
        self.fitness = 0
        self.total_steps = 0
        self.steps_since_last_food = 0
    
    def think(self, snack):
        ''''
        Performs:
        1. Forward pass through neural network.
        2. Moves as per output of neural network.
        3. Increments total_steps and steps_since_last_food variable.
        4. If steps_since_last_food exceeds 100 returns True.
        5. returns False otherwise.
        '''
        move, _ = self.brain.feedforward(self.vision(snack))
        move = nn.to_one_hot(move)
        self.total_steps += 1
        self.steps_since_last_food += 1
        if(self.steps_since_last_food == ROWS**2):
            return True

        if move == LEFT and (self.dirx != 1 and self.diry != 0):
            self.dirx = -1
            self.diry = 0
            self.turns[self.head.pos[:]] = [self.dirx, self.diry]

        elif move == RIGHT and (self.dirx != -1 and self.diry != 0):
            self.dirx = 1
            self.diry = 0
            self.turns[self.head.pos[:]] = [self.dirx, self.diry]

        elif move == UP and (self.dirx != 0 and self.diry != 1):
            self.dirx = 0
            self.diry = -1
            self.turns[self.head.pos[:]] = [self.dirx, self.diry]
        
        elif move == DOWN and (self.dirx != 0 and self.diry != -1):
            self.dirx = 0
            self.diry = 1
            self.turns[self.head.pos[:]] = [self.dirx, self.diry]
        
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirx, c.diry)
        return False

    def vision(self, snack):
        '''
        Calculates, in 8 directions, distance from walls, food and body parts (if there is any in that direction) and direction of head and tail.
        '''
        x, y = self.head.pos
        bodyPos = self.body[1:]
        
        # DISTANCES FROM WALLS
        north = y / ROWS
        south = (ROWS - y - 1) / ROWS
        west = x / ROWS
        east = (ROWS - x - 1) / ROWS
        nw = min(north, west) * SQRT2
        sw = min(south, west) * SQRT2
        ne = min(north, east) * SQRT2
        se = min(south, east) * SQRT2
        
        # DISTANCES FROM FOOD
        h = self.head.isHorizontal(snack, False) #/ ROWS
        v = self.head.isVertical(snack, False) #/ ROWS
        d1 = self.head.is135or315(snack, False) #/ ROWS
        d2 = self.head.is45or225(snack, False) #/ ROWS
                
        # DISTANCES FROM BODY PARTS
        left = [self.head.isHorizontal(x) for x in bodyPos if self.head.isHorizontal(x) > 0]
        if not left:
            left = 0.
        else: left = min(left) / ROWS
        
        right = [self.head.isHorizontal(x) for x in bodyPos if self.head.isHorizontal(x) < 0]
        if not right:
            right = 0.
        else: right = min(right) / ROWS
        
        above = [self.head.isVertical(x) for x in bodyPos if self.head.isVertical(x) > 0]
        if not above:
            above = 0.
        else: above = min(above) / ROWS
        
        below = [self.head.isVertical(x) for x in bodyPos if self.head.isVertical(x) < 0]
        if not below:
            below = 0.
        else: below = min(below) / ROWS
        
        d135 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) > 0]
        if not d135:
            d135 = 0.
        else: d135 = min(d135) / (ROWS*SQRT2)
        
        d315 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) < 0]
        if not d315:
            d315 = 0.
        else: d315 = min(d315) / (ROWS*SQRT2)
        
        d45 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) < 0]
        if not d45:
            d45 = 0.
        else: d45 = min(d45) / (ROWS*SQRT2)
        
        d225 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) > 0]
        if not d225:
            d225 = 0.
        else: d225 = min(d225) / (ROWS*SQRT2)

        # DIRECTION OF HEAD CUBE
        headDir = []            # [up, right, down, left]
        if self.diry == -1: headDir = [1., 0., 0., 0.]
        elif self.dirx == 1: headDir = [0., 1., 0., 0.]
        elif self.diry == 1: headDir = [0., 0., 1., 0.]
        elif self.dirx == -1: headDir = [0., 0., 0., 1.]
        
        tailDir = []            # [up, right, down, left]
        if self.body[-1].diry == -1: tailDir = [1., 0., 0., 0.]
        elif self.body[-1].dirx == 1: tailDir = [0., 1., 0., 0.]
        elif self.body[-1].diry == 1: tailDir = [0., 0., 1., 0.]
        elif self.body[-1].dirx == -1: tailDir = [0., 0., 0., 1.]

        dictionaryInputs = {'wall' : [north, ne, east, se, south, sw, west, nw],
                            'food' : [h, d1, v, d2],
                            'body' : [above, d45, right, d315, below, d225, left, d135],
                            'head' : headDir,
                            'tail' : tailDir}
        
        return dictionaryInputs['wall'] + dictionaryInputs['food'] + dictionaryInputs['body'] + dictionaryInputs['head']

    def move(self):
        '''
        This function enables user to control the snake.
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
            keys = pygame.key.get_pressed()
            for key in keys:
                if keys[pygame.K_LEFT] and (self.dirx != 1 and self.diry != 0):
                    self.dirx = -1
                    self.diry = 0
                    self.turns[self.head.pos[:]] = [self.dirx, self.diry]
                
                elif keys[pygame.K_RIGHT] and (self.dirx != -1 and self.diry != 0): #or len(self.body) == 1):
                    self.dirx = 1
                    self.diry = 0
                    self.turns[self.head.pos[:]] = [self.dirx, self.diry]

                elif keys[pygame.K_DOWN] and (self.dirx != 0 and self.diry != -1): #or len(self.body) == 1):
                    self.dirx = 0
                    self.diry = 1
                    self.turns[self.head.pos[:]] = [self.dirx, self.diry]

                elif keys[pygame.K_UP] and (self.dirx != 0 and self.diry != 1):# or len(self.body) == 1):
                    self.dirx = 0
                    self.diry = -1
                    self.turns[self.head.pos[:]] = [self.dirx, self.diry]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirx, c.diry)

    def reset(self, pos):
        '''
        After dying resets the snake.
        '''
        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirx = 0
        self.diry = 1
        self.score = 0
        self.fitness = 0
        self.total_steps = 0
        self.steps_since_last_food = 0
        self.brain = nn.NeuralNetwork([28, 20, 12, 4])

    def addCube(self):
        '''
        Adds a cube to body list.
        '''
        tail = self.body[-1]
        dx, dy = tail.dirx, tail.diry

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0]-1, tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0]+1, tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1]+1)))

        self.body[-1].dirx = dx
        self.body[-1].diry = dy

    def draw(self, window):
        '''
        Draws the snake on window
        '''
        for i, c in enumerate(self.body):
            c.draw(window)
