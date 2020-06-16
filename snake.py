import pygame
import random
import nn

SIZE_POP = 3
GENERATION = []

rows = 10
width = 300
sqrt2 = 2**0.5
UP = [1.0, 0.0, 0.0, 0.0]
RIGHT = [0.0, 1.0, 0.0, 0.0]
DOWN = [0.0, 0.0, 1.0, 0.0]
LEFT = [0.0, 0.0, 0.0, 1.0]

def apply_on_all(seq, method, *args, **kwargs):
    for obj in seq:
         getattr(obj, method)(*args, **kwargs)

class Cube(object):
    global rows, width, sqrt2
    def __init__(self, start, dirx=1, diry=0, color=(255,0,0)):
        self.pos = start
        self.dirx = 1
        self.diry = 0
        self.color = color

    def isHorizontal(self, cube):
        if self.pos[1] == cube.pos[1]:
            return self.pos[0] - cube.pos[0]
        return 0
    
    def isVertical(self, cube):
        if self.pos[0] == cube.pos[0]:
            return self.pos[1] - cube.pos[1]
        return 0

    def is135or315(self, cube):
        if self.pos[0] - cube.pos[0] == self.pos[1] - cube.pos[1]:
            return sqrt2 * (self.pos[0] - cube.pos[0])
        return 0

    def is45or225(self, cube):
        if self.pos[0] - cube.pos[0] == cube.pos[1] - self.pos[1]:
            return sqrt2 * (self.pos[0] - cube.pos[0])
        return 0

    def move(self, dirx, diry):
        self.dirx = dirx
        self.diry = diry
        self.pos = (self.pos[0]+self.dirx, self.pos[1]+self.diry)

    def draw(self, window):
        dis = width // rows
        i = self.pos[0]
        j = self.pos[1]
        pygame.draw.rect(window, self.color, (i*dis, j*dis, dis, dis))

class Snake(object):
    global rows, width, sqrt2, UP, DOWN, LEFT, RIGHT
        
    def __init__(self, color, pos):
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
        if(self.steps_since_last_food == 100):
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
        x, y = self.head.pos
        bodyPos = self.body[1:]
        
        # DISTANCES FROM WALLS
        north = y / rows
        south = (rows - y - 1) / rows
        west = x / rows
        east = (rows - x - 1) / rows
        nw = min(north, west) * sqrt2
        sw = min(south, west) * sqrt2
        ne = min(north, east) * sqrt2
        se = min(south, east) * sqrt2
        
        # DISTANCES FROM FOOD
        h = self.head.isHorizontal(snack) / rows
        v = self.head.isVertical(snack) / rows
        d1 = self.head.is135or315(snack) / rows
        d2 = self.head.is45or225(snack) / rows
        
        leftFood = rightFood = aboveFood = belowFood = d135Food = d315Food = d225Food = d45Food = 0.
        
        if h:
            if h > 0:
                leftFood = h / rows
            else:
                rightFood = -h / rows
        elif v:
            if v > 0:
                aboveFood = v / rows
            else:
                belowFood = -v / rows
        elif d1:
            if d1 > 0:
                d135Food = d1 / rows
            else:
                d315Food = -d1 / rows
        elif d2:
            if d2 > 0:
                d225Food = d2 / rows
            else:
                d45Food = -d2 / rows
                
        # DISTANCES FROM BODY PARTS
        left = [self.head.isHorizontal(x) for x in bodyPos if self.head.isHorizontal(x) > 0]
        if not left:
            left = 0.
        else: left = min(left) / rows
        
        right = [self.head.isHorizontal(x) for x in bodyPos if self.head.isHorizontal(x) < 0]
        if not right:
            right = 0.
        else: right = min(right) / rows
        
        above = [self.head.isVertical(x) for x in bodyPos if self.head.isVertical(x) > 0]
        if not above:
            above = 0.
        else: above = min(above) / rows
        
        below = [self.head.isVertical(x) for x in bodyPos if self.head.isVertical(x) < 0]
        if not below:
            below = 0.
        else: below = min(below) / rows
        
        d135 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) > 0]
        if not d135:
            d135 = 0.
        else: d135 = min(d135) / rows
        
        d315 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) < 0]
        if not d315:
            d315 = 0.
        else: d315 = min(d315) / rows
        
        d45 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) < 0]
        if not d45:
            d45 = 0.
        else: d45 = min(d45) / rows
        
        d225 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) > 0]
        if not d225:
            d225 = 0.
        else: d225 = min(d225) / rows

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
                            'food' : [aboveFood, d45Food, rightFood, d315Food, belowFood, d225Food, leftFood, d135Food],
                            'body' : [above, d45, right, d315, below, d225, left, d135],
                            'head' : headDir,
                            'tail' : tailDir}
        
        return dictionaryInputs['wall'] + dictionaryInputs['food'] + dictionaryInputs['body'] + dictionaryInputs['head']

    def move(self):
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
        for i, c in enumerate(self.body):
            c.draw(window)

def randomSnack(snake):
    global rows
    impossible_spawns = []
    x = None
    y = None
    for x, y in [x.pos for x in snake.body]:
        impossible_spawns.append((x,y))
    while (x, y) in impossible_spawns or x == None:
        x = random.randrange(0, rows)
        y = random.randrange(0, rows)
    return (x, y)

def redrawWindow(window, snake, snack):
    global rows, width
    window.fill((0,0,0))
    #snake.draw(window)
    #snack.draw(window)
    apply_on_all(snake, 'draw', (window))
    apply_on_all(snack, 'draw', (window))
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

def main():
    '''
    Runs the game i.e the current generation.
    '''
    global width, rows

    snake_population = []
    snacks = []
    to_be_killed = []

    snakePos = (5,5)
    win = pygame.display.set_mode((width, width))

    # Create SIZE_POP snakes and corresponding snacks. Snake [i] can only 'interact' with snack[i].
    for i in range(SIZE_POP):
        snake_population.append(Snake((255,0,0), snakePos))
        snacks.append(Cube(randomSnack(snake_population[i]), color=(0,0,255)))

    flag = True
    clock = pygame.time.Clock()
    
    while flag:
        pygame.time.delay(50)
        clock.tick(5)
        # snake.move()

        def check_conditions(snake, snack):
            '''
            Checks:
            1. If snake "ate" the food.
            2. If snake collides with wall. returns True.
            3. If snake collides with itself. returns True.
            
            returns False if snake did not die.
            '''
            if snake.body[0].pos == snack.pos:
                snake.addCube()
                snake.score += 1
                snake.steps_since_last_food = 0
                snack.pos = randomSnack(snake)
            
            if snake.body[0].pos[0] > rows-1 or snake.body[0].pos[0] < 0 or snake.body[0].pos[1] > rows-1 or snake.body[0].pos[1] < 0:
                return True
                #TODO: Instead of reset, save copy of snake and kill it (Remove from list maybe?).
            
            for x in range(len(snake.body)):
                if snake.body[x].pos in list(map(lambda z:z.pos, snake.body[x+1:])):
                    return True
                    #TODO: Instead of reset, save copy of snake and kill it (Remove from list maybe?).
            
            return False

        # Iterate through all snakes. Indices of dead snakes (if any) are appended to to_be_killed list
        for i, _ in enumerate(snake_population):
            snake_dead = snake_population[i].think(snacks[i])
            snake_dead2 = check_conditions(snake_population[i], snacks[i])
            if snake_dead or snake_dead2:
                to_be_killed.insert(0, i)
        
        # Iterate through snakes to be killed and remove them from list.
        if not to_be_killed == False:
            for i in to_be_killed:
                GENERATION.append(snake_population.pop(i))
                del snacks[i]
        to_be_killed = []

        if len(snake_population) == 0:
            # TODO: If all snakes are dead. Create new gen.
            print(GENERATION[0].total_steps)
            print(GENERATION[1].total_steps)
            print(GENERATION[2].total_steps)
            pygame.quit()
            return

        # Drawing logic comment if not wanted.
        redrawWindow(win, snake_population, snacks)


# TODO: Calculate fitness, perform crossover, mutation.
# TODO: OPTIONAL: Save all the weights of last population to continue training.
# TODO: OPTIONAL: Save the weights of best performing snake.

main()