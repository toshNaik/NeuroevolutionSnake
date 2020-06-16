import pygame
import random
import nn

SIZE_POP = 3
GENERATION = []

ROWS = 10
WIDTH = 300
SQRT2 = 2**0.5

UP = [1.0, 0.0, 0.0, 0.0]
RIGHT = [0.0, 1.0, 0.0, 0.0]
DOWN = [0.0, 0.0, 1.0, 0.0]
LEFT = [0.0, 0.0, 0.0, 1.0]

def apply_on_all(sequence, method, *args, **kwargs):
    '''
    Helper function to apply obj.method(*args(**kwargs)) on seq of objects
    '''
    for obj in sequence:
         getattr(obj, method)(*args, **kwargs)

class Cube(object):
    global ROWS, WIDTH, SQRT2
    def __init__(self, start, dirx=1, diry=0, color=(255,0,0)):
        self.pos = start
        self.dirx = 1
        self.diry = 0
        self.color = color

    def isHorizontal(self, cube):
        '''
        Returns horizontal distance between 2 cubes. (If their vertical distances are same). else returns 0
        '''
        if self.pos[1] == cube.pos[1]:
            return self.pos[0] - cube.pos[0]
        return 0
    
    def isVertical(self, cube):
        '''
        Returns vertical distance between 2 cubes. (If their horizontal distances are same). else returns 0
        '''
        if self.pos[0] == cube.pos[0]:
            return self.pos[1] - cube.pos[1]
        return 0

    def is135or315(self, cube):
        '''
        Returns distance between 2 cubes. (If they lie on the line x + y = 0). else returns 0
        '''
        if self.pos[0] - cube.pos[0] == self.pos[1] - cube.pos[1]:
            return SQRT2 * (self.pos[0] - cube.pos[0])
        return 0

    def is45or225(self, cube):
        '''
        Returns distance between 2 cubes. (If they lie on the line x - y = 0). else returns 0
        '''
        if self.pos[0] - cube.pos[0] == cube.pos[1] - self.pos[1]:
            return SQRT2 * (self.pos[0] - cube.pos[0])
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

class Snake(object):
    global ROWS, WIDTH, SQRT2, UP, DOWN, LEFT, RIGHT
        
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
        h = self.head.isHorizontal(snack) / ROWS
        v = self.head.isVertical(snack) / ROWS
        d1 = self.head.is135or315(snack) / ROWS
        d2 = self.head.is45or225(snack) / ROWS
        
        leftFood = rightFood = aboveFood = belowFood = d135Food = d315Food = d225Food = d45Food = 0.
        
        if h:
            if h > 0:
                leftFood = h / ROWS
            else:
                rightFood = -h / ROWS
        elif v:
            if v > 0:
                aboveFood = v / ROWS
            else:
                belowFood = -v / ROWS
        elif d1:
            if d1 > 0:
                d135Food = d1 / ROWS
            else:
                d315Food = -d1 / ROWS
        elif d2:
            if d2 > 0:
                d225Food = d2 / ROWS
            else:
                d45Food = -d2 / ROWS
                
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
        else: d135 = min(d135) / ROWS
        
        d315 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) < 0]
        if not d315:
            d315 = 0.
        else: d315 = min(d315) / ROWS
        
        d45 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) < 0]
        if not d45:
            d45 = 0.
        else: d45 = min(d45) / ROWS
        
        d225 = [self.head.is135or315(x) for x in bodyPos if self.head.is135or315(x) > 0]
        if not d225:
            d225 = 0.
        else: d225 = min(d225) / ROWS

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

def randomSnack(snake):
    '''
    Generates a random snack on window while avoiding snake
    '''
    global ROWS
    impossible_spawns = []
    x = None
    y = None
    for x, y in [x.pos for x in snake.body]:
        impossible_spawns.append((x,y))
    while (x, y) in impossible_spawns or x == None:
        x = random.randrange(0, ROWS)
        y = random.randrange(0, ROWS)
    return (x, y)

def redrawWindow(window, snake, snack):
    '''
    Drawing function
    '''
    global ROWS, WIDTH
    window.fill((0,0,0))
    #snake.draw(window)
    #snack.draw(window)
    apply_on_all(snake, 'draw', (window))
    apply_on_all(snack, 'draw', (window))
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

def play_game(gui = False, speed = 10, snakePos = (5,5)):
    '''
    Runs the game i.e the current generation.
    1. gui: To draw or not.
    2. speed: Speed of animation.
    3. snakePos: Spawn location of snake.
    '''
    global WIDTH, ROWS
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
        
        if snake.body[0].pos[0] > ROWS-1 or snake.body[0].pos[0] < 0 or snake.body[0].pos[1] > ROWS-1 or snake.body[0].pos[1] < 0:
            return True
            #TODO: Instead of reset, save copy of snake and kill it (Remove from list maybe?).
        
        for x in range(len(snake.body)):
            if snake.body[x].pos in list(map(lambda z:z.pos, snake.body[x+1:])):
                return True
                #TODO: Instead of reset, save copy of snake and kill it (Remove from list maybe?).
        
        return False

    snake_population = []
    snacks = []
    to_be_killed = []
    win = pygame.display.set_mode((WIDTH, WIDTH))

    # Create SIZE_POP snakes and corresponding snacks. Snake [i] can only 'interact' with snack[i].
    for i in range(SIZE_POP):
        snake_population.append(Snake((255,0,0), snakePos))
        snacks.append(Cube(randomSnack(snake_population[i]), color=(0,0,255)))

    flag = True
    clock = pygame.time.Clock()
    
    while flag:
        pygame.time.delay(50)
        clock.tick(speed)
        # snake.move()

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
            return GENERATION

        # Drawing logic.
        if gui:
            redrawWindow(win, snake_population, snacks)


# TODO: Calculate fitness, perform crossover, mutation.
# TODO: OPTIONAL: Save all the weights of last population to continue training.
# TODO: OPTIONAL: Save the weights of best performing snake.

GENERATION = play_game(True, 20)
pygame.quit()