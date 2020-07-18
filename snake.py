import pygame
import random
import nn
import ga
import pickle
from cube import Cube
from snake_class import Snake
from settings import SIZE_POP, ROWS, WIDTH, SQRT2, UP, DOWN, LEFT, RIGHT, NO_OF_GEN


best_individuals = []
GENERATION = []

def apply_on_all(sequence, method, *args, **kwargs):
    '''
    Helper function to apply obj.method(*args(**kwargs)) on seq of objects
    '''
    for obj in sequence:
        getattr(obj, method)(*args, **kwargs)

def save(filename1 = 'best_individuals.obj', filename2 = 'last_generation.obj'):
    '''
    Save the best individuals upto that generation along with entire last generation
    '''
    file1 = open(f'variables/individuals/{filename1}', 'wb')
    pickle.dump(best_individuals, file1)
    file1.close()
    
    file2 = open(f'variables/generations/{filename2}', 'wb')
    pickle.dump(GENERATION, file2)
    file2.close()

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
    apply_on_all(snake, 'draw', (window))
    apply_on_all(snack, 'draw', (window))
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            save()
    window.fill((0,0,0))

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
        snake.score += 2
        snake.steps_since_last_food = 0
        snack.pos = randomSnack(snake)
    
    if snake.body[0].pos[0] > ROWS-1 or snake.body[0].pos[0] < 0 or snake.body[0].pos[1] > ROWS-1 or snake.body[0].pos[1] < 0:
        return True
    
    for x in range(len(snake.body)):
        if snake.body[x].pos in list(map(lambda z:z.pos, snake.body[x+1:])):
            return True
    
    return False

def play_game(gui = False, speed = 10, snakePos = (5,5), number_of_gen = 1, view = False, brain = None):
    '''
    Runs the game i.e the current generation.
    1. gui: To draw or not.
    2. speed: Speed of animation.
    3. snakePos: Spawn location of snake.
    4. Number of generations
    5. Set to true when viewing individuals
    6. If viewing then provide brain
    '''
    global GENERATION
    count = 0
    snake_population = []
    snacks = []
    to_be_killed = []

    # Create SIZE_POP snakes and corresponding snacks. Snake [i] can only 'interact' with snack[i].
    if view == False:
        for i in range(SIZE_POP):
            snake_population.append(Snake((255,0,0), snakePos))
            snacks.append(Cube(randomSnack(snake_population[i]), color=(0,0,255)))

    elif view == True:
        snake_population.append(Snake((255,0,0), snakePos, brain = brain))
        snacks.append(Cube(randomSnack(snake_population[0]), color=(0,0,255)))
    
    if gui:
        win = pygame.display.set_mode((WIDTH, WIDTH))
        clock = pygame.time.Clock()
    
    while True:
        if gui:
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

        # End of a generation
        if len(snake_population) == 0:
            if view == True:
                pygame.quit()
            snake_population, best = ga.nextGeneration(GENERATION, SIZE_POP, Snake, ((255,0,0)), (snakePos))
            best_individuals.append(best)
            count += 1
            print(count)
            if count == number_of_gen:
                return
            if count%50 == 0:
                print('saving')
                save(f'best_individuals_upto{count}.obj', f'generation_no{count}.obj')
            GENERATION = []
            for i, snake in enumerate(snake_population):
                snacks.append(Cube(randomSnack(snake), color = (0,0,255)))

        # Drawing logic.
        if gui:
            redrawWindow(win, snake_population, snacks)

def main():
    play_game(True, speed = 40, number_of_gen=NO_OF_GEN)
    pygame.quit()
    save()

if __name__ == '__main__':
    main()
    print('Done!')