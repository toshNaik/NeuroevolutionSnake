import pickle
import snake

filehandler = open('best_individuals.obj', 'rb')
best_individuals = pickle.load(filehandler)
filehandler.close()

snake.play_game(gui=True, speed=10, snakePos=(5,5), view=True, brain=best_individuals[-1])
snake.play_game(gui=True, speed=10, snakePos=(5,5), view=True, brain=best_individuals[-2])
print(len(best_individuals))
# for individual in best_individuals:
#     snake.play_game(gui=True, speed=10, snakePos=(5,5), view=True, brain=individual)
    