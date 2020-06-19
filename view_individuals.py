import pickle
import snake

filehandler = open('variables/generations/last_generation.obj', 'rb')
best_individuals = pickle.load(filehandler)
filehandler.close()

# snake.play_game(gui=True, speed=20, snakePos=(5,5), view=True, brain=best_individuals[-1])
# snake.play_game(gui=True, speed=40, snakePos=(5,5), view=True, brain=best_individuals[-10])
# snake.play_game(gui=True, speed=40, snakePos=(5,5), view=True, brain=best_individuals[-20])
# snake.play_game(gui=True, speed=40, snakePos=(5,5), view=True, brain=best_individuals[-50])
# print(len(best_individuals))
# best_individuals = best_individuals[:50]
# for i, individual in enumerate(best_individuals):
#     if  i%5 == 0:
#         snake.play_game(gui=True, speed=40, snakePos=(5,5), view=True, brain=individual)
    