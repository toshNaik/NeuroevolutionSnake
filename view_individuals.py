import pickle
import snake

filehandler = open('variables/individuals/best_individuals.obj', 'rb')
best_individuals = pickle.load(filehandler)
filehandler.close()

print(len(best_individuals))
for i, individual in enumerate(best_individuals):
    snake.play_game(gui=True, speed=40, snakePos=(5,5), view=True, brain=individual)
    