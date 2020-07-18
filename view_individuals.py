import pickle
import snake

filehandler = open('variables/individuals/best_individuals_upto3300.obj', 'rb')
best_individuals = pickle.load(filehandler)
filehandler.close()

# for i, best_individual in enumerate(best_individuals):
#     if(i % 100 == 0):
snake.play_game(gui=True, speed=20, snakePos=(5,5), view=True, brain=best_individuals[-1])