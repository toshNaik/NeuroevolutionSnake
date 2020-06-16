import pickle
import snake

filehandler = open('best_individuals.obj', 'rb')
best_individuals = pickle.load(filehandler)
filehandler.close()

for individuals in best_individuals:
    snek = snake.Snake((255,0,0), (5,5), brain = individuals)
    snack = snake.randomSnack(snek)
    