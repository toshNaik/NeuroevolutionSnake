import random
import nn
import pickle

def nextGeneration(population, SIZE_POP, className, *args):
    new_gen = []
    best_individual = calculateFitness(population)
    parents1 = roulette_wheel(population, SIZE_POP)

    for i in range(SIZE_POP):
        child_brain = nn.mutate(parents1[i].brain.copy())
        new_gen.append(className(*args, brain=child_brain))

    return new_gen, best_individual

def calculateFitness(population):
    max_fitness = 0
    max_index = 0
    for i, individual in enumerate(population):
        steps = individual.total_steps
        score = individual.score
        fitness_score = steps + (2**score+500*score**2)#-(score**1.2*(0.25*steps)**1.3)
        individual.fitness = fitness_score
        if fitness_score > max_fitness:
            max_fitness = fitness_score
            max_index = i
    best_individual = population[max_index].brain
    return best_individual

def food_eating(population):
    max_index = 0
    max_score = 0
    for i, individual in enumerate(population):
        if individual.score > max_score:
            max_score = individual.score
            max_index = i
    return population[max_index]

def roulette_wheel(population, SIZE_POP):
    # TODO: Pick SIZE_POP parents from population.
    selection = []
    wheel = sum(individual.fitness for individual in population)
    for _ in range(SIZE_POP):
        pick = random.uniform(0, wheel)
        current = 0
        for individual in population:
            current += individual.fitness
            if current > pick:
                selection.append(individual)
                break
    return selection

    # TODO: parent1 and parent2 are neural networks perform crossover and return new brain.