import numpy as np
import copy

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def softmax(x):
    return np.exp(x)/sum(np.exp(x))
def relu(x):
    return np.maximum(0, x)
def to_one_hot(x):
    output = np.zeros(x.shape)
    max = np.argmax(x)
    output[max] = 1
    return np.squeeze(output).tolist()

class NeuralNetwork:
    def __init__(self, layer_dims, parameters_provided = None):
        self.layers = layer_dims
        parameters = {}
        L = len(layer_dims)
        if parameters_provided == None:
            for l in range(1, L):
                parameters['W' + str(l)] = np.random.uniform(-1, 1, (layer_dims[l], layer_dims[l-1]))
                parameters['b' + str(l)] = np.random.uniform(-1, 1, (layer_dims[l], 1))
            self.parameters = parameters
        else:
            self.parameters = parameters_provided


    def copy(self):
        return copy.deepcopy(self)

    def feedforward(self, inputs):
        def forward_one(A_prev, W, b):
            return np.dot(W, A_prev) + b
        
        A = np.array(inputs)
        A = np.expand_dims(A, axis=1)
        L = len(self.parameters) // 2

        for l in range(1, L):
            Z = forward_one(A, self.parameters['W'+str(l)], self.parameters['b'+str(l)])
            A = sigmoid(Z)

        ZL = forward_one(A, self.parameters['W'+str(L)], self.parameters['b'+str(L)])
        AL = softmax(ZL - max(ZL))
        return AL, self.parameters

def crossover(parent1, parent2):
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    mask = np.random.uniform(0, 1, size = offspring1.parameters['W1'].shape)
    offspring1.parameters['W1'][mask > 0.5] = parent2.parameters['W1'][mask > 0.5]
    offspring2.parameters['W1'][mask > 0.5] = parent1.parameters['W1'][mask > 0.5]

    mask = np.random.uniform(0, 1, size = offspring1.parameters['W2'].shape)
    offspring1.parameters['W2'][mask > 0.5] = parent2.parameters['W2'][mask > 0.5]
    offspring2.parameters['W2'][mask > 0.5] = parent1.parameters['W2'][mask > 0.5]

    mask = np.random.uniform(0, 1, size = offspring1.parameters['b1'].shape)
    offspring1.parameters['b1'][mask > 0.5] = parent2.parameters['b1'][mask > 0.5]
    offspring2.parameters['b1'][mask > 0.5] = parent1.parameters['b1'][mask > 0.5]

    mask = np.random.uniform(0, 1, size = offspring1.parameters['b2'].shape)
    offspring1.parameters['b2'][mask > 0.5] = parent2.parameters['b2'][mask > 0.5]
    offspring2.parameters['b2'][mask > 0.5] = parent1.parameters['b2'][mask > 0.5]

    return offspring1, offspring2

def mutate(individual, prob_mutation=0.05, mutation_type = 'gaussian'):
    if mutation_type == 'gaussian':
        for key, values in individual.parameters.items():
            gaussian_mutation(values, prob_mutation)
    elif mutation_type == 'uniform':
        for key, values in individual.parameters.items():
            random_uniform_mutation(values, prob_mutation)
    return individual

def gaussian_mutation(chromosome, prob_mutation):
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    gaussian_mutation = np.random.normal(size=chromosome.shape) 
    chromosome[mutation_array] += gaussian_mutation[mutation_array]

def random_uniform_mutation(chromosome, prob_mutation):
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(-1, 1, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]

# brain = NeuralNetwork([5,4,4])
# X = [0.123, -0.134, 0.23, 0.124, -0.451]
# AL, params = brain.feedforward(X)
# print(AL)