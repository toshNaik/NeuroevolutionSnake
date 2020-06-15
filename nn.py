import numpy as np

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
    def __init__(self, layer_dims):
        self.layers = layer_dims
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.standard_normal((layer_dims[l], layer_dims[l-1]))
            parameters['b' + str(l)] = np.random.standard_normal((layer_dims[l], 1))
        self.parameters = parameters

    def feedforward(self, inputs):
        def forward_one(A_prev, W, b):
            return np.dot(W, A_prev) + b
        
        A = np.array(inputs)
        A = np.expand_dims(A, axis=1)
        L = len(self.parameters) // 2

        for l in range(1, L):
            Z = forward_one(A, self.parameters['W'+str(l)], self.parameters['b'+str(l)])
            A = np.tanh(Z)

        ZL = forward_one(A, self.parameters['W'+str(L)], self.parameters['b'+str(L)])
        AL = softmax(ZL)
        
        return AL, self.parameters

# X1 = [0.1, 0.524, 1, -0.1, -0.415, 0.85, 0.12, 0.23, -0.211, -0.45]
# X2 = [-0.5234, -0.114, 1, 0.83754, 0.1, -0.901, -0.1, 0.5631, 0.4, -0.2131]
# X3 = [-0.512, 0.41, 0.123, -0.214, 0.4231]
# X4 = [-0.901, -0.1, 0.5631, 0.4, -0.2131]
# X5 = [0.85, 0.12, 0.23, -0.211, -0.45]

# brain = NeuralNetwork([10, 6, 4])
# A, _ = brain.feedforward(X1)
# print('----------------')

# A, _ = brain.feedforward(X2)

# # AL, _ = brain.feedforward(X3)
# # print(AL)
# # AL, _ = brain.feedforward(X4)
# # print(AL)
# # AL, _ = brain.feedforward(X5)
# # print(AL)