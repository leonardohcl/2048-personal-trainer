from enum import Enum
from math import floor
from random import random, sample, uniform
import numpy as np
from scipy.special import softmax

class CrossoverMode(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    RANDOM = 3

class ActivationFunction(Enum):
    NONE = 0
    ReLU = 1
    SOFTMAX = 2

def square(x):
    return x * x

def relu(vals):
    return np.array([max([0.0,x]) for x in vals])

def softmax(vals):
    return softmax(vals)

def mean_sqr_errors(errors):    
    return sum(map(square, errors)) / len(errors)

def batchnorm(x, gamma=1, beta=0, eps=1e-5):
    sample_mean = x.mean(axis=0)
    sample_var = x.var(axis=0)

    std = np.sqrt(sample_var + eps)
    x_centered = x - sample_mean
    x_norm = x_centered / std
    out = gamma * x_norm + beta

    return out

def matrix_random_crossover(m1, m2):
    rows, cols = np.shape(m1)
    param_count = rows * cols
    cross_point = random()
    m1_params = floor(param_count * cross_point)
    m2_params = param_count - m1_params

    result = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            if m1_params > 0 and random() < cross_point:
                result[row][col] = m1[row][col]
                m1_params -= 1
            elif m2_params > 0:
                result[row][col] = m2[row][col]
                m2_params -= 1
            else:
                result[row][col] = m1[row][col]
                m1_params -= 1
    return result

def matrix_simple_crossover(m1,m2, points = 1):
    rows, cols = np.shape(m1)
    param_count = rows * cols
    cross_points = sample(range(1, param_count), points)
    cross_points.sort()
    flat_m1 = m1.flatten().tolist()
    flat_m2 = m2.flatten().tolist()
    result = flat_m1[:cross_points[0]]
    for idx, point in enumerate(cross_points):
        ref = flat_m2 if idx % 2 == 0 else flat_m1
        if idx < points - 1:            
            result = result + ref[point:cross_points[idx + 1]]
        else:
            result = result + ref[point:]

    return np.reshape(result, (rows, cols))

def matrix_crossover(m1, m2, mode:CrossoverMode):
    if mode == CrossoverMode.ONE_POINT: return matrix_simple_crossover(m1, m2)
    if mode == CrossoverMode.TWO_POINT: return matrix_simple_crossover(m1, m2, points=2)
    if mode == CrossoverMode.RANDOM: return matrix_random_crossover(m1, m2)
    return m1

def matrix_mutation(m, mutation_prob):
    rows, cols = np.shape(m)
    result = m.copy()
    for x in range(rows):
        for y in range(cols):
            if random() < mutation_prob:
                result[x][y] = uniform(-1,1)
    return result
    
class NeuralNetwork():
    def __init__(self, layer_sizes: list, use_bias=True, use_batch_norm = True, activation_fn = ActivationFunction.ReLU, random_init=True) -> None:
        self.__layer_sizes = layer_sizes
        self.__use_bias = use_bias
        self.layers = [Layer(layer_sizes[idx], layer_sizes[idx+1], use_bias=use_bias, use_batch_norm=use_batch_norm, random_init=random_init)
                         for idx in range(len(layer_sizes) - 1)]
        if activation_fn == ActivationFunction.ReLU:
            self.__activation_fn = relu
        elif activation_fn == ActivationFunction.SOFTMAX:
            self.__activation_fn = softmax
        else:
            self.__activation_fn = lambda x: x

    def __backpropagate(self, input, errors, learning_rate):
        layer_inputs = [input]
        layer_errors = [errors]
        for layer in self.layers:
            layer_inputs.append(layer.process_input(layer_inputs[-1]))

        for idx in range(len(self.layers) - 1, 0, -1):
            error = np.matmul([layer_errors[0]], self.layers[idx].weights.transpose())
            layer_errors = [error] + layer_errors

        for idx in range(len(self.layers)):
            adjustment = np.matmul(np.matrix(layer_errors[idx]).transpose(), [layer_inputs[idx]]).transpose()
            self.layers[idx].weights += learning_rate * adjustment
            if self.__use_bias:
                self.layers[idx].bias = self.layers[idx].bias * layer_errors[idx]

    def process_input(self, input):
        next_input = input
        for layer in self.layers:
            next_input = layer.process_input(next_input)
            next_input = self.__activation_fn(next_input)
        return next_input
    
    def get_error(self, input, expected, error_fn = mean_sqr_errors):
        output = self.process_input(input)
        errors = np.subtract(expected, output)
        return error_fn(errors)

    def crossover(self, breeding_nn, mode=CrossoverMode.ONE_POINT):
        for idx in range(len(self.layers)):
            self.layers[idx].crossover(breeding_nn.layers[idx], mode=mode)

    def mutate(self, mutation_prob):
        for idx in range(len(self.layers)):
            self.layers[idx].mutate(mutation_prob)

    def learn(self, input, expected, learning_rate=0.1, error_fn = mean_sqr_errors):
        output = self.process_input(input)
        errors = np.subtract(expected, output)
        self.__backpropagate(input, errors, learning_rate)
        return error_fn(errors)       

    @property
    def layer_sizes(self):
        return self.__layer_sizes

    @property
    def use_bias(self):
        return self.__use_bias

class Layer():
    def __init__(self, input_size, output_size, use_bias=True, random_init=True, weights=None, use_batch_norm = True) -> None:
        self.__input_size = input_size
        self.__output_size = output_size
        self.__use_bias = use_bias
        if use_batch_norm:
            if use_bias:
                self.__process_output = lambda output: batchnorm((output + self.bias).flatten())
            else: 
                self.__process_output = lambda output: batchnorm(output)
        else:
            if use_bias:
                self.__process_output = lambda output: (output + self.bias).flatten()
            else: 
                self.__process_output = lambda output: output
        self.bias = None
        if random_init:
            self.weights = np.random.uniform(-1, 1, (input_size, output_size))
            if use_bias:
                self.bias = np.random.uniform(-1, 1, (1,output_size))
        else:
            self.weights = np.zeros((input_size, output_size))
            if use_bias:
                self.bias = np.zeros((1, output_size))

    def __str__(self) -> str:
        return f"Layer in:{self.__input_size} out:{self.__output_size}"

    def process_input(self, input):
        output = np.matmul(input, self.weights)
        return self.__process_output(output) 

    def crossover(self, breeding_layer, mode:CrossoverMode = CrossoverMode.ONE_POINT):
        weights_cross = matrix_crossover(self.weights, breeding_layer.weights, mode=mode)
        self.weights = weights_cross
        if self.__use_bias:
            bias_cross = matrix_crossover(self.bias, breeding_layer.bias, mode=mode)
            self.bias = bias_cross

    def mutate(self, mutation_prob):
        self.weights = matrix_mutation(self.weights, mutation_prob=mutation_prob)