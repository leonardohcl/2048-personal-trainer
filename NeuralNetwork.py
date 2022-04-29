from enum import Enum
from math import floor
from operator import mod
from random import random, sample
import numpy as np
from scipy.special import softmax

class CrossoverMode(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    RANDOM = 3

def batchnorm(x, gamma=1, beta=0, eps=1e-5):
    sample_mean = x.mean(axis=0)
    sample_var = x.var(axis=0)

    std = np.sqrt(sample_var + eps)
    x_centered = x - sample_mean
    x_norm = x_centered / std
    out = gamma * x_norm + beta

    cache = (x_norm, x_centered, std, gamma)

    return out, cache

def matrix_random_crossover(m1, m2, mutation_prob=0):
    rows, cols = np.shape(m1)
    param_count = rows * cols
    cross_point = random()
    m1_params = floor(param_count * cross_point)
    m2_params = param_count - m1_params

    result = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            if random() < mutation_prob:
                result[row][col] = random()
            elif m1_params > 0 and random() < cross_point:
                result[row][col] = m1[row][col]
                m1_params -= 1
            elif m2_params > 0:
                result[row][col] = m2[row][col]
                m2_params -= 1
            else:
                result[row][col] = m1[row][col]
                m1_params -= 1
    return result

def matrix_simple_crossover(m1,m2, points = 1, mutation_prob = 0):
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
    
    for idx in range(len(result)):
        if random() < mutation_prob:
            result[idx] = random

    return np.reshape(result, (rows, cols))

def matrix_crossover(m1, m2, mode:CrossoverMode, mutation_prob = 0):
    if mode == CrossoverMode.ONE_POINT: return matrix_simple_crossover(m1, m2, mutation_prob=mutation_prob)
    if mode == CrossoverMode.TWO_POINT: return matrix_simple_crossover(m1, m2, points=2, mutation_prob=mutation_prob)
    if mode == CrossoverMode.RANDOM: return matrix_random_crossover(m1, m2, mutation_prob)
    return m1

class NeuralNetwork():
    def __init__(self, layer_sizes: list, use_bias=True) -> None:
        self.__layer_sizes = layer_sizes
        self.__use_bias = use_bias
        self.layers = [Layer(layer_sizes[idx], layer_sizes[idx+1], use_bias)
                         for idx in range(len(layer_sizes) - 1)]

    def process_input(self, input, use_softmax = False):
        next_input = input
        for layer in self.layers:
            next_input = layer.process_input(next_input)
        return softmax(next_input) if use_softmax else next_input
        
    def crossover(self, breeding_nn, mutation_prob = 0, mode=CrossoverMode.ONE_POINT):
        for idx in range(len(self.layers)):
            self.layers[idx].crossover(breeding_nn.layers[idx], mutation_prob=mutation_prob, mode=mode)


    @property
    def layer_sizes(self):
        return self.__layer_sizes

    @property
    def use_bias(self):
        return self.__use_bias

class Layer():
    def __init__(self, input_size, output_size, use_bias=True, random_init=True, weights=None) -> None:
        self.__input_size = input_size
        self.__output_size = output_size
        self.__use_bias = use_bias
        self.bias = None
        if random_init:
            self.weights = np.random.rand(input_size, output_size)
            if use_bias:
                self.bias = np.random.rand(1, output_size)
        else:
            self.weights = np.zeros((input_size, output_size))
            if use_bias:
                self.bias = np.zeros(1, output_size)

    def __str__(self) -> str:
        return f"Layer in:{self.__input_size} out:{self.__output_size}"

    def process_input(self, input):
        mult = batchnorm(np.matmul(input, self.weights))[0]
        return (mult + self.bias).flatten() if self.__use_bias else mult

    def crossover(self, breeding_layer, mutation_prob=0, mode:CrossoverMode = CrossoverMode.ONE_POINT):
        weights_cross = matrix_crossover(self.weights, breeding_layer.weights, mode=mode, mutation_prob=mutation_prob)
        self.weights = weights_cross
        if self.__use_bias:
            bias_cross = matrix_crossover(self.bias, breeding_layer.bias, mode=mode, mutation_prob=mutation_prob)
            self.bias = bias_cross