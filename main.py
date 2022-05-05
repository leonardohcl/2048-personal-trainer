from cmath import log
from dis import dis
import math
from random import random
import numpy as np
from Game import Game
from NeuralNetwork import ActivationFunction, CrossoverMode
from Trainer import Robot, Trainer, WeightedRoulette
from matplotlib import pyplot as plt

def euclidian_distance(a, b):
    sqr_diff_sum = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sqr_diff_sum += diff * diff    
    return math.sqrt(sqr_diff_sum)
             
def matrix_center_of_mass(m):
    rows, cols = np.shape(m)
    mass = sum(m.flatten())
    x_coord = 0
    y_coord = 0
    for x in range(rows):
        for y in range(cols):
            x_coord += m[x][y] * x
            y_coord += m[x][y] * y
    return [x_coord / mass, y_coord/mass]

def robot_input(game: Game):
    return game.board + matrix_center_of_mass(game.matrix_board)

def board_entropy(board):
    entropy = 0.0
    for value in board:        
        entropy += (2048-value) if value > 0 else 0
    return entropy

BOARD_SIZE = 4    
CENTER = (BOARD_SIZE - 1) / 2
MAX_DIST = euclidian_distance([CENTER, CENTER], [0,0])
def robot_fitness(robot: Robot):
    robot.play()
    mass_center = matrix_center_of_mass(robot.game.matrix_board)
    dist = euclidian_distance([CENTER, CENTER], mass_center)
    return board_entropy(robot.game.board) * (1 + MAX_DIST - dist)          

def breeding_selection(fitnesses):
    cut_point = np.mean(fitnesses)
    adjusted_weights = [0 if x < cut_point else x for x in fitnesses]
    return WeightedRoulette(adjusted_weights).draw()[0]

def stop_condition(best_robot: Robot):
    last_best = []
    for _ in range(5):
        best_robot.play()
        last_best.append(max(best_robot.game.board))
    return last_best.count(2048) >= 1

trainer = Trainer(4, [18], use_bias=True, fitness_fn=robot_fitness, breeding_selection_fn=breeding_selection,
            stop_condition=stop_condition, nn_input=robot_input, crossover_mode=CrossoverMode.TWO_POINT,
            maximize=False, random_init=True, activation_fn=ActivationFunction.ReLU)
robot, score, goats, bests, means = trainer.train(
    pop_size=25, gen_count=3000, cross_prob=1, mutation_prob=0.7, use_elitism=True, parallel_workers=10)

for _ in range(10):
    robot.play()
    print(robot.game)

plt.plot(goats, label="GOAT")
plt.plot(bests, label="Gen Best")
plt.plot(means, label="Gen Avg")
plt.legend()
plt.show()
