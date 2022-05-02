from random import random
import numpy as np
from Game import Game
from NeuralNetwork import CrossoverMode
from Trainer import Robot, Trainer, WeightedRoulette
from matplotlib import pyplot as plt


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

def robot_fitness(robot: Robot):
    scores = []
    highest_blocks =[]
    for _ in range(3):
        robot.play()
        scores.append(sum(robot.game.board))
        highest_blocks.append(10* max(robot.game.board))
    return sum(scores) + sum(highest_blocks)

def breeding_selection(fitnesses):
    cut_point = np.median(fitnesses)
    adjusted_weights = [0 if x < cut_point else x for x in fitnesses]
    return WeightedRoulette(adjusted_weights).draw()[0]

def stop_condition(best_robot: Robot):
    last_best = []
    for _ in range(5):
        best_robot.play()
        last_best.append(max(best_robot.game.board))
    return last_best.count(2048) > 1

trainer = Trainer(4, [18,81,36,8], use_bias=True, fitness_fn=robot_fitness, breeding_selection_fn=breeding_selection,
            stop_condition=stop_condition, nn_input=robot_input, crossover_mode=CrossoverMode.TWO_POINT)
robot, score, goats, bests, means = trainer.train(
    pop_size=10, gen_count=100, cross_prob=0.75, mutation_prob=0.25, use_elitism=True, parallel_workers=12)

for _ in range(5):
    robot.play()
    print(robot.game)

plt.plot(goats, label="GOAT")
plt.plot(bests, label="Gen Best")
plt.plot(means, label="Gen Avg")
plt.legend()
plt.show()
