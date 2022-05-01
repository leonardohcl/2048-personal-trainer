import numpy as np
from Game import Game
from NeuralNetwork import CrossoverMode
from Trainer import Robot, Trainer
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
    return game.board + matrix_center_of_mass(game.matrix_board) + [game.moves_made]

def robot_fitness(robot: Robot):
    scores = [robot.play() for _ in range(5)]
    return np.mean(scores)

def stop_condition(best_robot: Robot):
    last_best = []
    for _ in range(5):
        best_robot.play()
        last_best.append(max(best_robot.game.board))
    return last_best.count(2048) > 1

trainer = Trainer(4, [19], use_bias=True, fitness_fn=robot_fitness,
            stop_condition=stop_condition, nn_input=robot_input, crossover_mode=CrossoverMode.TWO_POINT)
robot, score, goats, bests, means = trainer.train(
    pop_size=100, gen_count=10, cross_prob=0.8, mutation_prob=0.05, use_elitism=True, parallel_workers=8)

for _ in range(5):
    robot.play()
    print(robot.game)

plt.plot(goats, label="GOAT")
plt.plot(bests, label="Gen Best")
plt.plot(means, label="Gen Avg")
plt.legend()
plt.show()
