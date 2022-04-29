from statistics import mean
from NeuralNetwork import CrossoverMode
from Trainer import Robot, Trainer
from matplotlib import pyplot as plt

def robot_fitness(robot:Robot):
    scores = []
    for _ in range(5):
        robot.play()
        scores.append(robot.game.score / robot.game.moves_made)
    return mean(scores)

def stop_condition(best_robot:Robot):
    last_best = []
    for _ in range(5):
        best_robot.play()
        last_best.append(max(best_robot.game.board))
    return last_best.count(2048) > 1

x = Trainer(4, [16, 4], fitness_fn=robot_fitness, stop_condition=stop_condition, crossover_mode=CrossoverMode.TWO_POINT)
robot, score, goats, bests, means = x.train(pop_size=100, gen_count=10000, mutation_prob=0.15, parallel_workers = 10)

robot.play()
print(robot.game)

plt.plot(goats, label="GOAT")
plt.plot(bests, label="Gen Best")
plt.plot(means, label="Gen Avg")
plt.legend()
plt.show()