from cProfile import label
from cv2 import mean
import matplotlib
import numpy as np
from Game import Game, Direction as dir
from NeuralNetwork import Layer, NeuralNetwork
from Trainer import Robot, Trainer
from matplotlib import pyplot as plt

x = Trainer(5, [25, 64, 32,4])
robot, score, goats, bests, means = x.train(10, 100, 0.05)

robot.play()
print(robot.game)

plt.plot(goats, label="GOAT")
plt.plot(bests, label="Gen Best")
plt.plot(means, label="Gen Avg")
plt.legend()
plt.show()