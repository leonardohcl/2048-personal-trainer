from random import random
import numpy as np
from tqdm import tqdm
from Game import Direction, Game
from NeuralNetwork import NeuralNetwork


class Trainer():
    def __init__(self, board_size, brain_structure, use_bias=True) -> None:
        self.__board_size = board_size
        self.__brain_structure = brain_structure
        self.__use_bias = use_bias

    def train(self, pop_size, gen_count, mutation_prob, use_elitism=True):
        robots = [Robot(self.__board_size, self.__brain_structure,
                        self.__use_bias) for x in range(pop_size)]
        goat_score = 0
        goat_robot = robots[0]
        goat_track = []
        gen_best_track = []
        mean_track = []
        for gen in tqdm(range(gen_count), leave=False):
            scores = []
            best_score = 0
            best_robot = robots[0]
            for robot in tqdm(robots, leave=False):
                score = robot.play()
                if score >= best_score:
                    best_score = score
                    best_robot = robot
                    if score >= goat_score:
                        goat_score = score
                        goat_robot = robot

                scores.append(score)
            gen_best_track.append(best_score)
            mean_track.append(np.mean(score))
            goat_track.append(goat_score)

            roulette = WeightedRoulette(scores)

            for robot in robots:
                if use_elitism and robot == best_robot: continue
                breeding_robot = robots[roulette.draw()]
                if breeding_robot == robot: continue
                cross_point = random()
                robot.crossover(breeding_robot, cross_point, mutation_prob)

        return goat_robot, goat_score, goat_track, gen_best_track, mean_track



class Robot():
    def __init__(self, board_size, brain_structure, use_bias=True) -> None:
        self.__board_size = board_size
        self.__brain_structure = brain_structure
        self.__use_bias = use_bias
        self.game = Game(board_size)
        self.brain = NeuralNetwork(brain_structure, use_bias)

    def __get_next_move(self):
        next_move = Direction.UNK
        probs = self.brain.process_input(self.game.board).tolist()
        while next_move not in self.game.available_moves:
            highest_prob = max(probs)
            highest_prob_idx = probs.index(highest_prob)
            next_move = Direction(highest_prob_idx + 1)
            probs[highest_prob_idx] = float("-Inf")
        return next_move

    def play(self):
        self.game.start()
        while self.game.win == False and self.game.game_over == False:
            next_move = self.__get_next_move()
            self.game.move(next_move)
        return self.game.score

    def crossover(self, breeding_robot, cross_point, mutation_prob = 0):
        self.brain.crossover(breeding_robot.brain, cross_point, mutation_prob)

    @property
    def brain_structure(self):
        return self.__brain_structure


class WeightedRoulette():
    def __init__(self, weights) -> None:
        self.weights = weights

    def draw(self):
        accumulator = 0
        idx = 0
        draw = random()
        while accumulator < draw and idx < len(self.weights):
            accumulator += self.normalized_weights[idx]
            idx += 1

        return idx - 1

    @property
    def weights_sum(self):
        return sum(self.weights)

    @property
    def normalized_weights(self):
        return [x/self.weights_sum for x in self.weights]
