from random import random, choices
from xmlrpc.client import Boolean
import numpy as np
from tqdm import tqdm
from Game import Direction, Game
from NeuralNetwork import CrossoverMode, NeuralNetwork
from joblib import Parallel, delayed, cpu_count

class WeightedRoulette():
    def __init__(self, weights) -> None:
        self.weights = weights

    def draw(self):
        draw = choices([i for i in range(len(self.weights))], weights=self.weights, k=1)
        return draw[0]

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

    def crossover(self, breeding_robot, mutation_prob = 0, mode=CrossoverMode.ONE_POINT):
        self.brain.crossover(breeding_robot.brain, mutation_prob=mutation_prob, mode=mode)

    @property
    def brain_structure(self):
        return self.__brain_structure

class Trainer():
    def __init__(self, board_size, brain_structure, use_bias=True, fitness_fn = None, stop_condition = None, crossover_mode = CrossoverMode.ONE_POINT) -> None:
        self.__board_size = board_size
        self.__brain_structure = brain_structure
        self.__use_bias = use_bias
        self.__fitness_fn = fitness_fn
        self.__crossover_mode = crossover_mode
        self.__stop_condition = stop_condition

    def __get_robot_fitness(self, robot:Robot):
        return robot.play() if self.__fitness_fn == None else self.__fitness_fn(robot)

    def __should_stop(self, best_robot):
        return False if self.__stop_condition == None else self.__stop_condition(best_robot) 

    def __breed_robot(self, robot: Robot, candidates, roulette: WeightedRoulette, mutation_prob, use_elitism, best_robot):
        if use_elitism and robot == best_robot: return
        breeding_robot = candidates[roulette.draw()]
        if breeding_robot == robot: return
        robot.crossover(breeding_robot, mutation_prob=mutation_prob, mode=self.__crossover_mode)

    def __eval_generation(self, robots, parallel_workers = 1):
        fitness_fn = lambda robot: self.__get_robot_fitness(robot)
        fitness = Parallel(n_jobs=parallel_workers if parallel_workers < cpu_count() else cpu_count())(delayed(fitness_fn)(robot)for robot in tqdm(robots, leave=False, desc="Evaluation"))
        best_fitness = 0.0
        best_robot = robots[0]
        for idx in range(len(robots)):
            if(fitness[idx] > best_fitness):
                best_fitness = fitness[idx]
                best_robot = robots[idx]        
        return best_robot, best_fitness, fitness

    def __breed_generation(self, robots, best_robot, fitnesses, mutation_prob, use_elitism, parallel_workers):
        roulette = WeightedRoulette(fitnesses)
        breed = lambda robot: self.__breed_robot(robot, robots, roulette, mutation_prob, use_elitism, best_robot)
        Parallel(n_jobs=parallel_workers)(delayed(breed)(robot) for robot in tqdm(robots, leave=False, desc="  Breeding"))            
        
    def train(self, pop_size, gen_count, mutation_prob, use_elitism=True, parallel_workers = 1):
        robots = [Robot(self.__board_size, self.__brain_structure,
                        self.__use_bias) for _ in range(pop_size)]
        goat_fitness = 0
        goat_robot = robots[0]
        goat_track = []
        gen_best_track = []
        mean_track = []
            
        for _ in tqdm(range(gen_count), leave=False, desc="Generation"):
            
            best_robot, best_fitness, fitnesses = self.__eval_generation(robots, parallel_workers)

            if(best_fitness > goat_fitness):
                goat_fitness = best_fitness
                goat_robot = best_robot

            gen_best_track.append(best_fitness)
            mean_track.append(np.mean(fitnesses))
            goat_track.append(goat_fitness)

            if self.__should_stop(goat_robot):
                break

            self.__breed_generation(robots, goat_robot, fitnesses, mutation_prob, use_elitism, parallel_workers)



        return goat_robot, goat_fitness, goat_track, gen_best_track, mean_track