from random import choices, random
import time
import numpy as np
from tqdm import tqdm
from Game import Direction, Game
from NeuralNetwork import ActivationFunction, CrossoverMode, NeuralNetwork
from joblib import Parallel, delayed, cpu_count

def __default_input(game:Game):
    return game.board

def __default_fitness_fn(robot):
    return robot.play()

def __select_higher_fitness(fitnesses):
    return WeightedRoulette(fitnesses).draw()[0]

def __select_lowest_fitness(fitnesses):
    return WeightedRoulette(__reverse_fitnesses(fitnesses)).draw()[0]

def __reverse_fitnesses(fitnesses):
    top = max(fitnesses)
    return [top-x for x in fitnesses]

def __get_default_breeding_selection_fn(maximize):
    if maximize: 
        return __select_higher_fitness
    else:
        return __select_lowest_fitness
 
def __default_stop_condition(best_robot):
    return False
   
class WeightedRoulette():
    def __init__(self, weights) -> None:
        self.weights = weights

    def draw(self, amount=1):
        draw = choices([i for i in range(len(self.weights))],
                       weights=self.weights, k=amount)
        return draw

class Robot():
    def __init__(self, board_size, brain_structure, use_bias = True, get_input = lambda game: game.board, random_init=True, activation_fn=ActivationFunction.ReLU) -> None:
        self.__board_size = board_size
        self.__brain_structure = brain_structure + [4]
        self.__use_bias = use_bias
        self.__get_input = get_input
        self.game = Game(board_size)
        self.brain = NeuralNetwork(self.__brain_structure, self.__use_bias, random_init=random_init, activation_fn=activation_fn)

    def make_next_move(self):
        next_move = Direction.UNK
        probs = self.brain.process_input(self.__get_input(self.game)).tolist()
        while next_move not in self.game.available_moves:
            highest_prob = max(probs)
            highest_prob_idx = probs.index(highest_prob)
            next_move = Direction(highest_prob_idx + 1)
            probs[highest_prob_idx] = float("-Inf")
        return next_move

    def play(self):
        self.game.start()
        while self.game.win == False and self.game.game_over == False:
            next_move = self.make_next_move()
            self.game.move(next_move)
        return self.game.score

    def crossover(self, breeding_robot, mode=CrossoverMode.ONE_POINT):
        self.brain.crossover(breeding_robot.brain, mode=mode)

    def mutate(self, mutation_prob):
        self.brain.mutate(mutation_prob)

    @property
    def brain_structure(self):
        return self.__brain_structure

class Trainer():
    def __init__(self, board_size, brain_structure = [16], use_bias=True, fitness_fn=None, stop_condition = None, crossover_mode=CrossoverMode.ONE_POINT, nn_input=None, breeding_selection_fn = None, maximize=True, random_init=True, activation_fn=ActivationFunction.ReLU) -> None:
        self.__board_size = board_size
        self.__brain_structure = brain_structure
        self.__use_bias = use_bias
        self.__crossover_mode = crossover_mode
        self.__maximize = maximize
        self.__random_init=random_init
        self.__activation_fn = activation_fn
        self.__stop_condition = stop_condition if stop_condition != None else __default_stop_condition
        self.__breeding_selection_fn = breeding_selection_fn if breeding_selection_fn != None else __get_default_breeding_selection_fn(self.__maximize)
        self.__nn_input = nn_input = nn_input if nn_input != None else __default_input
        self.__fitness_fn = fitness_fn if fitness_fn != None else __default_fitness_fn
        self.__fitness_a_is_better_than_b = lambda fit_a, fit_b: fit_a > fit_b if self.__maximize else fit_a < fit_b

    def __get_robot_fitness(self, robot: Robot):
        return self.__fitness_fn(robot)

    def __should_stop(self, best_robot):
        return self.__stop_condition(best_robot)

    def __breed_robot(self, robot: Robot, candidates, best_robot, use_elitism, mutation_prob, cross_prob, fitnesses):
        if use_elitism and robot == best_robot:
            return

        breeding_index = self.__breeding_selection_fn(fitnesses)
        breeding_robot = candidates[breeding_index]
        if breeding_robot != robot and random() <= cross_prob:
            robot.crossover(breeding_robot,
                            mode=self.__crossover_mode)
        robot.mutate(mutation_prob)

    def __eval_generation(self, robots, parallel_workers):
        def fitness_fn(robot): return self.__get_robot_fitness(robot)
        fitness = Parallel(n_jobs=parallel_workers)(delayed(fitness_fn)(
            robot)for robot in tqdm(robots, leave=False, desc="Evaluation"))
        best_fitness = float("-inf") if self.__maximize else float("Inf")
        best_robot = robots[0]
        for idx in range(len(robots)):
            if self.__fitness_a_is_better_than_b(fitness[idx], best_fitness):
                best_fitness = fitness[idx]
                best_robot = robots[idx]
        return best_robot, best_fitness, fitness

    def __breed_generation(self, robots, best_robot, fitnesses, mutation_prob, use_elitism, parallel_workers, cross_prob):
        def breed(robot): return self.__breed_robot(robot, robots, best_robot,
                                                     use_elitism, mutation_prob, cross_prob, fitnesses)
        Parallel(n_jobs=parallel_workers)(delayed(breed)(robot)
                                          for robot in tqdm(robots, leave=False, desc="  Breeding"))

    def train(self, pop_size, gen_count, mutation_prob=0, cross_prob=0.8, use_elitism=True, parallel_workers=1):
        robots = [Robot(self.__board_size, self.__brain_structure,
                        self.__use_bias, get_input=self.__nn_input, random_init=self.__random_init, activation_fn=self.__activation_fn) for _ in range(pop_size)]
        parallel_workers_used = parallel_workers if parallel_workers <= cpu_count() else cpu_count()
        goat_fitness = float("-inf") if self.__maximize else float("Inf")
        goat_robot = robots[0]
        goat_track = []
        gen_best_track = []
        mean_track = []

        for it in tqdm(range(gen_count), leave=False, desc="Generation"):

            best_robot, best_fitness, fitnesses = self.__eval_generation(
                robots=robots, parallel_workers=parallel_workers_used)

            if self.__fitness_a_is_better_than_b(best_fitness, goat_fitness):
                goat_fitness = best_fitness
                goat_robot = best_robot

            gen_best_track.append(best_fitness)
            mean_track.append(np.mean(fitnesses))
            goat_track.append(goat_fitness)

            if self.__should_stop(goat_robot):
                break

            self.__breed_generation(robots, goat_robot, fitnesses=fitnesses, mutation_prob=mutation_prob,
                                    use_elitism=use_elitism, parallel_workers=parallel_workers, cross_prob=cross_prob)

        return goat_robot, goat_fitness, goat_track, gen_best_track, mean_track
