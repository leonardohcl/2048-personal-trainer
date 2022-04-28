from enum import Enum
import random
import numpy as np


class Direction(Enum):
    UNK = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Game():
    def __init__(self, size):
        self.__size = size
        self.__score = 0
        self.__moves_made = 0
        self.__game_over = True
        self.__win = False
        self.__board = Board(size)
        self.__available_moves = []

    def __str__(self) -> str:
        return f"Score: {self.score}\n{self.__board.formatted}\nwin: {self.win} | game over: {self.game_over}"

    def __tuple2idx(self, row, col):
        return row * self.__size + col

    def __allow_all_moves(self):
        self.__available_moves = [
            Direction.UP, Direction.RIGHT, Direction.LEFT, Direction.DOWN]

    def __move_right(self):
        next = Board(self.__size)
        score_increment = 0
        for row in range(self.__size):
            for col in range(self.__size - 1, -1, -1):
                current_idx = self.__tuple2idx(row, col)
                current_value = self.__board.content[current_idx].value
                if (current_value == 0):
                    continue
                next_idx = current_idx
                for next_neighbor_col in range(col, self.__size):
                    neighbor_idx = self.__tuple2idx(row, next_neighbor_col)
                    if (next.can_move(current_value, neighbor_idx)):
                        next_idx = neighbor_idx
                    else:
                        break
                score_increment += next.move(current_value, next_idx)
        return next, score_increment

    def __move_left(self):
        next = Board(self.__size)
        score_increment = 0
        for row in range(self.__size):
            for col in range(self.__size):
                current_idx = self.__tuple2idx(row, col)
                current_value = self.__board.content[current_idx].value
                if (current_value == 0):
                    continue
                next_idx = current_idx
                for next_neighbor_col in range(col, -1, -1):
                    neighbor_idx = self.__tuple2idx(row, next_neighbor_col)

                    if (next.can_move(current_value, neighbor_idx)):
                        next_idx = neighbor_idx
                    else:
                        break
                score_increment += next.move(current_value, next_idx)
        return next, score_increment

    def __move_down(self):
        next = Board(self.__size)
        score_increment = 0
        for row in range(self.__size - 1, -1, -1):
            for col in range(self.__size):
                current_idx = self.__tuple2idx(row, col)
                current_value = self.__board.content[current_idx].value
                if (current_value == 0):
                    continue
                next_idx = current_idx
                for next_neighbor_row in range(self.__size):
                    neighbor_idx = self.__tuple2idx(next_neighbor_row, col)

                    if (next.can_move(current_value, neighbor_idx)):
                        next_idx = neighbor_idx
                    else:
                        break
                score_increment += next.move(current_value, next_idx)
        return next, score_increment

    def __move_up(self):
        next = Board(self.__size)
        score_increment = 0
        for row in range(self.__size):
            for col in range(self.__size):
                current_idx = self.__tuple2idx(row, col)
                current_value = self.__board.content[current_idx].value
                if (current_value == 0):
                    continue
                next_idx = current_idx
                for next_neighbor_row in range(row - 1, -1, -1):
                    neighbor_idx = self.__tuple2idx(next_neighbor_row, col)

                    if (next.can_move(current_value, neighbor_idx)):
                        next_idx = neighbor_idx
                    else:
                        break
                score_increment += next.move(current_value, next_idx)
        return next, score_increment

    def __update_game_state(self):
        if len(self.__available_moves) == 0:
            self.__game_over = True
        elif self.__board.max == 2048:
            self.__win = True

    @property
    def available_moves(self):
        return self.__available_moves

    @property
    def game_over(self):
        return self.__game_over

    @property
    def win(self):
        return self.__win

    @property
    def score(self):
        return self.__score

    @property
    def moves_made(self):
        return self.__moves_made

    @property
    def board(self):
        return [x.value for x in self.__board.content]

    def start(self):
        self.__game_over = False
        self.__win = False
        self.__score = 0
        self.__moves_made = 0
        self.__board.clean_board()
        self.__board.spawn(2)
        self.__allow_all_moves()

    def move(self, dir):
        next_board = self.__board
        score_increment = 0
        if dir == Direction.UP:
            next_board, score_increment = self.__move_up()
        elif dir == Direction.RIGHT:
            next_board, score_increment = self.__move_right()
        elif dir == Direction.DOWN:
            next_board, score_increment = self.__move_down()
        elif dir == Direction.LEFT:
            next_board, score_increment = self.__move_left()

        if(next_board.serial == self.__board.serial):
            self.__available_moves.remove(dir)
        else:
            self.__moves_made += 1
            self.__score += score_increment
            self.__board = next_board
            self.__board.spawn()
            self.__allow_all_moves()

        self.__update_game_state()


class Board():
    def __init__(self, size):
        self.__size = size
        self.__square_count = size * size
        self.__content = self.get_empty_board()

    def __str__(self) -> str:
        return str(self.formatted)

    def __empty_squares(self):
        map = []
        for sqr in self.__content:
            if sqr.is_empty:
                map.append(sqr)
        return map

    def can_move(self, value, target_idx):
        if(self.__content[target_idx].is_empty):
            return True
        if(value == self.__content[target_idx].value and self.__content[target_idx].is_merge == False):
            return True
        return False

    def move(self, value, target_idx):
        if(not self.can_move(value, target_idx)):
            raise "Invalid movement"
        if(self.__content[target_idx].value == value):
            self.__content[target_idx].value = value + value
            self.__content[target_idx].is_merge = True
            return self.__content[target_idx].value
        else:
            self.__content[target_idx].value = value
            return 0

    def get_empty_board(self):
        return [Square(0) for x in range(self.__square_count)]

    def clean_board(self):
        self.__content = self.get_empty_board()

    def spawn(self, blocks=1):
        for block in range(blocks):
            empty_squares = self.__empty_squares()
            if len(empty_squares) == 0:
                break
            sqr = random.choice(empty_squares)
            sqr.value = random.choices([2, 4], weights=[9, 1], k=1)[0]

    @property
    def formatted(self):
        matrix = np.array([sqr.value for sqr in self.__content])
        return matrix.reshape((self.__size, self.__size))

    @property
    def serial(self):
        return "_".join([str(x) for x in self.__content])

    @property
    def content(self):
        return self.__content

    @property
    def max(self):
        return max([sqr.value for sqr in self.__content])


class Square():
    def __init__(self, value=0):
        self.value = value
        self.is_merge = False

    def __str__(self) -> str:
        return str(self.value)

    @property
    def is_empty(self):
        return self.value == 0
