import random
from Game import Game, Direction as dir

game = Game(4)

game.start()
while game.win == False and game.game_over == False:
    next_move = random.choice(game.available_moves)
    game.move(next_move)
    print(f"moves made: {game.moves_made} | last move:{next_move}\n{game}\nAvailable moves: {game.available_moves}\n")
