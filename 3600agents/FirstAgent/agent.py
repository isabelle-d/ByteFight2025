import math
from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple

import numpy as np
from engine.game import *

"""
min max algo with alpha pruning
"""
def evaluate(board):
    """
    Basic evaluation: difference in eggs, will improve next in next iteration
    """
    try:
        my_eggs_count = board.chicken_player.get_eggs_laid()
        opp_eggs_count = board.chicken_enemy.get_eggs_laid()
    except Exception:
        my_eggs_count = len(board.eggs_player)
        opp_eggs_count = len(board.eggs_enemy)

        # corner bonus: we can add +1 for each egg in a corner (encourage corner eggs)
    corner_bonus = 0
    corners = {(0,0), (0, board.game_map.MAP_SIZE-1),
               (board.game_map.MAP_SIZE-1, 0),
               (board.game_map.MAP_SIZE-1, board.game_map.MAP_SIZE-1)}
    corner_bonus += sum(1 for e in board.eggs_player if e in corners)

    return (my_eggs_count - opp_eggs_count) + 0.5 * corner_bonus

def minimax(board, depth, alpha, beta, isMaximizing):

    if depth == 0 or board.is_game_over():
        return evaluate(board)

    moves = board.get_valid_moves()
    #bad bad bad
    if len(moves) == 0:
        return -999999 if isMaximizing else 999999

    if isMaximizing:
        maxEval = -math.inf
        moves_sorted = sorted(moves, key=lambda mv: 0 if mv[1].name == "EGG" else 1)

        for direction, move_type in moves_sorted:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue
            # switch perspective for opponent evaluation
            child.reverse_perspective()

            eval_score = minimax(child, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval_score)
            alpha = max(alpha, eval_score)

            if beta <= alpha:
                break

        return maxEval

    else:
        minEval = math.inf
        moves_sorted = sorted(moves, key=lambda mv: 0 if mv[1].name == "TURD" else 1)
        for direction, move_type in moves_sorted:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue
            child.reverse_perspective()
            eval_score = minimax(child, depth - 1, alpha, beta, True)
            minEval = min(minEval, eval_score)
            beta = min(beta, eval_score)

            if beta <= alpha:
                break

        return minEval

class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        pass

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")
        #sleep(1.5)

        moves = board.get_valid_moves()
        best_move = None
        best_value = -math.inf

        for direction, move_type in moves:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue

            child.reverse_perspective()

            value = minimax(child, 3, -math.inf, math.inf, False)

            print(f"Move {direction.name}, {move_type.name} → value {value}")

            if value > best_value:
                best_value = value
                best_move = (direction, move_type)

            elif value == best_value and best_move is not None:
                order = {
                    "EGG": 3,
                    "PLAIN": 2,
                    "TURD": 1
                }
                if order[move_type.name] > order[best_move[1].name]:
                    print(f"   Tie-break: preferring {move_type.name} over {best_move[1].name}")
                    best_move = (direction, move_type)

        print(f"I have {time_left()} seconds left. Playing {best_move}.")
        return best_move
