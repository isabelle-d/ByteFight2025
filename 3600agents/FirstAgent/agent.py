import math
from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple
from .trapdoor_belief import TrapdoorBelief

import numpy as np
from engine.game import *

"""
min max algo with alpha pruning, bayes for updating belief for trapdoors
"""

def evaluate(board, belief):
    """
    added penalty for: trapdoor (based on bayes net), turds,
    bonus for corners, and movement for eggs
    """
    try:
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
    except:
        my_eggs = len(board.eggs_player)
        opp_eggs = len(board.eggs_enemy)

    score = my_eggs - opp_eggs

    #corner eggs good
    corners = {(0,0), (0, board.game_map.MAP_SIZE-1),
               (board.game_map.MAP_SIZE-1, 0),
               (board.game_map.MAP_SIZE-1, board.game_map.MAP_SIZE-1)}
    score += 0.5 * sum(1 for e in board.eggs_player if e in corners)

    my_moves = len(board.get_valid_moves())
    opp_moves = len(get_enemy_moves(board))
    score += 0.1 * (my_moves - opp_moves)

    #egg bump
    x,y = board.chicken_player.get_location()
    if ((x+y) % 2 == 0):
        score += 0.4

    #turd penalty, dont want to be next to turds b/c limit movement, small penalty depending
    for (tx,ty) in board.turds_enemy:
        d = abs(tx-x)+abs(ty-y)
        if d == 1:
            score -= 1.5
        elif d == 2:
            score -= 0.4

    # Trapdoor penalty
    if belief is not None:
        score += expected_trapdoor_penalty(board, belief)

    return score

def expected_trapdoor_penalty(board, belief):
    x, y = board.chicken_player.get_location()
    #out of bounds bad :(
    if not (0 <= x < board.game_map.MAP_SIZE and 0 <= y < board.game_map.MAP_SIZE):
        return 0
    p_fall_white = belief.white_probs[x,y]
    p_fall_black = belief.black_probs[x,y]
    p_total = p_fall_white + p_fall_black
    return -4 * p_total

def egg_opportunity(board):
    (x,y) = board.chicken_player.get_location()
    if ((x+y)%2 == 0):    # white squares = white lay
        return 0.5        # small bump ^_^
    return 0

def get_enemy_moves(board):
    moves = board.get_valid_moves()
    for direction, move_type in moves:
        child = board.forecast_move(direction, move_type)
        if child is not None:
            child.reverse_perspective()
            return child.get_valid_moves()
    return []

def minimax(board, depth, alpha, beta, isMaximizing, belief):
    if depth == 0 or board.is_game_over():
        return evaluate(board, belief)

    moves = board.get_valid_moves()

    if len(moves) == 0:
        return -999999 if isMaximizing else 999999

    if isMaximizing:
        maxEval = -math.inf
        moves_sorted = sorted(moves, key=lambda mv: 0 if mv[1].name == "EGG" else 1)

        for direction, move_type in moves_sorted:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue
            child.reverse_perspective()
            eval_score = minimax(child, depth - 1, alpha, beta, False, belief)
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

            eval_score = minimax(child, depth - 1, alpha, beta, True, belief)

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
        self.belief = TrapdoorBelief(board.game_map)

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

        #trapdoor belief
        (hw, fw) = sensor_data[0]
        (hb, fb) = sensor_data[1]

        self.belief.update(
            pos = location,
            heard_white = hw,
            felt_white  = fw,
            heard_black = hb,
            felt_black  = fb
        )
        moves = board.get_valid_moves()
        best_move = None
        best_value = -math.inf

        for direction, move_type in moves:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue

            child.reverse_perspective()
            value = minimax(child, 3, -math.inf, math.inf, False, self.belief)

            print(f"Move {direction.name}, {move_type.name} → value {value}")

            if value > best_value:
                best_value = value
                best_move = (direction, move_type)

            elif value == best_value and best_move is not None:
                #egg is best
                order = {"EGG": 3, "PLAIN": 2, "TURD": 1}
                if order[move_type.name] > order[best_move[1].name]:
                    print(f"   Tie-break: preferring {move_type.name} over {best_move[1].name}")
                    best_move = (direction, move_type)

        print(f"I have {time_left()} seconds left. Playing {best_move}.")
        return best_move

