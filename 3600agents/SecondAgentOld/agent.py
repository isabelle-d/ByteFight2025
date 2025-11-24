import math
from collections import deque
from collections.abc import Callable
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

    my_area = reachable_area(board)
    opp_area = reachable_area_enemy(board)

    score += 0.05 * (my_area - opp_area)

    #egg bump
    x,y = board.chicken_player.get_location()
    if ((x+y) % 2 == 0):
        score += 0.4



    # Offesne turd bonus
    t_left = board.chicken_player.get_turds_left()
    px, py = board.chicken_player.get_location()
    ex, ey = board.chicken_enemy.get_location()
    dist = abs(px - ex) + abs(py - ey)
    if board.turn_count < 12:
        if len(board.turds_player) > 0:
            if board.chicken_player.get_turds_left() >= 4:
                score -= 1.2

    if board.chicken_player.get_turds_left() > 0:
        if dist <= 2:
            score += 0.8
        else:
            score -= 0.4
    # standing next to your own turds
    for (tx, ty) in board.turds_player:
        d = abs(tx - x) + abs(ty - y)
        if d == 1:
            score -= 0.8

    #turd penalty, dont want to be next to turds b/c limit movement, small penalty depending
    for (tx,ty) in board.turds_enemy:
        d = abs(tx-x)+abs(ty-y)
        if d == 1:
            score -= 1.5
        elif d == 2:
            score -= 0.4
    # Avoid self-blocking with turds
    if board.chicken_player.get_turds_left() > 0:
        if len(board.get_valid_moves()) <= 2:   # very tight space
            score -= 2.0

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

#try and fix getting stuck
def reachable_area(board):
    start = board.chicken_player.get_location()
    visited = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            if not board.is_valid_cell((nx, ny)):
                continue
            if board.is_cell_blocked((nx, ny)):
                continue
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))

    return len(visited)

def reachable_area_enemy(board):
    start = board.chicken_enemy.get_location()
    visited = {start}
    q = deque([start])

    while q:
        x, y = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy

            # enemy movement uses enemy=True checks
            if not board.is_valid_cell((nx, ny)):
                continue

            # enemy cannot move into its own blocked zones
            # reuse board.is_valid_move by simulating a step
            if board.is_cell_blocked((nx, ny)):
                continue

            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))

    return len(visited)

def move_reachable_area_after(board, direction, move_type):
    child = board.forecast_move(direction, move_type)
    if child is None:
        return -1   # invalid move
    return reachable_area(child)
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
        moves_sorted = sorted(moves, key=lambda mv: 0 if mv[1].name == "EGG" else 1)

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

        (hw, fw) = sensor_data[0]
        (hb, fb) = sensor_data[1]

        self.belief.update(
            pos=location,
            heard_white=hw,
            felt_white=fw,
            heard_black=hb,
            felt_black=fb,
        )

        moves = board.get_valid_moves()
        best_value = -math.inf
        candidates = []

        for direction, move_type in moves:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue

            child.reverse_perspective()

            value = minimax(child, 3, -math.inf, math.inf, False, self.belief)
            print(f"Move {direction.name}, {move_type.name} → value {value}")

            candidates.append(((direction, move_type), value))
            best_value = max(best_value, value)

        if not candidates:
            return moves[0]

        best_move = max(candidates, key=lambda x: x[1])[0]

        def trap_prob_after(dir, mt):
            child = board.forecast_move(dir, mt)
            if child is None:
                return 1.0
            child.reverse_perspective()

            x, y = child.chicken_player.get_location()
            return self.belief.white_probs[x, y] + self.belief.black_probs[x, y]

        current_area = reachable_area(board)

        def move_area(dir, mt):
            child = board.forecast_move(dir, mt)
            if child is None:
                return -1
            child.reverse_perspective()
            return reachable_area(child)

        best_area = move_area(best_move[0], best_move[1])
        best_trap = trap_prob_after(best_move[0], best_move[1])

        if best_area >= max(1, 0.6 * current_area) and best_trap < 0.15:
            print(f"SAFE best move → {best_move}")
            return best_move

        safe = []
        for (d, m), val in candidates:
            a = move_area(d, m)
            p = trap_prob_after(d, m)
            if a >= max(1, 0.6 * current_area) and p < 0.15:
                safe.append(((d, m), val, a))

        if safe:
            # pick best value, then max area
            chosen = max(safe, key=lambda x: (x[1], x[2]))[0]
            print(f"Fallback SAFE move → {chosen}")
            return chosen


        emergency = []
        for (d, m), val in candidates:
            a = move_area(d, m)
            p = trap_prob_after(d, m)
            emergency.append(((d, m), a, -p, val))


        fallback = max(emergency, key=lambda x: (x[1], x[2], x[3]))[0]
        print(f"No safe options — using MAX AREA WITH TRAP AWARE fallback → {fallback}")
        return fallback

