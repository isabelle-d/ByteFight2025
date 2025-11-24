import math
from collections import deque
from collections.abc import Callable
from typing import List, Set, Tuple
from .trapdoor_belief import TrapdoorBelief
import numpy as np
import random
from game import *

"""
min max algo with alpha pruning, bayes for updating belief for trapdoors
"""

def evaluate(board, belief, pos_history=None):
    """
    added penalty for: trapdoor (based on bayes net), turds,
    bonus for corners, and movement for eggs
    """
    try:
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
    except Exception:
        my_eggs = len(board.eggs_player)
        opp_eggs = len(board.eggs_enemy)

    score = 10 * (my_eggs - opp_eggs)  # egg advantage dominates

    # corner eggs bonus
    corners = {(0,0), (0, board.game_map.MAP_SIZE-1),
               (board.game_map.MAP_SIZE-1, 0),
               (board.game_map.MAP_SIZE-1, board.game_map.MAP_SIZE-1)}
    score += 3.0 * sum(1 for e in board.eggs_player if e in corners)

    # movement / mobility
    my_moves = len(board.get_valid_moves())
    opp_moves = len(get_enemy_moves(board))
    score += 0.5 * (my_moves - opp_moves)

    # reachable area bonus
    my_area = reachable_area(board)
    opp_area = reachable_area_enemy(board)
    score += 0.2 * (my_area - opp_area)

    # egg bump for white squares
    x, y = board.chicken_player.get_location()
    if ((x + y) % 2 == 0):
        score += 1.0

    # offense turd bonus if close enough
    px, py = board.chicken_player.get_location()
    ex, ey = board.chicken_enemy.get_location()
    dist = abs(px - ex) + abs(py - ey)

    if board.chicken_player.get_turds_left() > 0:
        if dist == 2:
            score += 2.0
        elif dist == 3:
            score += 0.5

    # penalties for proximity to turds
    for tx, ty in board.turds_player:
        d = abs(tx - x) + abs(ty - y)
        if d == 1:
            score -= 1.0
    for tx, ty in board.turds_enemy:
        d = abs(tx - x) + abs(ty - y)
        if d == 1:
            score -= 2.0
        elif d == 2:
            score -= 0.5

    # self-blocking penalty
    if board.chicken_player.get_turds_left() > 0 and my_moves <= 2:
        score -= 5.0

    # trapdoor penalty
    if belief is not None:
        try:
            p_fall = belief.white_probs[x, y] + belief.black_probs[x, y]
            if p_fall > 0.1:  # anything non-negligible
                score -= 5.0 * p_fall
        except Exception:
            pass

    # repeat-position penalties
    if pos_history is not None and len(pos_history) >= 2:
        # penalize revisiting same spot multiple times
        repeats = pos_history.count((x, y))
        if repeats > 1:
            score -= 4.0 * (repeats - 1)
        # penalize immediate back-and-forth
        if pos_history[-1] == pos_history[-3] if len(pos_history) >= 3 else False:
            score -= 3.0

    return float(score)



def expected_trapdoor_penalty(board, belief):
    x, y = board.chicken_player.get_location()
    #out of bounds bad :(
    if not (0 <= x < board.game_map.MAP_SIZE and 0 <= y < board.game_map.MAP_SIZE):
        return 0
    p_fall_white = belief.white_probs[x,y]
    p_fall_black = belief.black_probs[x,y]
    p_total = p_fall_white + p_fall_black
    if p_total > 0.3:
        return -2 * p_total
    return 0

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
        return -1
    return reachable_area(child)

def minimax(board, depth, alpha, beta, isMaximizing, belief, pos_history = None):
    if depth == 0 or board.is_game_over():
        return evaluate(board, belief, pos_history)

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
            eval_score = minimax(child, depth - 1, alpha, beta, False, belief, pos_history)
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

            eval_score = minimax(child, depth - 1, alpha, beta, True, belief, pos_history)

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
        self.prev_pos = None
        self.pos_history = deque(maxlen=7)

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        current_pos = board.chicken_player.get_location()
        self.pos_history.append(current_pos)
        start_pos = board.chicken_player.get_spawn()
        if self.prev_pos is not None:
            if current_pos == start_pos and self.prev_pos != start_pos:
                print(f"*** TRAPDOOR FALL DETECTED at {self.prev_pos}! ***")

                try:
                    self.belief.set_trapdoor_at(self.prev_pos)
                except Exception as e:
                    print("Warning: set_trapdoor_at failed:", e)
        self.prev_pos = current_pos

        print(f"I'm at {current_pos}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")

        (hw, fw) = sensor_data[0]
        (hb, fb) = sensor_data[1]
        self.belief.update(
            pos=current_pos,
            heard_white=hw,
            felt_white=fw,
            heard_black=hb,
            felt_black=fb,
        )

        moves = board.get_valid_moves()
        best_value = -math.inf
        candidates = []  #((dir, move_type), score)

        # Evaluate moves normally using minimax
        for direction, move_type in moves:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue

            child.reverse_perspective()
            value = minimax(child, 3, -math.inf, math.inf, False, self.belief, self.pos_history)

            print(f"Move {direction.name}, {move_type.name} → value {value}")

            candidates.append(((direction, move_type), value))

            if value > best_value:
                best_value = value
        if not candidates:
            return moves[0]

        best_value = max(candidates, key=lambda x: x[1])[1]
        epsilon = 0.01
        top_candidates = [c for c in candidates if c[1] >= best_value - epsilon]
        best_by_value = random.choice(top_candidates)[0]
        #SAFETY FILTER :)
        current_area = reachable_area(board)

        def move_area(dir, mt):
            child = board.forecast_move(dir, mt)
            if child is None:
                return -1
            return reachable_area(child)
        area_after_best = move_area(best_by_value[0], best_by_value[1])

        if (area_after_best >= max(1, 0.35 * current_area)):
            print(f"SAFE best move → {best_by_value}")
            return best_by_value

        safe_moves = []
        for (d, m), v in candidates:
            a = move_area(d, m)
            if a >= max(1, 0.6 * current_area):
                safe_moves.append(((d, m), v, a))

        if safe_moves:
            best_safe_value = max(safe_moves, key=lambda x: x[1])[1]
            epsilon = 0.01
            top_safe = [s for s in safe_moves if s[1] >= best_safe_value - epsilon]
            chosen = random.choice(top_safe)[0]
            print(f"Fallback SAFE move → {chosen}")
            return chosen
        #tiebreak
        fallback = max( candidates, key=lambda x: (move_area(x[0][0], x[0][1]), x[1]))[0]
        print(f"No safe options — using MAX AREA fallback → {fallback}")
        return fallback

