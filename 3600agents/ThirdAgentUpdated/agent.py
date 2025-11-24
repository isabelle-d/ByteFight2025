import math
from collections import deque
from collections.abc import Callable
from typing import List, Set, Tuple
from .trapdoor_belief import TrapdoorBelief
import numpy as np
from game import *

def evaluate(board, belief, history=None):
    try:
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
    except:
        my_eggs = len(board.eggs_player)
        opp_eggs = len(board.eggs_enemy)
    score = my_eggs - opp_eggs
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
    x,y = board.chicken_player.get_location()
    if ((x+y) % 2 == 0):
        score += 0.4
    if history:
        pos = board.chicken_player.get_location()
        if pos in history:
            score -= 0.6
    if history:
        pos = board.chicken_player.get_location()
        if pos not in history:
            score += 2.0
        dist_sum = 0
        for (hx, hy) in history:
            dist_sum += abs(pos[0] - hx) + abs(pos[1] - hy)
        if len(history) > 0:
            score += 0.08 * (dist_sum / len(history))
    for (tx, ty) in board.turds_player:
        d = abs(tx - x) + abs(ty - y)
        if d == 1:
            score -= 0.8
    for (tx,ty) in board.turds_enemy:
        d = abs(tx-x)+abs(ty-y)
        if d == 1:
            score -= 1.5
        elif d == 2:
            score -= 0.4
    if board.chicken_player.get_turds_left() > 0:
        if len(board.get_valid_moves()) <= 2:
            score -= 2.0
    if belief is not None:
        score += expected_trapdoor_penalty(board, belief)
    return score

def expected_trapdoor_penalty(board, belief):
    x, y = board.chicken_player.get_location()
    if not (0 <= x < board.game_map.MAP_SIZE and 0 <= y < board.game_map.MAP_SIZE):
        return 0.0
    p_fall_white = float(belief.white_probs[x,y])
    p_fall_black = float(belief.black_probs[x,y])
    p_total = p_fall_white + p_fall_black
    return -4.0 * p_total

def egg_opportunity(board):
    (x,y) = board.chicken_player.get_location()
    if ((x+y)%2 == 0):
        return 0.5
    return 0

def get_enemy_moves(board):
    moves = board.get_valid_moves()
    for direction, move_type in moves:
        child = board.forecast_move(direction, move_type)
        if child is not None:
            child.reverse_perspective()
            return child.get_valid_moves()
    return []

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
            if not board.is_valid_cell((nx, ny)):
                continue
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

def minimax(board, depth, alpha, beta, isMaximizing, belief, history=None):
    if history is None:
        history = []
    if depth == 0 or board.is_game_over():
        return evaluate(board, belief, history)
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return -999999 if isMaximizing else 999999
    if isMaximizing:
        maxEval = -math.inf
        moves_sorted = sorted(moves, key=lambda mv: 0 if getattr(mv[1], "name", "") == "EGG" else 1)
        for direction, move_type in moves_sorted:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue
            landing_pos = child.chicken_player.get_location()
            p_fall_white = float(belief.white_probs[landing_pos])
            p_fall_black = float(belief.black_probs[landing_pos])
            p_total = p_fall_white + p_fall_black
            immediate_danger = -3.0 * p_total
            if history and landing_pos in history:
                immediate_danger -= 1.5
            child.reverse_perspective()
            eval_score = minimax(child, depth - 1, alpha, beta, False, belief, history)
            eval_score += immediate_danger
            maxEval = max(maxEval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = math.inf
        moves_sorted = sorted(moves, key=lambda mv: 0 if getattr(mv[1], "name", "") == "TURD" else 1)
        for direction, move_type in moves_sorted:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue
            landing_pos = child.chicken_player.get_location()
            p_fall_white = float(belief.white_probs[landing_pos])
            p_fall_black = float(belief.black_probs[landing_pos])
            p_total = p_fall_white + p_fall_black
            immediate_danger = -3.0 * p_total
            if history and landing_pos in history:
                immediate_danger -= 1.5
            child.reverse_perspective()
            eval_score = minimax(child, depth - 1, alpha, beta, True, belief, history)
            eval_score += immediate_danger
            minEval = min(minEval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return minEval

class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        self.belief = TrapdoorBelief(board.game_map)
        self.history = deque(maxlen=6)
    def play(
            self,
            board: board.Board,
            sensor_data: List[Tuple[bool, bool]],
            time_left: Callable,
    ):
        location = board.chicken_player.get_location()
        self.history.append(location)
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
        if not moves:
            return None
        candidates = []
        best_value = -math.inf
        SAFE_TRAP_THRESHOLD = 0.55
        filtered_moves = []
        for d, mt in moves:
            child = board.forecast_move(d, mt)
            if child is None:
                continue
            landing_pos = child.chicken_player.get_location()
            p_total = float(self.belief.white_probs[landing_pos] + self.belief.black_probs[landing_pos])
            if p_total >= SAFE_TRAP_THRESHOLD:
                print(f"Filtered out {d.name},{mt.name} due to trap prob {p_total:.2f}")
                continue
            filtered_moves.append((d, mt))
        if not filtered_moves:
            filtered_moves = moves
        for direction, move_type in filtered_moves:
            child = board.forecast_move(direction, move_type)
            if child is None:
                continue
            landing_pos = child.chicken_player.get_location()
            hist_for_eval = list(self.history)
            hist_for_eval.append(landing_pos)
            child.reverse_perspective()
            value = minimax(child, 3, -math.inf, math.inf, False, self.belief, hist_for_eval)
            print(f"Move {direction.name}, {move_type.name} → value {value}")
            candidates.append(((direction, move_type), value))
            if value > best_value:
                best_value = value
        if not candidates:
            return moves[0]
        best_by_value = max(candidates, key=lambda x: x[1])[0]
        current_area = reachable_area(board)
        def move_area(dir, mt):
            child = board.forecast_move(dir, mt)
            if child is None:
                return -1
            return reachable_area(child)
        area_after_best = move_area(best_by_value[0], best_by_value[1])
        if area_after_best >= max(1, 0.45 * current_area):
            print(f"SAFE best move → {best_by_value}")
            return best_by_value
        safe_moves = []
        for (d, m), v in candidates:
            a = move_area(d, m)
            if a >= max(1, 0.45 * current_area):
                safe_moves.append(((d, m), v, a))
        if safe_moves:
            chosen = max(safe_moves, key=lambda x: (x[1], x[2]))[0]
            print(f"Fallback SAFE move → {chosen}")
            return chosen
        fallback = max(candidates, key=lambda x: (move_area(x[0][0], x[0][1]), x[1]))[0]
        print(f"No safe options — using MAX AREA fallback → {fallback}")
        return fallback
