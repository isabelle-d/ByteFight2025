import numpy as np

class TrapdoorBelief:
    """
    Maintains belief distributions over white/black trapdoor locations.
    """
    def __init__(self, game_map):
        self.map_size = game_map.MAP_SIZE
        self.white_probs = self._initial_prior(color="white")
        self.black_probs = self._initial_prior(color="black")

    def _initial_prior(self, color):
        size = self.map_size
        probs = np.zeros((size, size), float)

        for i in range(size):
            for j in range(size):
                if (color=="white" and (i+j)%2!=0) or (color=="black" and (i+j)%2!=1):
                    continue

                # distance from nearest edge
                dist = min(i, j, size-1-i, size-1-j)

                weight = max(0, dist - 1)
                probs[i, j] = weight

        probs /= probs.sum()
        return probs

    #manhattan
    def _prob_hear_feel(self, dx, dy):
        man = dx + dy

        # shares edge
        if man == 1:
            return 0.50, 0.30

        # diagonal
        if dx == 1 and dy == 1:
            return 0.25, 0.15
        if man == 2:
            return 0.10, 0.00

        return 0.00, 0.00

    def update(self, pos, heard_white, felt_white, heard_black, felt_black):
        self.white_probs = self._update_single(self.white_probs, pos, heard_white, felt_white)
        self.black_probs = self._update_single(self.black_probs, pos, heard_black, felt_black)


    def _update_single(self, belief, pos, heard, felt):
        size = self.map_size
        x, y = pos

        likelihood = np.zeros_like(belief)

        for i in range(size):
            for j in range(size):
                if belief[i, j] == 0:
                    continue

                dx = abs(x - i)
                dy = abs(y - j)

                p_hear, p_feel = self._prob_hear_feel(dx, dy)

                L = (p_hear if heard else (1 - p_hear)) * \
                    (p_feel if felt else (1 - p_feel))

                likelihood[i, j] = L

        posterior = belief * likelihood
        s = posterior.sum()
        if s == 0:
            return belief
        return posterior / s

