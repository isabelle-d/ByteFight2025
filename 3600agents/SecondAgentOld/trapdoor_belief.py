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
        probs = np.zeros((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                if ((i + j) % 2 == 0 and color=="white") or ((i+j)%2==1 and color=="black"):
                    dist = min(i, j, size-1-i, size-1-j)
                    weight = max(0, dist - 1)   #near edge=0, 0, 1,  2
                    probs[i,j] = weight

        probs /= probs.sum()
        return probs

    def _prob_hear_feel(self, dist1, dist2):
        """
        Returns (P(hear|trapdoor), P(feel|trapdoor)).
        """

        #shares edges from those below
        hear = 0.10
        feel = 0.0
        #share edge
        if dist1 == 1:
            hear = 0.50
            feel = 0.30
        #diagonal
        elif dist2 == 1:
            hear = 0.25
            feel = 0.15

        return hear, feel

    def update(self, pos, heard_white, felt_white, heard_black, felt_black):
        self.white_probs = self._update_single(self.white_probs, pos, heard_white, felt_white)
        self.black_probs = self._update_single(self.black_probs, pos, heard_black, felt_black)

    def _update_single(self, belief, pos, heard, felt):
        size = self.map_size
        x, y = pos

        likelihood = np.zeros_like(belief)

        for i in range(size):
            for j in range(size):
                if belief[i,j] == 0:
                    continue
                dx = abs(x - i)
                dy = abs(y - j)
                if (dx + dy == 1):
                    dist1 = 1
                else:
                    dist1 = 0
                if (dx == 1 and dy == 1):
                    dist2 = 1
                else:
                    dist2 = 0

                p_hear, p_feel = self._prob_hear_feel(dist1, dist2)

                # P(obs | trapdoor at (i,j))
                L = (p_hear if heard else (1 - p_hear)) \
                    * (p_feel if felt else (1 - p_feel))
                likelihood[i,j] = L

        posterior = belief * likelihood
        s = posterior.sum()
        if s == 0:
            return belief
        return posterior / s

