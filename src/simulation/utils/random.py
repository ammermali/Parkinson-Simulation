from random import Random

class RNG:
    def __init__(self, seed: int = 42):
        random = Random()
        random.seed(seed)
        self._rng = random

    def random(self) -> float:
        return float(self._rng.random())

    def gaussian(self, mean: float, std: float) -> float:
        return mean + std * self._rng.gauss(0, 1)

    def choice(self, values):
        return self._rng.choice(values)