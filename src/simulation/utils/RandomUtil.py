from random import Random

RANDOM = Random()
RANDOM.seed(42) # TODO Param.yaml
class RNG:
    def __init__(self):
        self._rng = RANDOM

    def random(self) -> float:
        return float(self._rng.random())

    def gaussian(self, mean: float, std: float) -> float:
        return mean + std * self._rng.gauss(0, 1)

    def choice(self, values):
        return self._rng.choice(values)