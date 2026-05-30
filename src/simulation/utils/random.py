from random import Random
from typing import Optional

class RNG:
    """Shared random source used by simulation agents."""
    _rng = Random()
    _rng.seed(42)

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)

    @classmethod
    def seed(cls, seed: int = 42):
        cls._rng.seed(seed)

    @classmethod
    def random(cls) -> float:
        return float(cls._rng.random())

    @classmethod
    def gaussian(cls, mean: float, std: float) -> float:
        return mean + std * cls._rng.gauss(0, 1)

    @classmethod
    def choice(cls, values):
        return cls._rng.choice(list(values))
