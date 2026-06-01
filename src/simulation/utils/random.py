from random import Random
from typing import Optional

class RNG:
    """Shared random source used by simulation agents."""
    _rng = Random()
    _rng.seed(42)

    def __init__(self, seed: Optional[int] = None):
        """Optionally reseed the shared random source."""

        if seed is not None:
            self.seed(seed)

    @classmethod
    def seed(cls, seed: int = 42):
        """Seed the shared source used by all RNG instances."""

        cls._rng.seed(seed)

    @classmethod
    def random(cls) -> float:
        """Return a uniform random float in [0, 1)."""

        return float(cls._rng.random())

    @classmethod
    def gaussian(cls, mean: float, std: float) -> float:
        """Sample a Gaussian value with the configured mean and standard deviation."""

        return mean + std * cls._rng.gauss(0, 1)

    @classmethod
    def choice(cls, values):
        """Choose one item from an iterable using the shared source."""

        return cls._rng.choice(list(values))
