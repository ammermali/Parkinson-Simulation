from random import Random

RANDOM = Random()
RANDOM.seed(42) # TODO Param.yaml
def rng() -> float:
    return float(RANDOM.random())

def gaussian(mean: float, std: float) -> float:
    return mean + std * RANDOM.gauss(0, 1)