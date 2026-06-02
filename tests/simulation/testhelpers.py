import importlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from repast4py.space import DiscretePoint

def import_any(*module_names: str):
    last_error = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            last_error = e
    raise last_error

@dataclass(unsafe_hash=True)
class TestAgent:
    ptype: int = 1
    uid: int = 0

@dataclass(unsafe_hash=True)
class TestAggregateAgent:
    aggregate_weight: float = 1.0
    ptype: int = 99
    uid: int = 0

class TestRng:
    def __init__(self, random_value: float = 0.0, choice_index: int = 0):
        self.random_value = random_value
        self.choice_index = choice_index

    def random(self) -> float:
        return self.random_value

    def gaussian(self, mean: float, std: float) -> float:
        return mean + std * self.random_value

    def choice(self, values):
        return list(values)[self.choice_index]


def make_alpha_config(**overrides):
    alpha_module = import_any("src.simulation.agents.alphasynuclein")
    config = alpha_module.AlphaSynucleinConfig(
        perception_radius=1,
        move_radius=1,
        move_probability=0.5,
        oxidative_stress_high_threshold=0.6,
    )
    return replace(config, **overrides)


def make_mitochondrion_config(**overrides):
    mitochondrion_module = import_any("src.simulation.agents.mitochondrion")
    config = mitochondrion_module.MitochondrionConfig(
        perception_radius=1,
        energy_demand_high_threshold=0.7,
        oxidative_stress_high_threshold=0.7,
        oxidative_stress_low_threshold=0.3,
        aggregate_density_high_threshold=0.7,
        aggregate_density_low_threshold=0.3,
        debris_density_high_threshold=0.7,
        debris_density_low_threshold=0.3,
        irreversible_damage_threshold=0.85,
        stress_release_rate=0.03,
        damage_stress_release_rate=0.07,
        debris_release_rate=0.04,
        fusion_stress_reduction_rate=0.02,
        fusion_debris_reduction_rate=0.01,
        healthy_energy_demand_reduction_rate=0.04,
        consumed_energy_demand_reduction_rate=0.01,
        high_demand_reduction_multiplier=1.5,
    )
    return replace(config, **overrides)


def make_lysosome_config(**overrides):
    lysosome_module = import_any("src.simulation.agents.lysosome")
    config = lysosome_module.LysosomeConfig(
        perception_radius=1,
        move_radius=1,
        base_degradation_probability=0.8,
        protein_degradation_ticks=1,
        mitochondrion_repair_ticks=2,
        mitochondrion_repair_probability=0.8,
        aggregate_degradation_ticks_base=1,
        aggregate_degradation_ticks_per_member=1,
        aggregate_degradation_probability_base=0.35,
        aggregate_degradation_probability_per_member=0.05,
        aggregate_overwhelm_probability_base=0.02,
        aggregate_overwhelm_probability_per_member=0.01,
    )
    return replace(config, **overrides)


class TestBounds:
    xmin = 0
    ymin = 0
    xextent = 10
    yextent = 10


class TestRepastGrid:
    def __init__(self):
        self.locations = {}
        self.cells = {}

    def get_local_bounds(self):
        return TestBounds()

    def get_location(self, agent):
        return self.locations.get(agent)

    def set_location(self, agent, point: DiscretePoint):
        self.locations[agent] = point
        self.cells.setdefault((point.x, point.y), []).append(agent)

    def get_agents(self, point: DiscretePoint):
        return list(self.cells.get((point.x, point.y), []))

    def get_num_agents(self, point: DiscretePoint, agent_type: int):
        return sum(
            1
            for agent in self.get_agents(point)
            if getattr(agent, "ptype", None) == agent_type
        )


# This class is designed to mock the SubstantiaNigra environment for testing purposes. It provides methods that agents would call to interact with the environment, and it records calls and changes to the environment's state for assertions in tests.
class TestSubstantiaNigraLikeEnvironment:
    def __init__(
        self,
        position=None,
        nearby_alpha: float = 0.0,
        debris: float = 0.0,
        inflammation: float = 0.0,
    ):
        self.position = position
        self.nearby_alpha = nearby_alpha
        self.scalars = SimpleNamespace(
            extracellular_debris=debris,
            inflammation_level=inflammation,
            dopamine_output=0.0,
        )
        self.removed_debris = 0.0
        self.added_debris = 0.0
        self.added_inflammation = 0.0
        self.removed_inflammation = 0.0
        self.released_dopamine = 0.0
        self.moves = []
        self.neighbor_calls = []
        self.density_calls = []

    def position_of(self, agent):
        return self.position

    def density_of_type(self, center, radius, agent_type=None, include_center=True):
        self.density_calls.append((center, radius, agent_type, include_center))
        return self.nearby_alpha

    def neighbor_points(self, center, radius=1, include_center=True):
        self.neighbor_calls.append((center, radius, include_center))
        return [center, DiscretePoint(min(center.x + 1, 9), center.y)]

    def move_to(self, agent, point):
        self.moves.append((agent, point))
        self.position = point
        return point

    def remove_debris(self, amount: float):
        self.removed_debris += amount

    def add_debris(self, amount: float):
        self.added_debris += amount

    def remove_inflammation(self, amount: float):
        self.removed_inflammation += amount

    def add_inflammation(self, amount: float):
        self.added_inflammation += amount

    def release_dopamine(self, amount: float):
        self.released_dopamine += amount
