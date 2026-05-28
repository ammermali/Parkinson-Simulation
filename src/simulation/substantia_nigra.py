from dataclasses import dataclass
from typing import Optional, Iterable
from repast4py.space import DiscretePoint
from src.simulation.utils import clamp, LocalGrid


@dataclass(frozen=True)
class SNEnvironmentConfig:
    # Configuration derived from params.yaml
    initial_debris: float
    initial_inflammation: float
    initial_dopamine: float
    debris_decay: float
    inflammation_decay: float
    dopamine_smoothing: float


@dataclass
class SNScalars:
    # Scalars representing the state of the neuron
    extracellular_debris: float # amount of extracellular debris
    inflammation_level: float # level of inflammation
    dopamine_output: float # amount of the dopamine released in the Substantia Nigra


@dataclass
class SNEffects:
    # Effects of the agents in the environmental scalars for tick
    debris_added: float = 0.0
    debris_removed: float = 0.0
    inflammation_added: float = 0.0
    inflammation_removed: float = 0.0
    dopamine_released: float = 0.0


class SubstantiaNigra:

    def __init__(self, grid, config: SNEnvironmentConfig):
        self.grid = LocalGrid(repast_grid=grid)
        self.config = config
        self.scalars = SNScalars(
            extracellular_debris=config.initial_debris,
            inflammation_level=config.initial_inflammation,
            dopamine_output=config.initial_dopamine,
        )

        self.effects = SNEffects()

    # TICK CYCLE

    def begin_tick(self):
        self.effects = SNEffects()

    def commit_effects(self, max_possible_dopamine: float):
        cfg = self.config
        old = self.scalars
        eff = self.effects

        new_debris = (old.extracellular_debris + eff.debris_added - eff.debris_removed - cfg.debris_decay * old.extracellular_debris)
        new_inflammation = (old.inflammation_level + eff.inflammation_added - eff.inflammation_removed - cfg.inflammation_decay * old.inflammation_level)

        if max_possible_dopamine > 0:
            dopamine_raw = eff.dopamine_released / max_possible_dopamine
        else:
            dopamine_raw = 0.0

        new_dopamine = (
            (1.0 - cfg.dopamine_smoothing) * old.dopamine_output
            + cfg.dopamine_smoothing * dopamine_raw
        )

        self.scalars.extracellular_debris = clamp(new_debris)
        self.scalars.inflammation_level = clamp(new_inflammation)
        self.scalars.dopamine_output = clamp(new_dopamine)

    # AGENTS EFFECTS

    def add_debris(self, amount: float):
        self.effects.debris_added += amount

    def remove_debris(self, amount: float):
        self.effects.debris_removed += amount

    def add_inflammation(self, amount: float):
        self.effects.inflammation_added += amount

    def remove_inflammation(self, amount: float):
        self.effects.inflammation_removed += amount

    def release_dopamine(self, amount: float):
        self.effects.dopamine_released += amount

    # Grid wrapper functions
    def add_agent(self, agent, point: DiscretePoint):
        self.grid.add_agent(agent, point)

    def remove_agent(self, agent):
        self.grid.remove_agent(agent)

    def position_of(self, agent):
        return self.grid.position_of(agent)

    def move_to(self, agent, point: DiscretePoint):
        return self.grid.move_to(agent, point)

    def agents_at(self, point: DiscretePoint):
        return self.grid.agents_at(point)

    def agents_in_radius(self, center: DiscretePoint, radius: int):
        # TODO check again if it's equal to LocalGrid.agents_in_radius
        points = self.grid.neighbor_points(center, radius)
        agents = []
        for point in points:
            agents.extend(self.grid.agents_at(point))
        return agents

    def count_agents_in_radius(self, center: DiscretePoint, radius: int):
        return self.grid.count_agents_in_radius(center, radius)

    def neighbor_points(self, center: DiscretePoint, radius: int):
        return self.grid.neighbor_points(center, radius)

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: Optional[int], include_center: bool = True):
        return self.grid.density_of_type(center, radius, agent_type, include_center)