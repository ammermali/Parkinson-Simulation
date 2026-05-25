from dataclasses import dataclass
from typing import Optional, Iterable
from repast4py.space import DiscretePoint


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
        self.grid = grid
        self.config = config

        bounds = self.grid.get_local_bounds()

        self.xmin = bounds.xmin
        self.xmax = bounds.xmin + bounds.xextent
        self.ymin = bounds.ymin
        self.ymax = bounds.ymin + bounds.yextent

        self._offset_cache: dict[tuple[int, bool], list[tuple[int, int]]] = {}

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

        new_debris = (
            old.extracellular_debris
            + eff.debris_added
            - eff.debris_removed
            - cfg.debris_decay * old.extracellular_debris
        )

        new_inflammation = (
            old.inflammation_level
            + eff.inflammation_added
            - eff.inflammation_removed
            - cfg.inflammation_decay * old.inflammation_level
        )

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

    # GRID PRIMITIVES

    def position_of(self,agent) -> Optional[DiscretePoint]:
        return self.grid.get_location(agent)
    def move_to(self, agent, point: DiscretePoint) -> Optional[DiscretePoint]:
        return self.grid.move(agent, point)
    def agents_at(self, point: DiscretePoint) -> list:
        return list(self.grid.get_agents(point))


    # NEIGHBORHOOD

    def neighbor_points(self, center: DiscretePoint, radius:int = 1, include_center: bool = True) -> Iterable[DiscretePoint]:
        offsets = self._get_offsets(radius, include_center)
        for dx, dy in offsets:
            x = center.x + dx
            y = center.y + dy
            if self._inside_bounds(x,y):
                yield DiscretePoint(x,y)

    def agents_in_radius(self, center: DiscretePoint, radius: int = 1, include_center: bool = False) -> Iterable:
        for point in self.neighbor_points(center, radius, include_center):
            for agent in self.grid.get_agents(point):
                yield agent

    def count_agents_in_radius(self, center: DiscretePoint, radius: int = 1, agent_type: Optional[int] = None, include_center: bool = False) -> int:
        total = 0
        for point in self.neighbor_points(center, radius, include_center):
            total += (len(list(self.grid.get_agents(point))) if agent_type is None else self.grid.get_num_agents(point, agent_type))
        return total

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: int, include_center: bool = True,) -> float:
        neighborhood_size = self.neighbor_points(center=center, radius=radius, include_center=include_center).__sizeof__()
        if neighborhood_size <= 0:
            return 0.0

        count = self.count_agents_in_radius(center=center,radius=radius,agent_type=agent_type,include_center=include_center)
        return clamp(count / neighborhood_size)

    # Internal auxiliary functions
    def _inside_bounds(self, x: int, y: int) -> bool:
        return self.xmin <= x < self.xmax and self.ymin <= y < self.ymax
    def _get_offsets(self, radius: int, include_center: bool) -> list[tuple[int, int]]:
        key = (radius, include_center)

        if key in self._offset_cache:
            return self._offset_cache[key]

        offsets = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if not include_center and dx == 0 and dy == 0:
                    continue

                offsets.append((dx, dy))

        self._offset_cache[key] = offsets
        return offsets


# Auxiliary function
def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))