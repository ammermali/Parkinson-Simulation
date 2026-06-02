
from dataclasses import dataclass
from typing import Any, Optional

from repast4py.space import DiscretePoint


class AdaptiveAgentPerception:
    pass

@dataclass(frozen=True)
class AggregatePerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]


@dataclass(frozen=True)
class AlphaSynucleinPerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    oxidative_stress: float
    local_aggregate_density: float
    neighbors: list[Any]


@dataclass(frozen=True)
class AstrocytePerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    inflammation_level: float
    extracellular_debris: float


@dataclass(frozen=True)
class LysosomePerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    targets: list[Any]
    task: Optional[Any]
    local_aggregate_density: float
    target_pressure: float


@dataclass(frozen=True)
class MicrogliaPerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    extracellular_debris: float
    inflammation_level: float
    nearby_alpha: float


@dataclass(frozen=True)
class MitochondrionPerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    oxidative_stress: float
    energy_demand: float
    local_aggregate_density: float
    local_debris_density: float
    target_assigned: bool


@dataclass(frozen=True)
class NeuronPerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    nearby_alpha: float
    inflammatory_levels: float
    extracellular_debris: float
    oxidative_stress: float
    intracellular_debris: float
    energy_demand: float
    internal_damage: float
    alpha_load: float
    cell_damage: float

