from typing import Optional

from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum
from repast4py.space import DiscretePoint


# Internal State Set
class MitochondrionState(str, Enum):
    HEALTHY = "Healthy"
    CONSUMED = "Consumed"
    DAMAGED = "Damaged"
    DEBRIS = "Debris"


# Action Set
class MitochondrionAction(str, Enum):
    STRESS = "stress"
    FUSE = "fuse"
    DIVIDE = "divide"

@dataclass(frozen=True)
class MitochondrionPerception:
    position: Optional[DiscretePoint]
    oxidative_stress: float
    energy_demand: float
    local_aggregate_density: float
    local_debris_density: float
    target_assigned: bool

class Mitochondrion(AdaptiveAgent):
    pass