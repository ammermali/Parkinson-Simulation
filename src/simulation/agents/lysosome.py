from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum
from repast4py.space import DiscretePoint
from typing import Optional, List


# Internal State Set
class LysosomeState(str, Enum):
    INACTIVE = "Inactive"
    ACTIVE = "Active"
    OVERWHELMED = "Overwhelmed"

# Action Set
class LysosomeAction(str, Enum):
    SCAN = "scan"
    SELECT_TARGET = "select_target"
    DEGRADE = "degrade"

@dataclass(frozen=True)
class LysosomePerception:
    position: Optional[DiscretePoint]
    targets: List[AdaptiveAgent]
    # TODO tasks
    local_aggregate_density: float

class Lysosome(AdaptiveAgent):
    # Further Fields
    target: AdaptiveAgent = None # it indicates the current target of the lysosome