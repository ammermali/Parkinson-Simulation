from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

# Internal State Set
class AstrocyteState(str, Enum):
    SUPPORTIVE = "Supportive",
    REACTIVE = "Reactive"

# Action Set
class AstrocyteAction(str, Enum):
    SUPPORT = "provide_support",
    INFLAMMATION = "release_inflammation"

@dataclass(frozen=True)
class AstrocytePerception:
    position: Optional[DiscretePoint]
    inflammation_level: float
    extracellular_debris: float

class Astrocyte(AdaptiveAgent):
    pass