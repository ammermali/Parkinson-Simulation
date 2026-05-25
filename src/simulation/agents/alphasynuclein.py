from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from repast4py.space import DiscretePoint

# Internal State Set
class AlphaSynucleinState(str, Enum):
    MONOMER = "Monomer"
    MISFOLDED = "Misfolded"
    OLIGOMER = "Oligomer"
    CLEARED = "Cleared"
    LEWY_BODY = "LewyBody"

# Action Set
class AlphaSynucleinAction(str, Enum):
    MOVE = "move"
    STAY = "stay"

@dataclass(frozen=True)
class AlphaSynucleinPerception:
    position: Optional[DiscretePoint]
    oxidative_stress: float
    local_aggregate_density: float

class AlphaSynuclein(AdaptiveAgent):
    # Further Fields
    aggregate_id : int = None # saves the ID of the bigger aggregate it belongs to


    # Initialization
    def __init__(self, id, type, rank):
        super().__init__(id, type, rank)
        self.state = AlphaSynucleinState.MONOMER # e0
        self.aggregate_id = None
        # TODO complete inizialization of the agent