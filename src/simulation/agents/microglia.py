from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

class Microglia(AdaptiveAgent):
    # Internal State Set
    @dataclass
    class MicrogliaState(str, Enum):
        RESTING = "Resting",
        CLEARING = "Clearing",
        ACTIVATED = "Activated"
    # Action Set
    @dataclass
    class MicrogliaAction(str, Enum):
        MOVE = "scan",
        STAY = "clear_debris",
        INFLAMMATION = "release_inflammation"