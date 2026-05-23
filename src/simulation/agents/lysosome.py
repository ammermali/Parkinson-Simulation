from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

class Lysosome(AdaptiveAgent):
    # Internal State Set
    @dataclass
    class LysosomeState(str, Enum):
        INACTIVE = "Inactive",
        ACTIVE = "Active",
        OVERWHELMED = "Overwhelmed"
    # Action Set
    @dataclass
    class LysosomeAction(str, Enum):
        SCAN = "scan",
        SELECT_TARGET = "select_target",
        DEGRADE = "degrade"

    # Further Fields
    target: AdaptiveAgent = None # it indicates the current target of the lysosome