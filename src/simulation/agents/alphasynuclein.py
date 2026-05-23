from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

class AlphaSynuclein(AdaptiveAgent):
    # Internal State Set
    @dataclass
    class AlphaSynucleinState(str, Enum):
        MONOMER = "Monomer",
        MISFOLDED = "Misfolded",
        OLIGOMER = "Oligomer",
        CLEARED = "Cleared",
        LEWY_BODY = "LewyBody"
    # Action Set
    @dataclass
    class AlphaSynucleinAction(str, Enum):
        MOVE = "move",
        STAY = "stay"

    # Further Fields
    aggregate_id : int = None # saves the ID of the bigger aggregate it belongs to


    # Initialization
    def __init__(self):
        self.state = self.AlphaSynucleinState.MONOMER # e0
        aggregate_id = None
        # TODO complete inizialization of the agent