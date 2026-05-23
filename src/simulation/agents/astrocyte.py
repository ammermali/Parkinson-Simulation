from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

class Astrocyte(AdaptiveAgent):
    # Internal State Set
    @dataclass
    class AstrocyteState(str, Enum):
        SUPPORTIVE = "Supportive",
        REACTIVE = "Reactive"
    # Action Set
    @dataclass
    class AstrocyteAction(str, Enum):
        SUPPORT = "provide_support",
        INFLAMMATION = "release_inflammation",