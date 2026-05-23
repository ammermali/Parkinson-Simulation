from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

class Mitochondrion(AdaptiveAgent):
    # Internal State Set
    @dataclass
    class MitochondrionState(str, Enum):
        HEALTHY = "Healthy",
        CONSUMED = "Consumed",
        DAMAGED = "Damaged",
        DEBRIS = "Debris"
    # Action Set
    @dataclass
    class MitochondrionAction(str, Enum):
        STRESS = "stress",
        FUSE = "fuse",
        DIVIDE = "divide"