from enum import Enum

class AdaptiveAgentState(Enum):
    pass

class AggregateState(str, AdaptiveAgentState):
    OLIGOMER = "Oligomer"
    LEWY_BODY = "LewyBody"

class AlphaSynucleinState(str, AdaptiveAgentState):
    MONOMER = "Monomer"
    MISFOLDED = "Misfolded"
    OLIGOMER = "Oligomer"
    CLEARED = "Cleared"
    LEWY_BODY = "LewyBody"

class AstrocyteState(str, AdaptiveAgentState):
    SUPPORTIVE = "Supportive"
    REACTIVE = "Reactive"

class LysosomeState(str, AdaptiveAgentState):
    INACTIVE = "Inactive"
    ACTIVE = "Active"
    OVERWHELMED = "Overwhelmed"

class MicrogliaState(str, AdaptiveAgentState):
    RESTING = "Resting"
    CLEARING = "Clearing"
    ACTIVATED = "Activated"

class MitochondrionState(str, AdaptiveAgentState):
    HEALTHY = "Healthy"
    CONSUMED = "Consumed"
    DAMAGED = "Damaged"
    DEBRIS = "Debris"

class NeuronState(str, AdaptiveAgentState):
    HEALTHY = "Healthy"
    COMPROMISED = "Compromised"
    APOPTOTIC = "Apoptotic"
    RUPTURED = "Ruptured"
