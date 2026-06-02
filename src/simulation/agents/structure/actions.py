from enum import Enum

class AdaptiveAgentAction(Enum):
    pass

class AlphaSynucleinAction(str, AdaptiveAgentAction):
    MOVE = "move"
    STAY = "stay"

class NeuronAction(str, AdaptiveAgentAction):
    R_DOPAMINE = "release_dopamine"
    R_ALPHASYNUCLEIN = "release_alphasynuclein"
    DUMP_DEBRIS = "dump_debris"
    A_ALPHASYNUCLEIN = "absorb_alphasynuclein"
    STRESS = "signal_stress"
    IDLE = "idle"

class MitochondrionAction(str, AdaptiveAgentAction):
    REDUCE_DEMAND = "reduce_demand"
    STRESS = "stress"
    FUSE = "fuse"
    DIVIDE = "divide"

class AggregateAction(str, AdaptiveAgentAction):
    STAY = "stay"

class MicrogliaAction(str, AdaptiveAgentAction):
    SCAN = "scan"
    CLEAR_DEBRIS = "clear_debris"
    INFLAMMATION = "release_inflammation"

class LysosomeAction(str, AdaptiveAgentAction):
    SCAN = "scan"
    SELECT_TARGET = "select_target"
    DEGRADE = "degrade"
    IDLE = "idle"

class AstrocyteAction(str, AdaptiveAgentAction):
    SUPPORT = "provide_support"
    INFLAMMATION = "release_inflammation"
