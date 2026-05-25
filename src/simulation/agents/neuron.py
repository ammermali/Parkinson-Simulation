from typing import List, Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent
from enum import Enum
from dataclasses import dataclass

# Internal State Set
class NeuronState(str, Enum):
    HEALTHY = "Healthy",
    COMPROMISED = "Compromised",
    APOPTOPIC = "Apoptotic",
    RUPTURED = "Ruptured"


# Action Set
class NeuronAction(str, Enum):
    R_DOPAMINE = "release_dopamine",
    R_ALPHASYNUCLEIN = "release_alphasynuclein",
    DUMP_DEBRIS = "dump_debris",
    A_ALPHASYNUCLEIN = "absorb_alphasynuclein",
    STRESS = "signal_stress"

@dataclass(frozen=True)
class NeuronPerception:
    position: Optional[DiscretePoint]
    nearby_alpha: float
    inflammatory_levels: float
    extracellular_debris: float
    # Some internal scalar are formally considered a part of the perception

class Neuron(AdaptiveAgent):

    # Further Fields (for Intra environment)
    # Agent Registry - all the intracellular agents within the neuron are registered here
    agentRegistry: List<AdaptiveAgent>()

