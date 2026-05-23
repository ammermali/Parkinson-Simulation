from typing import List

from src.simulation.agents.adaptiveagent import AdaptiveAgent
from enum import Enum
from dataclasses import dataclass

class Neuron(AdaptiveAgent):
    # Internal State Set
    @dataclass
    class NeuronState(str, Enum):
        HEALTHY = "Healthy",
        COMPROMISED = "Compromised",
        APOPTOPIC = "Apoptotic",
        RUPTURED = "Ruptured"
    # Action Set
    @dataclass
    class NeuronAction(str, Enum):
        R_DOPAMINE = "release_dopamine",
        R_ALPHASYNUCLEIN = "release_alphasynuclein",
        DUMP_DEBRIS = "dump_debris",
        A_ALPHASYNUCLEIN = "absorb_alphasynuclein",
        STRESS = "signal_stress"

    # Further Fields (for Intra environment)
    # Agent Registry - all the intracellular agents within the neuron are registered here
    agentRegistry: List<AdaptiveAgent>()

