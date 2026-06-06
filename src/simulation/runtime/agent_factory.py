from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from src.simulation.agents.alphasynuclein import AlphaSynuclein, AlphaSynucleinCompartment
from src.simulation.agents.astrocyte import Astrocyte
from src.simulation.agents.microglia import Microglia
from src.simulation.agents.neuron import Neuron
from src.simulation.agents.mitochondrion import Mitochondrion
from src.simulation.agents.lysosome import Lysosome
from src.simulation.agents.structure.agenttypes import AgentType
from src.simulation.substantia_nigra import SubstantiaNigra
from src.simulation.utils import Params, RNG
from src.simulation.utils.config_factory import ConfigFactory

@dataclass
class AgentFactory:
    rank: int
    config_rng: RNG
    neuron_params: Params
    environment: SubstantiaNigra
    new_id: Callable[[], int]

    def create_neuron(self) -> Neuron:
        return Neuron(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.NEURON,
            config=ConfigFactory.build_neuron_config(self.neuron_params, rng=self.config_rng),
            alpha_type_id=AgentType.ALPHA,
            internal_config=ConfigFactory.build_neuron_internal_config(self.neuron_params),
            environment=self.environment
        )

    def create_microglia(self) -> Microglia:
        return Microglia(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.MICROGLIA,
            config=ConfigFactory.build_microglia_config(rng=self.config_rng),
            alpha_type_id=AgentType.ALPHA
        )

    def create_astrocyte(self) -> Astrocyte:
        return Astrocyte(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.ASTROCYTE,
            config=ConfigFactory.build_astrocyte_config(rng=self.config_rng)
        )

    def create_extracellular_alpha(self) -> AlphaSynuclein:
        return AlphaSynuclein(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.ALPHA,
            config=ConfigFactory.build_alpha_synuclein_config(rng=self.config_rng),
            compartment=AlphaSynucleinCompartment.EXTRACELLULAR,
            owner_neuron=None
        )

    def create_intracellular_alpha(self, owner_neuron: Neuron) -> AlphaSynuclein:
        return AlphaSynuclein(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.ALPHA,
            config=ConfigFactory.build_alpha_synuclein_config(rng=self.config_rng),
            compartment=AlphaSynucleinCompartment.INTRACELLULAR,
            owner_neuron=owner_neuron
        )

    def create_mitochondrion(self, owner_neuron: Neuron) -> Mitochondrion:
        return Mitochondrion(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.MITOCHONDRION,
            config=ConfigFactory.build_mitochondrion_config(rng=self.config_rng),
            owner_neuron=owner_neuron
        )

    def create_lysosome(self, owner_neuron: Neuron) -> Lysosome:
        return Lysosome(
            local_id=self.new_id(),
            rank=self.rank,
            type_id=AgentType.LYSOSOME,
            owner_neuron=owner_neuron,
            config=ConfigFactory.build_lysosome_config()
        )