from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from repast4py.space import DiscretePoint

# Internal State Set
class AlphaSynucleinState(str, Enum):
    MONOMER = "Monomer"
    MISFOLDED = "Misfolded"
    OLIGOMER = "Oligomer"
    CLEARED = "Cleared"
    LEWY_BODY = "LewyBody"

# Action Set
class AlphaSynucleinAction(str, Enum):
    MOVE = "move"
    STAY = "stay"

class AlphaSynucleinCompartment(str, Enum):
    INTRACELLULAR = "Intracellular"
    EXTRACELLULAR = "Extracellular"

@dataclass(frozen=True)
class AlphaSynucleinConfig:
    perception_radius: int = 1
    move_radius: int = 1
    move_probability: float = 0.5
    oxidative_stress_high_threshold: float = 0.6
    aggregate_density_high_threshold: float = 0.4
    lewy_body_density_high_threshold: float = 0.8

@dataclass(frozen=True)
class AlphaSynucleinPerception:
    position: Optional[DiscretePoint]
    oxidative_stress: float
    local_aggregate_density: float
    neighbors: Optional[List[AdaptiveAgent]]

class AlphaSynuclein(AdaptiveAgent):
    # Further Fields
    aggregate_id : Optional[int] = None # saves the ID of the bigger aggregate it belongs to

    # Initialization
    def __init__(
            self,
            id: int,
            type_id: int,
            rank: int,
            config: AlphaSynucleinConfig,
            compartment: AlphaSynucleinCompartment, # TODO
            owner_neuron: Optional[AdaptiveAgent] = None,
    ):
        super().__init__(id, type_id, rank)
        self.state = AlphaSynucleinState.MONOMER
        self.cfg = config
        self.compartment = compartment
        self.owner_neuron = owner_neuron
        self.aggregate_id: Optional[int] = None # TODO
        self.last_perception: Optional[AlphaSynucleinPerception] = None
        self.pending_action: Optional[AlphaSynucleinAction] = None

    @property
    def aggregate_weight(self) -> float:
        if self.state == AlphaSynucleinState.MISFOLDED:
            return 0.25 # TODO
        if self.state == AlphaSynucleinState.OLIGOMER:
            return 0.75 # TODO
        if self.state == AlphaSynucleinState.LEWY_BODY:
            return 1.0 # TODO
        return 0.0

    def see(self, model) -> AlphaSynucleinPerception:
        habitat = self._habitat(model)
        position = habitat.position_of(self)

        if position is None:
            perception = AlphaSynucleinPerception(
                position=None,
                oxidative_stress=0.0,
                local_aggregate_density=0.0,
                neighbors=None
            )
            self.last_perception = perception
            return perception

        oxidative_stress = habitat.oxidative_stress_at(position)
        local_aggregate_density = habitat.local_aggregate_density_at(
            position=position,
            radius=self.cfg.perception_radius,
            include_center=True
        )
        neighbors = list(
            agent
            for agent in habitat.agents_in_radius(
                center=position,
                radius=self.cfg.perception_radius,
                include_center=True
            )
            if agent is not self
        )
        perception = AlphaSynucleinPerception(
            position=position,
            oxidative_stress=oxidative_stress,
            local_aggregate_density=local_aggregate_density,
            neighbors=neighbors
        )
        self.last_perception = perception
        return perception

    def next(self) -> AlphaSynucleinState:
        if self.last_perception is None:
            raise RuntimeError("AlphaSynuclein.next() called before see().")

        # TODO: edit it to non-deterministic behavior
        p = self.last_perception
        if self.state == AlphaSynucleinState.MONOMER:
            if p.oxidative_stress >= self.cfg.oxidative_stress_high_threshold:
                self.state = AlphaSynucleinState.MISFOLDED
        elif self.state == AlphaSynucleinState.MISFOLDED:
            if p.local_aggregate_density >= self.cfg.aggregate_density_high_threshold:
                self.state = AlphaSynucleinState.OLIGOMER
        elif self.state == AlphaSynucleinState.OLIGOMER:
            if p.local_aggregate_density >= self.cfg.lewy_body_density_high_threshold:
                self.state = AlphaSynucleinState.LEWY_BODY
        return self.state

    def action(self) -> AlphaSynucleinAction:
        if self.state in (
            AlphaSynucleinState.CLEARED,
            AlphaSynucleinState.LEWY_BODY,
        ):
            self.pending_action = AlphaSynucleinAction.STAY
        else:
            self.pending_action = AlphaSynucleinAction.MOVE
        return self.pending_action

    def do(self, model):
        if self.pending_action is None:
            raise RuntimeError("AlphaSynuclein.do() called before action().")
        habitat = self._habitat(model)
        self._register_if_degradable(habitat)
        if self.pending_action == AlphaSynucleinAction.STAY:
            return
        if model.rng.random() > self.cfg.move_probability:
            return
        position = habitat.position_of(self)
        if position is None:
            return
        candidates = list(
            habitat.neighbor_points(
                center=position,
                radius=self.cfg.move_radius,
                include_center=True,
            )
        )
        if not candidates:
            return
        new_position = model.rng.choice(candidates)
        habitat.move_to(self, new_position)

    def step(self, model):
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def mark_cleared(self):
        self.state = AlphaSynucleinState.CLEARED
        self.pending_action = AlphaSynucleinAction.STAY

    def _register_if_degradable(self, habitat):
        if self.state not in (
            AlphaSynucleinState.MISFOLDED,
            AlphaSynucleinState.OLIGOMER,
            AlphaSynucleinState.LEWY_BODY
        ):
            return
        # Verify the presence of a "register_degradation_target" function in the habitat and call it if it exists
        register = getattr(habitat, "register_degradation_target", None)
        if callable(register):
            register(self)

    def _habitat(self, model):
        if self.compartment == AlphaSynucleinCompartment.INTRACELLULAR:
            if self.owner_neuron is None:
                raise RuntimeError("Intracellular AlphaSynuclein requires owner_neuron.")
            return self.owner_neuron
        if self.compartment == AlphaSynucleinCompartment.EXTRACELLULAR:
            return model.environment
        raise RuntimeError(f"Unknown AlphaSynuclein compartment.")