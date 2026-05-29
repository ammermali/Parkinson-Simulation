from typing import Optional
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from repast4py.space import DiscretePoint
from src.simulation.agents.neuron import Neuron
from src.simulation.utils import clamp, RNG


# Internal State Set
class MitochondrionState(str, AdaptiveAgentState):
    HEALTHY = "Healthy"
    CONSUMED = "Consumed"
    DAMAGED = "Damaged"
    DEBRIS = "Debris"


# Action Set
class MitochondrionAction(str, AdaptiveAgentAction):
    STRESS = "stress"
    FUSE = "fuse"
    DIVIDE = "divide"

@dataclass(frozen=True)
class MitochondrionPerception(AdaptiveAgentPerception):
    position: Optional[DiscretePoint]
    oxidative_stress: float
    energy_demand: float
    local_aggregate_density: float
    local_debris_density: float
    target_assigned: bool

@dataclass(frozen=True)
class MitochondrionConfig:
    perception_radius: int = 1
    energy_demand_high_threshold: float = 0.7
    oxidative_stress_high_threshold: float = 0.7
    oxidative_stress_low_threshold: float = 0.3
    aggregate_density_high_threshold: float = 0.7
    aggregate_density_low_threshold: float = 0.3
    debris_density_high_threshold: float = 0.7
    debris_density_low_threshold: float = 0.3
    irreversible_damage_threshold: float = 0.85
    stress_release_rate: float = 0.03
    damage_stress_release_rate: float = 0.07
    debris_release_rate: float = 0.04
    fusion_stress_reduction_rate: float = 0.02
    fusion_debris_reduction_rate: float = 0.01

class Mitochondrion(AdaptiveAgent):
    def __init__(self,
                 local_id: int,
                 rank: int,
                 type_id: int,
                 config: MitochondrionConfig,
                 owner_neuron: Neuron
                 ):
        super().__init__(local_id, type_id, rank)
        self.state = MitochondrionState.HEALTHY
        self.last_perception: Optional[MitochondrionPerception] = None
        self.pending_action: Optional[MitochondrionAction] = None
        self.cfg = config
        self.owner_neuron = owner_neuron
        self.rng = RNG()


    def see(self, model) -> MitochondrionPerception:
        habitat = self.owner_neuron
        position = habitat.position_of(self)
        if position is None:
            perception = MitochondrionPerception(
                position=None,
                oxidative_stress=0.0,
                energy_demand=0.0,
                local_aggregate_density=0.0,
                local_debris_density=0.0,
                target_assigned=False
            )
            self.last_perception = perception
            return perception
        perception = MitochondrionPerception(
            position=position,
            oxidative_stress=habitat.oxidative_stress_at(position),
            energy_demand=habitat.energy_demand_at(position),
            local_aggregate_density=habitat.local_aggregate_density_at(
                position=position,
                radius=self.cfg.perception_radius,
                include_center=True
            ),
            local_debris_density=habitat.local_debris_density_at(
                position=position,
                radius=self.cfg.perception_radius,
                include_center=True
            ),
            target_assigned=habitat.is_target_assigned(self)
        )
        self.last_perception = perception
        return perception

    def next(self) -> MitochondrionState:
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        rng = self.rng.random()

        if self.state == MitochondrionState.HEALTHY:
            if rng < self.pr_pathological_evolution():
                self.state = MitochondrionState.CONSUMED
        elif self.state == MitochondrionState.CONSUMED:
            pr_healthy = self.pr_consumed_to_damaged() # TODO rename to consumed_to_healthy
            pr_damage = self.pr_pathological_evolution()
            transition = self._sample_transition({
                MitochondrionState.HEALTHY: pr_healthy,
                MitochondrionState.DAMAGED: pr_damage,
                MitochondrionState.DEBRIS: 1.0
            })
            self.state = transition
        elif self.state == MitochondrionState.DAMAGED:
            if rng < self.pr_pathological_evolution():
                self.state = MitochondrionState.CONSUMED
        elif self.state == MitochondrionState.DEBRIS:
            self.state = MitochondrionState.DEBRIS

        self.last_transition = (old_state, self.state)
        return self.state

    def action(self) -> Optional[MitochondrionAction]:
        if self.last_perception is None:
            raise RuntimeError()
        p = self.last_perception
        if self.state == MitochondrionState.DEBRIS:
            self.pending_action = None
        elif self.state == MitochondrionState.DAMAGED:
            self.pending_action = MitochondrionAction.DIVIDE
        elif self.state == MitochondrionState.CONSUMED:
            if p.energy_demand < self.cfg.energy_demand_high_threshold and self._is_low_damage(p):
                self.pending_action = MitochondrionAction.FUSE
            else:
                self.pending_action = MitochondrionAction.STRESS
        elif self.state == MitochondrionState.HEALTHY:
            if p.energy_demand >= self.cfg.energy_demand_high_threshold:
                self.pending_action = MitochondrionAction.STRESS
            else:
                self.pending_action = None
        return self.pending_action

    def do(self, model):
        habitat = self.owner_neuron # TODO change all the environment variable to this
        if self.state in (MitochondrionState.DAMAGED, MitochondrionState.DEBRIS):
            self._register_if_degradable(habitat)
        if self.pending_action is None:
            return
        if self.pending_action == MitochondrionAction.STRESS:
            habitat.add_oxidative_stress(self.cfg.stress_release_rate)
        elif self.pending_action == MitochondrionAction.FUSE:
            habitat.add_oxidative_stress(-self.cfg.fusion_stress_reduction_rate)
            self._add_intracellular_debris(habitat, -self.cfg.fusion_debris_reduction_rate)
            self.state = MitochondrionState.HEALTHY
        elif self.pending_action == MitochondrionAction.DIVIDE:
            habitat.add_oxidative_stress(self.cfg.damage_stress_release_rate)
            self._add_intracellular_debris(habitat, self.cfg.debris_release_rate)
            self._register_if_degradable(habitat)

    def _is_high_damage(self, p):
        return (
            p.oxidative_stress >= self.cfg.oxidative_stress_high_threshold or
            p.local_aggregate_density >= self.cfg.aggregate_density_high_threshold or
            p.local_debris_density >= self.cfg.debris_density_high_threshold
        )

    def _is_low_damage(self, p):
        return (
            p.oxidative_stress <= self.cfg.oxidative_stress_low_threshold or
            p.local_aggregate_density <= self.cfg.aggregate_density_low_threshold or
            p.local_debris_density <= self.cfg.debris_density_low_threshold
        )

    def _toxicity(self, p: MitochondrionPerception):
        return clamp(
            0.4 * p.oxidative_stress
            + 0.4 * p.local_aggregate_density
            + 0.2 * p.local_debris_density
        )

    def _register_if_degradable(self, habitat):
        register = getattr(habitat, "register_degradation_target", None)
        if callable(register):
            register(self)

    def _add_intracellular_debris(self, habitat, amount: float):
        add_debris = getattr(habitat, "add_intracellular_debris", None)
        if callable(add_debris):
            add_debris(amount)
        else:
            habitat.internal_effects.debris_added += amount

    def pr_pathological_evolution(self) -> float:
        p = self.last_perception
        return clamp(p.energy_demand * p.oxidative_stress * p.local_aggregate_density)

    def pr_consumed_to_damaged(self) -> float:
        p = self.last_perception
        return clamp((1.0 - p.energy_demand) * (1.0 - p.oxidative_stress) * (1.0 - p.local_aggregate_density))


    # TODO double check this shit
    def _sample_transition(
            self,
            raw_probabilities: dict[MitochondrionState, float]
            ) -> MitochondrionState:
        raw_total = {
            state: clamp(probability)
            for state, probability in raw_probabilities.items()
        }
        total = sum(raw_total.values())
        if total <= 0.0:
            return self.state
        draw = self.rng.random()
        cumulative = 0.0
        for state, probability in raw_total.items():
            cumulative += probability / total
            if draw <= cumulative:
                return state
        return self.state
