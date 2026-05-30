from typing import Optional, TYPE_CHECKING
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from repast4py.space import DiscretePoint
from src.simulation.utils import clamp, RNG

if TYPE_CHECKING:
    from src.simulation.agents.neuron import Neuron


# Internal State Set
class MitochondrionState(str, AdaptiveAgentState):
    """Lifecycle state of one intracellular mitochondrion."""
    HEALTHY = "Healthy"
    CONSUMED = "Consumed"
    DAMAGED = "Damaged"
    DEBRIS = "Debris"


# Action Set
class MitochondrionAction(str, AdaptiveAgentAction):
    """Operations a mitochondrion can request during its action phase."""
    REDUCE_DEMAND = "reduce_demand"
    STRESS = "stress"
    FUSE = "fuse"
    DIVIDE = "divide"

@dataclass(frozen=True)
class MitochondrionPerception(AdaptiveAgentPerception):
    """Local intracellular context used by mitochondrial state transitions."""
    position: Optional[DiscretePoint]
    oxidative_stress: float
    energy_demand: float
    local_aggregate_density: float
    local_debris_density: float
    target_assigned: bool

@dataclass(frozen=True)
class MitochondrionConfig:
    """Parameters for mitochondrial stress, recovery and deficit reduction."""
    perception_radius: int
    energy_demand_high_threshold: float
    oxidative_stress_high_threshold: float
    oxidative_stress_low_threshold: float
    aggregate_density_high_threshold: float
    aggregate_density_low_threshold: float
    debris_density_high_threshold: float
    debris_density_low_threshold: float
    irreversible_damage_threshold: float
    stress_release_rate: float
    damage_stress_release_rate: float
    debris_release_rate: float
    fusion_stress_reduction_rate: float
    fusion_debris_reduction_rate: float
    healthy_energy_demand_reduction_rate: float
    consumed_energy_demand_reduction_rate: float
    high_demand_reduction_multiplier: float

class Mitochondrion(AdaptiveAgent):
    """Mitochondrial agent inside a neuron.
    The simulation tracks energy_demand as unmet energetic need, not physical
    energy production. Healthy mitochondria reduce that deficit through the
    neuron's buffered add_energy_demand API. Toxic local conditions push them
    toward a stressed CONSUMED state, then toward DAMAGED or DEBRIS. Damaged
    and debris states register with the neuron's degradation buffer so
    lysosomes can repair them.
    """
    def __init__(self, local_id: int, rank: int, type_id: int, config: MitochondrionConfig, owner_neuron: "Neuron"):
        super().__init__(local_id, type_id, rank)
        self.state: MitochondrionState = MitochondrionState.HEALTHY
        self.last_perception: Optional[MitochondrionPerception] = None
        self.pending_action: Optional[MitochondrionAction] = None
        self.cfg = config
        self.owner_neuron = owner_neuron
        self.rng = RNG


    def see(self, model) -> MitochondrionPerception:
        """Read neuron-level stress, energy demand and degradation assignment."""
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
        """Advance mitochondrial lifecycle from the last perception."""
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state

        if self.state == MitochondrionState.HEALTHY:
            if self.rng.random() < self.pr_pathological_evolution():
                self.state = MitochondrionState.CONSUMED
        elif self.state == MitochondrionState.CONSUMED:
            pr_healthy = self.pr_consumed_to_healthy()
            pr_damage = self.pr_consumed_to_damaged()
            transition = self._sample_transition({
                MitochondrionState.HEALTHY: pr_healthy,
                MitochondrionState.DAMAGED: pr_damage,
                MitochondrionState.DEBRIS: self.pr_irreversible_damage(),
            })
            self.state = transition
        elif self.state == MitochondrionState.DAMAGED:
            self.state = self._sample_transition({
                MitochondrionState.CONSUMED: self.pr_damaged_to_consumed(),
                MitochondrionState.DEBRIS: self.pr_irreversible_damage(),
            })
        elif self.state == MitochondrionState.DEBRIS:
            self.state = MitochondrionState.DEBRIS

        self.last_transition = (old_state, self.state)
        return self.state

    def action(self) -> Optional[MitochondrionAction]:
        """Choose energy, stress, recovery or degradation-related behavior."""
        if self.last_perception is None:
            raise RuntimeError()
        p = self.last_perception
        if p.target_assigned:
            self.pending_action = None
        elif self.state == MitochondrionState.DEBRIS:
            self.pending_action = None
        elif self.state == MitochondrionState.DAMAGED:
            self.pending_action = MitochondrionAction.DIVIDE
        elif self.state == MitochondrionState.CONSUMED:
            if p.energy_demand < self.cfg.energy_demand_high_threshold and self._is_low_damage(p):
                self.pending_action = MitochondrionAction.FUSE
            else:
                self.pending_action = MitochondrionAction.STRESS
        elif self.state == MitochondrionState.HEALTHY:
            if self._is_high_damage(p):
                self.pending_action = MitochondrionAction.STRESS
            else:
                self.pending_action = MitochondrionAction.REDUCE_DEMAND
        return self.pending_action

    def do(self, model):
        """Apply the selected mitochondrial effect to the owning neuron."""
        habitat = self.owner_neuron
        if self.state in (MitochondrionState.DAMAGED, MitochondrionState.DEBRIS):
            self._register_if_degradable(habitat)
        if self.pending_action is None:
            return
        if self.pending_action == MitochondrionAction.REDUCE_DEMAND:
            habitat.add_energy_demand(-self.energy_demand_reduction())
        elif self.pending_action == MitochondrionAction.STRESS:
            habitat.add_oxidative_stress(self.cfg.stress_release_rate)
            if self.state == MitochondrionState.CONSUMED:
                habitat.add_energy_demand(-self.energy_demand_reduction())
        elif self.pending_action == MitochondrionAction.FUSE:
            habitat.add_oxidative_stress(-self.cfg.fusion_stress_reduction_rate)
            self._add_intracellular_debris(habitat, -self.cfg.fusion_debris_reduction_rate)
            self.state = MitochondrionState.HEALTHY
        elif self.pending_action == MitochondrionAction.DIVIDE:
            habitat.add_oxidative_stress(self.cfg.damage_stress_release_rate)
            self._add_intracellular_debris(habitat, self.cfg.debris_release_rate)
            self._register_if_degradable(habitat)

    def _is_high_damage(self, p):
        """Return True when any local damage signal is above its high threshold."""
        return (
            p.oxidative_stress >= self.cfg.oxidative_stress_high_threshold or
            p.local_aggregate_density >= self.cfg.aggregate_density_high_threshold or
            p.local_debris_density >= self.cfg.debris_density_high_threshold
        )

    def _is_low_damage(self, p):
        """Return True only when all local damage signals are low."""
        return (
            p.oxidative_stress <= self.cfg.oxidative_stress_low_threshold and
            p.local_aggregate_density <= self.cfg.aggregate_density_low_threshold and
            p.local_debris_density <= self.cfg.debris_density_low_threshold
        )

    def _toxicity(self, p: MitochondrionPerception):
        """Weighted local pathology pressure acting on the mitochondrion."""
        return clamp(
            0.4 * p.oxidative_stress
            + 0.4 * p.local_aggregate_density
            + 0.2 * p.local_debris_density
        )

    def _register_if_degradable(self, habitat):
        """Expose damaged mitochondria to the neuron's lysosome buffer."""
        habitat.register_degradation_target(self)

    def _add_intracellular_debris(self, habitat, amount: float):
        """Buffer mitochondrial debris changes through the neuron API."""
        habitat.add_intracellular_debris(amount)

    def pr_pathological_evolution(self) -> float:
        """Probability that a healthy mitochondrion enters stressed state."""
        p = self.last_perception
        return clamp(0.5 * p.energy_demand + 0.5 * self._toxicity(p))

    def pr_consumed_to_healthy(self) -> float:
        """Recovery chance for a stressed mitochondrion in low-demand context."""

        p = self.last_perception
        return clamp((1.0 - p.energy_demand) * (1.0 - self._toxicity(p)))

    def pr_consumed_to_damaged(self) -> float:
        """Damage chance for a stressed mitochondrion under toxic pressure."""
        p = self.last_perception
        return clamp(self._toxicity(p) * (0.5 + 0.5 * p.energy_demand))

    def pr_damaged_to_consumed(self) -> float:
        """Partial recovery chance from damaged to stressed but viable state."""

        p = self.last_perception
        return clamp((1.0 - self._toxicity(p)) * (1.0 - p.energy_demand))

    def pr_irreversible_damage(self) -> float:
        """Chance that toxicity crosses into debris-producing damage."""

        p = self.last_perception
        toxicity = self._toxicity(p)
        if toxicity < self.cfg.irreversible_damage_threshold:
            return 0.0
        return clamp(toxicity)

    def energy_demand_reduction(self) -> float:
        """Amount of unmet energy demand reduced this tick."""

        if self.last_perception is None:
            demand_multiplier = 1.0
        elif self.last_perception.energy_demand >= self.cfg.energy_demand_high_threshold:
            demand_multiplier = self.cfg.high_demand_reduction_multiplier
        else:
            demand_multiplier = 1.0
        if self.state == MitochondrionState.HEALTHY:
            return self.cfg.healthy_energy_demand_reduction_rate * demand_multiplier
        if self.state == MitochondrionState.CONSUMED:
            return self.cfg.consumed_energy_demand_reduction_rate * demand_multiplier
        return 0.0

    def repair_by_lysosome(self):
        """Restore this mitochondrion after successful lysosomal cleanup."""
        old_state = self.state
        self.state = MitochondrionState.HEALTHY
        self.pending_action = None
        self.last_transition = (old_state, self.state)


    def _sample_transition(
            self,
            raw_probabilities: dict[MitochondrionState, float]
            ) -> MitochondrionState:
        """Sample outgoing transitions, keeping leftover probability as stay."""
        raw_total = {
            state: clamp(probability)
            for state, probability in raw_probabilities.items()
        }
        total = sum(raw_total.values())
        if total > 1.0:
            raw_total = {
                state: probability / total
                for state, probability in raw_total.items()
            }
            total = 1.0
        raw_total[self.state] = 1.0 - total
        draw = self.rng.random()
        cumulative = 0.0
        for state, probability in raw_total.items():
            cumulative += probability
            if draw <= cumulative:
                return state
        return self.state
