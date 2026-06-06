from typing import Optional, TYPE_CHECKING
from src.simulation.agents.structure import MitochondrionState, MitochondrionAction, MitochondrionPerception, MitochondrionConfig, AdaptiveAgent
from src.simulation.utils import clamp, RNG
from src.simulation.logger.agent_logging import bind_event_logger, event_logger_from
if TYPE_CHECKING:
    from src.simulation.agents.neuron import Neuron

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
        self.last_transition_sample = {}
        self.rng = RNG


    def see(self, model) -> MitochondrionPerception:
        """Read neuron-level stress, energy demand and degradation assignment."""
        bind_event_logger(self, model)
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
        self.last_transition_sample = {}

        if self.state == MitochondrionState.HEALTHY:
            probability = self.pr_pathological_evolution()
            draw = self.rng.random()
            self.last_transition_sample = {
                "check": "pathological_evolution",
                "probability": probability,
                "draw": draw,
            }
            if draw < probability:
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
        logger = event_logger_from(self)
        if logger is not None and old_state != self.state:
            logger.state_transition(
                self,
                old_state,
                self.state,
                "mitochondrion_lifecycle_transition",
                probability=self.last_transition_sample.get("probability"),
                rng_value=self.last_transition_sample.get("draw"),
                owner=self.owner_neuron,
                compartment="Intracellular"
            )
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
            if self._is_recovery_context(p):
                self.pending_action = MitochondrionAction.FUSE
            else:
                self.pending_action = MitochondrionAction.STRESS
        elif self.state == MitochondrionState.HEALTHY:
            if self._is_high_damage(p):
                self.pending_action = MitochondrionAction.STRESS
            else:
                self.pending_action = MitochondrionAction.REDUCE_DEMAND
        logger = event_logger_from(self)
        if logger is not None and self.pending_action is not None:
            logger.action_selection(
                self,
                self.pending_action,
                "mitochondrion_state_action_policy",
                owner=self.owner_neuron,
                compartment="Intracellular"
            )
        return self.pending_action

    def do(self, model):
        """Apply the selected mitochondrial effect to the owning neuron."""
        habitat = self.owner_neuron
        if self.state in (MitochondrionState.DAMAGED, MitochondrionState.DEBRIS):
            self._register_if_degradable(habitat)
        if self.pending_action is None:
            return
        if self.pending_action == MitochondrionAction.REDUCE_DEMAND:
            reduction = self.energy_demand_reduction()
            habitat.add_energy_demand(-reduction)
            logger = event_logger_from(self)
            if logger is not None:
                logger.internal_field_effect(
                    self,
                    self.owner_neuron,
                    "energy_demand",
                    -reduction,
                    "mitochondrion_energy_demand_reduction",
                    action=self.pending_action
                )
        elif self.pending_action == MitochondrionAction.STRESS:
            habitat.add_oxidative_stress(self.cfg.stress_release_rate)
            logger = event_logger_from(self)
            if logger is not None:
                logger.internal_field_effect(
                    self,
                    self.owner_neuron,
                    "oxidative_stress",
                    self.cfg.stress_release_rate,
                    "mitochondrion_stress_release",
                    action=self.pending_action
                )
            if self.state == MitochondrionState.CONSUMED:
                reduction = self.energy_demand_reduction()
                habitat.add_energy_demand(-reduction)
                if logger is not None:
                    logger.internal_field_effect(
                        self,
                        self.owner_neuron,
                        "energy_demand",
                        -reduction,
                        "consumed_mitochondrion_partial_demand_reduction",
                        action=self.pending_action
                    )
        elif self.pending_action == MitochondrionAction.FUSE:
            old_state = self.state
            habitat.add_oxidative_stress(-self.cfg.fusion_stress_reduction_rate)
            self._add_intracellular_debris(habitat, -self.cfg.fusion_debris_reduction_rate)
            self.state = MitochondrionState.HEALTHY
            logger = event_logger_from(self)
            if logger is not None:
                logger.internal_field_effect(
                    self,
                    self.owner_neuron,
                    "oxidative_stress",
                    -self.cfg.fusion_stress_reduction_rate,
                    "mitochondrion_fusion_stress_reduction",
                    action=self.pending_action
                )
                logger.internal_field_effect(
                    self,
                    self.owner_neuron,
                    "intracellular_debris",
                    -self.cfg.fusion_debris_reduction_rate,
                    "mitochondrion_fusion_debris_reduction",
                    action=self.pending_action
                )
                logger.state_transition(
                    self,
                    old_state,
                    self.state,
                    "mitochondrion_fusion_repair",
                    owner=self.owner_neuron,
                    compartment="Intracellular"
                )
        elif self.pending_action == MitochondrionAction.DIVIDE:
            habitat.add_oxidative_stress(self.cfg.damage_stress_release_rate)
            self._add_intracellular_debris(habitat, self.cfg.debris_release_rate)
            self._register_if_degradable(habitat)
            logger = event_logger_from(self)
            if logger is not None:
                logger.internal_field_effect(
                    self,
                    self.owner_neuron,
                    "oxidative_stress",
                    self.cfg.damage_stress_release_rate,
                    "damaged_mitochondrion_stress_release",
                    action=self.pending_action
                )
                logger.internal_field_effect(
                    self,
                    self.owner_neuron,
                    "intracellular_debris",
                    self.cfg.debris_release_rate,
                    "damaged_mitochondrion_debris_release",
                    action=self.pending_action
                )

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

    def _is_recovery_context(self, p):
        """Return True when a consumed mitochondrion can safely fuse."""
        low_energy_threshold = 0.5 * self.cfg.energy_demand_high_threshold
        return p.energy_demand <= low_energy_threshold and self._is_low_damage(p)

    def _toxicity(self, p: MitochondrionPerception):
        """Weighted local pathology pressure acting on the mitochondrion."""
        return clamp(
            0.4 * p.oxidative_stress
            + 0.4 * p.local_aggregate_density
            + 0.2 * p.local_debris_density
        )

    def _energy_pressure(self, p: MitochondrionPerception) -> float:
        """Energy-demand pressure above the configured high-demand threshold."""
        denominator = max(1e-9, 1.0 - self.cfg.energy_demand_high_threshold)
        return clamp((p.energy_demand - self.cfg.energy_demand_high_threshold) / denominator)

    def _register_if_degradable(self, habitat):
        """Expose damaged mitochondria to the neuron's lysosome buffer."""
        habitat.register_degradation_target(self)

    def _add_intracellular_debris(self, habitat, amount: float):
        """Buffer mitochondrial debris changes through the neuron API."""
        habitat.add_intracellular_debris(amount)

    def pr_pathological_evolution(self) -> float:
        """Probability that a healthy mitochondrion enters stressed state."""
        p = self.last_perception
        return clamp(0.35 * self._energy_pressure(p) + 0.65 * self._toxicity(p))

    def pr_consumed_to_healthy(self) -> float:
        """Recovery chance for a stressed mitochondrion in low-demand context."""

        p = self.last_perception
        if not self._is_recovery_context(p):
            return 0.0
        return clamp(0.25 * (1.0 - self._energy_pressure(p)) * (1.0 - self._toxicity(p)))

    def pr_consumed_to_damaged(self) -> float:
        """Damage chance for a stressed mitochondrion under toxic pressure."""
        p = self.last_perception
        return clamp(0.65 * self._toxicity(p) + 0.35 * self._energy_pressure(p))

    def pr_damaged_to_consumed(self) -> float:
        """Partial recovery chance from damaged to stressed but viable state."""

        p = self.last_perception
        if not self._is_low_damage(p):
            return 0.0
        return clamp(0.15 * (1.0 - self._toxicity(p)) * (1.0 - self._energy_pressure(p)))

    def pr_irreversible_damage(self) -> float:
        """Chance that toxicity crosses into debris-producing damage."""

        p = self.last_perception
        pressure = clamp(0.8 * self._toxicity(p) + 0.2 * self._energy_pressure(p))
        if pressure < self.cfg.irreversible_damage_threshold:
            return 0.0
        return pressure

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
        logger = event_logger_from(self)
        if logger is not None:
            logger.state_transition(
                self,
                old_state,
                self.state,
                "mitochondrion_lysosome_repair",
                owner=self.owner_neuron,
                compartment="Intracellular"
            )


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
        self.last_transition_sample = {
            "probabilities": raw_total,
            "draw": draw
        }
        cumulative = 0.0
        for state, probability in raw_total.items():
            cumulative += probability
            if draw <= cumulative:
                return state
        return self.state
