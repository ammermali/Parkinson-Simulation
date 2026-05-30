from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from repast4py.space import DiscretePoint
from typing import Optional, List
from src.simulation.agents.aggregate import AlphaAggregate, AggregateState
from src.simulation.agents.alphasynuclein import AlphaSynuclein
from src.simulation.agents.mitochondrion import Mitochondrion
from src.simulation.utils import clamp, RNG

# Internal State Set
class LysosomeState(str, AdaptiveAgentState):
    """Operational state of a lysosome inside a neuron."""

    INACTIVE = "Inactive"
    ACTIVE = "Active"
    OVERWHELMED = "Overwhelmed"

# Action Set
class LysosomeAction(str, AdaptiveAgentAction):
    """Actions a lysosome can perform during its execution phase."""

    SCAN = "scan"
    SELECT_TARGET = "select_target"
    DEGRADE = "degrade"
    IDLE = "idle"

@dataclass(frozen=True)
class LysosomePerception(AdaptiveAgentPerception):
    """Snapshot of the degradation context visible to a lysosome."""

    position: Optional[DiscretePoint]
    targets: List[AdaptiveAgent]
    task: Optional[AdaptiveAgent]
    local_aggregate_density: float
    target_pressure: float

@dataclass(frozen=True)
class LysosomeConfig:
    """Tunable lysosomal sensing, movement and cleanup parameters."""
    perception_radius: int
    move_radius: int
    base_degradation_probability: float
    protein_degradation_ticks: int
    mitochondrion_repair_ticks: int
    mitochondrion_repair_probability: float
    aggregate_degradation_ticks_base: int
    aggregate_degradation_ticks_per_member: int
    aggregate_degradation_probability_base: float
    aggregate_degradation_probability_per_member: float
    aggregate_overwhelm_probability_base: float
    aggregate_overwhelm_probability_per_member: float

class Lysosome(AdaptiveAgent):
    """Intracellular degradation agent coordinated by neuron target buffers.
    A lysosome does not discover all degradable agents directly. Instead,
    damaged proteins, aggregates and organelles register in the owning neuron;
    the lysosome claims one available target and repeatedly attempts to degrade
    it. Proteins and mitochondria use configurable repair/degradation durations;
    aggregate duration and risk are derived from aggregate size. Lewy bodies
    always overwhelm the lysosome.
    """
    def __init__(
            self,
            local_id,
            rank,
            type_id,
            owner_neuron,
            config: Optional[LysosomeConfig] = None,
    ):
        super().__init__(local_id, type_id, rank)

        raw_config = config or LysosomeConfig()
        self.state: LysosomeState = LysosomeState.INACTIVE
        self.owner_neuron = owner_neuron
        self.cfg = LysosomeConfig(
            perception_radius=raw_config.perception_radius,
            move_radius=raw_config.move_radius,
            base_degradation_probability=clamp(raw_config.base_degradation_probability),
            protein_degradation_ticks=max(1, raw_config.protein_degradation_ticks),
            mitochondrion_repair_ticks=max(1, raw_config.mitochondrion_repair_ticks),
            mitochondrion_repair_probability=clamp(raw_config.mitochondrion_repair_probability),
            aggregate_degradation_ticks_base=max(1, raw_config.aggregate_degradation_ticks_base),
            aggregate_degradation_ticks_per_member=max(0, raw_config.aggregate_degradation_ticks_per_member),
            aggregate_degradation_probability_base=clamp(raw_config.aggregate_degradation_probability_base),
            aggregate_degradation_probability_per_member=max(0.0, raw_config.aggregate_degradation_probability_per_member),
            aggregate_overwhelm_probability_base=clamp(raw_config.aggregate_overwhelm_probability_base),
            aggregate_overwhelm_probability_per_member=max(0.0, raw_config.aggregate_overwhelm_probability_per_member),
        )
        self.target: Optional[AdaptiveAgent] = None
        self._work_target: Optional[AdaptiveAgent] = None
        self.degradation_ticks_remaining: int = 0
        self.last_perception: Optional[LysosomePerception] = None
        self.pending_action: Optional[LysosomeAction] = None
        self.last_transition: tuple[LysosomeState, LysosomeState] = (
            self.state,
            self.state
        )
        self.rng = RNG

    def see(self, model) -> LysosomePerception:
        """Read the current assignment and pressure in the owning neuron."""

        habitat = self.owner_neuron
        position = habitat.position_of(self)
        task = habitat.target_for(self)
        if task is not None and habitat.position_of(task) is None:
            habitat.clear_degradation_assignment(self)
            task = None
        self.target = task
        if position is None:
            perception = LysosomePerception(
                position=None,
                targets=[],
                task=task,
                local_aggregate_density=0.0,
                target_pressure=0.0
            )
            self.last_perception = perception
            return perception
        targets = list(habitat.available_degradation_targets())
        local_aggregate_density = habitat.local_aggregate_density_at(
            position=position,
            radius=self.cfg.perception_radius,
            include_center=True
        )
        perception = LysosomePerception(
            position=position,
            targets=targets,
            task=task,
            local_aggregate_density=local_aggregate_density,
            target_pressure=self._target_pressure(habitat, targets)
        )
        self.last_perception = perception
        return perception

    def next(self) -> LysosomeState:
        """Advance the lysosome state before action selection."""
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        p = self.last_perception
        if self.state == LysosomeState.INACTIVE:
            self.state = self._sample_with_stay({
                LysosomeState.ACTIVE: self.pr_inactive_to_active(p)
            })
        elif self.state == LysosomeState.ACTIVE:
            self.state = self._sample_with_stay({
                LysosomeState.INACTIVE: self.pr_active_to_inactive(p)
            })
        elif self.state == LysosomeState.OVERWHELMED:
            self.state = LysosomeState.OVERWHELMED
        self.last_transition = (old_state, self.state)
        return self.state

    def action(self) -> LysosomeAction:
        """Choose the next operation from the current state and assignment."""

        if self.state == LysosomeState.INACTIVE:
            self.pending_action = LysosomeAction.SCAN
        elif self.state == LysosomeState.ACTIVE:
            if self.target is None:
                self.pending_action = LysosomeAction.SELECT_TARGET
            else:
                self.pending_action = LysosomeAction.DEGRADE
        elif self.state == LysosomeState.OVERWHELMED:
            self.pending_action = LysosomeAction.IDLE
        return self.pending_action

    def do(self, model):
        """Execute the selected lysosome operation."""

        habitat = self.owner_neuron
        if self.pending_action == LysosomeAction.SCAN:
            self._scan(habitat)
        elif self.pending_action == LysosomeAction.SELECT_TARGET:
            self._select_target(habitat)
        elif self.pending_action == LysosomeAction.DEGRADE:
            self._degrade_target(habitat)
        elif self.pending_action == LysosomeAction.IDLE:
            return

    def pr_inactive_to_active(self, p: LysosomePerception) -> float:
        """Activation probability rises with aggregate and target pressure."""

        return clamp(
            p.target_pressure + p.local_aggregate_density - p.target_pressure * p.local_aggregate_density
        )

    def pr_active_to_inactive(self, p: LysosomePerception) -> float:
        """Active lysosomes quiet down when there is little work to do."""

        task_pressure = 1.0 if p.task is not None else 0.0
        return clamp(
            (1.0 - task_pressure) * (1.0 - p.target_pressure) * (1.0 - p.local_aggregate_density)
        )

    def _sample_with_stay(
            self,
            outgoing: dict[LysosomeState, float]
    ) -> LysosomeState:
        """Sample one outgoing transition and keep remaining probability as stay."""

        probabilities = {
            state: clamp(probability)
            for state, probability in outgoing.items()
        }

        total = sum(probabilities.values())
        if total > 1.0:
            probabilities = {
                state: probability / total
                for state, probability in probabilities.items()
            }
            total = 1.0
        probabilities[self.state] = 1.0 - total
        draw = self.rng.random()
        cumulative = 0.0
        for state, probability in probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return state
        return self.state

    def _target_pressure(
            self,
            habitat,
            targets: List[AdaptiveAgent]
    ) -> float:
        """Normalize the current degradation workload by local agent count."""

        total_agents = len(habitat.grid.agent_registry)
        if total_agents <= 1:
            return 0.0
        return clamp(len(targets) / (total_agents - 1))

    def _select_target(self, habitat):
        """Claim one unassigned degradation target from the neuron buffer."""

        targets = list(habitat.available_degradation_targets())
        if not targets:
            self.target = None
            return
        target = self.rng.choice(targets)
        if habitat.assign_degradation_target(self, target):
            self.target = target

    def _degrade_target(self, habitat):
        """Attempt to degrade the assigned target.
        Work can span multiple ticks. Once enough work has accumulated, the
        target has mutually exclusive outcomes: overwhelm, successful cleanup,
        or failed attempt with the target returned to the neuron's pool.
        """
        target = habitat.target_for(self)
        if target is None:
            self.target = None
            self._reset_degradation_work()
            return
        if habitat.position_of(target) is None:
            habitat.clear_degradation_assignment(self)
            self.target = None
            self._reset_degradation_work()
            return
        if self._is_lewy_body(target):
            self._become_overwhelmed(habitat, target)
            return
        if not self._advance_degradation_work(target):
            self.target = target
            return
        self._resolve_degradation_attempt(habitat, target)

    def _resolve_degradation_attempt(self, habitat, target: AdaptiveAgent):
        """Resolve overwhelm, success or failure after work is complete."""
        overwhelm_probability = self.pr_overwhelmed_by_target(target)
        success_probability = min(self.pr_degradation_success(target), 1.0 - overwhelm_probability)
        draw = self.rng.random()
        if draw < overwhelm_probability:
            self._become_overwhelmed(habitat, target)
            return
        if draw < overwhelm_probability + success_probability:
            self._complete_degradation(habitat, target)
            return
        self._fail_degradation(habitat, target)

    def _complete_degradation(self, habitat, target: AdaptiveAgent):
        """Apply the successful cleanup effect for the target type."""
        if isinstance(target, AlphaAggregate):
            habitat.remove_agent(target)
        elif isinstance(target, Mitochondrion):
            target.repair_by_lysosome()
            habitat.unregister_degradation_target(target)
        elif isinstance(target, AlphaSynuclein):
            target.mark_cleared()
            habitat.unregister_degradation_target(target)
        else:
            habitat.remove_agent(target)
        self.target = None
        self._reset_degradation_work()

    def _fail_degradation(self, habitat, target: AdaptiveAgent):
        """Return a failed target to the neuron's available target pool."""
        habitat.clear_degradation_assignment(self, requeue_target=True)
        self.target = None
        self._reset_degradation_work()

    def _advance_degradation_work(self, target: AdaptiveAgent) -> bool:
        """Advance multi-tick cleanup and report whether resolution is due."""
        self._ensure_degradation_work(target)
        if self.degradation_ticks_remaining > 1:
            self.degradation_ticks_remaining -= 1
            return False
        self.degradation_ticks_remaining = 0
        return True

    def _ensure_degradation_work(self, target: AdaptiveAgent):
        """Initialize work duration when the lysosome starts a new target."""
        if self._work_target is target:
            return
        self._work_target = target
        self.degradation_ticks_remaining = self.degradation_ticks_required(target)

    def _reset_degradation_work(self):
        """Clear current multi-tick degradation bookkeeping."""
        self._work_target = None
        self.degradation_ticks_remaining = 0

    def degradation_ticks_required(self, target: AdaptiveAgent) -> int:
        """Ticks needed before a degradation attempt can be resolved."""
        if isinstance(target, AlphaAggregate):
            return self.cfg.aggregate_degradation_ticks_base + self.cfg.aggregate_degradation_ticks_per_member * max(0, target.size - 1)
        if isinstance(target, Mitochondrion):
            return self.cfg.mitochondrion_repair_ticks
        if isinstance(target, AlphaSynuclein):
            return self.cfg.protein_degradation_ticks
        return 1

    def _scan(self, habitat):
        """Move locally while searching for new degradation work."""
        position = habitat.position_of(self)
        if position is None:
            return
        candidate_points = list(habitat.neighbor_points(position, self.cfg.move_radius, True))
        if not candidate_points:
            return
        new_position = self.rng.choice(candidate_points)
        habitat.move_to(self, new_position)

    def pr_degradation_success(self, target: AdaptiveAgent) -> float:
        """Probability that one degradation attempt clears the assigned target."""
        if isinstance(target, AlphaAggregate):
            return clamp(
                self.cfg.aggregate_degradation_probability_base
                + self.cfg.aggregate_degradation_probability_per_member * target.size
            )
        if isinstance(target, Mitochondrion):
            return self.cfg.mitochondrion_repair_probability
        return self.cfg.base_degradation_probability

    def pr_overwhelmed_by_target(self, target: AdaptiveAgent) -> float:
        """Probability that the target disables the lysosome this attempt."""
        if self._is_lewy_body(target):
            return 1.0
        if isinstance(target, AlphaAggregate):
            return clamp(
                self.cfg.aggregate_overwhelm_probability_base
                + self.cfg.aggregate_overwhelm_probability_per_member * target.size
            )
        return 0.0

    def _is_lewy_body(self, target: AdaptiveAgent) -> bool:
        """Return True when the target is a Lewy body aggregate."""
        return isinstance(target, AlphaAggregate) and target.state == AggregateState.LEWY_BODY

    def _become_overwhelmed(self, habitat, target: AdaptiveAgent):
        """Mark this lysosome as non-functional after Lewy body contact."""
        self.state = LysosomeState.OVERWHELMED
        self.pending_action = LysosomeAction.IDLE
        self.target = None
        self._reset_degradation_work()
        habitat.clear_degradation_assignment(self, requeue_target=True)
