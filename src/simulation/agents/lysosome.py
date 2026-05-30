from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from repast4py.space import DiscretePoint
from typing import Optional, List
from src.simulation.agents.aggregate import AlphaAggregate, AggregateState
from src.simulation.agents.alphasynuclein import AlphaSynuclein
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

class Lysosome(AdaptiveAgent):
    """Intracellular degradation agent coordinated by neuron target buffers.
    A lysosome does not discover all degradable agents directly. Instead,
    damaged proteins, aggregates and organelles register in the owning neuron;
    the lysosome claims one available target and repeatedly attempts to degrade
    it. Aggregate degradation becomes harder with aggregate size, while a Lewy
    body overwhelms the lysosome and permanently stops its activity.
    """
    def __init__(
            self,
            local_id,
            rank,
            type_id,
            owner_neuron,
            perception_radius: int = 1,
            move_radius: int = 1,
            base_degradation_probability: float = 0.8,
            aggregate_size_penalty: float = 0.25
    ):
        super().__init__(local_id, type_id, rank)

        self.state = LysosomeState.INACTIVE
        self.owner_neuron = owner_neuron
        self.perception_radius = perception_radius
        self.move_radius = move_radius
        self.base_degradation_probability = clamp(base_degradation_probability)
        self.aggregate_size_penalty = max(0.0, aggregate_size_penalty)
        self.target: Optional[AdaptiveAgent] = None
        self.last_perception: Optional[LysosomePerception] = None
        self.pending_action: Optional[LysosomeAction] = None
        self.last_transition: tuple[LysosomeState, LysosomeState] = (
            self.state,
            self.state
        )
        self.rng = RNG()

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
            radius=self.perception_radius,
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
                LysosomeState.INACTIVE: self.pr_active_to_inactive(p),
                LysosomeState.OVERWHELMED: self.pr_active_to_overwhelmed(p)
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

    def pr_overwhelmed_to_active(self, p: LysosomePerception) -> float:
        """Overwhelmed lysosomes are terminally non-functional for now."""

        return 0.0

    def pr_active_to_overwhelmed(self, p: LysosomePerception) -> float:
        """Background overwhelm risk from crowded pathological neighborhoods."""

        return clamp(
            p.target_pressure * p.local_aggregate_density
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
        Lewy bodies are treated as too large and organized for the lysosome to
        clear; contact with one overwhelms the lysosome. Other aggregates use a
        size-sensitive success probability."""
        target = habitat.target_for(self)
        if target is None:
            self.target = None
            return
        if habitat.position_of(target) is None:
            habitat.clear_degradation_assignment(self)
            self.target = None
            return
        if self._is_lewy_body(target):
            self._become_overwhelmed(habitat, target)
            return
        if self.rng.random() > self.pr_degradation_success(target):
            self.target = target
            return
        if isinstance(target, AlphaAggregate):
            habitat.remove_agent(target)
            self.target = None
            return
        if isinstance(target, AlphaSynuclein):
            target.mark_cleared()
            habitat.clear_degradation_assignment(self)
        # TODO add cases for mitochondrion
        else:
            habitat.remove_agent(target)
        self.target = None

    def _scan(self, habitat):
        """Move locally while searching for new degradation work."""
        position = habitat.position_of(self)
        if position is None:
            return
        candidate_points = list(habitat.neighbor_points(position, self.move_radius, True))
        if not candidate_points:
            return
        new_position = self.rng.choice(candidate_points)
        habitat.move_to(self, new_position)

    def pr_degradation_success(self, target: AdaptiveAgent) -> float:
        """Probability that one degradation attempt clears the assigned target."""
        if isinstance(target, AlphaAggregate):
            size_penalty = 1.0 + self.aggregate_size_penalty * max(0, target.size - 1)
            return clamp(self.base_degradation_probability / size_penalty)
        return self.base_degradation_probability

    def _is_lewy_body(self, target: AdaptiveAgent) -> bool:
        return isinstance(target, AlphaAggregate) and target.state == AggregateState.LEWY_BODY

    def _become_overwhelmed(self, habitat, target: AdaptiveAgent):
        """Mark this lysosome as non-functional after Lewy body contact."""
        self.state = LysosomeState.OVERWHELMED
        self.pending_action = LysosomeAction.IDLE
        self.target = None
        habitat.clear_degradation_assignment(self, requeue_target=True)
