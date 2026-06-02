from enum import Enum
from typing import Optional
from src.simulation.agents.structure import AlphaSynucleinState, AlphaSynucleinPerception, AlphaSynucleinAction, AlphaSynucleinConfig, AdaptiveAgent
from src.simulation.agents.aggregate import AlphaAggregate
from src.simulation.utils import RNG, clamp
from src.simulation.logger.causal_trace_logger import bind_causal_logger, causal_logger_from, uid_of

class AlphaSynucleinCompartment(str, Enum):
    """The habitat where the protein currently exists."""
    INTRACELLULAR = "Intracellular"
    EXTRACELLULAR = "Extracellular"

class AlphaSynuclein(AdaptiveAgent):
    """Single alpha-synuclein protein agent.
    This class models only the free protein lifecycle:
    monomer -> misfolded -> willingness to join an aggregate. The actual
    conversion from free proteins into an AggregateAgent is deliberately owned
    by AggregateRegistry so that aggregation remains a collective process.
    """
    aggregate_id: Optional[int] = None
    def __init__(self, local_id: int, rank: int, type_id: int, config: AlphaSynucleinConfig, compartment: AlphaSynucleinCompartment, owner_neuron: Optional[AdaptiveAgent] = None):
        super().__init__(local_id, type_id, rank)
        self.state: AlphaSynucleinState = AlphaSynucleinState.MONOMER
        self.cfg = config
        self.compartment = compartment
        self.owner_neuron = owner_neuron
        self.aggregate_id: Optional[int] = None
        self.last_perception: Optional[AlphaSynucleinPerception] = None
        self.pending_action: Optional[AlphaSynucleinAction] = None
        self.rng = RNG
        # Intentions are reset in next() and consumed by AggregateRegistry.
        self.wants_oligomerization: bool = False
        self.misfolded_ticks: int = 0

    @property
    def is_free(self) -> bool:
        """Whether this protein is active as an individual grid agent."""
        return self.aggregate_id is None and self.state not in (AlphaSynucleinState.CLEARED, AlphaSynucleinState.LEWY_BODY)

    @property
    def can_seed_oligomerization(self) -> bool:
        """Whether the registry may use this protein to form an oligomer."""
        return (self.is_free and self.state == AlphaSynucleinState.MISFOLDED and self.wants_oligomerization)

    @property
    def aggregate_weight(self) -> float:
        """Contribution to local aggregate load when the protein is free.
        Once proteins are absorbed into AggregateAgent they are removed from the active grid, so these weights mostly matter before absorption or in unit tests with direct protein placement."""
        if self.state == AlphaSynucleinState.MISFOLDED:
            return 0.25
        if self.state == AlphaSynucleinState.OLIGOMER:
            return 0.75
        if self.state == AlphaSynucleinState.LEWY_BODY:
            return 1.0
        return 0.0

    def see(self, model) -> AlphaSynucleinPerception:
        """Read local stress and nearby agents from the current habitat."""
        bind_causal_logger(self, model)
        habitat = self._habitat(model)
        position = habitat.position_of(self)
        if self.compartment == AlphaSynucleinCompartment.EXTRACELLULAR:
            perception = AlphaSynucleinPerception(
                position=position,
                oxidative_stress=0.0,
                local_aggregate_density=0.0,
                neighbors=[]
            )
            self.last_perception = perception
            return perception
        if position is None:
            perception = AlphaSynucleinPerception(
                position=None,
                oxidative_stress=0.0,
                local_aggregate_density=0.0,
                neighbors=[]
            )
            self.last_perception = perception
            return perception

        perception = AlphaSynucleinPerception(
            position=position,
            oxidative_stress=habitat.oxidative_stress_at(position),
            local_aggregate_density=habitat.local_aggregate_density_at(
                position=position,
                radius=self.cfg.perception_radius,
                include_center=True
            ),
            neighbors=[
                agent
                for agent in habitat.agents_in_radius(
                    center=position,
                    radius=self.cfg.perception_radius,
                )
                if agent is not self
            ]
        )
        self.last_perception = perception
        return perception

    def next(self) -> AlphaSynucleinState:
        """Update free-protein state and aggregation intention.
        Misfolding is protein-local and depends on oxidative stress.
        Aggregation itself is not performed here; next() only sets wants_oligomerization so the registry can resolve all candidates in one collective pass.
        """
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        self.wants_oligomerization = False
        if not self.is_free or self.compartment == AlphaSynucleinCompartment.EXTRACELLULAR:
            return self.state
        p = self.last_perception
        draw = self.rng.random()
        if self.state == AlphaSynucleinState.MONOMER:
            probability = self.pr_misfolding()
            if draw < probability:
                self.state = AlphaSynucleinState.MISFOLDED
                self.misfolded_ticks = 0
                logger = causal_logger_from(self)
                if logger is not None:
                    source = logger.internal_field_node(
                        self.owner_neuron,
                        "oxidative_stress",
                        "1_perception",
                        p.oxidative_stress
                    )
                    logger.threshold_trigger(
                        source,
                        self,
                        self.state,
                        "alpha_misfolding_pressure",
                        "ALPHA_MISFOLDING",
                        "basal, oxidative_stress, or aggregate-seeded misfolding pressure",
                        owner=self.owner_neuron,
                        compartment=self.compartment
                    )
                    logger.state_transition(
                        self,
                        old_state,
                        self.state,
                        "alpha_misfolding",
                        rule_id="ALPHA_MISFOLDING",
                        probability=probability,
                        rng_value=draw,
                        owner=self.owner_neuron,
                        compartment=self.compartment
                    )
        elif self.state == AlphaSynucleinState.MISFOLDED:
            if self.misfolded_ticks < self.cfg.min_misfolded_ticks_before_oligomerization:
                probability = 0.0
            else:
                probability = self.pr_oligomerization()
            self.wants_oligomerization = draw < probability
            if self.wants_oligomerization:
                logger = causal_logger_from(self)
                if logger is not None:
                    source = logger.internal_field_node(
                        self.owner_neuron,
                        "local_aggregate_density",
                        "1_perception",
                        p.local_aggregate_density
                    )
                    action = logger.action_node(
                        self,
                        "wants_oligomerization",
                        "3_action_selection",
                        owner=self.owner_neuron,
                        compartment=self.compartment
                    )
                    logger.edge(
                        source,
                        action,
                        "action_selection",
                        "alpha_oligomerization_intention",
                        rule_id="ALPHA_OLIGOMERIZATION_INTENTION",
                        probability=probability,
                        rng_value=draw,
                        outcome="selected",
                        compartment=self.compartment,
                        owner_uid=uid_of(self.owner_neuron)
                    )
            self.misfolded_ticks += 1
        return self.state

    def action(self) -> AlphaSynucleinAction:
        """Free proteins move; cleared or already aggregated proteins stay."""
        if not self.is_free or self.compartment == AlphaSynucleinCompartment.EXTRACELLULAR:
            self.pending_action = AlphaSynucleinAction.STAY
        else:
            self.pending_action = AlphaSynucleinAction.MOVE
        return self.pending_action

    def do(self, model):
        """Move a free protein within its habitat.
        Degradable misfolded proteins register with the neuron, but aggregate
        formation is intentionally absent from this method.
        """
        if self.compartment == AlphaSynucleinCompartment.EXTRACELLULAR:
            return
        if self.pending_action is None:
            raise RuntimeError()
        habitat = self._habitat(model)
        self._register_if_degradable(habitat)
        if self.pending_action == AlphaSynucleinAction.STAY:
            return
        draw = self.rng.random()
        if draw > self.cfg.move_probability:
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
        new_position = self.rng.choice(candidates)
        habitat.move_to(self, new_position)

    def join_aggregate(self, aggregate_id: int, state: AlphaSynucleinState):
        """Mark this protein as biologically contained in an aggregate."""
        self.aggregate_id = aggregate_id
        self.state = state
        self.pending_action = AlphaSynucleinAction.STAY
        self.wants_oligomerization = False
        self.misfolded_ticks = 0

    def mark_cleared(self):
        """Mark a free protein as cleared by degradation machinery."""
        self.aggregate_id = None
        self.state = AlphaSynucleinState.CLEARED
        self.pending_action = AlphaSynucleinAction.STAY
        self.wants_oligomerization = False
        self.misfolded_ticks = 0

    def release_to_environment(self):
        """Freeze this protein as extracellular pathology."""
        self.compartment = AlphaSynucleinCompartment.EXTRACELLULAR
        self.owner_neuron = None
        self.pending_action = AlphaSynucleinAction.STAY
        self.wants_oligomerization = False

    def absorb_into_neuron(self, neuron):
        """Move this extracellular protein into a neuron's internal habitat."""
        self.compartment = AlphaSynucleinCompartment.INTRACELLULAR
        self.owner_neuron = neuron
        self.pending_action = AlphaSynucleinAction.STAY
        self.wants_oligomerization = False

    def _register_if_degradable(self, habitat):
        """Expose free misfolded intracellular proteins to lysosomes."""
        if self.compartment != AlphaSynucleinCompartment.INTRACELLULAR:
            return
        if self.state == AlphaSynucleinState.MISFOLDED and self.is_free:
            habitat.register_degradation_target(self)

    def _habitat(self, model):
        """Return the habitat matching the current compartment."""

        if self.compartment == AlphaSynucleinCompartment.INTRACELLULAR:
            if self.owner_neuron is None:
                raise RuntimeError("Intracellular AlphaSynuclein requires owner_neuron.")
            return self.owner_neuron
        if self.compartment == AlphaSynucleinCompartment.EXTRACELLULAR:
            return model.environment
        raise RuntimeError("Unknown AlphaSynuclein compartment.")

    def _neighbor_alpha_density(self) -> float:
        """Density of neighboring alpha-synuclein-like agents."""
        if self.last_perception is None:
            return 0.0
        agents = list(self.last_perception.neighbors)
        if not agents:
            return 0.0
        alpha_count = sum(1 for agent in agents if self._same_type(agent))
        return clamp(alpha_count / len(agents))

    def _neighbor_aggregate_density(self) -> float:
        """Weighted local aggregate pathology density around this protein."""
        if self.last_perception is None:
            return 0.0
        agents = list(self.last_perception.neighbors)
        if not agents:
            return 0.0
        aggregate_score = sum(self._aggregate_weight(agent) for agent in agents)
        return clamp(aggregate_score / len(agents))

    def _same_type(self, agent: AdaptiveAgent) -> bool:
        """Return True when another agent has the same Repast type id."""
        try:
            return agent.ptype == self.ptype
        except AttributeError:
            return False

    def _aggregate_weight(self, agent: AdaptiveAgent) -> float:
        """Return an aggregate-like weight for local probability estimates."""
        if isinstance(agent, (AlphaSynuclein, AlphaAggregate)):
            return agent.aggregate_weight
        return 0.0

    def pr_oligomerization(self) -> float:
        """Probability that a misfolded protein asks to join an aggregate."""
        alpha_density = self._neighbor_alpha_density()
        aggregate_density = self._neighbor_aggregate_density()
        return clamp(self.cfg.oligomerization_probability_scale * (alpha_density * 0.3 + aggregate_density * 0.7))

    def pr_misfolding(self) -> float:
        """Probability that a monomer misfolds during this tick."""
        if self.last_perception is None:
            return 0.0
        p = self.last_perception
        oxidative_component = self.cfg.oxidative_misfolding_weight * self._oxidative_pressure(p.oxidative_stress)
        seeded_component = (self.cfg.aggregate_seeded_misfold_weight * p.local_aggregate_density)
        return clamp(self.cfg.basal_misfold_probability + oxidative_component + seeded_component)

    def _oxidative_pressure(self, oxidative_stress: float) -> float:
        """Normalize oxidative stress above the sampled misfolding threshold."""
        threshold = self.cfg.oxidative_stress_high_threshold
        if oxidative_stress < threshold:
            return 0.0
        return clamp((oxidative_stress - threshold) / max(1e-9, 1.0 - threshold))
