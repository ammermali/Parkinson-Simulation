from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, TYPE_CHECKING
from src.simulation.agents.structure import AggregateAction, AggregatePerception, AggregateState, AdaptiveAgent
if TYPE_CHECKING:
    from src.simulation.agents.neuron import Neuron


@dataclass(eq=False)
class AlphaAggregate(AdaptiveAgent):
    """Simulation agent representing an alpha-synuclein aggregate.
    Free AlphaSynuclein agents are removed from the active grid when they join
    an aggregate. From that point onward this object is the simulation-visible
    unit for movement-free pathology, lysosomal targeting, local aggregate
    density and Lewy body maturation.
    """
    local_id: int
    rank: int
    type_id: int
    aggregate_id: int
    member_ids: set = field(default_factory=set)
    state: AggregateState = AggregateState.OLIGOMER
    owner_neuron: Optional[Neuron] = None
    member_agents: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        """Initialize runtime fields after dataclass construction."""
        AdaptiveAgent.__init__(self, self.local_id, self.type_id, self.rank)
        self.last_perception: Optional[AggregatePerception] = None
        self.pending_action: AggregateAction = AggregateAction.STAY
        self.wants_lewy_body_maturation: bool = self.state == AggregateState.LEWY_BODY

    @property
    def size(self) -> int:
        """Number of alpha-synuclein proteins represented by this aggregate."""
        return len(self.member_ids)

    @property
    def aggregate_weight(self) -> float:
        """Contribution to local aggregate density and neuron alpha load."""
        size_bonus = max(0, self.size - 1)
        if self.state == AggregateState.LEWY_BODY:
            return min(2.0, 1.0 + 0.10 * size_bonus)
        return min(1.25, 0.75 + 0.05 * size_bonus)

    @property
    def is_lewy_body(self) -> bool:
        """Whether this aggregate has matured into a Lewy body."""
        return self.state == AggregateState.LEWY_BODY

    @property
    def can_recruit(self) -> bool:
        """Whether this aggregate can recruit other pathology this tick."""
        return self.state == AggregateState.LEWY_BODY or self.wants_lewy_body_maturation

    def add_member(self, member_id, member_agent=None):
        """Add one represented protein by id and, when available, object."""
        self.member_ids.add(member_id)
        if member_agent is not None:
            self.member_agents.add(member_agent)

    def add_members(self, member_ids: Iterable[int], member_agents: Optional[Iterable[object]] = None):
        """Add protein identifiers absorbed into this aggregate."""
        self.member_ids.update(member_ids)
        if member_agents is not None:
            self.member_agents.update(member_agents)

    def mature_to_lewy_body(self):
        """Promote the aggregate to Lewy body state."""
        self.state = AggregateState.LEWY_BODY
        self.wants_lewy_body_maturation = True

    def see(self, model) -> AggregatePerception:
        """Record the aggregate position in its owning neuron."""
        position = None if self.owner_neuron is None else self.owner_neuron.position_of(self)
        perception = AggregatePerception(position=position)
        self.last_perception = perception
        return perception

    def next(self) -> AggregateState:
        """Keep Lewy bodies recruitment-competent between registry passes."""
        self.wants_lewy_body_maturation = self.state == AggregateState.LEWY_BODY
        return self.state

    def action(self) -> AggregateAction:
        """Aggregates do not currently move or choose active behavior."""
        self.pending_action = AggregateAction.STAY
        return self.pending_action

    def do(self, model):
        """Expose this aggregate as a lysosomal target.
        Growth and merging remain centralized in AggregateRegistry; this method
        only keeps the neuron's degradation buffer aware that the aggregate is a
        possible cleanup target."""
        if self.owner_neuron is not None:
            self.owner_neuron.register_degradation_target(self)
        return

    def release_to_environment(self):
        """Mark the aggregate as extracellular and detached from a neuron."""
        self.owner_neuron = None
        self.pending_action = AggregateAction.STAY

    def absorb_into_neuron(self, neuron):
        """Mark the aggregate as intracellular inside a new neuron."""
        self.owner_neuron = neuron
        self.pending_action = AggregateAction.STAY
