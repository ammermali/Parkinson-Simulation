from dataclasses import dataclass, field
from typing import Iterable, Optional

from repast4py.space import DiscretePoint

from src.simulation.agents.adaptiveagent import (
    AdaptiveAgent,
    AdaptiveAgentAction,
    AdaptiveAgentPerception,
    AdaptiveAgentState,
)


class AggregateState(str, AdaptiveAgentState):
    """State of an aggregate represented as one simulation entity."""
    OLIGOMER = "Oligomer"
    LEWY_BODY = "LewyBody"


class AggregateAction(str, AdaptiveAgentAction):
    """Aggregates are immobile for now; registry handles growth and merging."""
    STAY = "stay"


@dataclass(frozen=True)
class AggregatePerception(AdaptiveAgentPerception):
    """Minimal perception kept for consistency with the adaptive agent API."""
    position: Optional[DiscretePoint]


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
    owner_neuron: Optional[object] = None

    def __post_init__(self):
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
        if self.state == AggregateState.LEWY_BODY:
            return 1.0
        return 0.75

    @property
    def is_lewy_body(self) -> bool:
        """Whether this aggregate has matured into a Lewy body."""
        return self.state == AggregateState.LEWY_BODY

    @property
    def can_recruit(self) -> bool:
        """Whether this aggregate can recruit other pathology this tick."""
        return self.state == AggregateState.LEWY_BODY or self.wants_lewy_body_maturation

    def add_members(self, member_ids: Iterable[int]):
        """Add protein identifiers absorbed into this aggregate."""
        self.member_ids.update(member_ids)

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
        # The registry computes oligomer maturation probability; Lewy bodies
        # always remain recruitment-competent.
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
