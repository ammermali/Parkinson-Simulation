from typing import Iterable, Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent
from src.simulation.agents.aggregate import AlphaAggregate, AggregateState
from src.simulation.agents.alphasynuclein import AlphaSynuclein, AlphaSynucleinState
from src.simulation.utils import RNG, clamp

class AggregateRegistry:
    """Centralization of Alpha-Synuclein lifecycle.
    AlphaSynuclein agents only decide whether they want to oligomerize. This
    registry resolves the collective spatial event, like aggregating agents,
    absorbing free misfolded proteins, merging aggregates and maturing oligomers
    into Lewy bodies.
    """

    def __init__(self, lewy_body_size_threshold: int = 8, aggregate_type_id: Optional[int] = None, rng: Optional[RNG] = None):
        self._next_id: int = 0
        self._aggregates: dict[int, AlphaAggregate] = {}
        self._members: dict[int, set[AlphaSynuclein]] = {}
        self.lewy_body_size_threshold = max(1, lewy_body_size_threshold)
        self.aggregate_type_id = aggregate_type_id
        self.rng = rng if rng is not None else RNG()

    def process(self, habitat):
        """Process one aggregation phase for all intracellular agents."""

        for point, agents in self._agents_by_cell(habitat):
            free_candidates = [
                agent
                for agent in agents
                if isinstance(agent, AlphaSynuclein) and agent.can_seed_oligomerization
            ]
            aggregates = [
                agent
                for agent in agents
                if isinstance(agent, AlphaAggregate)
            ]
            self._process_cell(habitat, point, free_candidates, aggregates)

    def aggregates(self) -> list[AlphaAggregate]:
        return list(self._aggregates.values())

    def aggregate_for(self, aggregate_id: int) -> Optional[AlphaAggregate]:
        return self._aggregates.get(aggregate_id)

    def members(self, aggregate_id: int) -> set[AlphaSynuclein]:
        return set(self._members.get(aggregate_id, set()))

    def size(self, aggregate_id: int) -> int:
        aggregate = self._aggregates.get(aggregate_id)
        if aggregate is None:
            return 0
        return aggregate.size

    def create_aggregate(self, habitat, point: DiscretePoint, members: Iterable[AlphaSynuclein], state: AggregateState = AggregateState.OLIGOMER) -> Optional[AlphaAggregate]:
        """Create one AggregateAgent from two or more free proteins."""

        members = [
            member
            for member in members
            if member.can_seed_oligomerization
        ]
        if len(members) < 2:
            return None
        aggregate_id = self.new_id()
        first = members[0]
        aggregate = AlphaAggregate(
            local_id=aggregate_id,
            rank=self._agent_rank(first),
            type_id=self.aggregate_type_id if self.aggregate_type_id is not None else self._agent_type_id(first),
            aggregate_id=aggregate_id,
            member_ids=set(),
            state=state,
            owner_neuron=habitat,
        )
        self._aggregates[aggregate_id] = aggregate
        self._members[aggregate_id] = set()
        habitat.add_agent(aggregate, point)
        for member in members:
            self.add_alpha_to_aggregate(habitat, aggregate, member)
        if state == AggregateState.LEWY_BODY:
            self._set_members_state(aggregate, AlphaSynucleinState.LEWY_BODY)
        self._register_degradation_target(habitat, aggregate)
        return aggregate

    def add_alpha_to_aggregate(self, habitat, aggregate: AlphaAggregate, alpha: AlphaSynuclein) -> bool:
        """Absorb a free misfolded protein into an existing aggregate."""
        if not alpha.can_seed_oligomerization:
            return False
        if aggregate.aggregate_id not in self._aggregates:
            return False
        point = habitat.position_of(alpha)
        if point is not None:
            habitat.remove_agent(alpha)
        self._members[aggregate.aggregate_id].add(alpha)
        aggregate.add_members({self._member_id(alpha)})
        member_state = (
            AlphaSynucleinState.LEWY_BODY
            if aggregate.state == AggregateState.LEWY_BODY
            else AlphaSynucleinState.OLIGOMER
        )
        alpha.join_aggregate(aggregate.aggregate_id, member_state)
        return True

    def merge_aggregates(self, habitat, target: AlphaAggregate, source: AlphaAggregate) -> bool:
        """Merge source aggregate into target aggregate and remove source."""
        if target.aggregate_id == source.aggregate_id:
            return False
        if target.aggregate_id not in self._aggregates:
            return False
        if source.aggregate_id not in self._aggregates:
            return False
        source_members = self._members.get(source.aggregate_id, set())
        target.add_members(source.member_ids)
        self._members[target.aggregate_id].update(source_members)
        self._set_members_state(
            target,
            AlphaSynucleinState.LEWY_BODY
            if target.state == AggregateState.LEWY_BODY
            else AlphaSynucleinState.OLIGOMER,
        )
        self._untrack_degradation_target(habitat, source)
        habitat.grid.remove_agent(source)
        del self._aggregates[source.aggregate_id]
        del self._members[source.aggregate_id]
        self._register_degradation_target(habitat, target)
        return True

    def mature_to_lewy_body(self, aggregate: AlphaAggregate):
        """Promote an oligomer and every member protein to LewyBody."""
        aggregate.mature_to_lewy_body()
        self._set_members_state(aggregate, AlphaSynucleinState.LEWY_BODY)

    def remove(self, agent: AdaptiveAgent):
        """Remove aggregate bookkeeping for a cleared aggregate or member."""
        if isinstance(agent, AlphaAggregate):
            self._remove_aggregate(agent)
            return
        if isinstance(agent, AlphaSynuclein) and agent.aggregate_id is not None:
            self._remove_member(agent)

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _process_cell(
            self,
            habitat,
            point: DiscretePoint,
            free_candidates: list[AlphaSynuclein],
            aggregates: list[AlphaAggregate],
    ):
        self._refresh_aggregate_intentions(aggregates)

        lewy_bodies = [
            aggregate
            for aggregate in aggregates
            if aggregate.state == AggregateState.LEWY_BODY
        ]
        if lewy_bodies:
            seed = self._largest(lewy_bodies)
            for aggregate in aggregates:
                if aggregate is not seed and aggregate.can_recruit:
                    self.merge_aggregates(habitat, seed, aggregate)
            for alpha in list(free_candidates):
                self.add_alpha_to_aggregate(habitat, seed, alpha)
            return

        recruiting_oligomers = [
            aggregate
            for aggregate in aggregates
            if aggregate.state == AggregateState.OLIGOMER and aggregate.can_recruit
        ]
        if recruiting_oligomers:
            seed = self._largest(recruiting_oligomers)
            for aggregate in recruiting_oligomers:
                if aggregate is not seed:
                    self.merge_aggregates(habitat, seed, aggregate)
            for alpha in list(free_candidates):
                self.add_alpha_to_aggregate(habitat, seed, alpha)
            if seed.wants_lewy_body_maturation:
                self.mature_to_lewy_body(seed)
            return

        if len(free_candidates) >= 2:
            aggregate = self.create_aggregate(habitat, point, free_candidates)
            if aggregate is not None and self._should_mature(aggregate):
                self.mature_to_lewy_body(aggregate)

    def _agents_by_cell(self, habitat):
        cells: dict[tuple[int, int], tuple[DiscretePoint, list[AdaptiveAgent]]] = {}
        for agent in list(habitat.grid.agent_registry):
            point = habitat.position_of(agent)
            if point is None:
                continue
            key = (point.x, point.y)
            if key not in cells:
                cells[key] = (point, [])
            cells[key][1].append(agent)
        return cells.values()

    def _refresh_aggregate_intentions(self, aggregates: Iterable[AlphaAggregate]):
        for aggregate in aggregates:
            if aggregate.state == AggregateState.LEWY_BODY:
                aggregate.wants_lewy_body_maturation = True
            else:
                aggregate.wants_lewy_body_maturation = self._should_mature(aggregate)

    def _should_mature(self, aggregate: AlphaAggregate) -> bool:
        probability = clamp(aggregate.size / self.lewy_body_size_threshold)
        return self.rng.random() < probability

    def _largest(self, aggregates: Iterable[AlphaAggregate]) -> AlphaAggregate:
        return max(aggregates, key=lambda aggregate: aggregate.size)

    def _set_members_state(self, aggregate: AlphaAggregate, state: AlphaSynucleinState):
        for member in self._members.get(aggregate.aggregate_id, set()):
            member.join_aggregate(aggregate.aggregate_id, state)

    def _remove_aggregate(self, aggregate: AlphaAggregate):
        members = self._members.pop(aggregate.aggregate_id, set())
        for member in members:
            member.mark_cleared()
        self._aggregates.pop(aggregate.aggregate_id, None)

    def _remove_member(self, alpha: AlphaSynuclein):
        aggregate_id = alpha.aggregate_id
        if aggregate_id is None:
            return
        members = self._members.get(aggregate_id)
        if members is not None:
            members.discard(alpha)
        aggregate = self._aggregates.get(aggregate_id)
        if aggregate is not None:
            aggregate.member_ids.discard(self._member_id(alpha))
        alpha.aggregate_id = None

    def _member_id(self, alpha: AlphaSynuclein):
        try:
            return alpha.uid
        except AttributeError:
            return id(alpha)

    def _agent_rank(self, agent: AdaptiveAgent) -> int:
        """Read rank without assuming repast4py exposes it as an attribute."""

        for attr_name in ("rank", "agent_rank"):
            rank = getattr(agent, attr_name, None)
            if rank is not None and not callable(rank):
                return rank
        rank = self._uid_value(agent, index=2, attr_name="rank")
        if rank is not None:
            return rank
        return 0

    def _agent_type_id(self, agent: AdaptiveAgent) -> int:
        """Read type id from either the public ptype API or the Repast UID."""

        for attr_name in ("ptype", "type_id", "type"):
            ptype = getattr(agent, attr_name, None)
            if ptype is not None and not callable(ptype):
                return ptype
        ptype = self._uid_value(agent, index=1, attr_name="type")
        if ptype is not None:
            return ptype
        return 0

    def _uid_value(self, agent: AdaptiveAgent, index: int, attr_name: str):
        uid = getattr(agent, "uid", None)
        if uid is None:
            return None
        value = getattr(uid, attr_name, None)
        if value is not None and not callable(value):
            return value
        try:
            return uid[index]
        except (IndexError, KeyError, TypeError):
            return None

    def _register_degradation_target(self, habitat, aggregate: AlphaAggregate):
        register = getattr(habitat, "register_degradation_target", None)
        if callable(register):
            register(aggregate)

    def _untrack_degradation_target(self, habitat, aggregate: AlphaAggregate):
        unregister = getattr(habitat, "unregister_degradation_target", None)
        if callable(unregister):
            unregister(aggregate)
            return
        clear_target = getattr(habitat, "clear_assignments_for_target", None)
        if callable(clear_target):
            clear_target(aggregate)
