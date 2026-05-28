from typing import Optional

from src.simulation.agents.adaptiveagent import AdaptiveAgent
from src.simulation.agents.alphasynuclein import AlphaSynucleinState


class AggregateRegistry:
    def __init__(self):
        self._next_id: int = 0
        self._members: dict[int, set[AdaptiveAgent]] = {}

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def try_create_pair(
            self,
            first: AdaptiveAgent,
            second: AdaptiveAgent
    ) -> Optional[int]:
        if getattr(first, "aggregate_id", None) is not None:
            return None
        if getattr(second, "aggregate_id", None) is not None:
            return None
        if getattr(first, "state", None) != AlphaSynucleinState.MISFOLDED or getattr(second, "state", None) != AlphaSynucleinState.MISFOLDED:
            return None
        aggregate_id = self.new_id()
        self._members[aggregate_id] = {first, second}
        first.aggregate_id = aggregate_id
        second.aggregate_id = aggregate_id
        return aggregate_id

    def try_add_member(
            self,
            aggregate_id: int,
            member: AdaptiveAgent
    ) -> bool:
        if aggregate_id not in self._members:
            return False
        if getattr(member, "aggregate_id", None) is not None:
            return False
        if getattr(member, "state", None) != AlphaSynucleinState.MISFOLDED:
            return False
        self._members[aggregate_id].add(member)
        member.aggregate_id = aggregate_id
        member.state = AlphaSynucleinState.OLIGOMER
        return True

    def members(self, aggregate_id: int) -> set[AdaptiveAgent]:
        return set(self._members.get(aggregate_id, ()))

    def size(self, aggregate_id: int) -> int:
        return len(self._members.get(aggregate_id, set()))

    def remove(self, agent: AdaptiveAgent):
        aggregate_id = getattr(agent, "aggregate_id", None)
        if aggregate_id is None:
            return
        members = self._members.get(aggregate_id)
        if members is not None:
            members.remove(agent)
            if not members:
                del self._members[aggregate_id]
        agent.aggregate_id = None
