from typing import Iterable, Optional
from repast4py.space import DiscretePoint


class GridHabitatMixin:
    def add_agent(self, agent, point: DiscretePoint):
        return self.grid.add_agent(agent, point)

    def remove_agent(self, agent):
        return self.grid.remove_agent(agent)

    def position_of(self, agent) -> Optional[DiscretePoint]:
        return self.grid.position_of(agent)

    def move_to(self, agent, point: DiscretePoint) -> Optional[DiscretePoint]:
        return self.grid.move_to(agent, point)

    def agents_at(self, point: DiscretePoint) -> list:
        return self.grid.agents_at(point)

    def agents_in_radius(self, center: DiscretePoint, radius: int = 1, include_center: bool = False) -> Iterable:
        return self.grid.agents_in_radius(center, radius, include_center)

    def count_agents_in_radius(self, center: DiscretePoint, radius: int = 1, agent_type: Optional[int] = None, include_center: bool = False) -> int:
        return self.grid.count_agents_in_radius(center, radius, agent_type, include_center)

    def neighbor_points(self, center: DiscretePoint, radius: int = 1, include_center: bool = True) -> Iterable[DiscretePoint]:
        return self.grid.neighbor_points(center, radius, include_center)

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: Optional[int] = None, include_center: bool = True) -> float:
        return self.grid.density_of_type(center, radius, agent_type, include_center)
