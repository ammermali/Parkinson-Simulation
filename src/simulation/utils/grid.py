from repast4py.space import DiscretePoint
from typing import Iterable, Optional
from src.simulation.utils.clamp import clamp

class LocalGrid:
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None, repast_grid=None):
        self._repast_grid = repast_grid
        self._offset_cache: dict[tuple[int, bool], list[tuple[int, int]]] = {}
        if repast_grid is not None:
            bounds = repast_grid.get_local_bounds()
            self.xmin = bounds.xmin
            self.xmax = bounds.xmin + bounds.xextent
            self.ymin = bounds.ymin
            self.ymax = bounds.ymin + bounds.yextent
        else:
            self.xmin = 0
            self.xmax = width
            self.ymin = 0
            self.ymax = height

        self.agent_registry: list = []
        self._locations: dict[object, DiscretePoint] = {}
        self._cells: dict[tuple[int, int], list] = {}

    @classmethod
    def from_repast_grid(cls, grid) -> "LocalGrid":
        return cls(repast_grid=grid)

    @property
    def is_repast_backed(self) -> bool:
        return self._repast_grid is not None

    # AGENTS

    def add_agent(self, agent, point: DiscretePoint):
        if self.is_repast_backed:
            raise NotImplementedError()
        if not self._inside_bounds(point.x, point.y):
            raise ValueError()
        if agent not in self.agent_registry:
            self.agent_registry.append(agent)
        self._locations[agent] = point
        self._cells.setdefault((point.x, point.y), []).append(agent)

    def remove_agent(self, agent):
        if self.is_repast_backed:
            raise NotImplementedError()
        point = self._locations.get(agent)
        if point is not None:
            key = (point.x, point.y)
            self._cells[key].remove(agent)
            if not self._cells[key]:
                del self._cells[key]
            del self._locations[agent]

    # GRID PRIMITIVES

    def position_of(self, agent) -> Optional[DiscretePoint]:
        if self.is_repast_backed:
            return self._repast_grid.get_location(agent)
        return self._locations.get(agent)

    def move_to(self, agent, point: DiscretePoint) -> Optional[DiscretePoint]:
        if not self._inside_bounds(point.x, point.y):
            return None
        old_point = self._locations.get(agent)
        if old_point is not None:
            old_key = (old_point.x, old_point.y)
            self._cells[old_key].remove(agent)
            if not self._cells[old_key]:
                del self._cells[old_key]
        self._locations[agent] = point
        self._cells.setdefault((point.x, point.y), []).append(agent)
        return point

    def agents_at(self, point: DiscretePoint) -> list:
        if self.is_repast_backed:
            return list(self._repast_grid.get_agents(point))
        return list(self._cells.get((point.x, point.y), []))

    # NEIGHBORHOOD
    def neighbor_points(self, center: DiscretePoint, radius: int = 1, include_center: bool = True) -> Iterable[DiscretePoint]:
        for dx, dy in self._get_offsets(radius, include_center):
            x = center.x + dx
            y = center.y + dy
            if self._inside_bounds(x, y):
                yield DiscretePoint(x, y)

    def agents_in_radius(self, center: DiscretePoint, radius: int = 1, include_center: bool = False) -> Iterable:
        for point in self.neighbor_points(center, radius, include_center):
            for agent in self.agents_at(point):
                yield agent

    def count_agents_in_radius(self, center: DiscretePoint, radius: int = 1, agent_type: Optional[int] = None, include_center: bool = False) -> int:
        total = 0
        for point in self.neighbor_points(center, radius, include_center):
            if self.is_repast_backed and agent_type is not None:
                total += self._repast_grid.get_num_agents(point, agent_type)
            else:
                total += sum(1 for agent in self.agents_at(point) if agent_type is None or getattr(agent, "ptype", None) == agent_type)
        return total

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: Optional[int] = None, include_center: bool = True) -> float:
        points = list(self.neighbor_points(center, radius, include_center))
        if not points:
            return 0.0
        count = self.count_agents_in_radius(center=center, radius=radius, agent_type=agent_type, include_center=include_center)
        return clamp(count / len(points), 0.0, 1.0)

    # Aux Functions
    def _inside_bounds(self, x: int, y: int) -> bool:
        return self.xmin <= x < self.xmax and self.ymin <= y < self.ymax

    def _get_offsets(self, radius: int, include_center: bool) -> list[tuple[int, int]]:
        key = (radius, include_center)
        if key in self._offset_cache:
            return self._offset_cache[key]
        offsets = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and not include_center:
                    offsets.append((dx, dy))
        if include_center:
            offsets.append((0, 0))
        self._offset_cache[key] = offsets
        return offsets