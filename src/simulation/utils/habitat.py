from typing import Iterable, Optional
from repast4py.space import DiscretePoint
from src.simulation.utils.grid import LocalGrid


class GridHabitatMixin:
    grid: LocalGrid
    def _grid(self) -> LocalGrid:
        try:
            return self.grid
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__} must define self.grid before using GridHabitatMixin."
            ) from exc
    def add_agent(self, agent, point: DiscretePoint):
        return self._grid().add_agent(agent, point)

    def remove_agent(self, agent):
        return self._grid().remove_agent(agent)

    def position_of(self, agent) -> Optional[DiscretePoint]:
        return self._grid().position_of(agent)

    def move_to(self, agent, point: DiscretePoint) -> Optional[DiscretePoint]:
        return self._grid().move_to(agent, point)

    def agents_at(self, point: DiscretePoint) -> list:
        return self._grid().agents_at(point)

    def agents_in_radius(self, center: DiscretePoint, radius: int = 1, include_center: bool = False) -> Iterable:
        return self._grid().agents_in_radius(center, radius, include_center)

    def count_agents_in_radius(self, center: DiscretePoint, radius: int = 1, agent_type: Optional[int] = None, include_center: bool = False) -> int:
        return self._grid().count_agents_in_radius(center, radius, agent_type, include_center)

    def neighbor_points(self, center: DiscretePoint, radius: int = 1, include_center: bool = True) -> Iterable[DiscretePoint]:
        return self._grid().neighbor_points(center, radius, include_center)

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: Optional[int] = None, include_center: bool = True) -> float:
        return self._grid().density_of_type(center, radius, agent_type, include_center)


class InternalHabitatMixin(GridHabitatMixin):
    """Grid-backed habitat API expected by intracellular agents.
    Implementers expose global neuron scalars, local derived densities and the
    shared degradation buffers used by lysosomes.
    """
    def oxidative_stress_at(self, position: Optional[DiscretePoint] = None) -> float:
        raise NotImplementedError

    def energy_demand_at(self, position: Optional[DiscretePoint] = None) -> float:
        raise NotImplementedError

    def local_aggregate_density_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        raise NotImplementedError

    def local_debris_density_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        raise NotImplementedError

    def add_intracellular_debris(self, amount: float):
        raise NotImplementedError

    def add_energy_demand(self, amount: float):
        raise NotImplementedError

    def register_degradation_target(self, agent):
        raise NotImplementedError

    def available_degradation_targets(self) -> list:
        raise NotImplementedError

    def assign_degradation_target(self, lysosome, target) -> bool:
        raise NotImplementedError

    def target_for(self, lysosome):
        raise NotImplementedError

    def clear_degradation_assignment(self, lysosome, requeue_target: bool = False):
        raise NotImplementedError

    def unregister_degradation_target(self, target):
        raise NotImplementedError

    def is_target_assigned(self, target) -> bool:
        raise NotImplementedError
