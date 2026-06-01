from typing import Iterable, Optional
from repast4py.space import DiscretePoint
from src.simulation.utils.grid import LocalGrid


class GridHabitatMixin:
    """Mixin exposing a LocalGrid-like API through self.grid."""

    grid: LocalGrid
    def _grid(self) -> LocalGrid:
        """Return the bound grid or fail with a clear contract error."""

        try:
            return self.grid
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__} must define self.grid before using GridHabitatMixin."
            ) from exc
    def add_agent(self, agent, point: DiscretePoint):
        """Add an agent to this habitat grid."""

        return self._grid().add_agent(agent, point)

    def remove_agent(self, agent):
        """Remove an agent from this habitat grid."""

        return self._grid().remove_agent(agent)

    def position_of(self, agent) -> Optional[DiscretePoint]:
        """Return the position of an agent inside this habitat."""

        return self._grid().position_of(agent)

    def move_to(self, agent, point: DiscretePoint) -> Optional[DiscretePoint]:
        """Move an agent inside this habitat."""

        return self._grid().move_to(agent, point)

    def agents_at(self, point: DiscretePoint) -> list:
        """Return agents at one habitat point."""

        return self._grid().agents_at(point)

    def agents_in_radius(self, center: DiscretePoint, radius: int = 1, include_center: bool = False) -> Iterable:
        """Yield agents in a local neighborhood."""

        return self._grid().agents_in_radius(center, radius, include_center)

    def count_agents_in_radius(self, center: DiscretePoint, radius: int = 1, agent_type: Optional[int] = None, include_center: bool = False) -> int:
        """Count agents in a local neighborhood."""

        return self._grid().count_agents_in_radius(center, radius, agent_type, include_center)

    def neighbor_points(self, center: DiscretePoint, radius: int = 1, include_center: bool = True) -> Iterable[DiscretePoint]:
        """Yield valid neighboring habitat points."""

        return self._grid().neighbor_points(center, radius, include_center)

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: Optional[int] = None, include_center: bool = True) -> float:
        """Return local density for one optional agent type."""

        return self._grid().density_of_type(center, radius, agent_type, include_center)


class InternalHabitatMixin(GridHabitatMixin):
    """Grid-backed habitat API expected by intracellular agents.
    Implementers expose global neuron scalars, local derived densities and the
    shared degradation buffers used by lysosomes.
    """
    def oxidative_stress_at(self, position: Optional[DiscretePoint] = None) -> float:
        """Return oxidative stress, optionally localized by position."""

        raise NotImplementedError

    def energy_demand_at(self, position: Optional[DiscretePoint] = None) -> float:
        """Return unmet energy demand, optionally localized by position."""

        raise NotImplementedError

    def local_aggregate_density_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        """Return aggregate density around an intracellular position."""

        raise NotImplementedError

    def local_debris_density_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        """Return debris density around an intracellular position."""

        raise NotImplementedError

    def add_intracellular_debris(self, amount: float):
        """Buffer intracellular debris produced by an agent."""

        raise NotImplementedError

    def add_energy_demand(self, amount: float):
        """Buffer additional energy demand produced by an agent."""

        raise NotImplementedError

    def register_degradation_target(self, agent):
        """Expose an intracellular agent as available to lysosomes."""

        raise NotImplementedError

    def available_degradation_targets(self) -> list:
        """Return unassigned lysosomal degradation targets."""

        raise NotImplementedError

    def assign_degradation_target(self, lysosome, target) -> bool:
        """Assign a degradation target to one lysosome."""

        raise NotImplementedError

    def target_for(self, lysosome):
        """Return the target assigned to one lysosome."""

        raise NotImplementedError

    def clear_degradation_assignment(self, lysosome, requeue_target: bool = False):
        """Clear a lysosome assignment, optionally making target available."""

        raise NotImplementedError

    def unregister_degradation_target(self, target):
        """Remove a target from degradation buffers."""

        raise NotImplementedError

    def is_target_assigned(self, target) -> bool:
        """Return whether a target is already claimed by a lysosome."""

        raise NotImplementedError
