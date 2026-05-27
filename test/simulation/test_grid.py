import pytest
from repast4py.space import DiscretePoint
from testhelpers import TestAgent, import_any, TestRepastGrid


grid_module = import_any("src.simulation.utils.grid", "grid")
LocalGrid = grid_module.LocalGrid


class TestLocalGrid:
    def test_initializes_local_bounds(self):
        grid = LocalGrid(width=4, height=3)
        assert grid.xmin == 0
        assert grid.ymin == 0
        assert grid.xmax == 4
        assert grid.ymax == 3
        assert grid.is_repast_backed is False

    def test_add_agent_tracks_agent_in_registry_and_cells(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=7)
        point = DiscretePoint(1, 2)
        grid.add_agent(agent, point)
        assert agent in grid.agent_registry
        assert grid.agents_at(point) == [agent]

    def test_add_agent_outside_bounds_raises_value_error(self):
        grid = LocalGrid(width=2, height=2)
        with pytest.raises(ValueError):
            grid.add_agent(TestAgent(), DiscretePoint(2, 0))

    def test_move_to_updates_cell_membership(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=3)
        old_point = DiscretePoint(1, 1)
        new_point = DiscretePoint(2, 1)
        grid.add_agent(agent, old_point)
        returned_point = grid.move_to(agent, new_point)
        assert returned_point == new_point
        assert grid.agents_at(old_point) == []
        assert grid.agents_at(new_point) == [agent]

    def test_move_to_outside_bounds_returns_none_and_keeps_old_cell(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=3)
        old_point = DiscretePoint(1, 1)
        grid.add_agent(agent, old_point)
        returned_point = grid.move_to(agent, DiscretePoint(4, 1))
        assert returned_point is None
        assert grid.agents_at(old_point) == [agent]

    def test_density_of_type_clamps_to_one_for_full_cell(self):
        grid = LocalGrid(width=2, height=2)
        point = DiscretePoint(0, 0)
        grid.add_agent(TestAgent(ptype=5), point)
        assert grid.density_of_type(point, radius=0, agent_type=5, include_center=True) == 1.0

    def test_position_of_returns_local_position(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=1)
        point = DiscretePoint(1, 2)
        grid.add_agent(agent, point)
        assert grid.position_of(agent) == point

    def test_remove_agent_removes_agent_from_registry_and_cells(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=1)
        point = DiscretePoint(1, 2)
        grid.add_agent(agent, point)
        grid.remove_agent(agent)
        assert agent not in grid.agent_registry
        assert grid.agents_at(point) == []

    def test_neighbor_points_excludes_center_when_requested(self):
        grid = LocalGrid(width=5, height=5)
        center = DiscretePoint(2, 2)

        points = list(grid.neighbor_points(center, radius=1, include_center=False))

        assert center not in points

    def test_neighbor_points_does_not_duplicate_center(self):
        grid = LocalGrid(width=5, height=5)
        center = DiscretePoint(2, 2)
        points = list(grid.neighbor_points(center, radius=1, include_center=True))
        coords = [(point.x, point.y) for point in points]
        assert len(coords) == len(set(coords))
        assert coords.count((center.x, center.y)) == 1

    def test_position_of_returns_local_position_for_local_grid(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=1)
        point = DiscretePoint(1, 2)
        grid.add_agent(agent, point)
        assert grid.position_of(agent) == point

    def test_add_agent_moves_existing_agent_without_duplicate_cells(self):
        grid = LocalGrid(width=4, height=4)
        agent = TestAgent(ptype=1)

        old_point = DiscretePoint(1, 1)
        new_point = DiscretePoint(2, 2)

        grid.add_agent(agent, old_point)
        grid.add_agent(agent, new_point)

        assert agent not in grid.agents_at(old_point)
        assert grid.agents_at(new_point) == [agent]
        assert grid.agent_registry == [agent]
        assert grid.position_of(agent) == new_point
