import unittest
from dataclasses import dataclass
from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.substantia_nigra import SubstantiaNigra, SNEnvironmentConfig


@dataclass(frozen=True)
class DummyAgent:
    uid: int
    agent_type: int


@dataclass(frozen=True)
class FakeBounds:
    xmin: int
    xextent: int
    ymin: int
    yextent: int
    zmin: int = 0
    zextent: int = 0


class FakeGrid:
    def __init__(
        self,
        width: int,
        height: int,
        xmin: int = 0,
        ymin: int = 0,
    ):
        self._locations = {}
        self._cells = {}

        self._bounds = FakeBounds(
            xmin=xmin,
            xextent=width,
            ymin=ymin,
            yextent=height,
        )

    def get_local_bounds(self):
        return self._bounds

    def place(self, agent, point: DiscretePoint):
        self._locations[agent] = point
        self._cells.setdefault((point.x, point.y), []).append(agent)

    def get_location(self, agent) -> Optional[DiscretePoint]:
        return self._locations.get(agent)

    def move(self, agent, point: DiscretePoint) -> Optional[DiscretePoint]:
        old_point = self._locations.get(agent)

        if old_point is not None:
            old_key = (old_point.x, old_point.y)
            self._cells[old_key].remove(agent)

            if len(self._cells[old_key]) == 0:
                del self._cells[old_key]

        self._locations[agent] = point
        self._cells.setdefault((point.x, point.y), []).append(agent)

        return point

    def get_agents(self, point: DiscretePoint):
        return list(self._cells.get((point.x, point.y), []))

    def get_num_agents(self, point: DiscretePoint, agent_type: int) -> int:
        return sum(
            1
            for agent in self.get_agents(point)
            if agent.agent_type == agent_type
        )


class TestSubstantiaNigra(unittest.TestCase):

    def setUp(self):
        self.config = SNEnvironmentConfig(
            initial_debris=0.2,
            initial_inflammation=0.1,
            initial_dopamine=1.0,
            debris_decay=0.1,
            inflammation_decay=0.05,
            dopamine_smoothing=0.5,
        )

        self.grid = FakeGrid(width=5, height=5)
        self.env = SubstantiaNigra(self.grid, self.config)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initial_scalars_are_loaded_from_config(self):
        self.assertEqual(self.env.scalars.extracellular_debris, 0.2)
        self.assertEqual(self.env.scalars.inflammation_level, 0.1)
        self.assertEqual(self.env.scalars.dopamine_output, 1.0)

    def test_bounds_are_derived_from_grid_local_bounds(self):
        self.assertEqual(self.env.xmin, 0)
        self.assertEqual(self.env.xmax, 5)
        self.assertEqual(self.env.ymin, 0)
        self.assertEqual(self.env.ymax, 5)

    def test_offset_cache_is_initialized(self):
        self.assertTrue(hasattr(self.env, "_offset_cache"))
        self.assertEqual(self.env._offset_cache, {})

    def test_bounds_can_start_from_non_zero_coordinates(self):
        grid = FakeGrid(width=5, height=5, xmin=10, ymin=20)
        env = SubstantiaNigra(grid, self.config)

        self.assertEqual(env.xmin, 10)
        self.assertEqual(env.xmax, 15)
        self.assertEqual(env.ymin, 20)
        self.assertEqual(env.ymax, 25)

    # ------------------------------------------------------------------
    # Tick cycle and scalar update
    # ------------------------------------------------------------------

    def test_begin_tick_resets_effects(self):
        self.env.add_debris(0.4)
        self.env.remove_debris(0.1)
        self.env.add_inflammation(0.3)
        self.env.remove_inflammation(0.2)
        self.env.release_dopamine(5.0)

        self.env.begin_tick()

        self.assertEqual(self.env.effects.debris_added, 0.0)
        self.assertEqual(self.env.effects.debris_removed, 0.0)
        self.assertEqual(self.env.effects.inflammation_added, 0.0)
        self.assertEqual(self.env.effects.inflammation_removed, 0.0)
        self.assertEqual(self.env.effects.dopamine_released, 0.0)

    def test_commit_effects_updates_scalars_correctly(self):
        self.env.add_debris(0.3)
        self.env.remove_debris(0.1)

        self.env.add_inflammation(0.2)
        self.env.remove_inflammation(0.05)

        self.env.release_dopamine(4.0)

        self.env.commit_effects(max_possible_dopamine=10.0)

        # debris = 0.2 + 0.3 - 0.1 - 0.1 * 0.2 = 0.38
        self.assertAlmostEqual(
            self.env.scalars.extracellular_debris,
            0.38,
            places=6,
        )

        # inflammation = 0.1 + 0.2 - 0.05 - 0.05 * 0.1 = 0.245
        self.assertAlmostEqual(
            self.env.scalars.inflammation_level,
            0.245,
            places=6,
        )

        # dopamine_raw = 4 / 10 = 0.4
        # dopamine = 0.5 * 1.0 + 0.5 * 0.4 = 0.7
        self.assertAlmostEqual(
            self.env.scalars.dopamine_output,
            0.7,
            places=6,
        )

    def test_commit_effects_clamps_upper_bound(self):
        self.env.add_debris(10.0)
        self.env.add_inflammation(10.0)
        self.env.release_dopamine(100.0)

        self.env.commit_effects(max_possible_dopamine=1.0)

        self.assertEqual(self.env.scalars.extracellular_debris, 1.0)
        self.assertEqual(self.env.scalars.inflammation_level, 1.0)
        self.assertEqual(self.env.scalars.dopamine_output, 1.0)

    def test_commit_effects_clamps_lower_bound(self):
        self.env.remove_debris(10.0)
        self.env.remove_inflammation(10.0)
        self.env.release_dopamine(0.0)

        self.env.commit_effects(max_possible_dopamine=1.0)

        self.assertEqual(self.env.scalars.extracellular_debris, 0.0)
        self.assertEqual(self.env.scalars.inflammation_level, 0.0)

    def test_commit_effects_with_zero_max_possible_dopamine(self):
        self.env.release_dopamine(10.0)

        self.env.commit_effects(max_possible_dopamine=0.0)

        # dopamine_raw = 0.0
        # dopamine = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        self.assertAlmostEqual(
            self.env.scalars.dopamine_output,
            0.5,
            places=6,
        )

    def test_position_of_returns_agent_position(self):
        agent = DummyAgent(uid=1, agent_type=10)
        point = DiscretePoint(2, 2)

        self.grid.place(agent, point)

        position = self.env.position_of(agent)

        self.assertIsNotNone(position)
        self.assertEqual(position.x, 2)
        self.assertEqual(position.y, 2)

    def test_position_of_returns_none_for_unplaced_agent(self):
        agent = DummyAgent(uid=1, agent_type=10)

        self.assertIsNone(self.env.position_of(agent))

    def test_move_to_moves_agent(self):
        agent = DummyAgent(uid=1, agent_type=10)

        self.grid.place(agent, DiscretePoint(1, 1))

        new_position = self.env.move_to(agent, DiscretePoint(3, 4))

        self.assertIsNotNone(new_position)
        self.assertEqual(self.env.position_of(agent).x, 3)
        self.assertEqual(self.env.position_of(agent).y, 4)

        old_cell_agents = self.env.agents_at(DiscretePoint(1, 1))
        new_cell_agents = self.env.agents_at(DiscretePoint(3, 4))

        self.assertNotIn(agent, old_cell_agents)
        self.assertIn(agent, new_cell_agents)

    def test_agents_at_returns_all_agents_in_cell(self):
        agent_1 = DummyAgent(uid=1, agent_type=10)
        agent_2 = DummyAgent(uid=2, agent_type=20)

        point = DiscretePoint(1, 1)

        self.grid.place(agent_1, point)
        self.grid.place(agent_2, point)

        agents = self.env.agents_at(point)

        self.assertEqual(len(agents), 2)
        self.assertIn(agent_1, agents)
        self.assertIn(agent_2, agents)

    def test_agents_at_empty_cell_returns_empty_list(self):
        agents = self.env.agents_at(DiscretePoint(4, 4))

        self.assertEqual(agents, [])

    def test_inside_bounds_accepts_valid_points(self):
        self.assertTrue(self.env._inside_bounds(0, 0))
        self.assertTrue(self.env._inside_bounds(4, 4))
        self.assertTrue(self.env._inside_bounds(2, 3))

    def test_inside_bounds_rejects_invalid_points(self):
        self.assertFalse(self.env._inside_bounds(-1, 0))
        self.assertFalse(self.env._inside_bounds(0, -1))
        self.assertFalse(self.env._inside_bounds(5, 0))
        self.assertFalse(self.env._inside_bounds(0, 5))

    def test_neighbor_points_center_with_radius_one_include_center(self):
        center = DiscretePoint(2, 2)

        points = list(
            self.env.neighbor_points(
                center=center,
                radius=1,
                include_center=True,
            )
        )

        coords = {(p.x, p.y) for p in points}

        expected = {
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 2), (2, 3),
            (3, 1), (3, 2), (3, 3),
        }

        self.assertEqual(coords, expected)

    def test_neighbor_points_center_with_radius_one_exclude_center(self):
        center = DiscretePoint(2, 2)

        points = list(
            self.env.neighbor_points(
                center=center,
                radius=1,
                include_center=False,
            )
        )

        coords = {(p.x, p.y) for p in points}

        self.assertNotIn((2, 2), coords)
        self.assertEqual(len(coords), 8)

    def test_neighbor_points_corner_excludes_out_of_bounds_points(self):
        center = DiscretePoint(0, 0)

        points = list(
            self.env.neighbor_points(
                center=center,
                radius=1,
                include_center=True,
            )
        )

        coords = {(p.x, p.y) for p in points}

        expected = {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }

        self.assertEqual(coords, expected)

    def test_neighbor_points_with_non_zero_bounds(self):
        grid = FakeGrid(width=5, height=5, xmin=10, ymin=20)
        env = SubstantiaNigra(grid, self.config)

        center = DiscretePoint(10, 20)

        points = list(
            env.neighbor_points(
                center=center,
                radius=1,
                include_center=True,
            )
        )

        coords = {(p.x, p.y) for p in points}

        expected = {
            (10, 20),
            (10, 21),
            (11, 20),
            (11, 21),
        }

        self.assertEqual(coords, expected)

    def test_get_offsets_uses_cache(self):
        offsets_1 = self.env._get_offsets(radius=1, include_center=True)
        offsets_2 = self.env._get_offsets(radius=1, include_center=True)

        self.assertIs(offsets_1, offsets_2)

    def test_agents_in_radius_excluding_center(self):
        center = DiscretePoint(2, 2)

        center_agent = DummyAgent(uid=1, agent_type=10)
        near_agent = DummyAgent(uid=2, agent_type=20)
        far_agent = DummyAgent(uid=3, agent_type=20)

        self.grid.place(center_agent, center)
        self.grid.place(near_agent, DiscretePoint(3, 2))
        self.grid.place(far_agent, DiscretePoint(4, 4))

        nearby = list(
            self.env.agents_in_radius(
                center=center,
                radius=1,
                include_center=False,
            )
        )

        self.assertIn(near_agent, nearby)
        self.assertNotIn(center_agent, nearby)
        self.assertNotIn(far_agent, nearby)

    def test_agents_in_radius_including_center(self):
        center = DiscretePoint(2, 2)

        center_agent = DummyAgent(uid=1, agent_type=10)
        near_agent = DummyAgent(uid=2, agent_type=20)

        self.grid.place(center_agent, center)
        self.grid.place(near_agent, DiscretePoint(3, 2))

        nearby = list(
            self.env.agents_in_radius(
                center=center,
                radius=1,
                include_center=True,
            )
        )

        self.assertIn(center_agent, nearby)
        self.assertIn(near_agent, nearby)

    def test_count_agents_in_radius_without_type_filter(self):
        center = DiscretePoint(2, 2)

        agent_1 = DummyAgent(uid=1, agent_type=10)
        agent_2 = DummyAgent(uid=2, agent_type=20)
        agent_3 = DummyAgent(uid=3, agent_type=20)

        self.grid.place(agent_1, DiscretePoint(2, 2))
        self.grid.place(agent_2, DiscretePoint(3, 2))
        self.grid.place(agent_3, DiscretePoint(4, 4))

        count = self.env.count_agents_in_radius(
            center=center,
            radius=1,
            include_center=True,
        )

        self.assertEqual(count, 2)

    def test_count_agents_in_radius_with_type_filter(self):
        ALPHA_TYPE = 99

        center = DiscretePoint(2, 2)

        alpha_1 = DummyAgent(uid=1, agent_type=ALPHA_TYPE)
        alpha_2 = DummyAgent(uid=2, agent_type=ALPHA_TYPE)
        microglia = DummyAgent(uid=3, agent_type=10)

        self.grid.place(alpha_1, DiscretePoint(2, 2))
        self.grid.place(alpha_2, DiscretePoint(3, 2))
        self.grid.place(microglia, DiscretePoint(2, 3))

        count = self.env.count_agents_in_radius(
            center=center,
            radius=1,
            agent_type=ALPHA_TYPE,
            include_center=True,
        )

        self.assertEqual(count, 2)

    def test_density_of_type_returns_normalized_density(self):
        ALPHA_TYPE = 99

        center = DiscretePoint(2, 2)

        alpha_1 = DummyAgent(uid=1, agent_type=ALPHA_TYPE)
        alpha_2 = DummyAgent(uid=2, agent_type=ALPHA_TYPE)

        self.grid.place(alpha_1, DiscretePoint(2, 2))
        self.grid.place(alpha_2, DiscretePoint(3, 2))

        density = self.env.density_of_type(
            center=center,
            radius=1,
            agent_type=ALPHA_TYPE,
            normalization=4.0,
            include_center=True,
        )

        self.assertEqual(density, 0.5)

    def test_density_of_type_is_clamped_to_one(self):
        ALPHA_TYPE = 99

        center = DiscretePoint(2, 2)

        alpha_1 = DummyAgent(uid=1, agent_type=ALPHA_TYPE)
        alpha_2 = DummyAgent(uid=2, agent_type=ALPHA_TYPE)
        alpha_3 = DummyAgent(uid=3, agent_type=ALPHA_TYPE)

        self.grid.place(alpha_1, DiscretePoint(2, 2))
        self.grid.place(alpha_2, DiscretePoint(3, 2))
        self.grid.place(alpha_3, DiscretePoint(2, 3))

        density = self.env.density_of_type(
            center=center,
            radius=1,
            agent_type=ALPHA_TYPE,
            normalization=2.0,
            include_center=True,
        )

        self.assertEqual(density, 1.0)

    def test_density_of_type_returns_zero_with_invalid_normalization(self):
        ALPHA_TYPE = 99

        density = self.env.density_of_type(
            center=DiscretePoint(2, 2),
            radius=1,
            agent_type=ALPHA_TYPE,
            normalization=0.0,
            include_center=True,
        )

        self.assertEqual(density, 0.0)


if __name__ == "__main__":
    unittest.main()