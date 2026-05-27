from __future__ import annotations
import pytest
from repast4py.space import DiscretePoint
from testhelpers import TestAgent, TestRepastGrid, import_any


sn_module = import_any(
    "src.simulation.substantia_nigra",
    "src.simulation.substantia_nigra",
    "substantia_nigra",
)
SNEnvironmentConfig = sn_module.SNEnvironmentConfig
SubstantiaNigra = sn_module.SubstantiaNigra


def make_config() -> SNEnvironmentConfig:
    return SNEnvironmentConfig(
        initial_debris=0.4,
        initial_inflammation=0.5,
        initial_dopamine=0.2,
        debris_decay=0.1,
        inflammation_decay=0.2,
        dopamine_smoothing=0.5
    )


class TestSubstantiaNigra:
    def test_initializes_scalars_from_config(self):
        environment = SubstantiaNigra(TestRepastGrid(), make_config())
        assert environment.scalars.extracellular_debris == 0.4
        assert environment.scalars.inflammation_level == 0.5
        assert environment.scalars.dopamine_output == 0.2

    def test_effect_methods_accumulate_tick_buffers(self):
        environment = SubstantiaNigra(TestRepastGrid(), make_config())

        environment.add_debris(0.3)
        environment.remove_debris(0.1)
        environment.add_inflammation(0.2)
        environment.remove_inflammation(0.05)
        environment.release_dopamine(4.0)
        assert environment.effects.debris_added == 0.3
        assert environment.effects.debris_removed == 0.1
        assert environment.effects.inflammation_added == 0.2
        assert environment.effects.inflammation_removed == 0.05
        assert environment.effects.dopamine_released == 4.0

    def test_begin_tick_resets_effect_buffers(self):
        environment = SubstantiaNigra(TestRepastGrid(), make_config())
        environment.add_debris(0.3)
        environment.add_inflammation(0.2)
        environment.release_dopamine(4.0)
        environment.begin_tick()
        assert environment.effects.debris_added == 0.0
        assert environment.effects.inflammation_added == 0.0
        assert environment.effects.dopamine_released == 0.0

    def test_commit_effects_applies_decay_and_dopamine_smoothing(self):
        environment = SubstantiaNigra(TestRepastGrid(), make_config())
        environment.add_debris(0.3)
        environment.remove_debris(0.1)
        environment.add_inflammation(0.2)
        environment.remove_inflammation(0.1)
        environment.release_dopamine(4.0)
        environment.commit_effects(max_possible_dopamine=8.0)
        assert environment.scalars.extracellular_debris == pytest.approx(0.56)
        assert environment.scalars.inflammation_level == pytest.approx(0.50)
        assert environment.scalars.dopamine_output == pytest.approx(0.35)

    def test_commit_effects_clamps_scalar_values(self):
        environment = SubstantiaNigra(TestRepastGrid(), make_config())
        environment.add_debris(100.0)
        environment.add_inflammation(100.0)
        environment.release_dopamine(100.0)
        environment.commit_effects(max_possible_dopamine=1.0)
        assert environment.scalars.extracellular_debris == 1.0
        assert environment.scalars.inflammation_level == 1.0
        assert environment.scalars.dopamine_output == 1.0

    def test_grid_wrappers_delegate_position_and_agents_at(self):
        repast_grid = TestRepastGrid()
        environment = SubstantiaNigra(repast_grid, make_config())
        agent = TestAgent(ptype=9)
        point = DiscretePoint(2, 3)
        repast_grid.set_location(agent, point)
        assert environment.position_of(agent) == point
        assert environment.agents_at(point) == [agent]

    def test_density_of_type_respects_include_center_keyword(self):
        repast_grid = TestRepastGrid()
        environment = SubstantiaNigra(repast_grid, make_config())
        center = DiscretePoint(5, 5)
        alpha_at_center = TestAgent(ptype=3, uid=1)
        repast_grid.set_location(alpha_at_center, center)
        density_with_center = environment.density_of_type(
            center=center,
            radius=1,
            agent_type=3,
            include_center=True
        )
        density_without_center = environment.density_of_type(
            center=center,
            radius=1,
            agent_type=3,
            include_center=False
        )
        assert density_with_center > 0.0
        assert density_without_center == 0.0