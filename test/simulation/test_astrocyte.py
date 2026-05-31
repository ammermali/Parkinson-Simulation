from types import SimpleNamespace
import pytest
from repast4py.space import DiscretePoint
from testhelpers import TestRng, TestSubstantiaNigraLikeEnvironment, import_any


astrocyte_module = import_any("src.simulation.agents.astrocyte", "astrocyte")
Astrocyte = astrocyte_module.Astrocyte
AstrocyteAction = astrocyte_module.AstrocyteAction
AstrocyteConfig = astrocyte_module.AstrocyteConfig
AstrocyteState = astrocyte_module.AstrocyteState


def make_config() -> AstrocyteConfig:
    return AstrocyteConfig(
        inflammation_high_threshold=0.7,
        inflammation_low_threshold=0.2,
        debris_high_threshold=0.6,
        debris_low_threshold=0.1,
        support_inflammation_reduction_rate=0.05,
        inflammation_release_rate=0.08,
    )


class AstrocyteEnvironmentStub:
    def __init__(self, inflammation: float = 0.0, debris: float = 0.0):
        self.scalars = SimpleNamespace()
        self.effects = SimpleNamespace()
        self.scalars.inflammation_level = inflammation
        self.scalars.extracellular_debris = debris
        self.effects.removed_inflammation = 0.0
        self.effects.added_inflammation = 0.0
        self.position = DiscretePoint(1, 1)

    def position_of(self, agent):
        return self.position

    def remove_inflammation(self, amount: float):
        self.effects.removed_inflammation += amount

    def add_inflammation(self, amount: float):
        self.effects.added_inflammation += amount


class TestAstrocyte:
    def test_initial_state_is_supportive(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        assert astrocyte.state == AstrocyteState.SUPPORTIVE
        assert astrocyte.pending_action is None
        assert astrocyte.last_perception is None

    def test_see_collects_position_and_environment_scalars(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        environment = AstrocyteEnvironmentStub(inflammation=0.4, debris=0.3)
        model = SimpleNamespace(environment=environment)
        perception = astrocyte.see(model)
        assert perception is astrocyte.last_perception
        assert perception.position == environment.position
        assert perception.inflammation_level == 0.4
        assert perception.extracellular_debris == 0.3

    def test_next_changes_supportive_to_reactive_when_inflammation_is_high(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        astrocyte.last_perception = astrocyte_module.AstrocytePerception(
            position=None,
            inflammation_level=0.8,
            extracellular_debris=0.0,
        )
        astrocyte.next()
        assert astrocyte.state == AstrocyteState.REACTIVE

    def test_next_changes_reactive_to_supportive_when_inflammation_and_debris_are_low(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        astrocyte.state = AstrocyteState.REACTIVE
        astrocyte.last_perception = astrocyte_module.AstrocytePerception(
            position=None,
            inflammation_level=0.1,
            extracellular_debris=0.05,
        )
        astrocyte.next()
        assert astrocyte.state == AstrocyteState.SUPPORTIVE

    def test_action_maps_supportive_to_support(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        astrocyte.action()
        assert astrocyte.pending_action == AstrocyteAction.SUPPORT

    def test_action_maps_reactive_to_inflammation(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        astrocyte.state = AstrocyteState.REACTIVE
        astrocyte.action()
        assert astrocyte.pending_action == AstrocyteAction.INFLAMMATION

    def test_reactive_transition_can_be_delayed_by_memory_probability(self):
        config = make_config()
        config.stress_memory_decay = 0.8
        config.reactive_transition_rate = 0.25
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=config)
        astrocyte.rng = TestRng(random_value=0.9)
        astrocyte.last_perception = astrocyte_module.AstrocytePerception(
            position=None,
            inflammation_level=1.0,
            extracellular_debris=0.0,
        )

        astrocyte.next()

        assert astrocyte.state == AstrocyteState.SUPPORTIVE
        assert 0.0 < astrocyte.stress_memory < 1.0

    def test_reactive_astrocyte_can_remain_supportive_until_memory_is_high(self):
        config = make_config()
        config.inflammatory_memory_threshold = 0.6
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=config)
        astrocyte.state = AstrocyteState.REACTIVE
        astrocyte.stress_memory = 0.2

        astrocyte.action()

        assert astrocyte.pending_action == AstrocyteAction.SUPPORT

    def test_do_support_removes_inflammation(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        environment = AstrocyteEnvironmentStub()
        astrocyte.pending_action = AstrocyteAction.SUPPORT
        astrocyte.do(SimpleNamespace(environment=environment))
        assert environment.effects.removed_inflammation == pytest.approx(0.05)

    def test_do_reactive_adds_inflammation(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        environment = AstrocyteEnvironmentStub()
        astrocyte.pending_action = AstrocyteAction.INFLAMMATION
        astrocyte.do(SimpleNamespace(environment=environment))
        assert environment.effects.added_inflammation == pytest.approx(0.08)

    def test_step_runs_see_next_action_do(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        environment = AstrocyteEnvironmentStub(inflammation=0.8, debris=0.0)
        astrocyte.step(SimpleNamespace(environment=environment))
        assert astrocyte.state == AstrocyteState.REACTIVE
        assert astrocyte.pending_action == AstrocyteAction.INFLAMMATION
        assert environment.effects.added_inflammation == pytest.approx(0.08)

    def test_see_reads_substantia_nigra_style_scalars(self):
        astrocyte = Astrocyte(local_id=1, rank=0, type_id=2, config=make_config())
        environment = TestSubstantiaNigraLikeEnvironment(
            position=DiscretePoint(1, 1),
            debris=0.3,
            inflammation=0.4,
        )
        astrocyte.see(SimpleNamespace(environment=environment))
        assert astrocyte.last_perception.inflammation_level == 0.4
        assert astrocyte.last_perception.extracellular_debris == 0.3
