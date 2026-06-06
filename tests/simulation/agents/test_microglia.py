from types import SimpleNamespace
import pytest
from repast4py.space import DiscretePoint
from tests.simulation.testhelpers import TestRng, TestSubstantiaNigraLikeEnvironment, import_any


microglia_module = import_any("src.simulation.agents.microglia", "microglia")
Microglia = microglia_module.Microglia
MicrogliaAction = microglia_module.MicrogliaAction
MicrogliaConfig = microglia_module.MicrogliaConfig
MicrogliaPerception = microglia_module.MicrogliaPerception
MicrogliaState = microglia_module.MicrogliaState


def make_config(move_probability: float = 1.0) -> MicrogliaConfig:
    return MicrogliaConfig(
        per_radius=1,
        debris_high_threshold=0.6,
        debris_low_threshold=0.2,
        inflammation_high_threshold=0.7,
        inflammation_low_threshold=0.3,
        nearby_alpha_high_threshold=0.5,
        nearby_alpha_low_threshold=0.2,
        debris_clearance_rate=0.10,
        inflammation_release_rate=0.15,
        move_probability=move_probability
    )


class TestMicroglia:
    def test_initial_state_is_resting(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        assert microglia.state == MicrogliaState.RESTING
        assert microglia.pending_action is None
        assert microglia.last_perception is None

    def test_see_sets_nearby_alpha_to_zero_when_position_is_missing(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        environment = TestSubstantiaNigraLikeEnvironment(position=None, debris=0.4, inflammation=0.3)
        microglia.see(SimpleNamespace(environment=environment))
        assert microglia.last_perception.position is None
        assert microglia.last_perception.nearby_alpha == 0.0
        assert microglia.last_perception.extracellular_debris == 0.4
        assert microglia.last_perception.inflammation_level == 0.3

    def test_see_reads_density_when_position_is_present(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        position = DiscretePoint(1, 1)
        environment = TestSubstantiaNigraLikeEnvironment(position=position, nearby_alpha=0.75)
        microglia.see(SimpleNamespace(environment=environment))
        assert microglia.last_perception.nearby_alpha == 0.75
        assert environment.density_calls == [(position, 1, 9, True)]

    def test_next_resting_to_clearing_when_debris_is_high(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.7, inflammation_level=0.0, nearby_alpha=0.0)
        microglia.next()
        assert microglia.state == MicrogliaState.CLEARING

    def test_next_resting_to_activated_when_inflammation_is_high(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.0, inflammation_level=0.8, nearby_alpha=0.0)
        microglia.next()
        assert microglia.state == MicrogliaState.ACTIVATED

    def test_next_clearing_to_resting_when_debris_is_low(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        microglia.state = MicrogliaState.CLEARING
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.1, inflammation_level=0.0, nearby_alpha=0.0)
        microglia.next()
        assert microglia.state == MicrogliaState.RESTING

    def test_next_activated_to_resting_when_all_signals_are_low(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        microglia.state = MicrogliaState.ACTIVATED
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.1, inflammation_level=0.1, nearby_alpha=0.1)
        microglia.next()
        assert microglia.state == MicrogliaState.RESTING

    def test_next_can_remain_resting_when_probabilistic_activation_fails(self):
        config = make_config()
        config.activation_transition_rate = 0.1
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=config, alpha_type_id=9)
        microglia.rng = TestRng(random_value=1.0)
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.0, inflammation_level=0.8, nearby_alpha=0.0)

        microglia.next()

        assert microglia.state == MicrogliaState.RESTING
        assert microglia.last_transition_sample["check"] == "resting_to_activated"

    def test_next_activated_to_clearing_when_debris_persists_after_activation_signals_drop(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        microglia.rng = TestRng(random_value=0.0)
        microglia.state = MicrogliaState.ACTIVATED
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.7, inflammation_level=0.1, nearby_alpha=0.1)

        microglia.next()

        assert microglia.state == MicrogliaState.CLEARING

    @pytest.mark.parametrize(
        ("state", "expected_action"),
        [
            (MicrogliaState.RESTING, MicrogliaAction.SCAN),
            (MicrogliaState.CLEARING, MicrogliaAction.CLEAR_DEBRIS),
            (MicrogliaState.ACTIVATED, MicrogliaAction.INFLAMMATION)
        ],
    )
    def test_action_maps_state_to_action(self, state, expected_action):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        microglia.state = state
        microglia.action()
        assert microglia.pending_action == expected_action

    def test_activated_microglia_scans_when_context_pressure_is_low(self):
        config = make_config()
        config.inflammatory_action_threshold = 0.5
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=config, alpha_type_id=9)
        microglia.state = MicrogliaState.ACTIVATED
        microglia.last_perception = MicrogliaPerception(None, extracellular_debris=0.1, inflammation_level=0.1, nearby_alpha=0.1)

        microglia.action()

        assert microglia.pending_action == MicrogliaAction.SCAN

    def test_do_clear_debris_removes_debris_from_environment(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        environment = TestSubstantiaNigraLikeEnvironment()
        microglia.pending_action = MicrogliaAction.CLEAR_DEBRIS
        microglia.do(SimpleNamespace(environment=environment, rng=TestRng()))
        assert environment.removed_debris == pytest.approx(0.10)

    def test_do_inflammation_adds_inflammation_to_environment(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        environment = TestSubstantiaNigraLikeEnvironment()
        microglia.pending_action = MicrogliaAction.INFLAMMATION
        microglia.do(SimpleNamespace(environment=environment, rng=TestRng()))
        assert environment.added_inflammation == pytest.approx(0.15)

    def test_do_scan_moves_when_rng_is_within_move_probability(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(move_probability=1.0), alpha_type_id=9)
        microglia.rng = TestRng(random_value=0.0, choice_index=1)
        position = DiscretePoint(1, 1)
        environment = TestSubstantiaNigraLikeEnvironment(position=position)
        microglia.pending_action = MicrogliaAction.SCAN
        microglia.do(SimpleNamespace(environment=environment))
        assert environment.moves == [(microglia, DiscretePoint(2, 1))]

    def test_do_scan_does_not_move_when_rng_exceeds_move_probability(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(move_probability=0.0), alpha_type_id=9)
        microglia.rng = TestRng(random_value=1.0)
        position = DiscretePoint(1, 1)
        environment = TestSubstantiaNigraLikeEnvironment(position=position)
        microglia.pending_action = MicrogliaAction.SCAN
        microglia.do(SimpleNamespace(environment=environment))
        assert environment.moves == []

    def test_step_runs_see_next_action_do(self):
        microglia = Microglia(local_id=1, rank=0, type_id=2, config=make_config(), alpha_type_id=9)
        environment = TestSubstantiaNigraLikeEnvironment(
            position=DiscretePoint(1, 1),
            debris=0.7,
            inflammation=0.0
        )
        microglia.step(SimpleNamespace(environment=environment))
        assert microglia.state == MicrogliaState.CLEARING
        assert microglia.pending_action == MicrogliaAction.CLEAR_DEBRIS
        assert environment.removed_debris == pytest.approx(0.10)
