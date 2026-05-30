from types import SimpleNamespace
import pytest
from repast4py.space import DiscretePoint
from testhelpers import TestAgent, TestAggregateAgent, TestRng, TestSubstantiaNigraLikeEnvironment, import_any


neuron_module = import_any("src.simulation.agents.neuron", "neuron")
Neuron = neuron_module.Neuron
NeuronAction = neuron_module.NeuronAction
NeuronConfig = neuron_module.NeuronConfig
NeuronInternalEffects = neuron_module.NeuronInternalEffects
NeuronPerception = neuron_module.NeuronPerception
NeuronState = neuron_module.NeuronState
AdaptiveAgent = neuron_module.AdaptiveAgent


def make_config() -> NeuronConfig:
    return NeuronConfig(
        per_radius=1,
        nearby_alpha_high_threshold=0.5,
        inflammation_high_threshold=0.7,
        debris_high_threshold=0.6,
        alpha_load_release_threshold=0.02,
        damage_accumulation_rate=1.0,
        damage_recovery_rate=0.10,
        low_stress_threshold=0.05,
        inflammation_damage_weight=0.4,
        debris_damage_weight=0.3,
        alpha_damage_weight=0.3,
        compromised_threshold=0.3,
        apoptotic_threshold=0.6,
        ruptured_threshold=0.9,
        dopamine_release_rate=0.20,
        stress_inflammation_release_rate=0.15,
        debris_release_rate=0.25,
        alpha_absorption_rate=0.10,
        alpha_release_amount=0.05,
    )


def make_neuron(alpha_type_id: int = 99) -> Neuron:
    return Neuron(local_id=1, rank=0, type_id=10, config=make_config(), alpha_type_id=alpha_type_id)


class DebrisAgent:
    state = SimpleNamespace(value="Debris")


class PhaseAgent(AdaptiveAgent):
    def __init__(self, local_id: int, log: list[str], label: str):
        super().__init__(local_id, 77, 0)
        self.log = log
        self.label = label

    def see(self, model):
        self.log.append(f"{self.label}:see")

    def next(self):
        self.log.append(f"{self.label}:next")

    def action(self):
        self.log.append(f"{self.label}:action")

    def do(self, model):
        self.log.append(f"{self.label}:do")


def make_perception(
    nearby_alpha: float = 0.0,
    inflammation: float = 0.0,
    debris: float = 0.0,
    internal_damage: float = 0.0,
    alpha_load: float = 0.0,
    cell_damage: float = 0.0,
) -> NeuronPerception:
    return NeuronPerception(
        position=DiscretePoint(1, 1),
        nearby_alpha=nearby_alpha,
        inflammatory_levels=inflammation,
        extracellular_debris=debris,
        oxidative_stress=0.0,
        intracellular_debris=0.0,
        energy_demand=0.5,
        internal_damage=internal_damage,
        alpha_load=alpha_load,
        cell_damage=cell_damage,
    )


class TestNeuron:
    def test_initializes_as_healthy_macro_agent_with_internal_grid(self):
        neuron = make_neuron()
        assert neuron.state == NeuronState.HEALTHY
        assert neuron.cell_damage == 0.0
        assert neuron.grid.xmax == neuron.internal_cfg.width == 10
        assert neuron.grid.ymax == neuron.internal_cfg.height == 10
        assert neuron.degradation_targets == []
        assert neuron.degradation_assignment == {}

    def test_see_builds_external_and_internal_perception(self):
        neuron = make_neuron(alpha_type_id=99)
        neuron.internal_scalars.oxidative_stress = 0.4
        neuron.internal_scalars.intracellular_debris = 0.2
        neuron.internal_scalars.energy_demand = 0.8
        environment = TestSubstantiaNigraLikeEnvironment(
            position=DiscretePoint(2, 2),
            nearby_alpha=0.55,
            debris=0.25,
            inflammation=0.35,
        )
        perception = neuron.see(SimpleNamespace(environment=environment))
        assert perception.position == DiscretePoint(2, 2)
        assert perception.nearby_alpha == 0.55
        assert perception.inflammatory_levels == 0.35
        assert perception.extracellular_debris == 0.25
        assert perception.oxidative_stress == 0.4
        assert perception.intracellular_debris == 0.2
        assert perception.energy_demand == 0.8
        assert neuron.last_perception == perception

    def test_compute_internal_damage_uses_weighted_internal_scalars_and_clamps(self):
        neuron = make_neuron()
        neuron.internal_scalars.oxidative_stress = 1.0
        neuron.internal_scalars.intracellular_debris = 0.25
        for uid in range(50):
            neuron.add_agent(
                TestAggregateAgent(aggregate_weight=1.0, uid=uid),
                DiscretePoint(uid % 10, uid // 10),
            )
        assert neuron.compute_internal_damage() == pytest.approx(0.65)
        neuron.internal_scalars.oxidative_stress = 10.0
        assert neuron.compute_internal_damage() == 1.0

    def test_compute_alpha_load_counts_internal_alpha_agents_over_grid_capacity(self):
        neuron = make_neuron(alpha_type_id=99)
        neuron.add_agent(TestAggregateAgent(aggregate_weight=1.0, uid=1), DiscretePoint(0, 0))
        neuron.add_agent(TestAggregateAgent(aggregate_weight=1.0, uid=2), DiscretePoint(1, 0))
        neuron.add_agent(TestAggregateAgent(aggregate_weight=0.0, ptype=7, uid=3), DiscretePoint(2, 0))

        assert neuron.compute_alpha_load() == pytest.approx(2 / 100)

    def test_local_aggregate_density_is_derived_from_grid_neighborhood(self):
        neuron = make_neuron(alpha_type_id=99)
        center = DiscretePoint(1, 1)
        neuron.add_agent(TestAggregateAgent(aggregate_weight=1.0, uid=1), center)
        neuron.add_agent(TestAggregateAgent(aggregate_weight=0.5, uid=2), DiscretePoint(1, 2))

        density = neuron.local_aggregate_density_at(center, radius=1, include_center=True)

        assert density == pytest.approx(1.5 / 9)

    def test_local_debris_density_is_derived_from_grid_neighborhood(self):
        neuron = make_neuron(alpha_type_id=99)
        center = DiscretePoint(1, 1)
        neuron.add_agent(DebrisAgent(), center)
        neuron.add_agent(DebrisAgent(), DiscretePoint(1, 2))

        density = neuron.local_debris_density_at(center, radius=1, include_center=True)

        assert density == pytest.approx(2 / 9)
        assert neuron.local_debris_density_at(None) == 0.0

    def test_next_recovers_cell_damage_when_total_stress_is_low(self):
        neuron = make_neuron()
        neuron.cell_damage = 0.5
        neuron.last_perception = make_perception(inflammation=0.0, debris=0.0, nearby_alpha=0.0, internal_damage=0.0)
        neuron.next()
        assert neuron.cell_damage == pytest.approx(0.4)
        assert neuron.state == NeuronState.COMPROMISED

    @pytest.mark.parametrize(
        ("internal_damage", "expected_state"),
        [
            (0.4, NeuronState.HEALTHY),
            (0.8, NeuronState.COMPROMISED),
            (1.2, NeuronState.APOPTOTIC),
            (2.0, NeuronState.RUPTURED),
        ],
    )
    def test_next_sets_state_from_accumulated_damage_thresholds(self, internal_damage, expected_state):
        neuron = make_neuron()
        neuron.last_perception = make_perception(internal_damage=internal_damage)
        neuron.next()
        assert neuron.state == expected_state

    def test_action_prioritizes_ruptured_dump_debris(self):
        neuron = make_neuron()
        neuron.state = NeuronState.RUPTURED
        neuron.last_perception = make_perception(nearby_alpha=1.0, alpha_load=1.0)
        neuron.action()
        assert neuron.pending_action == NeuronAction.DUMP_DEBRIS

    def test_action_absorbs_alpha_when_nearby_alpha_is_high_and_not_apoptotic(self):
        neuron = make_neuron()
        neuron.state = NeuronState.HEALTHY
        neuron.last_perception = make_perception(nearby_alpha=0.6)
        neuron.action()
        assert neuron.pending_action == NeuronAction.A_ALPHASYNUCLEIN

    def test_action_releases_alpha_when_apoptotic(self):
        neuron = make_neuron()
        neuron.state = NeuronState.APOPTOTIC
        neuron.last_perception = make_perception(nearby_alpha=0.0, alpha_load=0.0)
        neuron.action()
        assert neuron.pending_action == NeuronAction.R_ALPHASYNUCLEIN

    def test_action_releases_alpha_when_alpha_load_is_high(self):
        neuron = make_neuron()
        neuron.state = NeuronState.HEALTHY
        neuron.last_perception = make_perception(nearby_alpha=0.0, alpha_load=0.5)
        neuron.action()
        assert neuron.pending_action == NeuronAction.R_ALPHASYNUCLEIN

    def test_action_healthy_neuron_signals_stress_when_inflammation_is_high(self):
        neuron = make_neuron()
        neuron.state = NeuronState.HEALTHY
        neuron.last_perception = make_perception(inflammation=0.8)
        neuron.action()
        assert neuron.pending_action == NeuronAction.STRESS

    def test_action_healthy_neuron_signals_stress_when_extracellular_debris_is_high(self):
        neuron = make_neuron()
        neuron.state = NeuronState.HEALTHY
        neuron.last_perception = make_perception(debris=0.7)
        neuron.action()
        assert neuron.pending_action == NeuronAction.STRESS

    def test_action_healthy_neuron_releases_dopamine_in_low_stress(self):
        neuron = make_neuron()
        neuron.state = NeuronState.HEALTHY
        neuron.last_perception = make_perception(inflammation=0.1, debris=0.1, nearby_alpha=0.0)
        neuron.action()
        assert neuron.pending_action == NeuronAction.R_DOPAMINE

    def test_do_release_dopamine_updates_environment(self):
        neuron = make_neuron()
        environment = TestSubstantiaNigraLikeEnvironment()
        neuron.last_perception = make_perception()
        neuron.pending_action = NeuronAction.R_DOPAMINE
        neuron.do(SimpleNamespace(environment=environment, rng=TestRng()))
        assert environment.released_dopamine == pytest.approx(0.20)

    def test_do_signal_stress_adds_inflammation(self):
        neuron = make_neuron()
        environment = TestSubstantiaNigraLikeEnvironment()
        neuron.last_perception = make_perception()
        neuron.pending_action = NeuronAction.STRESS
        neuron.do(SimpleNamespace(environment=environment, rng=TestRng()))
        assert environment.added_inflammation == pytest.approx(0.15)

    def test_do_dump_debris_exports_intracellular_debris_and_resets_it(self):
        neuron = make_neuron()
        environment = TestSubstantiaNigraLikeEnvironment()
        neuron.last_perception = make_perception()
        neuron.pending_action = NeuronAction.DUMP_DEBRIS
        neuron.internal_scalars.intracellular_debris = 0.42
        neuron.do(SimpleNamespace(environment=environment, rng=TestRng()))
        assert environment.added_debris == pytest.approx(0.42)
        assert neuron.internal_scalars.intracellular_debris == 0.0

    def test_begin_tick_resets_internal_effects(self):
        neuron = make_neuron()
        neuron.internal_effects = NeuronInternalEffects(oxidative_stress_added=1.0, debris_added=1.0)
        neuron.begin_tick()
        assert neuron.internal_effects.oxidative_stress_added == 0.0
        assert neuron.internal_effects.debris_added == 0.0

    def test_commit_effects_adds_internal_buffers_and_clamps(self):
        neuron = make_neuron()
        neuron.internal_scalars.oxidative_stress = 0.8
        neuron.internal_scalars.intracellular_debris = 0.1
        neuron.add_oxidative_stress(0.4)
        neuron.internal_effects.debris_added = 0.2
        neuron.commit_effects()
        assert neuron.internal_scalars.oxidative_stress == 1.0
        assert neuron.internal_scalars.intracellular_debris == pytest.approx(0.3)

    def test_commit_effects_applies_internal_decay_rates(self):
        neuron = make_neuron()
        neuron.internal_scalars.oxidative_stress = 1.0
        neuron.internal_scalars.intracellular_debris = 1.0
        neuron.commit_effects()
        assert neuron.internal_scalars.oxidative_stress < 1.0
        assert neuron.internal_scalars.intracellular_debris < 1.0

    def test_register_and_assign_degradation_target(self):
        neuron = make_neuron()
        lysosome = TestAgent(ptype=50, uid=1)
        target = TestAgent(ptype=99, uid=2)
        neuron.add_agent(lysosome, DiscretePoint(0, 0))
        neuron.add_agent(target, DiscretePoint(1, 0))
        neuron.register_degradation_target(target)
        neuron.assign_degradation_target(lysosome, target)
        assert neuron.target_for(lysosome) == target
        assert neuron.is_target_assigned(target) is True
        assert neuron.available_degradation_targets() == []

    def test_clear_degradation_assignment_removes_lysosome_mapping(self):
        neuron = make_neuron()
        lysosome = TestAgent(ptype=50, uid=1)
        target = TestAgent(ptype=99, uid=2)
        neuron.add_agent(lysosome, DiscretePoint(0, 0))
        neuron.add_agent(target, DiscretePoint(1, 0))
        neuron.assign_degradation_target(lysosome, target)

        neuron.clear_degradation_assignment(lysosome)

        assert neuron.target_for(lysosome) is None
        assert neuron.is_target_assigned(target) is False

    def test_remove_agent_clears_target_side_degradation_assignment(self):
        neuron = make_neuron()
        lysosome = TestAgent(ptype=50, uid=1)
        target = TestAgent(ptype=99, uid=2)
        neuron.add_agent(lysosome, DiscretePoint(0, 0))
        neuron.add_agent(target, DiscretePoint(1, 0))
        neuron.assign_degradation_target(lysosome, target)

        neuron.remove_agent(target)

        assert neuron.target_for(lysosome) is None
        assert neuron.is_target_assigned(target) is False
        assert target not in neuron.grid.agent_registry

    def test_remove_agent_ignores_non_target_agents(self):
        neuron = make_neuron()
        agent = TestAgent(ptype=7, uid=1)
        neuron.add_agent(agent, DiscretePoint(0, 0))

        neuron.remove_agent(agent)

        assert agent not in neuron.grid.agent_registry

    def test_step_runs_internal_agents_in_phases_before_neuron_step(self):
        neuron = make_neuron()
        log = []
        first = PhaseAgent(101, log, "first")
        second = PhaseAgent(102, log, "second")
        neuron.add_agent(first, DiscretePoint(0, 0))
        neuron.add_agent(second, DiscretePoint(1, 0))
        environment = TestSubstantiaNigraLikeEnvironment(position=DiscretePoint(2, 2))

        neuron.step(SimpleNamespace(environment=environment, rng=TestRng()))

        assert log == [
            "first:see",
            "second:see",
            "first:next",
            "second:next",
            "first:action",
            "second:action",
            "first:do",
            "second:do",
        ]
