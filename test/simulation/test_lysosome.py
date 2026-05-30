import pytest
from repast4py.space import DiscretePoint
from testhelpers import TestAgent, TestRng, import_any

alpha_module = import_any("src.simulation.agents.alphasynuclein")
AlphaSynuclein = alpha_module.AlphaSynuclein
AlphaSynucleinCompartment = alpha_module.AlphaSynucleinCompartment
AlphaSynucleinConfig = alpha_module.AlphaSynucleinConfig
AlphaSynucleinState = alpha_module.AlphaSynucleinState
aggregate_module = import_any("src.simulation.agents.aggregate")
AlphaAggregate = aggregate_module.AlphaAggregate
AggregateState = aggregate_module.AggregateState
lysosome_module = import_any("src.simulation.agents.lysosome")
Lysosome = lysosome_module.Lysosome
LysosomeAction = lysosome_module.LysosomeAction
LysosomeState = lysosome_module.LysosomeState
neuron_module = import_any("src.simulation.agents.neuron")
Neuron = neuron_module.Neuron
NeuronConfig = neuron_module.NeuronConfig

def make_config() -> NeuronConfig:
    return NeuronConfig(
        per_radius=1,
        nearby_alpha_high_threshold=0.5,
        inflammation_high_threshold=0.7,
        debris_high_threshold=0.6,
        alpha_load_release_threshold=0.02,
        damage_accumulation_rate=1.0,
        damage_recovery_rate=0.1,
        low_stress_threshold=0.05,
        inflammation_damage_weight=0.4,
        debris_damage_weight=0.3,
        alpha_damage_weight=0.3,
        compromised_threshold=0.3,
        apoptotic_threshold=0.6,
        ruptured_threshold=0.9,
        dopamine_release_rate=0.2,
        stress_inflammation_release_rate=0.15,
        debris_release_rate=0.25,
        alpha_absorption_rate=0.1,
        alpha_release_amount=0.05,
    )


def make_neuron() -> Neuron:
    return Neuron(local_id=1, rank=0, type_id=10, config=make_config(), alpha_type_id=99)


def make_lysosome(neuron: Neuron, **kwargs) -> Lysosome:
    lysosome = Lysosome(
        local_id=50,
        rank=0,
        type_id=60,
        owner_neuron=neuron,
        **kwargs
    )
    lysosome.state = LysosomeState.ACTIVE
    lysosome.rng = TestRng(random_value=0.0)
    neuron.add_agent(lysosome, DiscretePoint(0, 0))
    return lysosome


def make_alpha(local_id: int, owner_neuron: Neuron) -> AlphaSynuclein:
    alpha = AlphaSynuclein(
        local_id=local_id,
        rank=0,
        type_id=99,
        config=AlphaSynucleinConfig(),
        compartment=AlphaSynucleinCompartment.INTRACELLULAR,
        owner_neuron=owner_neuron
    )
    alpha.state = AlphaSynucleinState.MISFOLDED
    alpha.wants_oligomerization = True
    return alpha


class TestLysosome:
    def test_select_target_claims_one_neuron_buffer_target(self):
        neuron = make_neuron()
        lysosome = make_lysosome(neuron)
        target = TestAgent(ptype=99, uid=1)
        neuron.add_agent(target, DiscretePoint(1, 0))
        neuron.register_degradation_target(target)
        lysosome.see(model=None)
        lysosome.action()
        lysosome.do(model=None)
        assert lysosome.pending_action == LysosomeAction.SELECT_TARGET
        assert neuron.target_for(lysosome) == target
        assert neuron.available_degradation_targets() == []

    def test_aggregate_degradation_probability_decreases_with_size_and_failed_attempt_keeps_assignment(self):
        neuron = make_neuron()
        lysosome = make_lysosome(
            neuron,
            base_degradation_probability=1.0,
            aggregate_size_penalty=1.0
        )
        lysosome.rng = TestRng(random_value=0.5)
        aggregate = AlphaAggregate(
            local_id=100,
            rank=0,
            type_id=99,
            aggregate_id=1,
            member_ids={1, 2, 3},
            state=AggregateState.OLIGOMER,
            owner_neuron=neuron
        )
        neuron.add_agent(aggregate, DiscretePoint(1, 0))
        neuron.register_degradation_target(aggregate)
        neuron.assign_degradation_target(lysosome, aggregate)
        lysosome.pending_action = LysosomeAction.DEGRADE
        lysosome.do(model=None)

        assert lysosome.pr_degradation_success(aggregate) == pytest.approx(1 / 3)
        assert aggregate in neuron.grid.agent_registry
        assert neuron.target_for(lysosome) is aggregate

    def test_successful_aggregate_degradation_removes_aggregate_and_clears_members(self):
        neuron = make_neuron()
        lysosome = make_lysosome(
            neuron,
            base_degradation_probability=1.0,
            aggregate_size_penalty=0.0,
        )
        point = DiscretePoint(1, 1)
        members = [make_alpha(1, neuron), make_alpha(2, neuron)]
        for alpha in members:
            neuron.add_agent(alpha, point)
        aggregate = neuron.aggregate_registry.create_aggregate(neuron, point, members)
        neuron.assign_degradation_target(lysosome, aggregate)

        lysosome.pending_action = LysosomeAction.DEGRADE
        lysosome.do(model=None)

        assert aggregate not in neuron.grid.agent_registry
        assert neuron.target_for(lysosome) is None
        assert neuron.aggregate_registry.aggregate_for(aggregate.aggregate_id) is None
        assert {member.state for member in members} == {AlphaSynucleinState.CLEARED}

    def test_lewy_body_overwhelms_lysosome_and_is_requeued(self):
        neuron = make_neuron()
        lysosome = make_lysosome(neuron)
        lewy_body = AlphaAggregate(
            local_id=100,
            rank=0,
            type_id=99,
            aggregate_id=1,
            member_ids={1, 2, 3},
            state=AggregateState.LEWY_BODY,
            owner_neuron=neuron,
        )
        neuron.add_agent(lewy_body, DiscretePoint(1, 0))
        neuron.register_degradation_target(lewy_body)
        neuron.assign_degradation_target(lysosome, lewy_body)
        lysosome.pending_action = LysosomeAction.DEGRADE
        lysosome.do(model=None)
        assert lysosome.state == LysosomeState.OVERWHELMED
        assert lysosome.pending_action == LysosomeAction.IDLE
        assert neuron.target_for(lysosome) is None
        assert lewy_body in neuron.grid.agent_registry
        assert lewy_body in neuron.available_degradation_targets()

    def test_overwhelmed_lysosome_stays_idle(self):
        neuron = make_neuron()
        lysosome = make_lysosome(neuron)
        lysosome.state = LysosomeState.OVERWHELMED
        lysosome.last_perception = lysosome.see(model=None)
        lysosome.next()
        lysosome.action()
        assert lysosome.state == LysosomeState.OVERWHELMED
        assert lysosome.pending_action == LysosomeAction.IDLE
