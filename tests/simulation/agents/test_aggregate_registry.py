from types import SimpleNamespace
from repast4py.space import DiscretePoint
from tests.simulation.testhelpers import TestRng, import_any, make_alpha_config


alpha_module = import_any("src.simulation.agents.alphasynuclein")
AlphaSynuclein = alpha_module.AlphaSynuclein
AlphaSynucleinCompartment = alpha_module.AlphaSynucleinCompartment
AlphaSynucleinConfig = alpha_module.AlphaSynucleinConfig
AlphaSynucleinState = alpha_module.AlphaSynucleinState

aggregate_module = import_any("src.simulation.agents.aggregate")
AggregateState = aggregate_module.AggregateState

registry_module = import_any("src.simulation.agents.aggregate_registry")
AggregateRegistry = registry_module.AggregateRegistry
AggregateInvariantError = registry_module.AggregateInvariantError

neuron_module = import_any("src.simulation.agents.neuron")
Neuron = neuron_module.Neuron
NeuronConfig = neuron_module.NeuronConfig


def make_neuron() -> Neuron:
    config = NeuronConfig(
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
    environment = SimpleNamespace(aggregate_registry=AggregateRegistry())
    return Neuron(local_id=1, rank=0, type_id=10, config=config, alpha_type_id=99, environment=environment)


def make_alpha(local_id: int, owner_neuron: Neuron) -> AlphaSynuclein:
    alpha = AlphaSynuclein(
        local_id=local_id,
        rank=0,
        type_id=99,
        config=make_alpha_config(),
        compartment=AlphaSynucleinCompartment.INTRACELLULAR,
        owner_neuron=owner_neuron,
    )
    alpha.state = AlphaSynucleinState.MISFOLDED
    alpha.wants_oligomerization = True
    return alpha


class TestAggregateRegistry:
    def test_two_willing_misfolded_proteins_create_one_oligomer_agent(self):
        neuron = make_neuron()
        registry = AggregateRegistry(lewy_body_size_threshold=999, rng=TestRng(random_value=1.0))
        point = DiscretePoint(1, 1)
        first = make_alpha(1, neuron)
        second = make_alpha(2, neuron)
        neuron.add_agent(first, point)
        neuron.add_agent(second, point)

        registry.process(neuron)

        aggregates = registry.aggregates()
        assert len(aggregates) == 1
        aggregate = aggregates[0]
        assert aggregate.state == AggregateState.OLIGOMER
        assert aggregate.size == 2
        assert aggregate in neuron.grid.agent_registry
        assert first not in neuron.grid.agent_registry
        assert second not in neuron.grid.agent_registry
        assert first.aggregate_id == aggregate.aggregate_id
        assert second.aggregate_id == aggregate.aggregate_id
        assert first.state == AlphaSynucleinState.OLIGOMER
        assert second.state == AlphaSynucleinState.OLIGOMER

    def test_recruiting_oligomer_absorbs_misfolded_protein_and_matures(self):
        neuron = make_neuron()
        registry = AggregateRegistry(lewy_body_size_threshold=999, rng=TestRng(random_value=1.0))
        point = DiscretePoint(1, 1)
        first = make_alpha(1, neuron)
        second = make_alpha(2, neuron)
        neuron.add_agent(first, point)
        neuron.add_agent(second, point)
        registry.process(neuron)
        aggregate = registry.aggregates()[0]

        registry.lewy_body_size_threshold = 2
        registry.rng = TestRng(random_value=0.0)
        third = make_alpha(3, neuron)
        neuron.add_agent(third, point)
        registry.process(neuron)

        assert aggregate.size == 3
        assert aggregate.state == AggregateState.LEWY_BODY
        assert third not in neuron.grid.agent_registry
        assert third.aggregate_id == aggregate.aggregate_id
        assert third.state == AlphaSynucleinState.LEWY_BODY
        assert {member.state for member in registry.members(aggregate.aggregate_id)} == {
            AlphaSynucleinState.LEWY_BODY
        }

    def test_lewy_body_absorbs_recruiting_oligomer_in_same_cell(self):
        neuron = make_neuron()
        registry = AggregateRegistry(lewy_body_size_threshold=999, rng=TestRng(random_value=1.0))
        point = DiscretePoint(1, 1)
        lewy_members = [make_alpha(1, neuron), make_alpha(2, neuron)]
        oligomer_members = [make_alpha(3, neuron), make_alpha(4, neuron)]
        for alpha in lewy_members + oligomer_members:
            neuron.add_agent(alpha, point)

        lewy_body = registry.create_aggregate(
            neuron,
            point,
            lewy_members,
            state=AggregateState.LEWY_BODY,
        )
        oligomer = registry.create_aggregate(
            neuron,
            point,
            oligomer_members,
            state=AggregateState.OLIGOMER,
        )

        registry.lewy_body_size_threshold = 1
        registry.rng = TestRng(random_value=0.0)
        registry.process(neuron)

        assert lewy_body.size == 4
        assert lewy_body.state == AggregateState.LEWY_BODY
        assert {member.state for member in registry.members(lewy_body.aggregate_id)} == {
            AlphaSynucleinState.LEWY_BODY
        }
        assert oligomer not in neuron.grid.agent_registry
        assert registry.aggregate_for(oligomer.aggregate_id) is None
        assert registry.size(lewy_body.aggregate_id) == 4

    def test_mature_to_lewy_body_accepts_aggregate_id_and_updates_all_members(self):
        neuron = make_neuron()
        registry = AggregateRegistry(lewy_body_size_threshold=999, rng=TestRng(random_value=1.0))
        point = DiscretePoint(1, 1)
        members = [make_alpha(1, neuron), make_alpha(2, neuron)]
        for alpha in members:
            neuron.add_agent(alpha, point)
        aggregate = registry.create_aggregate(neuron, point, members)

        assert registry.mature_to_lewy_body(aggregate.aggregate_id) is True

        assert aggregate.state == AggregateState.LEWY_BODY
        assert {member.state for member in registry.members(aggregate.aggregate_id)} == {
            AlphaSynucleinState.LEWY_BODY
        }
        assert {member.aggregate_id for member in registry.members(aggregate.aggregate_id)} == {
            aggregate.aggregate_id
        }

    def test_validate_invariants_rejects_lewy_body_with_non_lewy_members(self):
        neuron = make_neuron()
        registry = AggregateRegistry(lewy_body_size_threshold=999, rng=TestRng(random_value=1.0))
        point = DiscretePoint(1, 1)
        members = [make_alpha(1, neuron), make_alpha(2, neuron)]
        for alpha in members:
            neuron.add_agent(alpha, point)
        aggregate = registry.create_aggregate(neuron, point, members, state=AggregateState.LEWY_BODY)
        next(iter(registry.members(aggregate.aggregate_id))).state = AlphaSynucleinState.OLIGOMER

        try:
            registry.validate_invariants(neuron)
        except AggregateInvariantError:
            pass
        else:
            raise AssertionError("Expected inconsistent Lewy body members to fail validation.")

    def test_removing_last_member_removes_aggregate_and_clears_member(self):
        neuron = make_neuron()
        registry = AggregateRegistry(lewy_body_size_threshold=999, rng=TestRng(random_value=1.0))
        point = DiscretePoint(1, 1)
        members = [make_alpha(1, neuron), make_alpha(2, neuron)]
        for alpha in members:
            neuron.add_agent(alpha, point)
        aggregate = registry.create_aggregate(neuron, point, members, state=AggregateState.LEWY_BODY)

        registry.remove(members[0], habitat=neuron)
        assert registry.aggregate_for(aggregate.aggregate_id) is aggregate
        assert members[0].state == AlphaSynucleinState.CLEARED
        registry.remove(members[1], habitat=neuron)

        assert registry.aggregate_for(aggregate.aggregate_id) is None
        assert aggregate not in neuron.grid.agent_registry
        assert {member.state for member in members} == {AlphaSynucleinState.CLEARED}
