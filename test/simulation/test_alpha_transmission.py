from types import SimpleNamespace
from repast4py.space import DiscretePoint
from testhelpers import TestRng, import_any


alpha_module = import_any("src.simulation.agents.alphasynuclein")
AlphaSynuclein = alpha_module.AlphaSynuclein
AlphaSynucleinAction = alpha_module.AlphaSynucleinAction
AlphaSynucleinCompartment = alpha_module.AlphaSynucleinCompartment
AlphaSynucleinConfig = alpha_module.AlphaSynucleinConfig
AlphaSynucleinState = alpha_module.AlphaSynucleinState
aggregate_module = import_any("src.simulation.agents.aggregate")
AlphaAggregate = aggregate_module.AlphaAggregate
AggregateState = aggregate_module.AggregateState
grid_module = import_any("src.simulation.utils.grid")
LocalGrid = grid_module.LocalGrid
neuron_module = import_any("src.simulation.agents.neuron")
Neuron = neuron_module.Neuron
NeuronAction = neuron_module.NeuronAction
NeuronConfig = neuron_module.NeuronConfig
NeuronPerception = neuron_module.NeuronPerception
NeuronState = neuron_module.NeuronState


class AlphaTransmissionEnvironment:
    """Grid-backed Substantia Nigra stand-in for alpha transfer tests."""
    def __init__(self):
        self.grid = LocalGrid(width=10, height=10)
        self.scalars = SimpleNamespace(
            extracellular_debris=0.0,
            inflammation_level=0.0,
            dopamine_output=0.0,
        )
        self.added_debris = 0.0
        self.added_inflammation = 0.0
        self.released_dopamine = 0.0
        self.moves = []

    def add_agent(self, agent, point: DiscretePoint):
        """Add one extracellular agent to the shared grid."""

        return self.grid.add_agent(agent, point)

    def remove_agent(self, agent):
        """Remove one extracellular agent from the shared grid."""

        return self.grid.remove_agent(agent)

    def position_of(self, agent):
        """Return an agent location in the shared grid."""

        return self.grid.position_of(agent)

    def agents_in_radius(self, center: DiscretePoint, radius: int = 1, include_center: bool = False):
        """Yield agents around a point in the shared grid."""

        return self.grid.agents_in_radius(center, radius, include_center)

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type=None, include_center: bool = True):
        """Return local density for one agent type in the shared grid."""

        return self.grid.density_of_type(center, radius, agent_type, include_center)

    def neighbor_points(self, center: DiscretePoint, radius: int = 1, include_center: bool = True):
        """Yield valid neighboring points in the shared grid."""

        return self.grid.neighbor_points(center, radius, include_center)

    def move_to(self, agent, point: DiscretePoint):
        """Move an extracellular agent and record the move for assertions."""

        self.moves.append((agent, point))
        return self.grid.move_to(agent, point)

    def add_debris(self, amount: float):
        """Record debris released by a ruptured neuron."""

        self.added_debris += amount
        self.scalars.extracellular_debris += amount

    def add_inflammation(self, amount: float):
        """Record inflammatory signal released by a neuron."""

        self.added_inflammation += amount
        self.scalars.inflammation_level += amount

    def release_dopamine(self, amount: float):
        """Record dopamine released by a neuron."""

        self.released_dopamine += amount
        self.scalars.dopamine_output += amount


def make_config(alpha_absorption_rate: float = 1.0, alpha_release_amount: float = 0.5) -> NeuronConfig:
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
        alpha_absorption_rate=alpha_absorption_rate,
        alpha_release_amount=alpha_release_amount,
    )


def make_neuron(alpha_absorption_rate: float = 1.0, alpha_release_amount: float = 0.5) -> Neuron:
    return Neuron(
        local_id=1,
        rank=0,
        type_id=10,
        config=make_config(alpha_absorption_rate, alpha_release_amount),
        alpha_type_id=99,
    )


def make_alpha(
    local_id: int,
    owner_neuron=None,
    compartment: AlphaSynucleinCompartment = AlphaSynucleinCompartment.INTRACELLULAR,
    state: AlphaSynucleinState = AlphaSynucleinState.MONOMER,
) -> AlphaSynuclein:
    alpha = AlphaSynuclein(
        local_id=local_id,
        rank=0,
        type_id=99,
        config=AlphaSynucleinConfig(),
        compartment=compartment,
        owner_neuron=owner_neuron,
    )
    alpha.state = state
    return alpha


def make_perception() -> NeuronPerception:
    return NeuronPerception(
        position=DiscretePoint(4, 4),
        nearby_alpha=0.0,
        inflammatory_levels=0.0,
        extracellular_debris=0.0,
        oxidative_stress=0.0,
        intracellular_debris=0.0,
        energy_demand=0.5,
        internal_damage=0.0,
        alpha_load=0.0,
        cell_damage=0.0,
    )


def make_intracellular_aggregate(neuron: Neuron, point: DiscretePoint):
    members = [
        make_alpha(2, neuron, state=AlphaSynucleinState.MISFOLDED),
        make_alpha(3, neuron, state=AlphaSynucleinState.MISFOLDED),
    ]
    for member in members:
        member.wants_oligomerization = True
        neuron.add_agent(member, point)
    aggregate = neuron.aggregate_registry.create_aggregate(neuron, point, members)
    assert aggregate is not None
    return aggregate, members


class TestAlphaTransmission:
    def test_ruptured_neuron_releases_all_uncleared_alpha_and_aggregates(self):
        neuron = make_neuron(alpha_release_amount=0.01)
        environment = AlphaTransmissionEnvironment()
        release_point = DiscretePoint(4, 4)
        environment.add_agent(neuron, release_point)
        internal_point = DiscretePoint(1, 1)
        free_alpha = make_alpha(1, neuron, state=AlphaSynucleinState.MONOMER)
        cleared_alpha = make_alpha(4, neuron, state=AlphaSynucleinState.CLEARED)
        neuron.add_agent(free_alpha, internal_point)
        neuron.add_agent(cleared_alpha, internal_point)
        aggregate, members = make_intracellular_aggregate(neuron, internal_point)
        neuron.state = NeuronState.RUPTURED
        neuron.last_perception = make_perception()
        neuron.pending_action = NeuronAction.DUMP_DEBRIS
        neuron.internal_scalars.intracellular_debris = 0.42

        neuron.do(SimpleNamespace(environment=environment))

        assert environment.added_debris == 0.42
        assert free_alpha in environment.grid.agent_registry
        assert aggregate in environment.grid.agent_registry
        assert cleared_alpha in neuron.grid.agent_registry
        assert cleared_alpha not in environment.grid.agent_registry
        assert free_alpha.compartment == AlphaSynucleinCompartment.EXTRACELLULAR
        assert free_alpha.owner_neuron is None
        assert aggregate.owner_neuron is None
        assert neuron.aggregate_registry.aggregate_for(aggregate.aggregate_id) is None
        assert all(member.compartment == AlphaSynucleinCompartment.EXTRACELLULAR for member in members)
        assert all(member.owner_neuron is None for member in members)

    def test_non_ruptured_release_is_gradual(self):
        neuron = make_neuron(alpha_release_amount=0.5)
        environment = AlphaTransmissionEnvironment()
        environment.add_agent(neuron, DiscretePoint(4, 4))
        alphas = [
            make_alpha(10, neuron),
            make_alpha(11, neuron),
            make_alpha(12, neuron),
        ]
        for index, alpha in enumerate(alphas):
            neuron.add_agent(alpha, DiscretePoint(index, 0))
        neuron.state = NeuronState.APOPTOTIC

        released = neuron.release_alpha(SimpleNamespace(environment=environment))

        assert len(released) == 2
        assert sum(alpha in environment.grid.agent_registry for alpha in alphas) == 2
        assert sum(alpha in neuron.grid.agent_registry for alpha in alphas) == 1

    def test_absorb_alpha_moves_one_external_protein_into_neuron(self):
        neuron = make_neuron(alpha_absorption_rate=1.0)
        neuron.rng = TestRng(random_value=0.0)
        environment = AlphaTransmissionEnvironment()
        environment.add_agent(neuron, DiscretePoint(4, 4))
        alpha = make_alpha(
            20,
            owner_neuron=None,
            compartment=AlphaSynucleinCompartment.EXTRACELLULAR,
            state=AlphaSynucleinState.MISFOLDED,
        )
        environment.add_agent(alpha, DiscretePoint(4, 4))

        absorbed = neuron.absorb_alpha(SimpleNamespace(environment=environment))

        assert absorbed is alpha
        assert alpha not in environment.grid.agent_registry
        assert alpha in neuron.grid.agent_registry
        assert alpha.compartment == AlphaSynucleinCompartment.INTRACELLULAR
        assert alpha.owner_neuron is neuron
        assert neuron.position_of(alpha) == neuron._default_internal_point()

    def test_absorb_aggregate_preserves_members_and_registers_new_owner(self):
        neuron = make_neuron(alpha_absorption_rate=1.0)
        neuron.rng = TestRng(random_value=0.0)
        environment = AlphaTransmissionEnvironment()
        environment.add_agent(neuron, DiscretePoint(4, 4))
        members = [
            make_alpha(
                30,
                owner_neuron=None,
                compartment=AlphaSynucleinCompartment.EXTRACELLULAR,
                state=AlphaSynucleinState.OLIGOMER,
            ),
            make_alpha(
                31,
                owner_neuron=None,
                compartment=AlphaSynucleinCompartment.EXTRACELLULAR,
                state=AlphaSynucleinState.OLIGOMER,
            ),
        ]
        aggregate = AlphaAggregate(
            local_id=40,
            rank=0,
            type_id=99,
            aggregate_id=40,
            state=AggregateState.OLIGOMER,
            owner_neuron=None,
        )
        for member in members:
            member.join_aggregate(aggregate.aggregate_id, AlphaSynucleinState.OLIGOMER)
            member.release_to_environment()
            aggregate.add_member(member.uid, member)
        environment.add_agent(aggregate, DiscretePoint(4, 4))

        absorbed = neuron.absorb_alpha(SimpleNamespace(environment=environment))

        assert absorbed is aggregate
        assert aggregate not in environment.grid.agent_registry
        assert aggregate in neuron.grid.agent_registry
        assert aggregate.owner_neuron is neuron
        assert neuron.aggregate_registry.aggregate_for(aggregate.aggregate_id) is aggregate
        assert neuron.aggregate_registry.size(aggregate.aggregate_id) == 2
        assert neuron.aggregate_registry.new_id() == 41
        assert aggregate in neuron.degradation_targets
        assert all(member.compartment == AlphaSynucleinCompartment.INTRACELLULAR for member in members)
        assert all(member.owner_neuron is neuron for member in members)

    def test_extracellular_alpha_is_frozen(self):
        environment = AlphaTransmissionEnvironment()
        alpha = make_alpha(
            50,
            owner_neuron=None,
            compartment=AlphaSynucleinCompartment.EXTRACELLULAR,
            state=AlphaSynucleinState.MISFOLDED,
        )
        point = DiscretePoint(4, 4)
        environment.add_agent(alpha, point)

        alpha.see(SimpleNamespace(environment=environment))
        alpha.next()
        action = alpha.action()
        alpha.do(SimpleNamespace(environment=environment))

        assert action == AlphaSynucleinAction.STAY
        assert environment.position_of(alpha) == point
        assert environment.moves == []
        assert alpha.wants_oligomerization is False
