import pytest
from typing import Optional
from repast4py.space import DiscretePoint

from testhelpers import TestAgent, TestRng, import_any, make_lysosome_config, make_mitochondrion_config

mitochondrion_module = import_any("src.simulation.agents.mitochondrion")
Mitochondrion = mitochondrion_module.Mitochondrion
MitochondrionAction = mitochondrion_module.MitochondrionAction
MitochondrionConfig = mitochondrion_module.MitochondrionConfig
MitochondrionPerception = mitochondrion_module.MitochondrionPerception
MitochondrionState = mitochondrion_module.MitochondrionState
lysosome_module = import_any("src.simulation.agents.lysosome")
Lysosome = lysosome_module.Lysosome
LysosomeAction = lysosome_module.LysosomeAction
LysosomeState = lysosome_module.LysosomeState
neuron_module = import_any("src.simulation.agents.neuron")
Neuron = neuron_module.Neuron
NeuronConfig = neuron_module.NeuronConfig
aggregate_module = import_any("src.simulation.agents.aggregate")
AlphaAggregate = aggregate_module.AlphaAggregate
AggregateState = aggregate_module.AggregateState


def make_neuron_config() -> NeuronConfig:
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
        alpha_release_amount=0.05
    )


def make_neuron() -> Neuron:
    return Neuron(
        local_id=1,
        rank=0,
        type_id=10,
        config=make_neuron_config(),
        alpha_type_id=99
    )


def make_aggregate(local_id: int) -> AlphaAggregate:
    return AlphaAggregate(
        local_id=local_id,
        rank=0,
        type_id=99,
        aggregate_id=local_id,
        member_ids={local_id},
        state=AggregateState.LEWY_BODY
    )


def make_mitochondrion(neuron: Neuron, config: Optional[MitochondrionConfig] = None) -> Mitochondrion:
    mitochondrion = Mitochondrion(
        local_id=20,
        rank=0,
        type_id=30,
        config=config or make_mitochondrion_config(),
        owner_neuron=neuron
    )
    mitochondrion.rng = TestRng(random_value=0.0)
    neuron.add_agent(mitochondrion, DiscretePoint(1, 1))
    return mitochondrion


def make_perception(
    energy_demand: float = 0.0,
    oxidative_stress: float = 0.0,
    local_aggregate_density: float = 0.0,
    local_debris_density: float = 0.0,
    target_assigned: bool = False
) -> MitochondrionPerception:
    return MitochondrionPerception(
        position=DiscretePoint(1, 1),
        oxidative_stress=oxidative_stress,
        energy_demand=energy_demand,
        local_aggregate_density=local_aggregate_density,
        local_debris_density=local_debris_density,
        target_assigned=target_assigned
    )


class TestMitochondrion:
    def test_see_reads_neuron_context_and_target_assignment(self):
        neuron = make_neuron()
        mitochondrion = make_mitochondrion(neuron)
        lysosome = TestAgent(ptype=50, uid=1)
        neuron.add_agent(lysosome, DiscretePoint(0, 0))
        neuron.internal_scalars.oxidative_stress = 0.4
        neuron.internal_scalars.energy_demand = 0.8
        neuron.add_agent(make_aggregate(2), DiscretePoint(1, 1))
        neuron.assign_degradation_target(lysosome, mitochondrion)
        perception = mitochondrion.see(model=None)
        assert perception.position == DiscretePoint(1, 1)
        assert perception.oxidative_stress == pytest.approx(0.4)
        assert perception.energy_demand == pytest.approx(0.8)
        assert perception.local_aggregate_density > 0.0
        assert perception.target_assigned is True

    def test_healthy_mitochondrion_reduces_neuron_energy_demand_on_commit(self):
        neuron = make_neuron()
        config = make_mitochondrion_config(
            healthy_energy_demand_reduction_rate=0.1,
            high_demand_reduction_multiplier=2.0,
        )
        mitochondrion = make_mitochondrion(neuron, config)
        neuron.internal_scalars.energy_demand = 0.8
        mitochondrion.see(model=None)
        mitochondrion.action()
        mitochondrion.do(model=None)
        neuron.commit_effects()
        assert mitochondrion.pending_action == MitochondrionAction.REDUCE_DEMAND
        assert neuron.internal_effects.energy_demand_added == pytest.approx(-0.2)
        assert neuron.internal_scalars.energy_demand < 0.8

    def test_healthy_mitochondrion_signals_stress_in_high_damage_context(self):
        neuron = make_neuron()
        mitochondrion = make_mitochondrion(neuron)
        neuron.internal_scalars.oxidative_stress = 0.9
        mitochondrion.see(model=None)
        mitochondrion.action()
        mitochondrion.do(model=None)
        assert mitochondrion.pending_action == MitochondrionAction.STRESS
        assert neuron.internal_effects.oxidative_stress_added == pytest.approx(mitochondrion.cfg.stress_release_rate)

    def test_consumed_mitochondrion_fuses_only_when_all_damage_signals_are_low(self):
        neuron = make_neuron()
        mitochondrion = make_mitochondrion(neuron)
        mitochondrion.state = MitochondrionState.CONSUMED
        mitochondrion.last_perception = make_perception(
            energy_demand=0.1,
            oxidative_stress=0.1,
            local_aggregate_density=0.1,
            local_debris_density=0.1
        )
        mitochondrion.action()
        mitochondrion.do(model=None)
        assert mitochondrion.pending_action == MitochondrionAction.FUSE
        assert mitochondrion.state == MitochondrionState.HEALTHY
        assert neuron.internal_effects.oxidative_stress_added < 0.0
        assert neuron.internal_effects.debris_added < 0.0

    def test_damaged_mitochondrion_registers_for_lysosomal_degradation(self):
        neuron = make_neuron()
        mitochondrion = make_mitochondrion(neuron)
        mitochondrion.state = MitochondrionState.DAMAGED
        mitochondrion.last_perception = make_perception()
        mitochondrion.action()
        mitochondrion.do(model=None)
        assert mitochondrion.pending_action == MitochondrionAction.DIVIDE
        assert mitochondrion in neuron.available_degradation_targets()
        assert neuron.internal_effects.debris_added == pytest.approx(mitochondrion.cfg.debris_release_rate)

    def test_assigned_mitochondrion_waits_for_lysosome_without_extra_damage_output(self):
        neuron = make_neuron()
        mitochondrion = make_mitochondrion(neuron)
        mitochondrion.state = MitochondrionState.DAMAGED
        mitochondrion.last_perception = make_perception(target_assigned=True)
        mitochondrion.action()
        mitochondrion.do(model=None)
        assert mitochondrion.pending_action is None
        assert neuron.internal_effects.debris_added == 0.0
        assert neuron.internal_effects.oxidative_stress_added == 0.0

    def test_lysosome_repairs_damaged_mitochondrion(self):
        neuron = make_neuron()
        mitochondrion = make_mitochondrion(neuron)
        mitochondrion.state = MitochondrionState.DAMAGED
        lysosome = Lysosome(
            local_id=50,
            rank=0,
            type_id=60,
            owner_neuron=neuron,
            config=make_lysosome_config(
                base_degradation_probability=1.0,
                mitochondrion_repair_ticks=1,
                mitochondrion_repair_probability=1.0
            )
        )
        lysosome.state = LysosomeState.ACTIVE
        lysosome.rng = TestRng(random_value=0.0)
        neuron.add_agent(lysosome, DiscretePoint(0, 0))
        neuron.register_degradation_target(mitochondrion)
        neuron.assign_degradation_target(lysosome, mitochondrion)
        lysosome.pending_action = LysosomeAction.DEGRADE
        lysosome.do(model=None)
        assert mitochondrion in neuron.grid.agent_registry
        assert mitochondrion.state == MitochondrionState.HEALTHY
        assert neuron.target_for(lysosome) is None