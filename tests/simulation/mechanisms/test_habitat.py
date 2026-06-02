import inspect
import pytest
from repast4py.space import DiscretePoint
from testhelpers import TestRepastGrid, import_any

habitat_module = import_any("src.simulation.utils.habitat")
GridHabitatMixin = habitat_module.GridHabitatMixin
InternalHabitatMixin = habitat_module.InternalHabitatMixin
sn_module = import_any("src.simulation.substantia_nigra")
SubstantiaNigra = sn_module.SubstantiaNigra
SNEnvironmentConfig = sn_module.SNEnvironmentConfig
neuron_module = import_any("src.simulation.agents.neuron")
Neuron = neuron_module.Neuron
NeuronConfig = neuron_module.NeuronConfig


def make_environment_config() -> SNEnvironmentConfig:
    return SNEnvironmentConfig(
        initial_debris=0.0,
        initial_inflammation=0.0,
        initial_dopamine=0.0,
        debris_decay=0.0,
        inflammation_decay=0.0,
        dopamine_smoothing=0.5,
    )


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
        alpha_release_amount=0.05,
    )


class TestGridHabitatContract:
    def test_mixin_fails_clearly_when_grid_is_missing(self):
        with pytest.raises(AttributeError, match="must define self.grid"):
            GridHabitatMixin().agents_at(DiscretePoint(0, 0))

    @pytest.mark.parametrize(
        "method_name",
        [
            "agents_in_radius",
            "count_agents_in_radius",
            "neighbor_points",
            "density_of_type",
        ],
    )
    def test_neuron_and_substantia_nigra_expose_same_grid_method_parameters(self, method_name):
        environment = SubstantiaNigra(TestRepastGrid(), make_environment_config())
        neuron = Neuron(
            local_id=1,
            rank=0,
            type_id=10,
            config=make_neuron_config(),
            alpha_type_id=99,
        )

        environment_params = inspect.signature(getattr(environment, method_name)).parameters
        neuron_params = inspect.signature(getattr(neuron, method_name)).parameters

        assert environment_params.keys() == neuron_params.keys()
        assert "include_center" in environment_params

    def test_neuron_implements_internal_habitat_contract(self):
        neuron = Neuron(
            local_id=1,
            rank=0,
            type_id=10,
            config=make_neuron_config(),
            alpha_type_id=99,
        )

        assert isinstance(neuron, InternalHabitatMixin)
        assert neuron.energy_demand_at() == 0.5
        assert neuron.local_debris_density_at(None) == 0.0
