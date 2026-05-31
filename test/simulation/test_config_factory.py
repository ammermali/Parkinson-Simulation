from pathlib import Path

import pytest

from testhelpers import TestRng, import_any


factory = import_any("src.simulation.utils.config_factory")
params_module = import_any("src.simulation.utils.paramsloader")
Params = params_module.Params
alpha_module = import_any("src.simulation.agents.alphasynuclein")
AlphaSynucleinConfig = alpha_module.AlphaSynucleinConfig
astrocyte_module = import_any("src.simulation.agents.astrocyte")
AstrocyteConfig = astrocyte_module.AstrocyteConfig
lysosome_module = import_any("src.simulation.agents.lysosome")
LysosomeConfig = lysosome_module.LysosomeConfig
microglia_module = import_any("src.simulation.agents.microglia")
MicrogliaConfig = microglia_module.MicrogliaConfig
mitochondrion_module = import_any("src.simulation.agents.mitochondrion")
MitochondrionConfig = mitochondrion_module.MitochondrionConfig
neuron_module = import_any("src.simulation.agents.neuron")
NeuronConfig = neuron_module.NeuronConfig
sn_module = import_any("src.simulation.substantia_nigra")
SNEnvironmentConfig = sn_module.SNEnvironmentConfig


def test_params_resolves_bare_name_and_yaml_filename():
    bare = Params("microglia")
    filename = Params("microglia.yaml")
    assert bare.path == filename.path
    assert bare.path.name == "microglia.yaml"
    assert bare.section("perception") == filename.section("perception")


def test_params_honors_explicit_path(tmp_path: Path):
    path = tmp_path / "custom.yaml"
    path.write_text("section:\n  value: 3\n", encoding="utf-8")
    params = Params(str(path))
    assert params.path == path
    assert params.section("section") == {"value": 3}


def test_config_factory_builds_all_configs_from_default_yaml_files():
    rng = TestRng(random_value=0.0)
    assert isinstance(factory.build_substantia_nigra_config(), SNEnvironmentConfig)
    assert isinstance(factory.build_microglia_config(rng=rng), MicrogliaConfig)
    assert isinstance(factory.build_astrocyte_config(rng=rng), AstrocyteConfig)
    assert isinstance(factory.build_neuron_config(rng=rng), NeuronConfig)
    assert isinstance(factory.build_alpha_synuclein_config(rng=rng), AlphaSynucleinConfig)
    assert isinstance(factory.build_mitochondrion_config(rng=rng), MitochondrionConfig)
    assert isinstance(factory.build_lysosome_config(), LysosomeConfig)


def test_thresholds_are_sampled_from_nested_mean_std(tmp_path: Path):
    path = tmp_path / "microglia.yaml"
    path.write_text(
        """
perception:
  per_radius: 2
thresholds:
  debris_high_threshold: {mean: 0.5, std: 0.1}
  debris_low_threshold: {mean: 0.2, std: 0.1}
  inflammation_high_threshold: {mean: 0.6, std: 0.1}
  inflammation_low_threshold: {mean: 0.3, std: 0.1}
  nearby_alpha_high_threshold: {mean: 0.4, std: 0.1}
  nearby_alpha_low_threshold: {mean: 0.1, std: 0.1}
rates:
  debris_clearance_rate: 0.11
  inflammation_release_rate: 0.12
movement:
  move_probability: 0.8
dynamics:
  activation_transition_rate: 0.21
  clearing_transition_rate: 0.22
  recovery_transition_rate: 0.23
""",
        encoding="utf-8",
    )
    config = factory.build_microglia_config(Params(str(path)), rng=TestRng(random_value=2.0))
    assert config.per_radius == 2
    assert config.debris_high_threshold == pytest.approx(0.7)
    assert config.debris_low_threshold == pytest.approx(0.4)
    assert config.debris_clearance_rate == pytest.approx(0.11)
    assert config.move_probability == pytest.approx(0.8)
    assert config.activation_transition_rate == pytest.approx(0.21)
    assert config.clearing_transition_rate == pytest.approx(0.22)
    assert config.recovery_transition_rate == pytest.approx(0.23)


def test_sampled_thresholds_are_clamped(tmp_path: Path):
    path = tmp_path / "astrocyte.yaml"
    path.write_text(
        """
thresholds:
  inflammation_high_threshold: {mean: 0.9, std: 0.2}
  inflammation_low_threshold: {mean: 0.1, std: 0.2}
  debris_high_threshold: {mean: 0.9, std: 0.2}
  debris_low_threshold: {mean: 0.1, std: 0.2}
rates:
  support_inflammation_reduction_rate: 0.05
  inflammation_release_rate: 0.08
dynamics:
  debris_stress_weight: 0.25
""",
        encoding="utf-8",
    )
    high = factory.build_astrocyte_config(Params(str(path)), rng=TestRng(random_value=2.0))
    low = factory.build_astrocyte_config(Params(str(path)), rng=TestRng(random_value=-2.0))
    assert high.inflammation_high_threshold == 1.0
    assert high.debris_stress_weight == pytest.approx(0.25)
    assert low.inflammation_low_threshold == 0.0


def test_alpha_config_reads_non_random_misfolding_rates(tmp_path: Path):
    path = tmp_path / "alpha.yaml"
    path.write_text(
        """
perception:
  perception_radius: 2
movement:
  move_radius: 3
  move_probability: 0.4
rates:
  basal_misfold_probability: 0.006
  aggregate_seeded_misfold_weight: 0.25
  oxidative_misfolding_weight: 0.7
  oligomerization_probability_scale: 0.4
  min_misfolded_ticks_before_oligomerization: 3
thresholds:
  oxidative_stress_high_threshold: {mean: 0.5, std: 0.1}
""",
        encoding="utf-8",
    )

    config = factory.build_alpha_synuclein_config(Params(str(path)), rng=TestRng(random_value=0.0))

    assert config.perception_radius == 2
    assert config.move_radius == 3
    assert config.move_probability == pytest.approx(0.4)
    assert config.oxidative_stress_high_threshold == pytest.approx(0.5)
    assert config.basal_misfold_probability == pytest.approx(0.006)
    assert config.aggregate_seeded_misfold_weight == pytest.approx(0.25)
    assert config.oxidative_misfolding_weight == pytest.approx(0.7)
    assert config.oligomerization_probability_scale == pytest.approx(0.4)
    assert config.min_misfolded_ticks_before_oligomerization == 3


def test_substantia_nigra_config_reads_saturating_effects(tmp_path: Path):
    path = tmp_path / "substantia_nigra.yaml"
    path.write_text(
        """
initial:
  debris: 0.1
  inflammation: 0.2
  dopamine: 0.3
decay:
  debris: 0.01
  inflammation: 0.02
dopamine:
  smoothing: 0.5
effects:
  debris_added_max_delta: 0.11
  debris_removed_max_delta: 0.12
  debris_effect_scale: 0.8
  inflammation_added_max_delta: 0.13
  inflammation_removed_max_delta: 0.14
  inflammation_effect_scale: 0.9
""",
        encoding="utf-8",
    )

    config = factory.build_substantia_nigra_config(Params(str(path)))

    assert config.debris_added_max_delta == pytest.approx(0.11)
    assert config.debris_removed_max_delta == pytest.approx(0.12)
    assert config.debris_effect_scale == pytest.approx(0.8)
    assert config.inflammation_added_max_delta == pytest.approx(0.13)
    assert config.inflammation_removed_max_delta == pytest.approx(0.14)
    assert config.inflammation_effect_scale == pytest.approx(0.9)
