from pathlib import Path
from typing import Optional, Union
from src.simulation.agents.alphasynuclein import AlphaSynucleinConfig
from src.simulation.agents.astrocyte import AstrocyteConfig
from src.simulation.agents.lysosome import LysosomeConfig
from src.simulation.agents.microglia import MicrogliaConfig
from src.simulation.agents.mitochondrion import MitochondrionConfig
from src.simulation.agents.neuron import NeuronConfig
from src.simulation.substantia_nigra import SNEnvironmentConfig
from src.simulation.utils import RNG, clamp
from src.simulation.utils.paramsloader import Params
ParamsSource = Union[Params, str, Path]


def build_substantia_nigra_config(params: ParamsSource = "substantia_nigra") -> SNEnvironmentConfig:
    """Build Substantia Nigra environment config from YAML parameters."""
    params = _coerce_params(params)
    initial = params.section("initial")
    decay = params.section("decay")
    dopamine = params.section("dopamine")
    return SNEnvironmentConfig(
        initial_debris=_number(initial, "debris"),
        initial_inflammation=_number(initial, "inflammation"),
        initial_dopamine=_number(initial, "dopamine"),
        debris_decay=_number(decay, "debris"),
        inflammation_decay=_number(decay, "inflammation"),
        dopamine_smoothing=_number(dopamine, "smoothing")
    )

def build_microglia_config(params: ParamsSource = "microglia", rng: Optional[RNG] = None) -> MicrogliaConfig:
    """Build one microglia config, sampling perception thresholds once."""
    params = _coerce_params(params)
    rng = RNG()
    perception = params.section("perception")
    rates = params.section("rates")
    movement = params.section("movement")
    return MicrogliaConfig(
        per_radius=_integer(perception, "per_radius"),
        debris_high_threshold=_threshold(params, "debris_high_threshold", rng),
        debris_low_threshold=_threshold(params, "debris_low_threshold", rng),
        inflammation_high_threshold=_threshold(params, "inflammation_high_threshold", rng),
        inflammation_low_threshold=_threshold(params, "inflammation_low_threshold", rng),
        nearby_alpha_high_threshold=_threshold(params, "nearby_alpha_high_threshold", rng),
        nearby_alpha_low_threshold=_threshold(params, "nearby_alpha_low_threshold", rng),
        debris_clearance_rate=_number(rates, "debris_clearance_rate"),
        inflammation_release_rate=_number(rates, "inflammation_release_rate"),
        move_probability=_number(movement, "move_probability")
    )

def build_astrocyte_config(params: ParamsSource = "astrocyte", rng: Optional[RNG] = None) -> AstrocyteConfig:
    """Build one astrocyte config, sampling perception thresholds once."""
    params = _coerce_params(params)
    rng = RNG()
    rates = params.section("rates")
    return AstrocyteConfig(
        inflammation_high_threshold=_threshold(params, "inflammation_high_threshold", rng),
        inflammation_low_threshold=_threshold(params, "inflammation_low_threshold", rng),
        debris_high_threshold=_threshold(params, "debris_high_threshold", rng),
        debris_low_threshold=_threshold(params, "debris_low_threshold", rng),
        support_inflammation_reduction_rate=_number(rates, "support_inflammation_reduction_rate"),
        inflammation_release_rate=_number(rates, "inflammation_release_rate")
    )


def build_neuron_config(params: ParamsSource = "neuron", rng: Optional[RNG] = None) -> NeuronConfig:
    """Build one neuron config, sampling extracellular perception thresholds once."""
    params = _coerce_params(params)
    rng = rng or RNG()
    perception = params.section("perception")
    thresholds = params.section("thresholds")
    damage = params.section("damage")
    rates = params.section("rates")
    alpha = params.section("alpha")
    return NeuronConfig(
        per_radius=_integer(perception, "per_radius"),
        nearby_alpha_high_threshold=_threshold(params, "nearby_alpha_high_threshold", rng),
        inflammation_high_threshold=_threshold(params, "inflammation_high_threshold", rng),
        debris_high_threshold=_threshold(params, "debris_high_threshold", rng),
        alpha_load_release_threshold=_threshold(params, "alpha_load_release_threshold", rng),
        damage_accumulation_rate=_number(damage, "damage_accumulation_rate"),
        damage_recovery_rate=_number(damage, "damage_recovery_rate"),
        low_stress_threshold=_threshold(params, "low_stress_threshold", rng),
        inflammation_damage_weight=_number(damage, "inflammation_damage_weight"),
        debris_damage_weight=_number(damage, "debris_damage_weight"),
        alpha_damage_weight=_number(damage, "alpha_damage_weight"),
        compromised_threshold=_number(thresholds, "compromised_threshold"),
        apoptotic_threshold=_number(thresholds, "apoptotic_threshold"),
        ruptured_threshold=_number(thresholds, "ruptured_threshold"),
        dopamine_release_rate=_number(rates, "dopamine_release_rate"),
        stress_inflammation_release_rate=_number(rates, "stress_inflammation_release_rate"),
        debris_release_rate=_number(rates, "debris_release_rate"),
        alpha_absorption_rate=_number(alpha, "alpha_absorption_rate"),
        alpha_release_amount=_number(alpha, "alpha_release_amount")
    )


def build_alpha_synuclein_config(params: ParamsSource = "alpha", rng: Optional[RNG] = None) -> AlphaSynucleinConfig:
    """Build one alpha-synuclein config, sampling misfolding threshold once."""
    params = _coerce_params(params)
    rng = rng or RNG()
    perception = params.section("perception")
    movement = params.section("movement")
    return AlphaSynucleinConfig(
        perception_radius=_integer(perception, "perception_radius"),
        move_radius=_integer(movement, "move_radius"),
        move_probability=_number(movement, "move_probability"),
        oxidative_stress_high_threshold=_threshold(params, "oxidative_stress_high_threshold", rng)
    )


def build_mitochondrion_config(params: ParamsSource = "mitochondrion", rng: Optional[RNG] = None) -> MitochondrionConfig:
    """Build one mitochondrion config, sampling perception thresholds once."""
    params = _coerce_params(params)
    rng = rng or RNG()
    perception = params.section("perception")
    rates = params.section("rates")
    return MitochondrionConfig(
        perception_radius=_integer(perception, "perception_radius"),
        energy_demand_high_threshold=_threshold(params, "energy_demand_high_threshold", rng),
        oxidative_stress_high_threshold=_threshold(params, "oxidative_stress_high_threshold", rng),
        oxidative_stress_low_threshold=_threshold(params, "oxidative_stress_low_threshold", rng),
        aggregate_density_high_threshold=_threshold(params, "aggregate_density_high_threshold", rng),
        aggregate_density_low_threshold=_threshold(params, "aggregate_density_low_threshold", rng),
        debris_density_high_threshold=_threshold(params, "debris_density_high_threshold", rng),
        debris_density_low_threshold=_threshold(params, "debris_density_low_threshold", rng),
        irreversible_damage_threshold=_threshold(params, "irreversible_damage_threshold", rng),
        stress_release_rate=_number(rates, "stress_release_rate"),
        damage_stress_release_rate=_number(rates, "damage_stress_release_rate"),
        debris_release_rate=_number(rates, "debris_release_rate"),
        fusion_stress_reduction_rate=_number(rates, "fusion_stress_reduction_rate"),
        fusion_debris_reduction_rate=_number(rates, "fusion_debris_reduction_rate"),
        healthy_energy_demand_reduction_rate=_number(rates, "healthy_energy_demand_reduction_rate"),
        consumed_energy_demand_reduction_rate=_number(rates, "consumed_energy_demand_reduction_rate"),
        high_demand_reduction_multiplier=_number(rates, "high_demand_reduction_multiplier")
    )


def build_lysosome_config(params: ParamsSource = "lysosome") -> LysosomeConfig:
    """Build lysosome config from scalar YAML parameters."""
    params = _coerce_params(params)
    perception = params.section("perception")
    movement = params.section("movement")
    degradation = params.section("degradation")
    repair = params.section("repair")
    aggregate = params.section("aggregate")
    return LysosomeConfig(
        perception_radius=_integer(perception, "perception_radius"),
        move_radius=_integer(movement, "move_radius"),
        base_degradation_probability=_number(degradation, "base_degradation_probability"),
        protein_degradation_ticks=_integer(degradation, "protein_degradation_ticks"),
        mitochondrion_repair_ticks=_integer(repair, "mitochondrion_repair_ticks"),
        mitochondrion_repair_probability=_number(repair, "mitochondrion_repair_probability"),
        aggregate_degradation_ticks_base=_integer(aggregate, "degradation_ticks_base"),
        aggregate_degradation_ticks_per_member=_integer(aggregate, "degradation_ticks_per_member"),
        aggregate_degradation_probability_base=_number(aggregate, "degradation_probability_base"),
        aggregate_degradation_probability_per_member=_number(aggregate, "degradation_probability_per_member"),
        aggregate_overwhelm_probability_base=_number(aggregate, "overwhelm_probability_base"),
        aggregate_overwhelm_probability_per_member=_number(aggregate, "overwhelm_probability_per_member")
    )


def _coerce_params(params: ParamsSource) -> Params:
    """Return Params regardless of whether a Params object or name was passed."""
    if isinstance(params, Params):
        return params
    return Params(params)


def _threshold(params: Params, name: str, rng: RNG) -> float:
    """Sample a threshold from {mean, std} and clamp it into [0, 1]."""
    thresholds = params.section("thresholds")
    if name not in thresholds:
        raise KeyError(f"Missing threshold parameter: {name}")
    spec = thresholds[name]
    if isinstance(spec, dict):
        mean = _number(spec, "mean")
        std = _number(spec, "std")
        return clamp(rng.gaussian(mean, std))
    return clamp(float(spec))


def _number(section: dict, name: str) -> float:
    """Read one numeric value from a YAML mapping."""
    if name not in section:
        raise KeyError(f"Missing parameter: {name}")
    return float(section[name])


def _integer(section: dict, name: str) -> int:
    """Read one integer value from a YAML mapping."""
    if name not in section:
        raise KeyError(f"Missing parameter: {name}")
    return int(section[name])