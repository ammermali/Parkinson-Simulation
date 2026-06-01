from pathlib import Path
from typing import Optional, Union
from src.simulation.agents.alphasynuclein import AlphaSynucleinConfig
from src.simulation.agents.astrocyte import AstrocyteConfig
from src.simulation.agents.lysosome import LysosomeConfig
from src.simulation.agents.microglia import MicrogliaConfig
from src.simulation.agents.mitochondrion import MitochondrionConfig
from src.simulation.agents.neuron import NeuronConfig, NeuronInternalConfig
from src.simulation.substantia_nigra import SNEnvironmentConfig
from src.simulation.utils import clamp, RNG
from src.simulation.utils.paramsloader import Params
ParamsSource = Union[Params, str, Path]


class ConfigFactory:
    """Build runtime config dataclasses from YAML-backed Params objects."""

    @staticmethod
    def build_substantia_nigra_config(params: ParamsSource = "substantia_nigra") -> SNEnvironmentConfig:
        """Build Substantia Nigra environment config from YAML parameters."""
        params = _coerce_params(params)
        initial = params.section("initial")
        decay = params.section("decay")
        dopamine = params.section("dopamine")
        effects = _optional_section(params, "effects")
        return SNEnvironmentConfig(
            initial_debris=_number(initial, "debris"),
            initial_inflammation=_number(initial, "inflammation"),
            initial_dopamine=_number(initial, "dopamine"),
            debris_decay=_number(decay, "debris"),
            inflammation_decay=_number(decay, "inflammation"),
            dopamine_smoothing=_number(dopamine, "smoothing"),
            debris_added_max_delta=_optional_number(effects, "debris_added_max_delta"),
            debris_removed_max_delta=_optional_number(effects, "debris_removed_max_delta"),
            debris_effect_scale=_number_or(effects, "debris_effect_scale", 1.0),
            inflammation_added_max_delta=_optional_number(effects, "inflammation_added_max_delta"),
            inflammation_removed_max_delta=_optional_number(effects, "inflammation_removed_max_delta"),
            inflammation_effect_scale=_number_or(effects, "inflammation_effect_scale", 1.0),
            baseline_debris_input=_number_or(effects, "baseline_debris_input", 0.0)
        )

    @staticmethod
    def build_microglia_config(params: ParamsSource = "microglia", rng: Optional[RNG] = None) -> MicrogliaConfig:
        """Build one microglia config, sampling perception thresholds once."""
        params = _coerce_params(params)
        rng = rng or RNG()
        perception = params.section("perception")
        rates = params.section("rates")
        movement = params.section("movement")
        dynamics = _optional_section(params, "dynamics")
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
            move_probability=_number(movement, "move_probability"),
            activation_transition_rate=_number_or(dynamics, "activation_transition_rate", 1.0),
            clearing_transition_rate=_number_or(dynamics, "clearing_transition_rate", 1.0),
            recovery_transition_rate=_number_or(dynamics, "recovery_transition_rate", 1.0),
            inflammatory_action_threshold=_number_or(dynamics, "inflammatory_action_threshold", 0.0)
        )

    @staticmethod
    def build_astrocyte_config(params: ParamsSource = "astrocyte", rng: Optional[RNG] = None) -> AstrocyteConfig:
        """Build one astrocyte config, sampling perception thresholds once."""
        params = _coerce_params(params)
        rng = rng or RNG()
        rates = params.section("rates")
        dynamics = _optional_section(params, "dynamics")
        return AstrocyteConfig(
            inflammation_high_threshold=_threshold(params, "inflammation_high_threshold", rng),
            inflammation_low_threshold=_threshold(params, "inflammation_low_threshold", rng),
            debris_high_threshold=_threshold(params, "debris_high_threshold", rng),
            debris_low_threshold=_threshold(params, "debris_low_threshold", rng),
            support_inflammation_reduction_rate=_number(rates, "support_inflammation_reduction_rate"),
            inflammation_release_rate=_number(rates, "inflammation_release_rate"),
            stress_memory_decay=_number_or(dynamics, "stress_memory_decay", 0.0),
            reactive_transition_rate=_number_or(dynamics, "reactive_transition_rate", 1.0),
            supportive_recovery_rate=_number_or(dynamics, "supportive_recovery_rate", 1.0),
            inflammatory_memory_threshold=_number_or(dynamics, "inflammatory_memory_threshold", 0.0),
            inflammation_memory_weight=_number_or(dynamics, "inflammation_memory_weight", 0.0),
            debris_stress_weight=_number_or(dynamics, "debris_stress_weight", 1.0)
        )

    @staticmethod
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
            alpha_release_amount=_number(alpha, "alpha_release_amount"),
            max_damage_increment_per_tick=_number_or(
                damage,
                "max_damage_increment_per_tick",
                1.0
            ),
            apoptotic_internal_damage_threshold=_number_or(
                damage,
                "apoptotic_internal_damage_threshold",
                0.0
            ),
            dopamine_factor_healthy=_number_or(
                rates,
                "dopamine_factor_healthy",
                1.0
            ),
            dopamine_factor_compromised=_number_or(
                rates,
                "dopamine_factor_compromised",
                0.6
            ),
            dopamine_factor_apoptotic=_number_or(
                rates,
                "dopamine_factor_apoptotic",
                0.0
            ),
            dopamine_factor_ruptured=_number_or(
                rates,
                "dopamine_factor_ruptured",
                0.0
            ),
            alpha_release_dopamine_fraction=_number_or(
                alpha,
                "alpha_release_dopamine_fraction",
                0.35
            ),
            min_ticks_compromised_before_apoptotic=_integer_or(
                damage,
                "min_ticks_compromised_before_apoptotic",
                0
            ),
            min_ticks_apoptotic_before_ruptured=_integer_or(
                damage,
                "min_ticks_apoptotic_before_ruptured",
                0
            ),
            rupture_internal_damage_threshold=_number_or(
                damage,
                "rupture_internal_damage_threshold",
                0.0
            ),
            rupture_intracellular_debris_threshold=_number_or(
                damage,
                "rupture_intracellular_debris_threshold",
                0.0
            )
        )

    @staticmethod
    def build_neuron_internal_config(params: ParamsSource = "neuron") -> NeuronInternalConfig:
        """Build the neuron's intracellular grid and scalar dynamics config."""
        params = _coerce_params(params)
        defaults = NeuronInternalConfig()
        intracellular = _optional_section(params, "intracellular")
        grid = _mapping(intracellular, "grid", _optional_section(params, "grid"))
        scalars = _mapping(intracellular, "scalars", {})
        decay = _mapping(intracellular, "decay", {})
        damage = _mapping(intracellular, "damage", {})

        return NeuronInternalConfig(
            width=_integer_or(grid, "width", defaults.width),
            height=_integer_or(grid, "height", defaults.height),
            energy_demand_baseline=_number_or(
                scalars,
                "energy_demand_baseline",
                defaults.energy_demand_baseline,
            ),
            energy_demand_recovery_rate=_number_or(
                decay,
                "energy_demand_recovery_rate",
                defaults.energy_demand_recovery_rate,
            ),
            oxidative_stress_decay=_number_or(
                decay,
                "oxidative_stress_decay",
                defaults.oxidative_stress_decay,
            ),
            intracellular_debris_decay=_number_or(
                decay,
                "intracellular_debris_decay",
                defaults.intracellular_debris_decay,
            ),
            internal_damage_oxidative_weight=_number_or(
                damage,
                "internal_damage_oxidative_weight",
                defaults.internal_damage_oxidative_weight,
            ),
            internal_damage_aggregate_weight=_number_or(
                damage,
                "internal_damage_aggregate_weight",
                defaults.internal_damage_aggregate_weight,
            ),
            internal_damage_debris_weight=_number_or(
                damage,
                "internal_damage_debris_weight",
                defaults.internal_damage_debris_weight,
            ),
        )

    @staticmethod
    def build_alpha_synuclein_config(params: ParamsSource = "alpha", rng: Optional[RNG] = None) -> AlphaSynucleinConfig:
        """Build one alpha-synuclein config, sampling misfolding threshold once."""
        params = _coerce_params(params)
        rng = rng or RNG()
        perception = params.section("perception")
        movement = params.section("movement")
        rates = _optional_section(params, "rates")
        defaults = AlphaSynucleinConfig(1, 1, 0.5, 0.6)
        return AlphaSynucleinConfig(
            perception_radius=_integer(perception, "perception_radius"),
            move_radius=_integer(movement, "move_radius"),
            move_probability=_number(movement, "move_probability"),
            oxidative_stress_high_threshold=_threshold(params, "oxidative_stress_high_threshold", rng),
            basal_misfold_probability=_number_or(
                rates,
                "basal_misfold_probability",
                defaults.basal_misfold_probability
            ),
            aggregate_seeded_misfold_weight=_number_or(
                rates,
                "aggregate_seeded_misfold_weight",
                defaults.aggregate_seeded_misfold_weight
            ),
            oxidative_misfolding_weight=_number_or(
                rates,
                "oxidative_misfolding_weight",
                defaults.oxidative_misfolding_weight
            ),
            oligomerization_probability_scale=_number_or(
                rates,
                "oligomerization_probability_scale",
                defaults.oligomerization_probability_scale
            ),
            min_misfolded_ticks_before_oligomerization=_integer_or(
                rates,
                "min_misfolded_ticks_before_oligomerization",
                defaults.min_misfolded_ticks_before_oligomerization
            )
        )

    @staticmethod
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

    @staticmethod
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


def _optional_section(params: Params, name: str) -> dict:
    """Return a top-level mapping section, or an empty mapping when absent."""
    value = params.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Parameter section '{name}' must be a mapping.")
    return value


def _mapping(section: dict, name: str, default: dict) -> dict:
    """Read a nested mapping, returning default when the section is omitted."""
    value = section.get(name, default)
    if value is None:
        return default
    if not isinstance(value, dict):
        raise TypeError(f"Parameter section '{name}' must be a mapping.")
    return value


def _number(section: dict, name: str) -> float:
    """Read one numeric value from a YAML mapping."""
    if name not in section:
        raise KeyError(f"Missing parameter: {name}")
    return float(section[name])


def _number_or(section: dict, name: str, default: float) -> float:
    """Read a numeric value when present, otherwise return a default."""
    if name not in section:
        return default
    return float(section[name])


def _optional_number(section: dict, name: str) -> Optional[float]:
    """Read an optional numeric value from a YAML mapping."""
    if name not in section or section[name] is None:
        return None
    return float(section[name])


def _integer(section: dict, name: str) -> int:
    """Read one integer value from a YAML mapping."""
    if name not in section:
        raise KeyError(f"Missing parameter: {name}")
    return int(section[name])


def _integer_or(section: dict, name: str, default: int) -> int:
    """Read an integer value when present, otherwise return a default."""
    if name not in section:
        return default
    return int(section[name])


build_substantia_nigra_config = ConfigFactory.build_substantia_nigra_config
build_microglia_config = ConfigFactory.build_microglia_config
build_astrocyte_config = ConfigFactory.build_astrocyte_config
build_neuron_config = ConfigFactory.build_neuron_config
build_neuron_internal_config = ConfigFactory.build_neuron_internal_config
build_alpha_synuclein_config = ConfigFactory.build_alpha_synuclein_config
build_mitochondrion_config = ConfigFactory.build_mitochondrion_config
build_lysosome_config = ConfigFactory.build_lysosome_config
