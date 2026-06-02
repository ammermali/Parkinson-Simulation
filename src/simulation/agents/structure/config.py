from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaSynucleinConfig:
    perception_radius: int
    move_radius: int
    move_probability: float
    oxidative_stress_high_threshold: float
    basal_misfold_probability: float = 0.006
    aggregate_seeded_misfold_weight: float = 0.25
    oxidative_misfolding_weight: float = 0.7
    oligomerization_probability_scale: float = 0.4
    min_misfolded_ticks_before_oligomerization: int = 0


@dataclass
class AstrocyteConfig:
    inflammation_high_threshold: float
    inflammation_low_threshold: float
    debris_high_threshold: float
    debris_low_threshold: float
    support_inflammation_reduction_rate: float
    inflammation_release_rate: float
    stress_memory_decay: float = 0.0
    reactive_transition_rate: float = 1.0
    supportive_recovery_rate: float = 1.0
    inflammatory_memory_threshold: float = 0.0
    inflammation_memory_weight: float = 1.0
    debris_stress_weight: float = 1.0


@dataclass(frozen=True)
class LysosomeConfig:
    perception_radius: int = 1
    move_radius: int = 1
    base_degradation_probability: float = 0.8
    protein_degradation_ticks: int = 1
    mitochondrion_repair_ticks: int = 1
    mitochondrion_repair_probability: float = 0.8
    aggregate_degradation_ticks_base: int = 1
    aggregate_degradation_ticks_per_member: int = 1
    aggregate_degradation_probability_base: float = 0.35
    aggregate_degradation_probability_per_member: float = 0.05
    aggregate_overwhelm_probability_base: float = 0.02
    aggregate_overwhelm_probability_per_member: float = 0.01


@dataclass
class MicrogliaConfig:
    per_radius: int
    debris_high_threshold: float
    debris_low_threshold: float
    inflammation_high_threshold: float
    inflammation_low_threshold: float
    nearby_alpha_high_threshold: float
    nearby_alpha_low_threshold: float
    debris_clearance_rate: float
    inflammation_release_rate: float
    move_probability: float
    activation_transition_rate: float = 1.0
    clearing_transition_rate: float = 1.0
    recovery_transition_rate: float = 1.0
    inflammatory_action_threshold: float = 0.0


@dataclass(frozen=True)
class MitochondrionConfig:
    perception_radius: int
    energy_demand_high_threshold: float
    oxidative_stress_high_threshold: float
    oxidative_stress_low_threshold: float
    aggregate_density_high_threshold: float
    aggregate_density_low_threshold: float
    debris_density_high_threshold: float
    debris_density_low_threshold: float
    irreversible_damage_threshold: float
    stress_release_rate: float
    damage_stress_release_rate: float
    debris_release_rate: float
    fusion_stress_reduction_rate: float
    fusion_debris_reduction_rate: float
    healthy_energy_demand_reduction_rate: float
    consumed_energy_demand_reduction_rate: float
    high_demand_reduction_multiplier: float


@dataclass
class NeuronConfig:
    per_radius: int
    nearby_alpha_high_threshold: float
    inflammation_high_threshold: float
    debris_high_threshold: float
    alpha_load_release_threshold: float
    damage_accumulation_rate: float
    damage_recovery_rate: float
    low_stress_threshold: float
    inflammation_damage_weight: float
    debris_damage_weight: float
    alpha_damage_weight: float
    compromised_threshold: float
    apoptotic_threshold: float
    ruptured_threshold: float
    dopamine_release_rate: float
    stress_inflammation_release_rate: float
    debris_release_rate: float
    alpha_absorption_rate: float
    alpha_release_amount: float
    max_damage_increment_per_tick: float = 1.0
    apoptotic_internal_damage_threshold: float = 0.0
    dopamine_factor_healthy: float = 1.0
    dopamine_factor_compromised: float = 0.6
    dopamine_factor_apoptotic: float = 0.0
    dopamine_factor_ruptured: float = 0.0
    alpha_release_dopamine_fraction: float = 0.35
    min_ticks_compromised_before_apoptotic: int = 0
    min_ticks_apoptotic_before_ruptured: int = 0
    rupture_internal_damage_threshold: float = 0.0
    rupture_intracellular_debris_threshold: float = 0.0


@dataclass
class NeuronInternalConfig:
    width: int = 10
    height: int = 10
    energy_demand_baseline: float = 0.5
    energy_demand_recovery_rate: float = 0.02
    oxidative_stress_decay: float = 0.01
    intracellular_debris_decay: float = 0.005
    internal_damage_oxidative_weight: float = 0.4
    internal_damage_aggregate_weight: float = 0.4
    internal_damage_debris_weight: float = 0.2


@dataclass
class NeuronInternalScalars:
    oxidative_stress: float = 0.0
    intracellular_debris: float = 0.0
    energy_demand: float = 0.5


@dataclass
class NeuronInternalEffects:
    oxidative_stress_added: float = 0.0
    debris_added: float = 0.0
    energy_demand_added: float = 0.0
