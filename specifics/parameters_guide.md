# Parkinson MAS Parameter Guide

This document explains the effect of every parameter currently defined in `src/configuration/param`.

## Threshold Schema

Perception thresholds use:

```yaml
threshold_name:
  mean: 0.5
  std: 0.05
```

`mean` is the population average. `std` controls agent-to-agent heterogeneity. Each agent samples its resolved threshold once at config construction with `rng.gaussian(mean, std)`, then clamps it to `[0.0, 1.0]`. Higher means usually make agents less sensitive to the signal. Higher std values reduce synchronized transitions but can create very sensitive or very insensitive outliers.

## `system.yaml`

Global runtime, population, world and logging parameters.

| Parameter                        |       Current | Effect                                                                                                                     |
|----------------------------------|--------------:|----------------------------------------------------------------------------------------------------------------------------|
| `stop.at`                        |         `500` | Number of scheduled ticks. Larger values expose slower or late-stage pathology.                                            |
| `random.seed`                    |          `42` | Global reproducibility seed for positions, sampled thresholds and stochastic events.                                       |
| `external.population.neurons`    |          `20` | Global neuron count. MPI ranks partition this population, so the value describes the whole simulated system.               |
| `external.population.microglia`  |          `15` | Global microglia count. More microglia increase debris clearance and eventual inflammatory release.                        |
| `external.population.astrocytes` |          `15` | Global astrocyte count. More astrocytes increase supportive anti-inflammatory capacity and possible reactive inflammation. |
| `external.population.alpha`      |           `0` | Initial extracellular alpha-synuclein count. Values above zero seed extracellular pathology at startup.                    |
| `world.height`                   |          `10` | Extracellular grid height.                                                                                                 |
| `world.width`                    |          `10` | Extracellular grid width.                                                                                                  |
| `world.buffer_size`              |           `2` | Repast/MPI ghost-cell buffer.                                                                                              |
| `logging.enabled`                |        `true` | Default logging switch used when a sub-logger does not override it.                                                        |
| `logging.run_id`                 | `default_run` | Identifier written into output logs.                                                                                       |
| `logging.output_dir`             |      `output` | Root directory for runtime logs, rank logs, initialization logs and metrics.                                               |
| `logging.causal.enabled`         |        `true` | Enables semantic event logging for G0 projection and mechanism analysis.                                                   |
| `logging.initialization.enabled` |        `true` | Writes initial agent conditions and baseline alpha nodes.                                                                  |
| `logging.scalar_stdout`          |        `true` | Prints main extracellular scalars every tick.                                                                              |
| `logging.tick_metrics_csv`       |        `true` | Writes `tick_metrics.csv` with global per-tick scalars and compact state counts.                                           |
| `logging.progress_stdout`        |        `true` | Prints progress and final summary from rank 0.                                                                             |
| `logging.progress_interval`      |          `25` | Tick interval for progress messages.                                                                                       |
| `logging.summary_stdout`         |        `true` | Prints end-of-run counts for neurons, glia, alpha and aggregates.                                                          |

## `substantia_nigra.yaml`

Controls shared extracellular scalars and their population-level normalization.

| Parameter | Current | Effect |
|---|---:|---|
| `initial.debris` | `0.1` | Initial extracellular debris. Higher values activate debris-sensitive agents earlier. |
| `initial.inflammation` | `0.08` | Initial inflammation. Higher values push glia and neurons toward stress earlier. |
| `initial.dopamine` | `0.8` | Initial normalized dopamine before neuron release dynamics dominate. |
| `decay.debris` | `0.03` | Natural debris decay per tick. Higher values make debris self-resolve faster. |
| `decay.inflammation` | `0.08` | Natural inflammation decay per tick. Higher values make inflammatory plateaus harder to maintain. |
| `dopamine.smoothing` | `0.35` | Blend between previous dopamine and current normalized release. Higher values make dopamine more reactive and volatile. |
| `effects.baseline_debris_input` | `0.003` | Basal extracellular debris turnover per tick. Prevents a perfectly sterile zero-debris attractor. |
| `effects.debris_added_max_delta` | `0.12` | Maximum committed debris increase per tick after saturating population effects. |
| `effects.debris_removed_max_delta` | `0.05` | Maximum committed debris removal per tick. Higher values strengthen microglial cleanup. |
| `effects.debris_effect_scale` | `0.8` | Saturation scale for debris additions/removals. Higher values make raw population sums translate into larger scalar changes. |
| `effects.inflammation_added_max_delta` | `0.08` | Maximum committed inflammation increase per tick. |
| `effects.inflammation_removed_max_delta` | `0.06` | Maximum committed inflammation removal per tick. |
| `effects.inflammation_effect_scale` | `0.8` | Saturation scale for inflammatory additions/removals. |

## `neuron.yaml`

Controls extracellular perception, cumulative damage, alpha transmission and each neuron's intracellular habitat.

| Parameter | Current | Effect |
|---|---:|---|
| `perception.per_radius` | `1` | Radius for sensing extracellular alpha density. |
| `thresholds.nearby_alpha_high_threshold.mean` | `0.58` | Mean nearby-alpha threshold for absorption/stress decisions. Higher values reduce alpha-driven response. |
| `thresholds.nearby_alpha_high_threshold.std` | `0.08` | Heterogeneity in nearby-alpha sensitivity. |
| `thresholds.inflammation_high_threshold.mean` | `0.7` | Mean inflammation threshold for stress signaling. |
| `thresholds.inflammation_high_threshold.std` | `0.05` | Heterogeneity in inflammatory sensitivity. |
| `thresholds.debris_high_threshold.mean` | `0.6` | Mean extracellular debris threshold for stress signaling. |
| `thresholds.debris_high_threshold.std` | `0.05` | Heterogeneity in debris sensitivity. |
| `thresholds.alpha_load_release_threshold.mean` | `0.10` | Mean intracellular alpha-load threshold for non-rupture alpha release. Higher values delay transmission. |
| `thresholds.alpha_load_release_threshold.std` | `0.02` | Heterogeneity in alpha-load release threshold. |
| `thresholds.low_stress_threshold.mean` | `0.12` | Total stress threshold below which cell damage can recover. |
| `thresholds.low_stress_threshold.std` | `0.02` | Heterogeneity in recovery tolerance. |
| `thresholds.compromised_threshold` | `0.28` | Cumulative damage required for `Compromised`. |
| `thresholds.apoptotic_threshold` | `0.78` | Cumulative damage required for `Apoptotic`. Higher values keep neurons `Compromised` longer. |
| `thresholds.ruptured_threshold` | `0.985` | Cumulative damage required for `Ruptured`. |
| `damage.damage_accumulation_rate` | `0.22` | Converts total stress into cumulative cell damage. Higher values accelerate pathology. |
| `damage.damage_recovery_rate` | `0.022` | Damage removed per low-stress tick for `Healthy` and `Compromised` neurons; it does not recover `Apoptotic` neurons. |
| `damage.max_damage_increment_per_tick` | `0.05` | Upper bound on damage gained in one tick, preventing rapid threshold skipping. |
| `damage.min_ticks_compromised_before_apoptotic` | `25` | Minimum ticks a neuron must remain `Compromised` before it can become `Apoptotic`. |
| `damage.min_ticks_apoptotic_before_ruptured` | `20` | Minimum ticks a neuron must remain `Apoptotic` before it can become `Ruptured`. |
| `damage.apoptotic_internal_damage_threshold` | `0.35` | Minimum internal damage required for a `Compromised` neuron to become `Apoptotic`; blocks purely extracellular apoptotic jumps. |
| `damage.rupture_internal_damage_threshold` | `0.45` | Minimum internal damage required for rupture once cumulative damage is high enough. |
| `damage.rupture_intracellular_debris_threshold` | `0.10` | Minimum intracellular debris required for rupture once cumulative damage is high enough. |
| `damage.inflammation_damage_weight` | `0.35` | Inflammation contribution to extracellular neuron stress. |
| `damage.debris_damage_weight` | `0.3` | Extracellular debris contribution to neuron stress. |
| `damage.alpha_damage_weight` | `0.31` | Nearby alpha contribution to neuron stress. Lower values reduce alpha-driven rupture pressure. |
| `rates.dopamine_release_rate` | `0.18` | Dopamine released by a healthy neuron before normalization. |
| `rates.dopamine_factor_healthy` | `1.0` | State-specific dopamine factor for `Healthy` neurons. |
| `rates.dopamine_factor_compromised` | `0.55` | State-specific dopamine factor for `Compromised` neurons. |
| `rates.dopamine_factor_apoptotic` | `0.15` | Residual dopamine factor for `Apoptotic` neurons when they perform dopamine-coupled release. |
| `rates.dopamine_factor_ruptured` | `0.0` | Dopamine factor for `Ruptured` neurons. |
| `rates.stress_inflammation_release_rate` | `0.04` | Inflammation released by a neuron choosing the stress action. |
| `rates.debris_release_rate` | `0.1` | One-shot debris payload released by rupture. |
| `alpha.alpha_absorption_rate` | `0.08` | Probability that a candidate extracellular alpha/aggregate is absorbed. |
| `alpha.alpha_release_amount` | `0.04` | Fraction of eligible alpha pathology released by non-ruptured leakage. |
| `alpha.alpha_release_dopamine_fraction` | `0.35` | Fraction of state-dependent dopamine retained while a functional neuron releases alpha. |
| `intracellular.grid.height` | `5` | Internal grid height. Larger values dilute local intracellular encounters. |
| `intracellular.grid.width` | `5` | Internal grid width. Larger values dilute local intracellular encounters. |
| `intracellular.population.alpha` | `40` | Initial alpha-synuclein proteins per neuron. Higher values increase misfolding and aggregation substrate. |
| `intracellular.population.mitochondria` | `10` | Initial mitochondria per neuron. |
| `intracellular.population.lysosomes` | `5` | Initial lysosomes per neuron. Higher values increase cleanup and mitochondrial repair capacity. |
| `intracellular.scalars.energy_demand_baseline` | `0.5` | Baseline unmet energy demand. |
| `intracellular.decay.energy_demand_recovery_rate` | `0.02` | Pull of energy demand back toward baseline. |
| `intracellular.decay.oxidative_stress_decay` | `0.005` | Natural oxidative stress decay. Higher values reduce alpha misfolding pressure. |
| `intracellular.decay.intracellular_debris_decay` | `0.005` | Natural intracellular debris decay. |
| `intracellular.damage.internal_damage_oxidative_weight` | `0.35` | Oxidative stress contribution to internal neuron damage. |
| `intracellular.damage.internal_damage_aggregate_weight` | `0.5` | Alpha aggregate/load contribution to internal neuron damage. |
| `intracellular.damage.internal_damage_debris_weight` | `0.15` | Intracellular debris contribution to internal neuron damage. |

## `alpha.yaml`

Controls free alpha-synuclein proteins before they are absorbed into an aggregate.

| Parameter | Current | Effect |
|---|---:|---|
| `perception.perception_radius` | `1` | Radius for reading local oxidative stress, aggregate density and neighbors. |
| `movement.move_radius` | `1` | Maximum movement radius per tick. |
| `movement.move_probability` | `0.35` | Probability of moving when free. Higher values increase mixing and encounters. |
| `rates.basal_misfold_probability` | `0.002` | Baseline per-tick misfolding probability without stress. Higher values create widespread spontaneous pathology. |
| `rates.aggregate_seeded_misfold_weight` | `0.10` | Misfolding pressure from local aggregates. Higher values strengthen prion-like seeding. |
| `rates.oxidative_misfolding_weight` | `0.45` | Misfolding pressure from oxidative stress above threshold. |
| `rates.oligomerization_probability_scale` | `0.25` | Probability scale for misfolded alpha wanting oligomerization. |
| `rates.min_misfolded_ticks_before_oligomerization` | `4` | Minimum ticks spent misfolded before oligomerization can be attempted. |
| `thresholds.oxidative_stress_high_threshold.mean` | `0.58` | Mean oxidative stress threshold for stress-driven misfolding. Higher values reduce sensitivity. |
| `thresholds.oxidative_stress_high_threshold.std` | `0.07` | Heterogeneity in oxidative-stress sensitivity. |

## Aggregate Dynamics

Aggregates currently do not have a dedicated YAML file. Their behavior depends on:

| Source | Effect |
|---|---|
| `AggregateRegistry.lewy_body_size_threshold` | Minimum aggregate size for Lewy-body maturation checks when configured in code/tests. |
| `AlphaAggregate.aggregate_weight` | Code-derived contribution to alpha load and local aggregate density. |
| `lysosome.yaml` aggregate parameters | Degradation time, degradation probability and overwhelm probability. |
| `neuron.yaml` alpha-load and damage parameters | How strongly aggregates affect neuron stress, release and rupture. |

## `lysosome.yaml`

Controls lysosomal targeting, degradation, mitochondrial repair and overwhelm.

| Parameter | Current | Effect |
|---|---:|---|
| `perception.perception_radius` | `1` | Radius for reading local aggregate density and nearby targets. |
| `movement.move_radius` | `1` | Movement radius during scanning. |
| `degradation.base_degradation_probability` | `0.5` | Probability of clearing non-aggregate protein targets. |
| `degradation.protein_degradation_ticks` | `2` | Ticks required before resolving base protein degradation. |
| `repair.mitochondrion_repair_ticks` | `3` | Ticks required before resolving mitochondrial repair. |
| `repair.mitochondrion_repair_probability` | `0.55` | Probability that mitochondrial repair succeeds. |
| `aggregate.degradation_ticks_base` | `1` | Base ticks required before attempting aggregate degradation. |
| `aggregate.degradation_ticks_per_member` | `1` | Additional ticks per aggregate member. |
| `aggregate.degradation_probability_base` | `0.18` | Base probability of degrading a non-Lewy aggregate. |
| `aggregate.degradation_probability_per_member` | `0.035` | Degradation probability added per member. |
| `aggregate.overwhelm_probability_base` | `0.11` | Base probability that an aggregate overwhelms the lysosome. |
| `aggregate.overwhelm_probability_per_member` | `0.06` | Overwhelm probability added per aggregate member. |

## `mitochondrion.yaml`

Controls mitochondrial lifecycle, oxidative stress, debris and energy-demand interaction.

| Parameter | Current | Effect |
|---|---:|---|
| `perception.perception_radius` | `1` | Radius for sensing local aggregate and debris density. |
| `thresholds.energy_demand_high_threshold.mean` | `0.7` | Mean threshold for high unmet energy demand. |
| `thresholds.energy_demand_high_threshold.std` | `0.05` | Heterogeneity in energy-demand sensitivity. |
| `thresholds.oxidative_stress_high_threshold.mean` | `0.7` | Mean high oxidative-stress threshold. |
| `thresholds.oxidative_stress_high_threshold.std` | `0.05` | Heterogeneity in stress vulnerability. |
| `thresholds.oxidative_stress_low_threshold.mean` | `0.45` | Mean low oxidative-stress threshold for recovery-oriented behavior. |
| `thresholds.oxidative_stress_low_threshold.std` | `0.03` | Heterogeneity in low-stress threshold. |
| `thresholds.aggregate_density_high_threshold.mean` | `0.55` | Mean high local aggregate-density threshold. |
| `thresholds.aggregate_density_high_threshold.std` | `0.05` | Heterogeneity in aggregate sensitivity. |
| `thresholds.aggregate_density_low_threshold.mean` | `0.3` | Mean low local aggregate-density threshold. |
| `thresholds.aggregate_density_low_threshold.std` | `0.03` | Heterogeneity in low aggregate-density threshold. |
| `thresholds.debris_density_high_threshold.mean` | `0.55` | Mean high local debris-density threshold. |
| `thresholds.debris_density_high_threshold.std` | `0.05` | Heterogeneity in debris sensitivity. |
| `thresholds.debris_density_low_threshold.mean` | `0.3` | Mean low local debris-density threshold. |
| `thresholds.debris_density_low_threshold.std` | `0.03` | Heterogeneity in low debris-density threshold. |
| `thresholds.irreversible_damage_threshold.mean` | `0.65` | Mean threshold for irreversible mitochondrial damage risk. |
| `thresholds.irreversible_damage_threshold.std` | `0.03` | Heterogeneity in irreversible damage vulnerability. |
| `rates.stress_release_rate` | `0.1` | Oxidative stress added by pathological mitochondrial behavior. |
| `rates.damage_stress_release_rate` | `0.15` | Oxidative stress added by damaged mitochondria. |
| `rates.debris_release_rate` | `0.04` | Intracellular debris added by damaged/debris mitochondria. |
| `rates.fusion_stress_reduction_rate` | `0.02` | Oxidative stress reduction from fusion/repair behavior. |
| `rates.fusion_debris_reduction_rate` | `0.01` | Intracellular debris reduction from fusion/repair behavior. |
| `rates.healthy_energy_demand_reduction_rate` | `0.008` | Energy-demand reduction from healthy mitochondria. |
| `rates.consumed_energy_demand_reduction_rate` | `0.003` | Energy-demand reduction from consumed mitochondria. |
| `rates.high_demand_reduction_multiplier` | `1.5` | Multiplier applied when energy demand is high. |

## `microglia.yaml`

Controls extracellular immune sensing, transitions and actions.

| Parameter | Current | Effect |
|---|---:|---|
| `perception.per_radius` | `1` | Radius for perceiving nearby extracellular alpha. |
| `thresholds.debris_high_threshold.mean` | `0.6` | Mean threshold for high debris. Lower values make clearing more likely. |
| `thresholds.debris_high_threshold.std` | `0.05` | Heterogeneity in high-debris sensitivity. |
| `thresholds.debris_low_threshold.mean` | `0.2` | Mean threshold below which debris pressure is low. |
| `thresholds.debris_low_threshold.std` | `0.03` | Heterogeneity in low-debris threshold. |
| `thresholds.inflammation_high_threshold.mean` | `0.7` | Mean threshold for inflammation-driven activation. |
| `thresholds.inflammation_high_threshold.std` | `0.05` | Heterogeneity in inflammatory sensitivity. |
| `thresholds.inflammation_low_threshold.mean` | `0.3` | Mean threshold below which inflammation pressure is low. |
| `thresholds.inflammation_low_threshold.std` | `0.03` | Heterogeneity in low-inflammation threshold. |
| `thresholds.nearby_alpha_high_threshold.mean` | `0.5` | Mean threshold for alpha-driven activation. |
| `thresholds.nearby_alpha_high_threshold.std` | `0.05` | Heterogeneity in alpha sensitivity. |
| `thresholds.nearby_alpha_low_threshold.mean` | `0.2` | Mean threshold below which nearby alpha pressure is low. |
| `thresholds.nearby_alpha_low_threshold.std` | `0.03` | Heterogeneity in low-alpha threshold. |
| `rates.debris_clearance_rate` | `0.035` | Raw debris removal contributed by a clearing microglia. |
| `rates.inflammation_release_rate` | `0.024` | Raw inflammation added by activated microglia when action pressure is high enough. |
| `movement.move_probability` | `0.5` | Probability of moving while scanning. |
| `dynamics.activation_transition_rate` | `0.35` | Scale for probabilistic transition to `Activated`. |
| `dynamics.clearing_transition_rate` | `0.45` | Scale for probabilistic transition to `Clearing`. |
| `dynamics.recovery_transition_rate` | `0.25` | Scale for recovery to `Resting` when pressure is low. |
| `dynamics.inflammatory_action_threshold` | `0.28` | Minimum contextual pressure required for activated microglia to release inflammation. |

## `astrocyte.yaml`

Controls Supportive/Reactive transitions and inflammation modulation.

| Parameter | Current | Effect |
|---|---:|---|
| `thresholds.inflammation_high_threshold.mean` | `0.55` | Mean threshold for becoming reactive due to inflammation. |
| `thresholds.inflammation_high_threshold.std` | `0.1` | Heterogeneity in inflammatory reactivity. |
| `thresholds.inflammation_low_threshold.mean` | `0.25` | Mean threshold below which inflammation is low for recovery logic. |
| `thresholds.inflammation_low_threshold.std` | `0.05` | Heterogeneity in low-inflammation threshold. |
| `thresholds.debris_high_threshold.mean` | `0.55` | Mean threshold for debris-driven reactivity. |
| `thresholds.debris_high_threshold.std` | `0.1` | Heterogeneity in debris sensitivity. |
| `thresholds.debris_low_threshold.mean` | `0.15` | Mean threshold below which debris is low. |
| `thresholds.debris_low_threshold.std` | `0.04` | Heterogeneity in low-debris threshold. |
| `rates.support_inflammation_reduction_rate` | `0.014` | Raw inflammation removal by supportive astrocytes. |
| `rates.inflammation_release_rate` | `0.015` | Raw inflammation added by reactive astrocytes when stress memory is high enough. |
| `dynamics.stress_memory_decay` | `0.85` | Persistence of astrocyte stress memory. Higher values make reactivity longer-lasting. |
| `dynamics.reactive_transition_rate` | `0.45` | Scale for probabilistic transition to `Reactive`. |
| `dynamics.supportive_recovery_rate` | `0.16` | Scale/probability of recovery to `Supportive`. |
| `dynamics.inflammatory_memory_threshold` | `0.5` | Stress-memory level required for a reactive astrocyte to release inflammation. |
| `dynamics.inflammation_memory_weight` | `0.8` | Inflammation contribution to astrocyte stress memory. |
| `dynamics.debris_stress_weight` | `0.2` | Debris contribution to astrocyte stress memory. |
