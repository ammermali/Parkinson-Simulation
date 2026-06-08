# Agent: Neuron

**Implementation:** `src/simulation/agents/neuron.py`

`Neuron` is the main macro-agent of the simulation.
It lives in the extracellular grid of the Substantia Nigra, but also contains an internal grid
where mitochondria, lysosomes, alpha-synuclein proteins and aggregates interact with each other.

## Responsibilities

- Integrate extracellular stress and intracellular damage in a cumulative value `cell_damage`.
- Produce dopamine.
- Absorb or release pathological alpha.
- Expose an internal habitat.
- Coordinate the cycle of its internal agents before its own.

## State

| State         | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `Healthy`     | functional                                                              |
| `Compromised` | reduced dopamine release                                                |
| `Apoptotic`   | advanced damage, alpha-synuclein release                                |
| `Ruptured`    | terminal damage, releases all alpha-synuclein and debris and stays idle |

## Perception - `see()`

`NeuronPerception` contains:

| Field                  | Source                                     |
|------------------------|--------------------------------------------|
| `position`             | position in the extracellular grid         |
| `nearby_alpha`         | local density of alpha-synuclein proteins  |
| `inflammatory_levels`  | global extracellular inflammatory          |
| `extracellular_debris` | global extracellular debris                |
| `oxidative_stress`     | internal scalar                            |
| `intracellular_debris` | internal scalar                            |
| `energy_demand`        | unsatisfied energy demand                  |
| `internal_damage`      | derived from stress, alpha_load and debris |
| `alpha_load`           | pathological alpha load                    |
| `cell_damage`          | cellular cumulative damage                 |

## State transitions - `next()`

The neuron computes:

```text
external_stress =
  inflammation * inflammation_damage_weight
  + extracellular_debris * debris_damage_weight
  + nearby_alpha * alpha_damage_weight

total_stress = 0.5 * external_stress + 0.5 * internal_damage
```

If `total_stress <= low_stress_threshold`, `cell_damage` can recover through
`damage_recovery_rate` in `Healthy` and `Compromised` states. Otherwise the damage increases by `damage_accumulation_rate`, limited by
`max_damage_increment_per_tick`.

| Condition                              | Candidate state |
|----------------------------------------|-----------------|
| `cell_damage >= ruptured_threshold`    | `Ruptured`      |
| `cell_damage >= apoptotic_threshold`   | `Apoptotic`     |
| `cell_damage >= compromised_threshold` | `Compromised`   |
| otherwise                              | `Healthy`       |

## Actions - `action()` and `do()`

| Action                   | Selection reasoning                                   | Effect                              |
|--------------------------|-------------------------------------------------------|-------------------------------------|
| `release_dopamine`       | context not critical                                  | increases `dopamine_output`         |
| `signal_stress`          | healthy with high inflammation/debris                 | increase extracellular inflammation |
| `absorb_alphasynuclein`  | extracellular alpha above threshold (not `Apoptotic`) | transfer outside the pathology      |
| `release_alphasynuclein` | `Apoptotic` or high internal alpha load               | gradually release alpha             |
| `dump_debris`            | `Ruptured`                                            | releases debris and alpha in the SN |
| `idle`                   | `Ruptured` after spill                                | no effect                           |

The amount of `release_dopamine` is derived from the parameters `dopamine_factor_healthy`, `dopamine_factor_compromised`,
`dopamine_factor_apoptotic` and `dopamine_factor_ruptured`.

## Intracellular habitat

Every neuron exposes a `LocalGrid`. The internal scalars are:

| Scalar                 | Role                                     |
|------------------------|------------------------------------------|
| `oxidative_stress`     | internal toxicity                        |
| `intracellular_debris` | debris produced by mitochondrial failure |
| `energy_demand`        | unsatisfied energy demand                |

## Degradation buffers

The neuron contains:

- `degradation_targets`: available degradable targets;
- `degradation_assignment`: mapping lysosome -> assigned target.

Damaged mitochondria, misfolded proteins and aggregates register themselves as targets.
Lysosomes select unassigned targets, work on them and, according to the outcome, the neuron either clears or re-queues them.

## Alpha transfer

`absorb_alpha(...)` searches in the extracellular grid for absorbable `AlphaSynuclein` or `AlphaAggregate` in `per_radius`. 
Every candidate gets accepted with probability `alpha_absorption_rate`.

`release_alpha(...)` transfers pathological alpha from the internal habitat to the environment:

- if the neuron is `Ruptured`, release all alpha agents;
- otherwise, release a quota derived from `alpha_release_amount`.

## Main parameters

**Source:** `src/configuration/param/neuron.yaml`

| Category          | Parameters                                                                         |
|-------------------|------------------------------------------------------------------------------------|
| Perceptions       | `per_radius`, thresholds for alpha, debris and inflammation                        |
| Damage            | state transition threshold, recovery/accumulation rate                             |
| Dopamine          | `dopamine_release_rate`, value per states                                          |
| Stress and debris | `stress_inflammation_release_rate`, `debris_release_rate`                          |
| Alpha             | `alpha_absorption_rate`, `alpha_release_amount`, `alpha_release_dopamine_fraction` |
| Intracellular     | grid dimension, internal population, decay and damage weight                       |

## Interactions

- With `Mitochondrion`: receives energy demand, oxidative stress and debris reduction.
- With `Lysosome`: provides the target assignment buffers.
- With `AlphaSynuclein` and `AlphaAggregate`: absorbs, releases and computes the pathological load.
- With `AggregateRegistry`: delegates identity, fusion and maturation of aggregates.
