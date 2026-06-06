# Agent: Mitochondrion

**Implementation:** `src/simulation/agents/mitochondrion.py`


`Mitochondrion` is an intracellular agent. 
The simulation uses `energy_demand` as a deficit in the energy demand, healthy mitochondria are responsible for overcoming this deficit.

## Responsabilities

- Reduce the internal energy demand.
- Release oxidative stress when stress or in toxic environment.
- Produce debris in damaged situations.
- Registers itself as a degradation target.
- Recover through fusion or lysosomal reparation.

## State

| State      | Meaning                                 |
|------------|-----------------------------------------|
| `Healthy`  | functional                              |
| `Consumed` | stressed but still partially functional |
| `Damaged`  | damaged, needs to be repaired           |
| `Debris`   | irreversible damage                     |

## Perception - `see()`

`MitochondrionPerception` contains:

| Field                     | Source                                             |
|---------------------------|----------------------------------------------------|
| `position`                | position on the intracellular grid                 |
| `oxidative_stress`        | oxidative stress in the habitat                    |
| `energy_demand`           | internal energy demand                             |
| `local_aggregate_density` | local aggregate density                            |
| `local_debris_density`    | local debris density                               |
| `target_assigned`         | true if a lysosome has this mitochondrion assigned |

## State transitions - `next()`

Transitions are probabilistic and depend on two pressures:

```text
toxicity = 0.4 * oxidative_stress
         + 0.4 * local_aggregate_density
         + 0.2 * local_debris_density

energy_pressure =
  (energy_demand - energy_demand_high_threshold)
  / (1 - energy_demand_high_threshold)
```

| Current State | Transition                                             |
|---------------|--------------------------------------------------------|
| `Healthy`     | may become `Consumed` with `pr_pathological_evolution` |
| `Consumed`    | may become `Healthy`, `Damaged` o `Debris`             |
| `Damaged`     | may become `Consumed` or  `Debris`                     |
| `Debris`      | stays `Debris`                                         |

## Actions - `action()` and `do()`

| State                            | Action          | Effect                                                |
|----------------------------------|-----------------|-------------------------------------------------------|
| target assigned to a lysosome    | `None`          | wait without producing any effect                     |
| `Healthy` with low damage        | `reduce_demand` | reduce `energy_demand`                                |
| `Healthy` with high damage       | `stress`        | increase `oxidative_stress`                           |
| `Consumed` in recovery state     | `fuse`          | reduce stress and debris, turns into `Healthy`        |
| `Consumed` in irreversible state | `stress`        | increases stress                                      |
| `Damaged`                        | `divide`        | increases stress and debris, register for degradation |
| `Debris`                         | `None`          | idle                                                  |

## Main parameters

**Source:** `src/configuration/param/mitochondrion.yaml`

| Category            | Parameters                                                                                                          |
|---------------------|---------------------------------------------------------------------------------------------------------------------|
| Perception          | `perception_radius`                                                                                                 |
| Thresholds          | `energy_demand_high_threshold`,                                                                                     |
| Toxic thresholds    | high/low thresholds for oxidative_stress, aggregate_density, debris_density                                         |
| Irreversible damage | `irreversible_damage_threshold`                                                                                     |
| Release rates       | `stress_release_rate`, `damage_stress_release_rate`, `debris_release_rate`                                          |
| Recovery            | `fusion_stress_reduction_rate`, `fusion_debris_reduction_rate`                                                      |
| Energy              | `healthy_energy_demand_reduction_rate`, `consumed_energy_demand_reduction_rate`, `high_demand_reduction_multiplier` |

## Interactions

- With `Neuron`: uses the neurons as an habitat.
- With `Lysosome`: can be assigned as a target for the lysosome.