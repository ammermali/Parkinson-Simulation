# Agent: Microglia

**Implementation:** `src/simulation/agents/microglia.py`

`Microglia` is an immunitaty extracellular agent. 
Responds to debris, inflammation and local alpha-synuclein with clearance, scan or inflammatory release.

## Responsibilities

- Read global extracellular debris.
- Read global extracellular inflammation.
- Remove debris when in Clearing state.
- Amplify inflammation when Activated and the context is pathological enough.
- Move during scan.

## State

| State       | Meaning                  |
|-------------|--------------------------|
| `Resting`   | idle                     |
| `Clearing`  | busy removing debris     |
| `Activated` | in an inflammatory state |

## Perception - `see()`

`MicrogliaPerception` contains:

| Field                  | Source                               |
|------------------------|--------------------------------------|
| `position`             | position in the extracellular grid   |
| `extracellular_debris` | global substantia nigra debris       |
| `inflammation_level`   | global substantia nigra inflammation |
| `nearby_alpha`         | local alpha agent density            |

## State transitions - `next()`

The transitions are probabilistic and sampled for agent.

| Current State | Condition                                      | Possible transition |
|---------------|------------------------------------------------|---------------------|
| `Resting`     | debris pressure > 0                            | `Clearing`          |
| `Resting`     | inflammation or alpha pressure > 0             | `Activated`         |
| `Clearing`    | activation pressure > 0                        | `Activated`         |
| `Clearing`    | debris pressure == 0                           | `Resting`           |
| `Activated`   | activation pressure == 0 e debris pressure > 0 | `Clearing`          |
| `Activated`   | all the signals low                            | `Resting`           |

The probabilities are derived from:
- `clearing_transition_rate`;
- `activation_transition_rate`;
- `recovery_transition_rate`.

## Actions - `action()` and `do()`

| Context                          | Action                | Effect                                   |
|----------------------------------|-----------------------|------------------------------------------|
| `Resting`                        | `scan`                | move with probability `move_probability` |
| `Clearing`                       | `clear_debris`        | reduce extracellular debris              |
| `Activated` with enough pressure | `release_inflammation` | increases extracellular inflammation     |
| `Activated` with low pressure    | `scan`                | move with probability `move_probability` |

The inflammatory decision uses:
```text
max(alpha_pressure, inflammation_pressure, 0.5 * debris_pressure)
>= inflammatory_action_threshold
```

## Main parameters

**Source:** `src/configuration/param/microglia.yaml`

| Category    | Parameter                                                                                                                                                                 |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Perception  | `per_radius`                                                                                                                                                              |
| Threshold   | `debris_low_threshold`, `debris_high_threshold`, `inflammation_low_threshold`, `inflammation_high_threshold`, `nearby_alpha_low_threshold`, `nearby_alpha_high_threshold` |
| Effects     | `debris_clearance_rate`, `inflammation_release_rate`                                                                                                                      |
| Movement    | `move_probability`                                                                                                                                                        |
| Transitions | `activation_transition_rate`, `clearing_transition_rate`, `recovery_transition_rate`, `inflammatory_action_threshold`                                                     |

## Interactions

- With `SubstantiaNigra`: reads and modify inflammation/debris.
- With `Astrocyte`: both handle inflammatory response.
