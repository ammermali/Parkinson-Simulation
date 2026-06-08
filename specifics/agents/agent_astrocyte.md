# Agent: Astrocyte

**Implementation:** `src/simulation/agents/astrocyte.py`

`Astrocyte` is a supportive extracellular agent.
In normal conditions, it reduces inflammation; if the stress is persistent, it can become Reactive and contribute to the inflammatory release.

## Responsibilities

- Read inflammation and extracellular debris.
- Integrate a stress memory over time.
- Switch between `Supportive` and `Reactive` states in a probabilistic way.
- Reduce inflammation when supportive.
- Release inflammation when reactive.

## State

| State        | Meaning                                              |
|--------------|------------------------------------------------------|
| `Supportive` | protective, reduces inflammation                     |
| `Reactive`   | reactive, can either release or support inflammation |

## Perception - `see()`

`AstrocytePerception` contains:

| Field                  | Source                               |
|------------------------|--------------------------------------|
| `position`             | position in the extracellular grid   |
| `inflammation_level`   | global substantia nigra inflammation |
| `extracellular_debris` | global substantia nigra debris       |

## Stress memory

The astrocyte normalizes inflammation and debris between low and high threshold. 
The total pressure is:
```text
stress_pressure =
  max(inflammation_pressure, debris_stress_weight * debris_pressure)
```
The memory gets updated as:
```text
stress_memory =
  stress_memory_decay * stress_memory
  + (1 - stress_memory_decay) * stress_pressure
```

## State transitions - `next()`

| Current state | Rule                                                                                  |
|---------------|---------------------------------------------------------------------------------------|
| `Supportive`  | gets to `Reactive` with probability `reactive_transition_rate * stress_memory`        |
| `Reactive`    | gets to `Supportive` if the inflammation is low and the recovery sample is successful |

The recovery sample is:
```text
supportive_recovery_rate * (1 - stress_memory)
```

## Actions

| State                                  | Action                | Effect                 |
|----------------------------------------|-----------------------|------------------------|
| `Supportive`                           | `provide_support`     | reduce inflammation    |
| `Reactive` with memory under threshold | `provide_support`     | reduce inflammation    |
| `Reactive` with memory over threshold  | `release_inflammation` | increases inflammation |

The release rate is derived from the stress memory:
```text
release =
  inflammation_release_rate
  * ((1 - inflammation_memory_weight)
     + inflammation_memory_weight * stress_memory)
```

## Main parameters

**Source:** `src/configuration/param/astrocyte.yaml`

| Category    | Parameters                                                                                                            |
|-------------|-----------------------------------------------------------------------------------------------------------------------|
| Thresholds  | `inflammation_low_threshold`, `inflammation_high_threshold`, `debris_low_threshold`, `debris_high_threshold`          |
| Effects     | `support_inflammation_reduction_rate`, `inflammation_release_rate`                                                    |
| Memory      | `stress_memory_decay`, `debris_stress_weight`                                                                         |
| Transitions | `reactive_transition_rate`, `supportive_recovery_rate`, `inflammatory_memory_threshold`, `inflammation_memory_weight` |

## Interactions

- With `SubstantiaNigra`: reads debris/inflammation and edits inflammation.
- With `Microglia`: either balance or amplify the microglia inflammation.
- With `Neuron`: reducing the inflammation may limit neuronal damage.
