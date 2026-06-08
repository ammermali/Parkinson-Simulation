# Agent: AlphaSynuclein

**Implementation:** `src/simulation/agents/alphasynuclein.py`

`AlphaSynuclein` represents a single free alpha protein.
The creation of the aggregate doesn't happen inside this agent: it's entirely handled through the `AggregateRegistry`.

## Compartments

| Compartment     | Meaning                      |
|-----------------|------------------------------|
| `Intracellular` | inside a proprietary neuron  |
| `Extracellular` | free in the Substantia Nigra |


## State

| State       | Meaning                                 |
|-------------|-----------------------------------------|
| `Monomer`   | initial and healthy state               |
| `Misfolded` | free and degradable pathological state  |
| `Oligomer`  | represented by a pathological aggregate |
| `LewyBody`  | represented by a pathological Lewy Body |
| `Cleared`   | removed due to degradation              |

## Perception - `see()`

`AlphaSynucleinPerception` contains:

| Field                     | Source                                                    |
|---------------------------|-----------------------------------------------------------|
| `position`                | position in the current compartment                       |
| `oxidative_stress`        | oxidative stress in the current intracellular compartment |
| `local_aggregate_density` | density of aggregate in the current cell                  |
| `neighbors`               | nearby agents within `perception_radius`                  |

## State transitions - `next()`

From `Monomer`, the protein can get to `Misfolded` with a probability of:

```text
pr_misfolding =
  basal_misfold_probability
  + oxidative_misfolding_weight * oxidative_pressure
  + aggregate_seeded_misfold_weight * local_aggregate_density
```

`oxidative_pressure` is zero below `oxidative_stress_high_threshold`, then it increases until the normal value of 1.

From `Misfolded`, the protein doesn't directly create an aggregate. After
`min_misfolded_ticks_before_oligomerization`, it samples:

```text
pr_oligomerization =
  oligomerization_probability_scale
  * (0.3 * neighbor_alpha_density + 0.7 * neighbor_aggregate_density)
```

If the sample is successful, it sets `wants_oligomerization = True`.
The aggregate registry will consume this intention during the collective phase.

## Actions - `action()` and `do()`

| Action | Selection reasoning                         | Effect                                                         |
|--------|---------------------------------------------|----------------------------------------------------------------|
| `move` | intracellular free protein                  | moves to a nearby position with probability `move_probability` |
| `stay` | extracellular, cleared or aggregate protein | no movement                                                    |

During `do`, a misfolded free intracellular protein also gets registered as a target for degradation.

## Pathological weight

`aggregate_weight` contributes to local alpha load:

| State        | Weight |
|--------------|-------:|
| `Misfolded`  |   0.25 |
| `Oligomer`   |   0.75 |
| `LewyBody`   |    1.0 |
| other states |    0.0 |

## Transfer

| Function                 | Effect                                                   |
|--------------------------|----------------------------------------------------------|
| `join_aggregate`         | assign `aggregate_id`, state and freezes the protein     |
| `mark_cleared`           | marks the protein as Cleared                             |
| `release_to_environment` | makes the protein extracellular and removes owner_neuron |
| `absorb_into_neuron`     | makes the protein intracellular and adds owner_neuron    |

## Main parameters

**Source:** `src/configuration/param/alpha.yaml`

| Category        | Parameters                                                                                                                       |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------|
| Perception      | `perception_radius`                                                                                                              |
| Movement        | `move_radius`, `move_probability`                                                                                                |
| Misfolding      | `basal_misfold_probability`, `oxidative_misfolding_weight`, `aggregate_seeded_misfold_weight`, `oxidative_stress_high_threshold` |
| Oligomerization | `oligomerization_probability_scale`, `min_misfolded_ticks_before_oligomerization`                                                |

## Interaction

- With `Neuron`: uses the owner neuron as a habitat and registers to the degradation buffer when it is misfolded.
- With `AggregateRegistry`: delivers the intention to oligomerize and can be absorbed into aggregate members.
- With `Lysosome`: can be degraded if misfolded and free, or as a member of an aggregate.
- With `AlphaAggregate`: stops being an active agent when it is absorbed into an aggregate.
