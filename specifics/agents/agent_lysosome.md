# Agent: Lysosome

**Implementazione:** `src/simulation/agents/lysosome.py`

`Lysosome` is an intracellular agent responsible for degradation and reparation.
It doesn't directly know its targets: misfolded proteins, aggregates and damaged mitochondria register themself to a buffer inside the neuron;
the lysosome then selects an available target and degrade it over time.

## Responsibilities

- Scan the intracellular grid when inactive.
- Activate in presence of target/aggregate pressure.
- Gets assigned to a degradable target by the owner neuron.
- Degrade alpha proteins and aggregates.
- Repair damaged mitochondria.
- Becomes `Overwhelmed` if the target is too pathological.

## State

| State         | Meaning                             |
|---------------|-------------------------------------|
| `Inactive`    | not active in any degradation task  |
| `Active`      | ready to select or degrade a target |
| `Overwhelmed` | disabled lysosome, stays idle       |

## Perception - `see()`

`LysosomePerception` contains:

| Field                     | Source                                           |
|---------------------------|--------------------------------------------------|
| `position`                | position in the intracellular grid               |
| `targets`                 | unassigned degradable targets                    |
| `task`                    | assigned target to this lysosome                 |
| `local_aggregate_density` | local aggregate density                          |
| `target_pressure`         | available target over the number of total agents |

## State transitions - `next()`

From `Inactive`:
```text
pr_inactive_to_active =
  target_pressure + local_aggregate_density
  - target_pressure * local_aggregate_density
```

Da `Active`:
```text
pr_active_to_inactive =
  (1 - task_pressure)
  * (1 - target_pressure)
  * (1 - local_aggregate_density)
```

`task_pressure` is 1 if the lysosome already has a target, 0 otherwise.

The agent becomes `Overwhelmed` if it meets a non-degradable target.

## Actions - `action()` and `do()`

| State                   | Action          | Effect                             |
|-------------------------|-----------------|------------------------------------|
| `Inactive`              | `scan`          | local random movement              |
| `Active` without target | `select_target` | claims a target from the buffer    |
| `Active` with target    | `degrade`       | proceeds or finish the degradation |
| `Overwhelmed`           | `idle`          | no effect                          |

## Multi-tick degradation

Every target requires a number of tick before complete degradation:

| Target           | Time                                                                                     |
|------------------|------------------------------------------------------------------------------------------|
| `AlphaSynuclein` | `protein_degradation_ticks`                                                              |
| `Mitochondrion`  | `mitochondrion_repair_ticks`                                                             |
| `AlphaAggregate` | `aggregate_degradation_ticks_base + aggregate_degradation_ticks_per_member * (size - 1)` |
| other target     | 1                                                                                        |

After the completing the task, the lysosome samples three possible, mutuably exclusive, outcomes:
1. overwhelm;
2. success;
3. failing (the target gets re-queued).

For the aggregates:

```text
pr_success =
  aggregate_degradation_probability_base
  + aggregate_degradation_probability_per_member * size

pr_overwhelm =
  aggregate_overwhelm_probability_base
  + aggregate_overwhelm_probability_per_member * size
```

For the Lewy bodies, `pr_overwhelm = 1.0`.

## Outcome for target

| Target                            | Success                                            |
|-----------------------------------|----------------------------------------------------|
| `AlphaAggregate`                  | removes the aggregate and mark is as `Cleared`     |
| free `AlphaSynuclein`             | `mark_cleared` and removal from buffer             |
| `AlphaSynuclein` aggregate member | update of the aggregate registry                   |
| `Mitochondrion`                   | call `repair_by_lysosome` and leave it on the grid |
| other agent                       | removal from the grid                              |

A failure cleans the assignment and re-queue the target as available.

## Main parameters

**Source:** `src/configuration/param/lysosome.yaml`

| Category     | Parameter                                                                                                |
|--------------|----------------------------------------------------------------------------------------------------------|
| Perception   | `perception_radius`                                                                                      |
| Movement     | `move_radius`                                                                                            |
| Proteins     | `base_degradation_probability`, `protein_degradation_ticks`                                              |
| Mitochondria | `mitochondrion_repair_ticks`, `mitochondrion_repair_probability`                                         |
| Aggregates   | `degradation_ticks_per_member`, `degradation_probability_per_member`, `overwhelm_probability_per_member` |

## Interactions

- With `Neuron`: reads and modify `degradation_targets` and
  `degradation_assignment` buffers.
- With `AlphaSynuclein`: degrade misfolded proteins.
- With `AlphaAggregate`: degrade aggregate, but risks overwhelming.
- With `Mitochondrion`: repairs damaged mitochondria instead of removing them.
