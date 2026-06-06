# Agent: AlphaAggregate

**Implementation:** `src/simulation/agents/aggregate.py`

**Coordinator:** `src/simulation/agents/aggregate_registry.py`

`AlphaAggregate` represents a set of alpha-synuclein proteins aggregated. It's the unit used for the grid representation, lysosomal degradation, local aggregate density and Lewy bodies maturation.

## Responsibility

Single misfolded `AlphaSynuclein` can declare the intention to oligomerize, but the actual decision is collective as it requires more proteins in the same cell or the presence of a recruiting aggregate. 
`AggregateRegistry` centralizes this phase in order to maintain identity, member and biological state consistency.

## State

| Stato | Description                                                |
|---|------------------------------------------------------------|
| `Oligomer` | initial aggregate, degradable and potentially recruiting   |
| `LewyBody` | mature aggregate; always degrades and overwhelms lysosomes |

The aggregate keeps:

- `aggregate_id`: biological identity assigned from the registry;
- `member_ids`: IDs of the aggregated proteins;
- `member_agents`: reference to the `AlphaSynuclein` instances when available;
- `owner_neuron`: neuron to which it belongs to, or `None` if it's extracellular.

## Pathological Weight

Weight contributes to `local_aggregate_density` and `alpha_load`.

| State      | Formula |
|------------|---|
| `Oligomer` | `min(1.25, 0.75 + 0.05 * (size - 1))` |
| `LewyBody` | `min(2.0, 1.0 + 0.10 * (size - 1))` |

`size` is the number of alpha proteins represented.

## AggregateRegistry

`AggregateRegistry` handles:

- aggregate creation;
- absorption of misfolded proteins into existing aggregates;
- merge of multiple aggregates in the same cell;
- maturation from `Oligomer` to `LewyBody`;
- synchronization of members' state;
- invariant validation;
- removal after degradation.

## Collective Rules

During `process(habitat)`, the registry collects agents cell-by-cell and applies these rules:

1. Remove completely cleared aggregates.
2. Collects `AlphaSynuclein` proteins that are free, misfolded and with `wants_oligomerization = True`
3. Collects aggregates in the same cell
4. If a Lewy body exists, chooses the largest one as a seed, then merges the other recruiting aggregates and absorb candidate proteins.
5. If there are recruiting oligomers, chooses the largest one, merges the others and absorb the candidate proteins.
6. If there are at least two candidate proteins and not a recruiting aggregate, it creates a new oligomer.
7. Samples the maturation into Lewy body.
8. Validate invariants.

The maturation probability is:

```text
pr_maturation = clamp(aggregate.size / lewy_body_size_threshold)
```

## Invariants

The registry fails with `AggregateInvariantError` when it finds incoherence, like:

- memberless aggregate;
- `member_ids` not synchronized to tracked members;
- member that point to a different `aggregate_id`;
- member still tracked after becoming `Cleared`;
- registred aggregate but absent from the grid of the owner neuron;
- Lewy body with members not `LewyBody`.

## Degradation

Aggregates are exposed to the lysosomial buffer within the neuron. If a `Lysosome` successfully degrades an aggregate:
- the aggregate gets removed from the grid;
- the members tracked get marked as `Cleared`;
- the registry removes bookkeeping and membership.

## Intra-extracellular transfer

An aggregate can be released in the Substantia Nigra by a neuron or get absorbed by another neuron.

| Function                            | Effect                                               |
|-------------------------------------|------------------------------------------------------|
| `release_to_environment`            | removes `owner_neuron`                               |
| `absorb_into_neuron`                | assign a new neuron to the aggregate                 |
| `register_existing_aggregate`       | register the aggregate in the new habitat registry   |
| `unregister_aggregate_for_transfer` | unregister an aggregate without deleting the members |


## Interazioni

- With `AlphaSynuclein`: represented members that are no more active in a grid.
- With `Neuron`: lives in the intracellular grid, contributes to alpha_load and gets targetted. 
- With `Lysosome`: can overwhelm the lysosome or get degraded by it.
- With other aggregates: gets merged by the registry.
