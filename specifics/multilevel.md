# Multilevel Graph Pipeline

This document describes the current G0 -> G1 -> G2 graph pipeline.
`src/analysis/graph/g0_builder.py` is now the only component responsible for
building G0 from simulation logs.

## External Library

The project targets `marco-caputo/multilevel-graphs`, installed as the
`multilevelgraphs` Python package. The package builds a `MultilevelGraph` from
a NetworkX directed graph and an ordered list of contraction schemes.

The local implementation keeps the same conceptual sequence:

```text
G0 --TimeContractionScheme--> G1 --AgentClusteringScheme--> G2
```

The schemes are compatible with `multilevelgraphs` when the package is
available, and they also expose a NetworkX path used for export and tests.

## Output Location

Graph outputs live in:

```text
output/analysis/graphs
```

The current build command is:

```powershell
python main.py graphs
```

Default outputs:

```text
output/analysis/graphs/g1.gexf
output/analysis/graphs/g2.gexf
output/analysis/graphs/multilevel_report.md
```

`g0.gexf` is not rewritten by default because it can be very large. To export it
in the same pass, add:

```powershell
--write-g0
```

## G0

G0 is the full temporal causal graph produced by `g0_builder`.

Each node is one temporally situated entity:

```text
AgentNameID_State@Tick
EnvironmentField@Tick
```

Examples:

```text
AlphaSynuclein_19_3_0_Misfolded@34
Neuron_112_0_0_Compromised@34
SN_dopamine_output@34
Neuron_112_0_0_oxidative_stress@34
```

Agent identities preserve the full runtime UID. In MPI runs that UID commonly
contains rank/type components such as `local_id:type_id:rank`; G0 and G1 must
not truncate it, otherwise agents with the same local id on different ranks
would be merged too early.

Action nodes emitted by the runtime logger are collapsed into edge attributes.
This keeps G0 biological: nodes are agents or fields, while actions are causal
relations.

G0 direct agent-agent relations are intentionally narrow:

- `degradation`: `Lysosome -> Target`.
- `aggregation`: `AlphaSynuclein -> AlphaAggregate`.

`target_assignment` is not treated as degradation and is not retained as a
direct agent-agent edge in G0.

## G1: TimeContractionScheme

`TimeContractionScheme` removes temporal repetition while preserving identity
and state.

The key rule is:

```text
AgentNameID_State@t
AgentNameID_State@t+1
    -> AgentNameID_State
```

Environment fields are contracted similarly:

```text
SN_dopamine_output@t
SN_dopamine_output@t+1
    -> SN_dopamine_output
```

Different states are not merged. For example:

```text
AlphaSynuclein_19_Monomer
AlphaSynuclein_19_Misfolded
```

remain distinct G1 nodes.

### G1 Node Attributes

Each G1 node summarizes the G0 nodes it absorbed:

- `observation_count`: number of temporal observations.
- `first_seen`: first tick observed.
- `last_seen`: last tick observed.
- `absorbed_node_count`: number of lower nodes absorbed.
- `absorbed_edge_count`: number of lower edges absorbed inside the supernode.
- `original_node_ids`: source G0 node ids.

### G1 Edge Attributes

Repeated lower-level edges are compacted into one summary edge:

```python
{
    "count": int,
    "lower_edge_count": int,
    "total_effect": float,
    "mean_effect": float,
    "mean_of_mean_effect": float,
    "first_seen": int,
    "last_seen": int,
    "sign": "+" | "-" | "state" | "structural" | "mixed",
}
```

The edge also keeps compact lists of observed `relations`, `mechanisms`,
`causal_kinds`, `actions`, `outcomes`, and source edge ids.

## G2: AgentClusteringScheme

`AgentClusteringScheme` contracts agents with the same class and state by
dropping the individual id.

Example:

```text
AlphaSynuclein_1_3_0_Misfolded
AlphaSynuclein_2_3_0_Misfolded
    -> AlphaSynuclein_Misfolded
```

Non-agent field nodes remain singleton nodes at this stage. This is deliberate:
field clustering should be a separate future contraction, because merging all
neuron-local fields too early would hide local intracellular context.

G2 edges are compacted using the same summary fields as G1. The `label`
attribute is designed for Gephi and includes the dominant relation, total
count, and mean effect:

```text
relation n=<count> mean=<mean_effect>
```

## Current Implementation

Relevant files:

```text
src/analysis/graph/g0_builder.py
src/analysis/graph/multilevel_builder.py
src/analysis/schemes/time_contractionscheme.py
src/analysis/schemes/agent_clustering_scheme.py
src/analysis/schemes/contraction_utils.py
```

Current generated levels from the latest logs:

```text
G0: 264269 nodes, 372093 edges
G1: 4071 nodes, 5971 edges
G2: 414 nodes, 343 edges
```

## Future Levels

The intended hierarchy is:

- `G0`: full temporal causal trace.
- `G1`: time-contracted entity-state graph.
- `G2`: agent class/state clustered graph.
- `G3`: process, mechanism, or motif graph.

G1 and G2 are conservative enough to preserve causal interpretability while
reducing temporal and agent-level redundancy.
