# Multilevel Graphs

This document describes the current multilevel graph plan for the Parkinson
simulation. The implementation follows the idea in `specifics/presentation.pdf`:
G0 is the full temporal causal trace and higher levels are progressively
contracted abstractions of the same process.

## External Library

The library used for the implementation of multilevel graphs is `marco-caputo/multilevel-graphs`, distributed as the
`multilevelgraphs` Python package. It is based on NetworkX and represents a
multilevel graph as a base graph plus an ordered sequence of contraction
schemes. 

## Output Location

Graph outputs live in:

```text
output/analysis/graphs
```

The build command exports G0 in Gephi-ready formats:

```powershell
python src/analysis/build_multilevel_graphs.py --summary
```

Default outputs:

```text
output/analysis/graphs/g0.gexf
output/analysis/graphs/g0.graphml
output/analysis/graphs/g0.json
```

GEXF and GraphML are intended for Gephi. JSON is kept for inspection and future
analysis code.

## G0

G0 is the most detailed graph.

Each node is one temporally situated entity:

```text
AgentNameID_State@Time
Environment_Field@Time
```

Examples:

```text
AlphaSynuclein_a:12_Misfolded@34.2
Neuron_n:3_Compromised@34.2
SN_dopamine@34.5
Neuron_n:3_oxidative_stress@34.5
```

G0 edges are typed causal links:

- `perception`: an environmental or internal field contributes to an agent
  decision or transition.
- `action`: an agent affects an environmental or intracellular field.
- `transition`: an agent changes state.
- `agent_relation`: an agent targets or acts on another agent.
- `aggregation`: alpha-synuclein agents contribute to an aggregate.
- `continuity`: the same entity continues from one observed moment to the next.

Runtime action nodes can be collapsed by `g0_lexer.py`, so an action is stored
as an edge attribute rather than as a biological node.

## G1: TimeContraption

`TimeContraption` is the first contraction scheme. It creates G1 by contracting
time while preserving agent state identity.

The key rule is:

```text
AgentNameID_State@t
AgentNameID_State@t+1
    -> AgentNameID_State
```

Environment fields are contracted similarly:

```text
SN_dopamine@t
SN_dopamine@t+1
    -> SN_dopamine
```

Important consequence: different states are not merged. For example:

```text
AlphaSynuclein_a:12_Monomer
AlphaSynuclein_a:12_Misfolded
```

remain two different G1 supernodes. The edge between them preserves the state
transition evidence.

## G1 Node Attributes

Each G1 node summarizes the G0 nodes it absorbed:

- `observation_count`: number of temporal observations in G0.
- `first_seen`: first tick in which the supernode was observed.
- `last_seen`: last tick in which the supernode was observed.
- `g0_node_ids`: source G0 node ids.
- `absorbed_edge_count`: number of G0 edges that became internal to the
  supernode during contraction.
- `absorbed_total_effect`: total numeric effect of internal absorbed edges.
- `absorbed_mean_effect`: mean numeric effect of internal absorbed edges.
- `absorbed_relations`: relation labels hidden inside the supernode.

Internal continuity edges usually end up here.

## G1 Edge Attributes

If several G0 edges connect the same pair of G1 supernodes, they are compacted
into one superedge:

```python
G1_edge = {
    "count": int,
    "total_effect": float,
    "mean_effect": float,
    "first_seen": int,
    "last_seen": int,
    "sign": "+" | "-" | "state" | "structural",
}
```

Additional attributes are kept for analysis:

- `relations`: distinct G0 relation types.
- `mechanisms`: distinct mechanisms observed on the edge.
- `causal_kinds`: normalized G0 causal classes.
- `actions`: collapsed action labels, when available.
- `g0_edge_ids`: source G0 edge ids.
- `weight`: equal to `count`, useful for Gephi visualization.

The compacted `sign` is derived as follows:

- `+` when the total effect is positive.
- `-` when the total effect is negative.
- `state` when the edge is a state transition with no numeric effect.
- `structural` when the edge is causal or temporal but has no numeric direction.

## Current Implementation

Relevant files:

```text
src/analysis/g0_lexer.py
src/analysis/schemes/time_contractionscheme.py
src/analysis/build_multilevel_graphs.py
```

`g0_lexer.py` builds G0 from `g0_nodes` and `g0_edges`.
`time_contractionscheme.py` defines `TimeContraption`.
`build_multilevel_graphs.py` exports G0 and G1 together.

## Future Levels

The intended hierarchy is:

- `G0`: full temporal causal trace.
- `G1`: time-contracted graph.
- `G2`: agent/mechanism-clustered graph.
- `G3`: motif or process graph.

G1 is intentionally conservative: it removes repeated time observations but does
not merge biological states, mechanisms, or agent classes. This keeps enough
detail for later contractions to choose biologically meaningful abstractions.
