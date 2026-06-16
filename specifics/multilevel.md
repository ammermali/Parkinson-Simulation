# Multilevel Graph Report

This document describes the complete multilevel graph approach currently used by the project. The pipeline starts from simulation logs and builds a sequence
of progressively coarses directed graphs:

```text
G0 --time contraction--> G1 --agent/state clustering--> G2 --topological SCC contraction--> G3
```

The implementation lives in the following files:

```text
src/analysis/graph/g0_builder.py
src/analysis/graph/multilevel_builder.py
src/analysis/schemes/time_contractionscheme.py
src/analysis/schemes/agent_clustering_scheme.py
src/analysis/schemes/topological_scc_scheme.py
src/analysis/schemes/contraction_utils.py
```

The graph exports are written to:

```text
output/graphs
```

The relevant CLI commands are:

```powershell
python main.py graph-g0
python main.py graph-g1
python main.py graph-g2
python main.py graphs
python main.py graph-g3 --g2 output/graphs/g2.gexf --output-dir output/graphs
```

`graph-g0` builds only the base temporal graph. G0 is full by default: event
logs provide causal edges, while spatial snapshots add agent-state nodes for
observed entities even when they are edge-isolated.

`graph-g1` and `graph-g2` build only the requested intermediate contraction and
write the corresponding Gephi-ready GEXF file.

`graphs` rebuilds the full G0 -> G1 -> G2 -> G3 pipeline from simulation logs.
It writes only these graph artifacts:

```text
g0.gexf
g0.lite.gexf
g1.gexf
g2.gexf
g3.gexf
```

`graph-g0` writes:

```text
g0.gexf
g0.lite.gexf
```

`g0.lite.gexf` keeps the same node ids and directed edges as G0, but removes
rich node attributes and keeps only the edge relation label. It is meant for
fast PyVis/dashboard visualization, not for analytical contraction.

Pass `--no-snapshot-nodes` to `graph-g0`, `graph-g1`, `graph-g2` or `graphs`
when you need an event-only projection for comparison.

Long provenance lists become count attributes, which keeps Gephi files loadable.

`graph-g3` is a fast final-stage command that reads an already exported G2 and
applies only the topological contraction.

## Multilevel Library

The project uses `marco-caputo/multilevel-graphs` as the main library for multilevel contractions.

The package provides the base abstractions used by the project:

- `MultilevelGraph`
- `ContractionScheme`
- `EdgeBasedContractionScheme`
- `CompTable`
- `ComponentSet`

The project defines concrete domain-aware contraction schemes on top of these
abstractions. The export path calls each scheme through `contract(...)`; with
NetworkX inputs the schemes use an adapter that materializes standard directed
graphs for GEXF output. `TopologicalSCCContractionScheme` inherits from
the library's `EdgeBasedContractionScheme`; this is important because G3 is not
an arbitrary semantic merge, but an edge/topology-driven contraction.

## G0: Temporal Causal Trace

G0 is the most detailed graph. It is built by
`src/analysis/graph/g0_builder.py` from:

```text
output/run_logs/events.jsonl
output/run_logs/spatial_snapshots.jsonl
```

Each G0 node is a simulation entity at a specific tick:

```text
AgentNameID_State@Tick
Environment_Field@Tick
```

Examples:

```text
AlphaSynuclein_19_3_0_Misfolded@34
Neuron_112_0_0_Compromised@34
SN_dopamine_output@34
Neuron_112_0_0_oxidative_stress@34
```

Runtime decision nodes are not represented in G0. This keeps G0 focused:
nodes are agents or fields, while effects are represented as causal relations.

### G0 Nodes

`agent_state`

An agent in a state at a specific tick. Examples:

```text
Neuron_0_0_0_Healthy@1
AlphaSynuclein_12_3_0_Misfolded@45
Lysosome_51_5_0_Active@45
```

`aggregate`

An aggregate represented as an agent-like state node. It is treated as a
biological object rather than as the registry itself. The registry is not a G0
node.

`env_field`

A global Substantia Nigra scalar at a specific tick. Examples:

```text
SN_extracellular_debris@10
SN_inflammation_level@10
SN_dopamine_output@10
```

`internal_field`

A neuron-local scalar at a specific tick. Examples:

```text
Neuron_0_0_0_oxidative_stress@10
Neuron_0_0_0_intracellular_debris@10
Neuron_0_0_0_energy_demand@10
```

### G0 Node Attributes

Common attributes include:

- `formal_id`: canonical graph id.
- `display_id`: readable display id.
- `semantic_kind`: `agent_state` or `environment_field`.
- `kind`: raw log kind, such as `agent_state`, `env_field`, or `internal_field`.
- `agent_type`: biological/simulation class, such as `Neuron`, `AlphaSynuclein`, `Lysosome`.
- `uid`: runtime uid.
- `state`: agent state, if present.
- `field`: field name, if present.
- `value`: field value, if present.
- `tick`: simulation tick.
- `rank`: MPI rank.
- `run_id`: run identifier.
- `level`: raw logger level, such as `environment` or `intracellular`.
- `owner_uid`: owning neuron for intracellular agents/fields.
- `compartment`: biological compartment.
- `entity_key`: identity used for continuity contraction.
- `source_log_node_ids`: source log rows absorbed into this G0 node.

### G0 Edges

`threshold_trigger`

A field threshold influences an agent transition.
Typical relation:

```text
threshold_trigger
```

Direction:

```text
EnvironmentField@t -> AgentState@t
```

`field_effect`

An agent affects an environment or intracellular field.

Direction:

```text
AgentState@t -> EnvironmentField@t
```

Typical relations:

```text
field_effect
internal_field_effect
```

`state_transition`

An agent changes state.

Direction:

```text
AgentStateA@t -> AgentStateB@t
```

`degradation`

Direct agent-agent edge for lysosomal degradation.

Direction:

```text
Lysosome -> Target
```

This is intentionally narrow: target assignment alone is not treated as
degradation.

`aggregation`

Direct agent-agent edge for alpha aggregation.

Direction:

```text
AlphaSynuclein -> AlphaAggregate
```

The aggregate registry is not represented as a graph node. Its effect is
represented through edges from alpha-synuclein members into the aggregate.

`continuity`

Temporal identity edge added by `g0_builder`.

Direction:

```text
Entity_State@t -> Entity_State@t+1
```

Continuity is useful because it tells later contractions which repeated nodes
represent the same persistent biological/simulation entity. It can be disabled
when building graphs if needed.

### G0 Edge Attributes

Common edge attributes include:

- `edge_id`: source log edge id.
- `relation`: raw relation class.
- `mechanism`: biological/simulation mechanism.
- `causal_kind`: normalized causal class.
- `tick`: simulation tick.
- `rank`: MPI rank.
- `run_id`: run identifier.
- `rule_id`: rule id, if logged.
- `predicate`: predicate or threshold check, if logged.
- `effect_value`: numeric effect when available.
- `sign`: normalized sign after contraction.
- `effect_unit`: unit label, if present.
- `probability`: probability used by stochastic mechanisms.
- `rng_value`: sampled random value.
- `outcome`: event outcome.
- `source_kind`, `target_kind`: semantic endpoint kinds.
- `source_type`, `target_type`: endpoint agent types.
- `source_uid`, `target_uid`: endpoint identities.
- `owner_uid`: neuron owner, when applicable.
- `compartment`: biological compartment.
- `count`, `total_effect`, `mean_effect`, `first_tick`, `last_tick`: summary values when lower edges merge.

## G1: Time Contraction

G1 is built by `TimeContractionScheme`.

The purpose of G1 is to remove raw tick repetition while preserving:

- entity identity,
- state,
- field identity,
- causal direction,
- first/last observation range.

The basic contraction rule is:

```text
AgentNameID_State@t
AgentNameID_State@t+1
    -> AgentNameID_State
```

For fields:

```text
SN_dopamine_output@t
SN_dopamine_output@t+1
    -> SN_dopamine_output
```

### G1 Possible Situations

#### Persistent same-state agent

If the same agent remains in the same state across ticks, its temporal nodes
are absorbed into one G1 node.

#### State-changing agent

Different states are not merged. Example:

```text
AlphaSynuclein_19_Monomer
AlphaSynuclein_19_Misfolded
```

These remain distinct nodes, usually connected by a state-transition edge.

#### Persistent field

Repeated observations of the same field are contracted into one field node.

#### Windowed time contraction

If `window_size` is set, time contraction is not global over the whole trace.
Instead, nodes are contracted inside fixed tick windows:

```text
AlphaSynuclein_19_Monomer_t0_9
AlphaSynuclein_19_Monomer_t10_19
```

This is useful when the user wants temporal windows rather than a fully timeless
G1.

#### Internal continuity absorbed

If continuity edges connect nodes that are contracted into the same G1 node,
they are absorbed into the node's `absorbed_edge_count`.

### G1 Node Attributes

G1 nodes use the shared summary attributes created by `summarize_nodes`:

- `label`: supernode label.
- `contraction`: `time`.
- `level`: `G1`.
- `semantic_kind`: common semantic kind or `mixed`.
- `agent_type`: common agent type or `mixed`.
- `uid`: common uid or `mixed`.
- `state`: common state or `mixed`.
- `field`: common field or `mixed`.
- `entity_key`: common entity key or `mixed`.
- `member_count`: number of direct lower nodes absorbed.
- `observation_count`: total lower observations represented.
- `absorbed_node_count`: recursive absorbed node count.
- `absorbed_edge_count`: lower edges absorbed inside the supernode.
- `first_seen`: first tick represented.
- `last_seen`: last tick represented.
- `original_node_ids`: source lower node ids.

### G1 Edge Attributes

G1 edges summarize repeated lower G0 edges:

- `count`: total number of lower events represented.
- `lower_edge_count`: number of direct lower graph edges merged.
- `total_effect`: sum of represented effects.
- `mean_effect`: `total_effect / count`.
- `mean_of_mean_effect`: mean of lower edge means.
- `first_seen`: first tick represented.
- `last_seen`: last tick represented.
- `sign`: `+`, `-`, `state`, `structural`, or `mixed`.
- `relation`: compact relation label or `mixed`.
- `mechanism`: compact mechanism label or `mixed`.
- `causal_kind`: compact causal kind or `mixed`.
- `outcome`: compact outcome or `mixed`.
- `relations`, `mechanisms`, `causal_kinds`, `outcomes`: provenance lists.
- `source_edge_ids`: source lower edge ids.
- `weight`: equal to `count`.
- `label`: compact Gephi label.

## G2: Agent/State Clustering

G2 is built by `AgentClusteringScheme`.

The purpose of G2 is to remove individual agent identity and expose class/state
behavior.

Example:

```text
AlphaSynuclein_1_3_0_Misfolded
AlphaSynuclein_2_3_0_Misfolded
    -> AlphaSynuclein_Misfolded
```

### G2 Possible Situations

#### Agent-state cluster

All agents with the same `agent_type` and `state` are merged.

Examples:

```text
Neuron_Healthy
Neuron_Compromised
AlphaSynuclein_Misfolded
Lysosome_Active
```

Different states remain separated. This matters biologically: `Healthy`,
`Compromised`, `Apoptotic`, and `Ruptured` are not equivalent.

#### Substantia Nigra fields

SN fields remain separate singleton nodes.

Examples:

```text
SN_dopamine_output
SN_extracellular_debris
SN_inflammation_level
```

#### Neuron internal fields

Neuron-local fields are clustered into:

```text
Neuron_internal_environment
```

This prevents G2 from being dominated by one node per neuron-local field while
still preserving the idea that intracellular environment participates in causal
patterns.

#### Mixed edge summaries

If many lower edges with different mechanisms collapse into the same G2 edge,
the compact label can become `mixed`. The provenance lists still retain the set
of observed mechanisms, relations, and outcomes.

### G2 Node Attributes

G2 uses the same core summary attributes as G1, plus:

- `cluster_kind`: `agent_state_cluster` for clustered agent states, otherwise `singleton`.

### G2 Edge Attributes

G2 uses the same summary edge attributes as G1:

- `count`
- `lower_edge_count`
- `total_effect`
- `mean_effect`
- `mean_of_mean_effect`
- `first_seen`
- `last_seen`
- `sign`
- `relation`
- `mechanism`
- `causal_kind`
- `outcome`
- provenance lists and counts
- `weight`
- `label`

## G3: Topological SCC Contraction

G3 is built by `TopologicalSCCContractionScheme`.

This scheme inherits from `multilevelgraphs.contraction_schemes.EdgeBasedContractionScheme`
and contracts strongly connected components of G2.

The motivation is topological rather than biological. G2 already tells us which
agent classes/states interact. G3 asks a different question:

```text
Which parts of the G2 causal graph form recurring directed feedback patterns?
```

In a directed graph, a strongly connected component is a conservative definition
of recurrence: every node in the component can reach every other node through
directed paths. This avoids false positives from mere co-occurrence or weak
connectivity.

### G3 Possible Situations

#### `feedback_component`

A component with more than one G2 node. This is the central G3 situation.

Interpretation:

```text
These processes are mutually reachable and form a recurring directed causal pattern.
```

Example conceptual pattern:

```text
SN_inflammation_level -> Microglia_Activated -> SN_inflammation_level
```

#### `self_feedback`

A singleton node with an internal self-loop.

Interpretation:

```text
The process has a self-reinforcing edge after previous contractions.
```

This can occur when repeated lower-level relations collapse into a self-edge at
G2.

#### `singleton_process`

A node that is not part of a directed feedback component.

Interpretation:

```text
The process is present, but it is not topologically recurrent at G2.
```

Singletons are kept because they preserve the boundary of feedback components:
they can be sources, sinks, inputs, or downstream consequences.

#### Internal absorbed edge

An edge whose source and target are inside the same SCC is absorbed into the G3
node. These edges define the pattern's internal recurrence.

#### Boundary incoming edge

An edge from another G3 node into a component.

Interpretation:

```text
External process -> recurring pattern
```

#### Boundary outgoing edge

An edge from a component to another G3 node.

Interpretation:

```text
Recurring pattern -> downstream process
```

#### Acyclic inter-pattern edge

Edges between different G3 nodes remain as G3 edges. These represent causal
flow between patterns or singleton processes.

### G3 Node Attributes

G3 keeps the shared summary node attributes:

- `label`
- `contraction`: `topological_scc`.
- `level`: `G3`.
- `semantic_kind`
- `agent_type`
- `uid`
- `state`
- `field`
- `entity_key`
- `member_count`
- `observation_count`
- `absorbed_node_count`
- `absorbed_edge_count`
- `first_seen`
- `last_seen`
- `original_node_ids`

G3 also adds topological pattern attributes:

- `pattern_kind`: `feedback_component`, `self_feedback`, or `singleton_process`.
- `component_size`: number of G2 nodes inside this G3 node.
- `is_feedback_pattern`: boolean.
- `internal_edge_count`: number of direct G2 edges inside the SCC.
- `boundary_in_edge_count`: number of direct G2 edges entering the SCC.
- `boundary_out_edge_count`: number of direct G2 edges leaving the SCC.
- `boundary_in_superedge_count`: number of G3 incoming superedges after contraction.
- `boundary_out_superedge_count`: number of G3 outgoing superedges after contraction.
- `internal_event_count`: sum of represented event counts on internal edges.
- `boundary_in_event_count`: sum of represented event counts on incoming boundary edges.
- `boundary_out_event_count`: sum of represented event counts on outgoing boundary edges.
- `total_internal_effect`: summed internal effect.
- `mean_internal_effect`: `total_internal_effect / internal_event_count`.
- `total_boundary_in_effect`: summed incoming boundary effect.
- `mean_boundary_in_effect`: incoming mean effect.
- `total_boundary_out_effect`: summed outgoing boundary effect.
- `mean_boundary_out_effect`: outgoing mean effect.
- `dominant_internal_relation`: most frequent internal relation weighted by event count.
- `dominant_internal_mechanism`: most frequent internal mechanism weighted by event count.
- `dominant_internal_causal_kind`: most frequent internal causal kind weighted by event count.
- `dominant_internal_sign`: dominant internal sign.
- `dominant_boundary_in_relation`: dominant incoming relation.
- `dominant_boundary_out_relation`: dominant outgoing relation.
- `node_signature`: compact count of node roles inside the pattern.
- `internal_edge_signature`: compact count of internal edge roles.
- `boundary_in_signature`: compact count of incoming edge roles.
- `boundary_out_signature`: compact count of outgoing edge roles.
- `topological_signature`: combined node and internal-edge signature.
- `component_node_ids`: lower G2 nodes represented.

For Gephi export, long provenance attributes such as `component_node_ids`,
`original_node_ids`, `source_edge_ids`, and edge provenance lists are replaced
with count attributes to avoid oversized GEXF fields.

### G3 Edge Attributes

G3 edge attributes are produced by the same edge summarization logic used for
G1 and G2:

- `count`
- `lower_edge_count`
- `total_effect`
- `mean_effect`
- `mean_of_mean_effect`
- `first_seen`
- `last_seen`
- `sign`
- `relation`
- `mechanism`
- `causal_kind`
- `outcome`
- provenance list counts in Gephi exports
- `weight`
- `label`

The meaning changes slightly at G3: an edge is no longer a single direct
biological relation, but a summarized causal channel between patterns.

## Why SCC For G3

Several topological contractions are possible: weak components, connected
components, cliques, cycles, motifs, or SCCs.

SCC is the best current choice because:

- the graph is directed;
- causal direction matters;
- recurrence should require mutual reachability;
- singleton non-pattern nodes remain visible;
- boundary edges make feedback inputs and outputs analyzable;
- it is conservative and avoids merging large weakly connected regions.

Weakly connected components would be too aggressive. In a causal simulation,
almost everything may be weakly connected through environment fields, causing
G3 to collapse into a small number of uninformative giant components.

Cliques would be too strict. Biological causal graphs rarely contain complete
directed connectivity between all process pairs, even when they are clearly
part of a feedback loop.

Simple cycle enumeration would be difficult to summarize because cycles can
overlap heavily. SCCs provide a stable, compact supernode representation for
overlapping cycles.

## Graph Export Policy

Graph exports are made Gephi-safe by converting complex provenance attributes
into scalar values.

Long lists are not written directly. Instead, attributes such as:

```text
original_node_ids
component_node_ids
source_edge_ids
relations
mechanisms
causal_kinds
outcomes
```

are exported as:

```text
original_node_ids_count
component_node_ids_count
source_edge_ids_count
relations_count
mechanisms_count
causal_kinds_count
outcomes_count
```

This keeps GEXF files loadable in Gephi while preserving the amount of
provenance represented by each node or edge.

## Practical Reading Strategy

Use G0 when the question is:

```text
What exactly happened at a specific tick?
```

Use G1 when the question is:

```text
How did a specific entity/state behave across time?
```

Use G2 when the question is:

```text
How do biological classes and states interact in aggregate?
```

Use G3 when the question is:

```text
Which recurring causal feedback patterns exist in the class/state graph?
```

G3 should not replace G2. It is a higher-level view intended to highlight
recurrence, feedback and pattern boundaries.
