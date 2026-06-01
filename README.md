# Parkinson Simulation

Agent-based Parkinson's disease simulation and analysis toolkit.

This repository models a simplified Substantia Nigra environment with neurons,
alpha-synuclein proteins, aggregates, mitochondria, lysosomes, microglia and
astrocytes. The current focus is not final biological calibration yet, but a
simulation-ready architecture with explicit mechanisms, YAML-driven parameters,
causal logging and graph-based analysis.

## Current Simulation

Implemented biological mechanisms include:

- Alpha-synuclein misfolding, oligomerization and Lewy body formation.
- Environment-level `AggregateRegistry` ownership for aggregate identity,
  membership tracking and invariant validation.
- Lysosome targeting, degradation, failure and overwhelming logic.
- Mitochondrion lifecycle, damage progression and lysosome-mediated recovery.
- Neuron internal environment, damage estimation, dopamine release, rupture,
  alpha release into the Substantia Nigra and gradual alpha absorption.
- Microglia and astrocyte extracellular response to inflammation, debris and
  nearby alpha-related signals.
- Shared Substantia Nigra scalar lifecycle for debris, inflammation and
  dopamine.
- Per-agent configuration sampling for perception thresholds.

## Project Layout

```text
src/simulation/agents/       Agent classes and biological mechanisms
src/simulation/logger/       Causal and initialization loggers
src/simulation/params/       YAML parameter files
src/simulation/utils/        Params, RNG and config factory utilities
src/analysis/                Log analysis, G0 lexer and graph builders
src/analysis/schemes/        Multilevel graph contraction schemes
src/visualization/           Plotters for tick_metrics.csv
specifics/                  Project notes and specifications
test/                       Unit tests
output/simulation/          Simulation outputs
output/analysis/             Analysis, plots and graph outputs
```

## Runtime Dependencies

The simulation uses:

- Python 3.10 or newer.
- `repast4py`.
- `mpi4py`.
- `PyYAML`.

Analysis and graph export additionally use:

- `networkx`.
- `matplotlib`.
- `multilevelgraphs`, optional for the future full multilevel hierarchy.

## Running The Simulation

From the repository root:

```powershell
mpiexec -n 4 python main.py simulate
```

The CLI loads `src/simulation/params/system.yaml` by default and delegates to
the Repast4Py engine. The engine can still be run directly when needed:

```powershell
mpiexec -n 4 python src/simulation/engine.py
```

Main system settings include:

- `stop.at`: final simulation tick.
- `random.seed`: global seed.
- `external.population`: global population counts.
- `world.width`, `world.height`, `world.buffer_size`: Repast grid shape.
- `logging.output_dir`: runtime output directory.
- `logging.causal.enabled`: enables G0 causal nodes and edges.
- `logging.initialization.enabled`: enables initial-condition logs.
- `logging.tick_metrics_csv`: writes compact per-tick metrics.
- `logging.progress_stdout`: prints progress during long runs.
- `logging.summary_stdout`: prints the final simulation summary.

Default simulation outputs are written to:

```text
output/simulation/logs
```

## Configuration

Parameters are stored in one YAML file per configurable agent or mechanism:

```text
src/simulation/params/system.yaml
src/simulation/params/neuron.yaml
src/simulation/params/microglia.yaml
src/simulation/params/astrocyte.yaml
src/simulation/params/alpha.yaml
src/simulation/params/mitochondrion.yaml
src/simulation/params/lysosome.yaml
src/simulation/params/substantia_nigra.yaml
```

`Params` loads YAML files safely by name, filename or explicit path.
`ConfigFactory` converts YAML values into runtime config dataclasses.

Perception thresholds use nested `mean` and `std` entries and are sampled once
per agent config creation:

```yaml
thresholds:
  inflammation_high_threshold:
    mean: 0.7
    std: 0.05
```

The sampled value is clamped to `[0.0, 1.0]`. Non-threshold rates,
probabilities, counts and decay values remain scalar unless explicitly modeled
otherwise.

For a full parameter reference, see:

```text
specifics/parameters_guide.md
```

or click [here](specifics/parameters_guide.md).

## Logging

The simulation currently writes three complementary log families.

### Causal G0 Logs

When causal logging is enabled, the engine writes:

```text
g0_nodes.jsonl
g0_edges.jsonl
```

or rank-local files that are merged at the end of a distributed run.

G0 nodes represent timed states:

```text
AgentNameID_State@Time
Environment_Field@Time
```

G0 edges represent causal relations such as perception, action, transition,
targeting, degradation and aggregation.

### Initialization Logs

Initialization logs describe the starting state, including alpha-synuclein
agents that do not immediately misfold. These logs are useful for checking that
population counts, configs and initial placement were imported correctly.

### Tick Metrics

`tick_metrics.csv` stores compact global values per tick:

```text
debris,inflammation,dopamine,
neurons_healthy,neurons_compromised,neurons_apoptotic,neurons_ruptures,
free_alpha,alpha_aggregate
```

`free_alpha` counts free alpha-synuclein proteins. `alpha_aggregate` counts
alpha-synuclein proteins represented inside aggregates, not the number of
aggregate agents. This is the main lightweight file for calibration plots.

## Analysis Commands

Mechanism counts from G0 causal traces:

```powershell
python main.py mechanisms
```

Intervention-oriented scalar and event summary:

```powershell
python main.py intervention
```

Intervention analysis is used to compare completed simulation runs in which a
mechanism, probability, threshold or parameter group has been deliberately
changed. It is not a correctness test by itself. It helps answer questions such
as whether weakening lysosome degradation, changing alpha aggregation
probabilities, or altering glial thresholds produces a measurable difference in
pathology progression. The report summarizes threshold crossings, final
environmental values, key event counts and agent-state counts over time.

Validate G0 trace consistency:

```powershell
python main.py validate-g0
```

Validate initialization logs:

```powershell
python main.py validate-init
```

Default analysis outputs are written to:

```text
output/analysis
```

## Visualization

Plotters read `output/simulation/logs/tick_metrics.csv` by default and write
PNG plots to `output/plots`.

Neuron states:

```powershell
python main.py plot neurons
```

Free alpha:

```powershell
python main.py plot alpha-free
```

Aggregate alpha:

```powershell
python main.py plot alpha-aggregate
```

Both alpha plots, written as separate figures:

```powershell
python main.py plot alpha
```

Substantia Nigra scalar trends:

```powershell
python main.py plot sn
```

All default plots:

```powershell
python main.py plot all
```

Default plot outputs are written to:

```text
output/plots
```

## Multilevel Graphs

The graph analysis pipeline now supports:

- `G0`: full timed causal trace.
- `G1`: first contraction level created by `TimeContraption`.

`G0` is built from causal node and edge logs. `TimeContraption` contracts time
while preserving state identity:

```text
AgentNameID_State@t
AgentNameID_State@t+1
    -> AgentNameID_State
```

Different states remain distinct supernodes, so a transition such as
`Monomer -> Misfolded` is preserved as a G1 edge.

G1 superedges summarize repeated interactions with:

```python
{
    "count": int,
    "total_effect": float,
    "mean_effect": float,
    "first_seen": int,
    "last_seen": int,
    "sign": "+" | "-" | "state" | "structural"
}
```

Build and export both G0 and G1:

```powershell
python main.py graphs
```

Default graph outputs:

```text
output/analysis/graphs/g0.gexf
output/analysis/graphs/g0.graphml
output/analysis/graphs/g0.json
output/analysis/graphs/g1_time.gexf
output/analysis/graphs/g1_time.graphml
output/analysis/graphs/g1_time.json
```

GEXF and GraphML are Gephi-ready. JSON is kept for inspection and downstream
analysis.

For the full multilevel plan, see:

```text
specifics/multilevel.md
```

or click [here](specifics/multilevel.md).

## Testing

Run tests from the repository root:

```powershell
python -m pytest
```

Some graph tests skip automatically when `networkx` is not installed. Simulation
tests require the Repast4Py/MPI runtime.

## Current Modeling Notes

This is still an exploratory simulation. Parameter tuning is expected. The
current logging and graph infrastructure is designed to make that tuning
observable: if a run converges too quickly, produces too much pathology or hides
an expected mechanism, the causal logs and mechanism metrics should make the
failure mode inspectable.
