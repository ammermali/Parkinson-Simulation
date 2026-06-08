# Parkinson Simulation

Agent-based framework for simulating, analysing and visualising simplified mechanisms involved in Parkinson’s disease progression.

The project provides:

* a distributed simulation built with Repast4Py and MPI;
* a command-line interface for simulation and post-processing;
* a Streamlit dashboard for configuring, running and exploring experiments;
* causal and initialization logging;
* time-series plots;
* multilevel causal graph generation;
* automated tests.

> This is an exploratory computational model and is not intended for clinical use, diagnosis or biological prediction.

---

## Requirements

### Software

* Python 3.10 or newer
* Git
* An MPI implementation:

  * Open MPI, or
  * MPICH

The project uses the following Python dependencies:

* `repast4py`
* `mpi4py`
* `PyYAML`
* `networkx`
* `matplotlib`
* `Pillow`
* `pytest`
* `streamlit`
* `multilevelgraphs`

All Python dependencies are listed in `requirements.txt`.

### Installing MPI

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```

#### macOS

Using Homebrew:

```bash
brew install open-mpi
```

#### Windows

Running Repast4Py and MPI directly on Windows may require additional configuration. Using WSL with Ubuntu is recommended.

After installing MPI, verify that it is available:

```bash
mpiexec --version
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ammermali/Parkinson-Simulation.git
cd Parkinson-Simulation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Verify the installation:

```bash
python main.py --help
```

---

## Quick Start

### Start the dashboard

From the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local address, usually:

```text
http://localhost:8501
```

Open it in a browser if it does not open automatically.

The dashboard includes pages for:

* parameter configuration;
* simulation execution;
* spatial reconstruction;
* initialization inspection;
* post-run metrics;
* G0 temporal causal graph;
* G1 time-contracted graph;
* G2 agent-state graph;
* G3 topological pattern graph;
* project documentation.

### Run a simulation from the CLI

```bash
mpiexec -n 4 python main.py simulate
```

The value after `-n` is the number of MPI processes. For example, to run with two processes:

```bash
mpiexec -n 2 python main.py simulate
```

For a minimal local run, you can use:

```bash
mpiexec -n 1 python main.py simulate
```

---

## Dashboard

Launch the dashboard with:

```bash
streamlit run app.py
```

The dashboard is the recommended interface for interactive usage. It allows users to configure parameters, start simulations and inspect generated results without invoking each analysis command manually.

The dashboard is divided into four main areas.

### Run

* **Parameters** — inspect and modify simulation parameters.
* **Simulation** — start and monitor simulation runs.

### Post-run

* **Reconstruction** — inspect spatial reconstruction data.
* **Initialization Overview** — explore the initial simulation state.
* **Post-run Metrics** — visualise metrics generated during execution.

### Graphs

* **Temporal Causal Graph** — inspect the full G0 causal graph.
* **Time-contracted Graph** — inspect the G1 graph.
* **Agent-state Graph** — inspect the G2 graph.
* **Topological Pattern Graph** — inspect the G3 graph.

### Docs

* **Specifics and Docs** — read the project documentation from the dashboard.

Stop the dashboard with:

```text
Ctrl+C
```

---

## Command-Line Interface

`main.py` is the main CLI entry point.

Display the available commands:

```bash
python main.py --help
```

Display help for a specific command:

```bash
python main.py <command> --help
```

For example:

```bash
python main.py simulate --help
python main.py graphs --help
python main.py postprocess --help
```

### Available commands

| Command         | Description                                   |
| --------------- | --------------------------------------------- |
| `simulate`      | Run the distributed simulation                |
| `validate-g0`   | Validate causal graph logs                    |
| `validate-init` | Validate initialization logs                  |
| `mechanisms`    | Count biological mechanisms in causal traces  |
| `intervention`  | Generate an intervention-oriented run summary |
| `plot`          | Generate plots from tick metrics              |
| `graphs`        | Generate multilevel causal graphs             |
| `postprocess`   | Run the standard post-simulation workflow     |

---

## Running the Simulation

The standard command is:

```bash
mpiexec -n 4 python main.py simulate
```

The simulation reads its default system configuration from:

```text
src/simulation/params/system.yaml
```

Agent- and mechanism-specific parameters are stored in:

```text
src/simulation/params/
```

The main configuration files include:

```text
system.yaml
neuron.yaml
microglia.yaml
astrocyte.yaml
alpha.yaml
mitochondrion.yaml
lysosome.yaml
substantia_nigra.yaml
```

Before running an experiment, review at least:

* final simulation tick;
* random seed;
* agent population sizes;
* grid dimensions;
* output directory;
* causal logging settings;
* initialization logging settings;
* tick metrics settings.

The simulation engine can also be started directly for debugging:

```bash
mpiexec -n 4 python src/simulation/engine.py
```

However, the recommended entry point is:

```bash
python main.py
```

---

## Running the Tests

Run the complete test suite from the repository root:

```bash
python -m pytest
```

Run the tests with verbose output:

```bash
python -m pytest -v
```

Stop after the first failure:

```bash
python -m pytest -x
```

Run a single test file:

```bash
python -m pytest tests/path_to_test.py
```

Run a specific test:

```bash
python -m pytest tests/path_to_test.py::test_name
```

Show printed output while running tests:

```bash
python -m pytest -s
```

Some tests require:

* a working MPI installation;
* `mpi4py`;
* `repast4py`;
* optional graph dependencies.

Tests that rely on unavailable optional dependencies may be skipped automatically.

---

## Post-processing

After completing a simulation, run the standard analysis workflow:

```bash
python main.py postprocess
```

Generate plots as part of post-processing:

```bash
python main.py postprocess --plots
```

Generate plots and multilevel graphs:

```bash
python main.py postprocess --plots --graphs
```

---

## Plot Generation

Plots are generated from:

```text
output/simulation/logs/tick_metrics.csv
```

Generate neuron-state plots:

```bash
python main.py plot neurons
```

Generate the free alpha-synuclein plot:

```bash
python main.py plot alpha-free
```

Generate the aggregated alpha-synuclein plot:

```bash
python main.py plot alpha-aggregate
```

Generate all alpha-synuclein plots:

```bash
python main.py plot alpha
```

Generate Substantia Nigra environmental plots:

```bash
python main.py plot sn
```

Generate all available plots:

```bash
python main.py plot all
```

Generated plots are written by default to:

```text
output/plots/
```

---

## Graph Generation

Generate the default multilevel graphs:

```bash
python main.py graphs
```

This produces the G1 and G2 graph levels.

To export G0 as well:

```bash
python main.py graphs --write-g0
```

G0 is optional because the complete temporal causal graph can become very large.

The main graph outputs are written to:

```text
output/analysis/graphs/
```

Typical generated files include:

```text
g1.gexf
g2.gexf
multilevel_report.md
```

GEXF files can be opened directly with Gephi.

To inspect all graph options:

```bash
python main.py graphs --help
```

---

## Validation and Analysis

Validate causal G0 logs:

```bash
python main.py validate-g0
```

Validate initialization logs:

```bash
python main.py validate-init
```

Count biological mechanisms found in causal traces:

```bash
python main.py mechanisms
```

Generate an intervention-oriented summary:

```bash
python main.py intervention
```

Analysis outputs are written by default to:

```text
output/analysis/
```

---

## Output Structure

Simulation and analysis results are stored under `output/`.

```text
output/
├── simulation/
│   └── logs/
├── analysis/
│   └── graphs/
└── plots/
```

### Simulation logs

The default simulation log directory is:

```text
output/simulation/logs/
```

Depending on the enabled configuration, it may contain:

```text
tick_metrics.csv
g0_nodes.jsonl
g0_edges.jsonl
```

In distributed runs, rank-local log files may be generated and merged at the end of the simulation.

### Tick metrics

`tick_metrics.csv` contains global values recorded during the simulation, including:

* debris;
* inflammation;
* dopamine;
* healthy neurons;
* compromised neurons;
* apoptotic neurons;
* neuronal ruptures;
* free alpha-synuclein;
* alpha-synuclein contained in aggregates.

---

## Project Structure

```text
Parkinson-Simulation/
├── app.py                  # Streamlit dashboard entry point
├── main.py                 # Main command-line interface
├── requirements.txt        # Python dependencies
├── dashboard/              # Streamlit pages and dashboard utilities
├── src/
│   ├── simulation/         # Simulation engine and biological agents
│   ├── analysis/           # Validation, metrics and graph generation
│   └── visualization/      # Plot generation
├── specifics/              # Technical specifications and documentation
├── tests/                  # Automated tests
└── output/                 # Generated runtime and analysis outputs
```

---

## Troubleshooting

### `mpiexec: command not found`

MPI is not installed or is not available in `PATH`.

On Ubuntu or Debian:

```bash
sudo apt install openmpi-bin libopenmpi-dev
```

On macOS:

```bash
brew install open-mpi
```

### `ModuleNotFoundError`

Make sure the virtual environment is active and reinstall the dependencies:

```bash
pip install -r requirements.txt
```

### `mpi4py` installation fails

Install MPI before installing the Python requirements, then retry:

```bash
pip install --force-reinstall mpi4py
```

### Streamlit does not start

Verify that Streamlit is installed:

```bash
python -m streamlit --version
```

Then start the dashboard through Python:

```bash
python -m streamlit run app.py
```

### Port 8501 is already in use

Start Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

### Graph export uses too much memory

Avoid exporting G0 unless it is required:

```bash
python main.py graphs
```

instead of:

```bash
python main.py graphs --write-g0
```

---

## Documentation

Additional technical information is available under:

```text
specifics/
```

In particular:

```text
specifics/parameters_guide.md
specifics/multilevel.md
```

The same documentation can also be accessed from the **Docs** section of the Streamlit dashboard.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.

---

## Academic Context

This framework was developed as part of the Distributed Calculus and Coordination and Multi-Agent Systems Lab courses held by Prof. Emanuela Merelli at the University of Camerino.
