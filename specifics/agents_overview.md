# Agents Overview

This documentation describes the implemented agents in `src/simulation/agents` and the interaction mechanisms in the simulation.

All the agents are adaptive agents that implement the cycle:

```text
see -> next -> action -> do
```

- `see` builds the perception of the agent.
- `next` updates the internal state. 
- `action` selects the action. 
- `do` apply the action to the environment. 

The common contract is defined in `AdaptiveAgent`(`src/simulation/agents/structure/adaptiveagent.py`).

## Simulated agents

| Agent          | File                                      | Document                                                    |
|----------------|-------------------------------------------|-------------------------------------------------------------|
| Neuron         | `src/simulation/agents/neuron.py`         | [agent_neuron.md](agents/agent_neuron.md)                   |
| AlphaSynuclein | `src/simulation/agents/alphasynuclein.py` | [agent_alpha_synuclein.md](agents/agent_alpha_synuclein.md) |
| AlphaAggregate | `src/simulation/agents/aggregate.py`      | [agent_alpha_aggregate.md](agents/agent_alpha_aggregate.md) |
| Mitochondrion  | `src/simulation/agents/mitochondrion.py`  | [agent_mitochondrion.md](agents/agent_mitochondrion.md)     |
| Lysosome       | `src/simulation/agents/lysosome.py`       | [agent_lysosome.md](agents/agent_lysosome.md)               |
| Microglia      | `src/simulation/agents/microglia.py`      | [agent_microglia.md](agents/agent_microglia.md)             |
| Astrocyte      | `src/simulation/agents/astrocyte.py`      | [agent_astrocyte.md](agents/agent_astrocyte.md)             |

`AggregateRegistry`, defined in `src/simulation/agents/aggregate_registry.py`, is not an autonomous Repast agent, but the coordinator of the collective alpha-synuclein lifecycles.
Its logic is documented in `AlphaAggregate`.

## Compartments and types

`AgentType` defines the external types:

| Type            | Id | Compartment                     |
|-----------------|---:|---------------------------------|
| `NEURON`        | 0 | Extracellular                   |
| `MICROGLIA`     | 1 | Extracellular                   |
| `ASTROCYTE`     | 2 | Extracellular                   |
| `ALPHA`         | 3 | Extracellular and intracellular |
| `MITOCHONDRION` | 4 | intracellular                   |
| `LYSOSOME`      | 5 | intracellular                   |

## Simulation levels

The simulation works on two spatial levels:

- Extracellular level: `SubstantiaNigra` contains neurons, microglia, astrocytes and extracellular alpha-synuclein.
- Intracellular level: every `Neuron` contains a `LocalGrid` where mitochondria, lysosomes, intracellular alpha-synuclein and aggregates live.

The neuron is both an agent in the external environment and an habitat for internal agents. It's the bridge between the two levels.

## Intracellular sequence

During `Neuron.step(...)`, if the neuron is not Ruptured:

1. The neuron empties the internal effect buffer (`begin_tick`).
2. All the internal agents execute `see`.
3. All the internal agents execute `next`.
4. `AggregateRegistry.process(...)` solves oligomerization, recruiting, merge and maturation into Lewy bodies.
5. All the internal agents still active execute `action`.
6. All the internal agents still active execute `do`.
7. The neuron applies the effect buffer (`commit_effects`).
8. The neuron execute it's own cycle (`see -> next -> action -> do`)

If the neuron is `Ruptured`, the internal agents get skipped and the neuron handles only the extracellular spilling and the following idling.

## Configuration files

The runtime parameters are loaded from `src/configuration/param` through `ConfigFactory`:

| Agent                       | Parameter file         |
|-----------------------------|------------------------|
| Neuron and internal habitat | `neuron.yaml`          |
| AlphaSynuclein              | `alpha.yaml`           |
| Mitochondrion               | `mitochondrion.yaml`   |
| Lysosome                    | `lysosome.yaml`        |
| Microglia                   | `microglia.yaml`       |
| Astrocyte                   | `astrocyte.yaml`       |
| External environment        | `substantia_nigra.yaml` |

For more details on the complete parameter setting, see `specifics/parameters_guide.md`.
