"""Repast4Py entrypoint for the Parkinson simulation.

This module does not contain biological transition logic.
Its job is to wire together the simulation runtime:
1. load parameters from YAML;
2. create the shared extracellular grid;
3. wrap that grid as a SubstantiaNigra environment;
4. instantiate extracellular agents;
5. populate each neuron with its intracellular agents;
6. schedule and execute one model tick at a time.

The actual agent behavior remains inside the agent classes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _ensure_project_root_on_path() -> None:
    """Allow direct execution from src/simulation with python engine.py."""
    project_root = Path(__file__).resolve().parents[2]
    project_root_path = str(project_root)
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)


if __package__ in (None, ""):
    _ensure_project_root_on_path()

from mpi4py import MPI
from repast4py import context as ctx
from repast4py import random
from repast4py import schedule
from repast4py import space
from repast4py.space import DiscretePoint
from src.simulation.agents.alphasynuclein import AlphaSynuclein, AlphaSynucleinCompartment
from src.simulation.agents.astrocyte import Astrocyte
from src.simulation.agents.lysosome import Lysosome
from src.simulation.agents.microglia import Microglia
from src.simulation.agents.mitochondrion import Mitochondrion
from src.simulation.agents.neuron import Neuron, NeuronState
from src.simulation.substantia_nigra import SubstantiaNigra
from src.simulation.utils import Params, RNG
from src.simulation.utils.config_factory import ConfigFactory
from src.simulation.logger import CausalTraceLogger, InitializationLogger


@dataclass(frozen=True)
class AgentType:
    """Numeric type ids used by repast4py.core.Agent."""
    NEURON: int = 0
    MICROGLIA: int = 1
    ASTROCYTE: int = 2
    ALPHA: int = 3
    MITOCHONDRION: int = 4
    LYSOSOME: int = 5


class ParkinsonModel:
    """Top-level simulation model executed by Repast4Py.

    A model instance exists per MPI rank. Each instance owns:
    - a rank-local SharedContext containing local agents;
    - a rank-local view of the distributed SharedGrid;
    - one SubstantiaNigra wrapper around that grid;
    - a Repast schedule runner.

    The model exposes itself to agents during .step() so they can access the
    environment, the shared grid wrapper and other model-level services without
    importing the engine directly."""

    def __init__(self, comm: MPI.Intracomm, params: dict[str, Any]):
        """Create the simulation runtime from already-loaded system params."""
        self.comm = comm
        self.rank = comm.Get_rank()
        self.params = params
        self.agent_type = AgentType()
        self.seed = int(_param(params, "random.seed", 42))
        self._init_repast_rng(self.seed)
        # This RNG is used only while building config dataclasses.
        # It is shared across config factory calls so thresholds are sampled once
        # and differ from one created agent to the next.
        self.config_rng = RNG(self.seed + self.rank)
        # Repast's global RNG is the object expected by SharedGrid helpers such
        # as get_random_local_pt(). It's separate from ConfigFactory's RNG
        # because it belongs to the Repast runtime rather than our YAML layer.
        self.rng = random.default_rng
        # Neuron-specific YAML is loaded once because it is used both for the
        # neuron config dataclasses and for initial intracellular population.
        self.neuron_params = Params("neuron")
        self.neuron_param_values = self.neuron_params.as_dict()
        self.tick_count = 0
        # SharedContext owns the active agents for this rank.
        self.context = ctx.SharedContext(comm)
        # Schedule runner calls self.step() repeatedly and stops at stop.at.
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(_param(params, "stop.at", 100))
        # Create the distributed extracellular grid and make it a projection of
        # the context, which lets Repast keep agent locations synchronized.
        self.grid = self._create_grid(params)
        self.context.add_projection(self.grid)
        # SubstantiaNigra is the biological wrapper around the Repast grid.
        self.environment = SubstantiaNigra(grid=self.grid, config=ConfigFactory.build_substantia_nigra_config())
        # Runtime causal traces and initialization logs are intentionally
        # separate: G0 edges stay compact, while initial conditions stay rich.
        self.causal_logger, self.initialization_logger = self._create_loggers(params)
        # Local ids only need to be unique per rank and type id.
        self._next_local_id = 0
        self._create_agents(params)
        self.initialization_logger.close()

    # Initialization

    def _create_grid(self, params: dict[str, Any]) -> space.SharedGrid:
        """Create the shared extracellular grid.
        world.width and world.height are the biological space size.
        world.buffer_size is the number of neighboring cells mirrored from adjacent ranks.
        It is not a biological buffer and it is not related to neuron / lysosome target buffers."""
        width = int(_param(params, "world.width", 50))
        height = int(_param(params, "world.height", 50))
        buffer_size = int(_param(params, "world.buffer_size", 2))
        bounds = space.BoundingBox(0, width, 0, height, 0, 0)
        return space.SharedGrid(
            name="substantia_nigra_grid",
            bounds=bounds,
            borders=space.BorderType.Sticky,
            occupancy=space.OccupancyType.Multiple, # Multiple agents can occupy the same cell.
            buffer_size=buffer_size,
            comm=self.comm
        )

    def _create_agents(self, params: dict[str, Any]) -> None:
        """Create every extracellular agent family for this rank."""
        self._create_neurons(params)
        self._create_microglia(params)
        self._create_astrocytes(params)
        self._create_extracellular_alpha(params)

    def _create_neurons(self, params: dict[str, Any]) -> None:
        """Create neuron macro-agents and populate their internal habitats."""
        count = int(_param(params, "external.population.neurons", 10))
        for _ in range(count):
            agent = Neuron(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.NEURON,
                config=ConfigFactory.build_neuron_config(self.neuron_params, rng=self.config_rng),
                alpha_type_id=self.agent_type.ALPHA, # Type that gets recognized from the neuron as ALPHA (pathologic one).
                internal_config=ConfigFactory.build_neuron_internal_config(self.neuron_params)
            )
            self._add_agent_randomly(agent)
            self._populate_neuron(agent, params)

    def _create_microglia(self, params: dict[str, Any]) -> None:
        """Create extracellular microglia agents."""
        count = int(_param(params, "external.population.microglia", 5))
        for _ in range(count):
            agent = Microglia(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.MICROGLIA,
                config=ConfigFactory.build_microglia_config(rng=self.config_rng),
                alpha_type_id=self.agent_type.ALPHA
            )
            self._add_agent_randomly(agent)

    def _create_astrocytes(self, params: dict[str, Any]) -> None:
        """Create extracellular astrocyte agents."""
        count = int(_param(params, "external.population.astrocytes", 5))
        for _ in range(count):
            agent = Astrocyte(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.ASTROCYTE,
                config=ConfigFactory.build_astrocyte_config(rng=self.config_rng),
            )
            self._add_agent_randomly(agent)

    def _create_extracellular_alpha(self, params: dict[str, Any]) -> None:
        """Seed extracellular alpha-synuclein directly in Substantia Nigra.
        Alpha-synuclein agents should start inside neurons, but this hook is useful for
        experiments that begin with extracellular pathology already present.
        Extracellular alpha is frozen by the AlphaSynuclein class itself.
        """
        count = int(_param(params, "external.population.alpha", 0))
        for _ in range(count):
            agent = AlphaSynuclein(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.ALPHA,
                config=ConfigFactory.build_alpha_synuclein_config(rng=self.config_rng),
                compartment=AlphaSynucleinCompartment.EXTRACELLULAR,
                owner_neuron=None
            )
            self._add_agent_randomly(agent)

    def _populate_neuron(self, neuron: Neuron, params: dict[str, Any]) -> None:
        """Fill one neuron's internal grid with configured intracellular agents.
        The preferred source is now neuron.yaml:
            intracellular.population.alpha
            intracellular.population.mitochondria
            intracellular.population.lysosomes
        """
        alpha_count = self._intracellular_count(params, "alpha", "alpha")
        mitochondria_count = self._intracellular_count(params,"mitochondria","mitochondria")
        lysosome_count = self._intracellular_count(params,"lysosomes","lysosomes")
        for index in range(alpha_count):
            alpha = AlphaSynuclein(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.ALPHA,
                config=ConfigFactory.build_alpha_synuclein_config(rng=self.config_rng),
                compartment=AlphaSynucleinCompartment.INTRACELLULAR,
                owner_neuron=neuron
            )
            self._add_internal_agent(neuron, alpha, self._internal_point(neuron, index))
        for index in range(mitochondria_count):
            mitochondrion = Mitochondrion(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.MITOCHONDRION,
                config=ConfigFactory.build_mitochondrion_config(
                    rng=self.config_rng),
                owner_neuron=neuron
            )
            self._add_internal_agent(neuron, mitochondrion, self._internal_point(neuron, alpha_count + index))

        for index in range(lysosome_count):
            lysosome = Lysosome(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.LYSOSOME,
                owner_neuron=neuron,
                config=ConfigFactory.build_lysosome_config()
            )
            self._add_internal_agent(neuron, lysosome, self._internal_point(neuron, alpha_count + mitochondria_count + index))

    # Simulation loop

    def step(self) -> None:
        """Advance the whole simulation by one tick.
        Order matters:
        1. reset extracellular effect buffers;
        2. let each extracellular macro-agent run one step;
        3. commit extracellular scalar effects;
        4. optionally synchronize distributed agents;
        5. emit a runtime log. #TODO
        Neurons run their own intracellular phase inside Neuron.step()
        That keeps internal alpha, aggregate, mitochondrion and lysosome logic
        synchronized before the neuron itself acts on the extracellular space."""
        self.tick_count = getattr(self, "tick_count", 0) + 1
        causal_logger = getattr(self, "causal_logger", None)
        if causal_logger is not None:
            causal_logger.set_tick(self.tick_count)
        self.environment.begin_tick()
        # Use a list snapshot because agents can release or absorb alpha during
        # a step. Iterating over the live context while it mutates would be
        # brittle and can skip agents.
        for agent in list(self.context.agents()):
            if hasattr(agent, "step"):
                agent.step(self)
        self.environment.commit_effects(max_possible_dopamine=self._max_possible_dopamine())
        # Future distributed runs that move agents across MPI rank boundaries
        # should synchronize here with a restore_agent function.
        # self.context.synchronize(restore_agent) # TODO ?
        self._record_scalar_tick()
        self._log_tick()

    def start(self) -> None:
        """Start the Repast schedule runner."""
        try:
            self.runner.execute()
        finally:
            causal_logger = getattr(self, "causal_logger", None)
            if causal_logger is not None:
                causal_logger.close()

    # Helpers

    def _new_id(self) -> int:
        """Return the next rank-local id for a newly created agent."""
        local_id = self._next_local_id
        self._next_local_id += 1
        return local_id

    def _init_repast_rng(self, seed: int) -> None:
        """Seed Repast4Py's global RNG when the installed version supports it."""
        if hasattr(random, "init"):
            random.init(seed)

    def _add_agent_randomly(self, agent) -> None:
        """Add an extracellular agent to both Repast context and SN grid."""
        point = self.grid.get_random_local_pt(self.rng)
        self.context.add(agent)
        # Keep the agent's convenience pt field aligned with the Repast grid.
        # The biological grid API still queries the grid as source of truth.
        agent.pt = point
        self.environment.add_agent(agent, point)
        self.initialization_logger.record_agent(agent, position=point, raw_details={"habitat": "SubstantiaNigra"})

    def _add_internal_agent(self, neuron: Neuron, agent, point: DiscretePoint) -> None:
        """Add an intracellular agent and log its initial placement."""

        neuron.add_agent(agent, point)
        self.initialization_logger.record_agent(agent, position=point, owner=neuron, target=neuron, raw_details={"habitat": "Neuron"})

    def _internal_point(self, neuron: Neuron, index: int) -> DiscretePoint:
        """Return a deterministic initial point inside a neuron's local grid.
        Agents are placed row by row and wrap around when there are more agents
        than cells. Multiple occupancy is acceptable inside the current local
        grid model.
        """
        width = max(1, neuron.internal_cfg.width)
        height = max(1, neuron.internal_cfg.height)
        cell = index % (width * height)
        return DiscretePoint(cell % width, cell // width)

    def _intracellular_count(self, system_params: dict[str, Any], neuron_key: str, legacy_key: str) -> int:
        """Read initial intracellular population from neuron.yaml first."""
        return int(_param(self.neuron_param_values, f"intracellular.population.{neuron_key}", _param(system_params, f"intracellular.population.{legacy_key}", 0)))

    def _max_possible_dopamine(self) -> float:
        """Return dopamine capacity from currently viable neurons.
        SubstantiaNigra.commit_effects() normalizes released dopamine by a maximum
        possible amount. Ruptured and apoptotic neurons are excluded because they
        they should no longer contribute normal dopamine output."""
        total = 0.0
        for agent in self.context.agents():
            if (isinstance(agent, Neuron) and agent.state not in (NeuronState.APOPTOTIC, NeuronState.RUPTURED)):
                total += agent.cfg.dopamine_release_rate
        return total

    def _create_loggers(self, params: dict[str, Any]) -> tuple[CausalTraceLogger, InitializationLogger]:
        """Create separated causal and initialization loggers."""

        output_dir = self._resolve_output_dir(_param(params, "logging.output_dir", "src/simulation/output/logs"))
        run_id = str(_param(params, "logging.run_id", f"run_seed_{self.seed}"))
        agent_type_map = {
            self.agent_type.NEURON: "Neuron",
            self.agent_type.MICROGLIA: "Microglia",
            self.agent_type.ASTROCYTE: "Astrocyte",
            self.agent_type.ALPHA: "AlphaSynuclein",
            self.agent_type.MITOCHONDRION: "Mitochondrion",
            self.agent_type.LYSOSOME: "Lysosome"
        }
        causal_logger = CausalTraceLogger(
            run_id=run_id,
            comm=self.comm,
            rank=self.rank,
            output_dir=output_dir,
            enabled=bool(_param(params, "logging.causal.enabled", _param(params, "logging.enabled", False))),
            agent_type_map=agent_type_map,
            params=params
        )
        initialization_logger = InitializationLogger(
            run_id=run_id,
            comm=self.comm,
            rank=self.rank,
            output_dir=output_dir,
            enabled=bool(_param(params, "logging.initialization.enabled", _param(params, "logging.enabled", False)))
        )
        return causal_logger, initialization_logger

    def _resolve_output_dir(self, value: str) -> Path:
        """Resolve relative output paths from the project root."""
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    def _record_scalar_tick(self) -> None:
        """Record one per-tick scalar snapshot in the structured event log."""
        logger = getattr(self, "causal_logger", None)
        if logger is None:
            return
        scalars = self.environment.scalars
        effects = self.environment.effects
        logger.snapshot_field("extracellular_debris", scalars.extracellular_debris)
        logger.snapshot_field("inflammation_level", scalars.inflammation_level)
        logger.snapshot_field("dopamine_output", scalars.dopamine_output)
        logger.buffer_commit("debris_added", "extracellular_debris", effects.debris_added, "positive", "sn_debris_added_commit")
        logger.buffer_commit("debris_removed", "extracellular_debris", effects.debris_removed, "negative", "sn_debris_removed_commit")
        logger.buffer_commit("inflammation_added", "inflammation_level", effects.inflammation_added, "positive", "sn_inflammation_added_commit")
        logger.buffer_commit("inflammation_removed", "inflammation_level", effects.inflammation_removed, "negative", "sn_inflammation_removed_commit")
        logger.buffer_commit("dopamine_released", "dopamine_output", effects.dopamine_released, "positive", "sn_dopamine_release_commit")

    def _log_tick(self) -> None:
        """Print a compact scalar log from rank 0 only."""
        if not bool(_param(getattr(self, "params", {}), "logging.scalar_stdout", True)):
            return
        if self.rank != 0:
            return
        scalars = self.environment.scalars
        print(
            f"debris={scalars.extracellular_debris:.3f}, "
            f"inflammation={scalars.inflammation_level:.3f}, "
            f"dopamine={scalars.dopamine_output:.3f}"
        )

def run(params: Optional[dict[str, Any]] = None) -> None:
    """Create and run a ParkinsonModel.
    When no params are passed, system.yaml is loaded through Params.
    This keeps command-line execution simple while still allowing tests or
    notebooks to inject a prebuilt parameter dictionary.
    """

    if params is None:
        params = Params("system").as_dict()
    model = ParkinsonModel(MPI.COMM_WORLD, params)
    model.start()


def _param(params: dict[str, Any], key: str, default=None):
    """Read a parameter by dotted path.
    The dotted-section support is useful because some current YAML files group
    related values under keys such as external.population.
    """
    if key in params:
        return params[key]
    parts = key.split(".")
    for split_at in range(len(parts), 0, -1):
        prefix = ".".join(parts[:split_at])
        if prefix not in params:
            continue
        value = params[prefix]
        for part in parts[split_at:]:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value
    value = params
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value

if __name__ == "__main__":
    run()
