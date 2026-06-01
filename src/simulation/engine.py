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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _ensure_project_root_on_path() -> None:
    """Allow direct execution with python src/simulation/engine.py."""
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
from src.simulation.agents.aggregate import AlphaAggregate, AggregateState
from src.simulation.agents.alphasynuclein import AlphaSynuclein, AlphaSynucleinCompartment, AlphaSynucleinState
from src.simulation.agents.astrocyte import Astrocyte
from src.simulation.agents.lysosome import Lysosome
from src.simulation.agents.microglia import Microglia
from src.simulation.agents.mitochondrion import Mitochondrion
from src.simulation.agents.neuron import Neuron, NeuronState
from src.simulation.substantia_nigra import SubstantiaNigra
from src.simulation.utils import Params, RNG
from src.simulation.utils.config_factory import ConfigFactory
from src.simulation.logger import CausalTraceLogger, InitializationLogger


class SimulationCompleted(Exception):
    """Internal control signal used to leave Repast's runner at stop.at."""


@dataclass(frozen=True)
class AgentType:
    """Numeric type ids used by repast4py.core.Agent."""
    NEURON: int = 0
    MICROGLIA: int = 1
    ASTROCYTE: int = 2
    ALPHA: int = 3
    MITOCHONDRION: int = 4
    LYSOSOME: int = 5


NEURON_METRIC_STATES = ("Healthy", "Compromised", "Apoptotic", "Ruptured", "Unknown")
ASTROCYTE_METRIC_STATES = ("Supportive", "Reactive", "Unknown")
MICROGLIA_METRIC_STATES = ("Resting", "Clearing", "Activated", "Unknown")
ALPHA_METRIC_STATES = ("Monomer", "Misfolded", "Oligomer", "LewyBody", "Cleared", "Unknown")
AGGREGATE_METRIC_STATES = ("Oligomer", "LewyBody", "Unknown")

FINAL_SUM_METRIC_KEYS = (
    *(f"neurons.{state}" for state in NEURON_METRIC_STATES),
    "neurons.transitions.healthy_to_compromised.count",
    "neurons.transitions.compromised_to_apoptotic.count",
    "neurons.transitions.apoptotic_to_ruptured.count",
    "neurons.state_time.compromised_ticks_total",
    "neurons.state_time.apoptotic_ticks_total",
    "neurons.state_time.compromised_neuron_count",
    "neurons.state_time.apoptotic_neuron_count",
    "neurons.recoveries.compromised_to_healthy",
    "neurons.blocks.min_ticks_compromised",
    "neurons.blocks.apoptotic_internal_damage_threshold",
    "neurons.ever_compromised",
    "neurons.ever_apoptotic",
    "neurons.ever_recovered",
    *(f"astrocytes.{state}" for state in ASTROCYTE_METRIC_STATES),
    *(f"microglia.{state}" for state in MICROGLIA_METRIC_STATES),
    *(f"alpha.free.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.intracellular.free.total",
    *(f"alpha.intracellular.free.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.extracellular.free.total",
    *(f"alpha.extracellular.free.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.members",
    "alpha.intracellular.members",
    "alpha.extracellular.members",
    *(f"alpha.members.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.orphan_lewy",
    "aggregates.total",
    *(f"aggregates.{state}" for state in AGGREGATE_METRIC_STATES),
    "aggregates.size_total",
    "aggregates.intracellular.total",
    *(f"aggregates.intracellular.{state}" for state in AGGREGATE_METRIC_STATES),
    "aggregates.intracellular.size_total",
    "aggregates.extracellular.total",
    *(f"aggregates.extracellular.{state}" for state in AGGREGATE_METRIC_STATES),
    "aggregates.extracellular.size_total",
    "aggregates.invariant_failures",
    "aggregates.intracellular.invariant_failures",
    "aggregates.extracellular.invariant_failures",
)

FINAL_MAX_METRIC_KEYS = (
    "aggregates.max_size",
    "aggregates.intracellular.max_size",
    "aggregates.extracellular.max_size",
)

TICK_METRIC_COUNT_KEYS = (
    "neurons_healthy",
    "neurons_compromised",
    "neurons_apoptotic",
    "neurons_ruptures",
    "free_alpha",
    "alpha_aggregate",
)

TICK_METRIC_COLUMNS = (
    "tick",
    "debris",
    "inflammation",
    "dopamine",
    *TICK_METRIC_COUNT_KEYS,
)

DEFAULT_SIMULATION_OUTPUT_DIR = "output/simulation/logs"


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
        self.stop_at = int(_param(params, "stop.at", 100))
        self.progress_interval = max(
            1,
            int(_param(params, "logging.progress_interval", 25))
        )
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
        self._finalized = False
        # SharedContext owns the active agents for this rank.
        self.context = ctx.SharedContext(comm)
        # Schedule runner calls self.step() repeatedly and stops at stop.at.
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(self.stop_at)
        # Create the distributed extracellular grid and make it a projection of
        # the context, which lets Repast keep agent locations synchronized.
        self.grid = self._create_grid(params)
        self.context.add_projection(self.grid)
        # SubstantiaNigra is the biological wrapper around the Repast grid.
        self.environment = SubstantiaNigra(grid=self.grid, config=ConfigFactory.build_substantia_nigra_config())
        # Runtime causal traces and initialization logs are intentionally
        # separate: G0 edges stay compact, while initial conditions stay rich.
        self.causal_logger, self.initialization_logger = self._create_loggers(params)
        self._create_tick_metrics_csv(params)
        # Local ids only need to be unique per rank and type id.
        self._next_local_id = 0
        self._create_agents(params)
        # Dopamine is normalized against the global initial dopaminergic
        # capacity, not the currently surviving capacity. This lets neuron loss
        # appear as reduced output instead of being hidden by renormalization.
        self.initial_dopamine_capacity = self._global_sum(
            self._local_dopamine_capacity()
        )
        self.initialization_logger.close()
        self._log_startup()

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
        count = self._local_population_count(
            int(_param(params, "external.population.neurons", 10))
        )
        for _ in range(count):
            agent = Neuron(
                local_id=self._new_id(),
                rank=self.rank,
                type_id=self.agent_type.NEURON,
                config=ConfigFactory.build_neuron_config(self.neuron_params, rng=self.config_rng),
                alpha_type_id=self.agent_type.ALPHA, # Type that gets recognized from the neuron as ALPHA (pathologic one).
                internal_config=ConfigFactory.build_neuron_internal_config(self.neuron_params),
                environment=self.environment
            )
            self._add_agent_randomly(agent)
            self._populate_neuron(agent, params)

    def _create_microglia(self, params: dict[str, Any]) -> None:
        """Create extracellular microglia agents."""
        count = self._local_population_count(
            int(_param(params, "external.population.microglia", 5))
        )
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
        count = self._local_population_count(
            int(_param(params, "external.population.astrocytes", 5))
        )
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
        count = self._local_population_count(
            int(_param(params, "external.population.alpha", 0))
        )
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
        5. emit a runtime log.
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
        self._synchronize_environment_effects()
        self.environment.commit_effects(max_possible_dopamine=self._max_possible_dopamine())
        # Future distributed runs that move agents across MPI rank boundaries
        # should synchronize here with a restore_agent function.
        # self.context.synchronize(restore_agent)
        self._record_scalar_tick()
        self._record_tick_metrics_csv()
        self._log_tick()
        self._log_progress_tick()
        self._stop_if_complete()

    def start(self) -> None:
        """Start the Repast schedule runner."""
        try:
            self.runner.execute()
        except SimulationCompleted:
            pass
        finally:
            self._finalize_run()

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
        self._record_initial_g0_alpha(agent)

    def _add_internal_agent(self, neuron: Neuron, agent, point: DiscretePoint) -> None:
        """Add an intracellular agent and log its initial placement."""

        neuron.add_agent(agent, point)
        self.initialization_logger.record_agent(agent, position=point, owner=neuron, target=neuron, raw_details={"habitat": "Neuron"})
        self._record_initial_g0_alpha(agent, owner=neuron)

    def _record_initial_g0_alpha(self, agent, owner: Optional[Neuron] = None) -> None:
        """Add baseline alpha-synuclein nodes to G0 at tick zero."""
        if not isinstance(agent, AlphaSynuclein):
            return
        logger = getattr(self, "causal_logger", None)
        if logger is None:
            return
        logger.agent_state_node(
            agent,
            getattr(agent, "state", None),
            "0_pre_state",
            owner=owner,
            compartment=getattr(agent, "compartment", None)
        )

    def _internal_point(self, neuron: Neuron, index: int) -> DiscretePoint:
        """Return a deterministic initial point inside a neuron's local grid."""
        width = max(1, neuron.internal_cfg.width)
        height = max(1, neuron.internal_cfg.height)
        cell = index % (width * height)
        return DiscretePoint(cell % width, cell // width)

    def _intracellular_count(self, system_params: dict[str, Any], neuron_key: str, legacy_key: str) -> int:
        """Read initial intracellular population from neuron.yaml first."""
        return int(_param(self.neuron_param_values, f"intracellular.population.{neuron_key}", _param(system_params, f"intracellular.population.{legacy_key}", 0)))

    def _local_population_count(self, global_count: int) -> int:
        """Return this rank's share of a global agent population.
        YAML population values describe the whole biological system. In MPI
        runs, each rank creates only its deterministic slice so ``-n 4`` still
        creates one large simulation rather than four copies of a smaller one.
        Remainders are assigned to the lowest ranks for stable reproducibility.
        """
        global_count = max(0, int(global_count))
        size = max(1, self._comm_size())
        base_count, remainder = divmod(global_count, size)
        return base_count + (1 if self.rank < remainder else 0)

    def _comm_size(self) -> int:
        """Return MPI world size, falling back to serial execution."""
        get_size = getattr(getattr(self, "comm", None), "Get_size", None)
        if callable(get_size):
            return int(get_size())
        return 1

    def _global_sum(self, value: float) -> float:
        """Sum a scalar across MPI ranks when the communicator supports it."""
        allreduce = getattr(getattr(self, "comm", None), "allreduce", None)
        if not callable(allreduce):
            return value
        try:
            return allreduce(value, op=getattr(MPI, "SUM", None))
        except TypeError:
            return allreduce(value)

    def _global_max(self, value: float) -> float:
        """Return a maximum across MPI ranks when supported."""
        allreduce = getattr(getattr(self, "comm", None), "allreduce", None)
        if not callable(allreduce):
            return value
        try:
            return allreduce(value, op=getattr(MPI, "MAX", None))
        except TypeError:
            return allreduce(value)

    def _stop_if_complete(self) -> None:
        """Finalize and exit the schedule when the configured tick is reached."""
        stop_at = getattr(self, "stop_at", None)
        if stop_at is None:
            return
        if getattr(self, "tick_count", 0) < stop_at:
            return
        self._request_runner_stop()
        self._finalize_run()
        raise SimulationCompleted()

    def _request_runner_stop(self) -> None:
        """Ask the installed schedule runner to stop when it exposes an API."""
        runner = getattr(self, "runner", None)
        if runner is None:
            return
        stop = getattr(runner, "stop", None)
        if callable(stop):
            stop()

    def _finalize_run(self) -> None:
        """Print final summaries and close loggers exactly once."""
        if getattr(self, "_finalized", False):
            return
        self._finalized = True
        self._log_completion()
        tick_metrics_file = getattr(self, "_tick_metrics_file", None)
        if tick_metrics_file is not None:
            tick_metrics_file.close()
            self._tick_metrics_file = None
        causal_logger = getattr(self, "causal_logger", None)
        if causal_logger is not None:
            causal_logger.close()

    def _synchronize_environment_effects(self) -> None:
        """Make extracellular effect buffers global before scalar commit."""
        effects = getattr(self.environment, "effects", None)
        if effects is None:
            return
        for field in (
            "debris_added",
            "debris_removed",
            "inflammation_added",
            "inflammation_removed",
            "dopamine_released"
        ):
            if hasattr(effects, field):
                setattr(effects, field, self._global_sum(getattr(effects, field)))

    def _local_dopamine_capacity(self) -> float:
        """Return this rank's initial dopamine capacity contribution."""
        total = 0.0
        for agent in self.context.agents():
            if isinstance(agent, Neuron):
                total += agent.cfg.dopamine_release_rate
        return total

    def _max_possible_dopamine(self) -> float:
        """Return the global baseline used to normalize dopamine output."""
        capacity = getattr(self, "initial_dopamine_capacity", None)
        if capacity is not None:
            return capacity
        return self._global_sum(self._local_dopamine_capacity())

    def _create_loggers(self, params: dict[str, Any]) -> tuple[CausalTraceLogger, InitializationLogger]:
        """Create separated causal and initialization loggers."""
        output_dir = self._resolve_output_dir(_param(params, "logging.output_dir", DEFAULT_SIMULATION_OUTPUT_DIR))
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

    def _create_tick_metrics_csv(self, params: dict[str, Any]) -> None:
        """Open the global per-tick CSV writer on rank 0.

        Counts are reduced across ranks before writing, so this file describes
        the whole distributed simulation even when MPI partitions the agent
        population.
        """

        default_enabled = bool(_param(params, "logging.enabled", True))
        self.tick_metrics_enabled = bool(_param(params, "logging.tick_metrics_csv", default_enabled))
        self._tick_metrics_file = None
        if not self.tick_metrics_enabled or self.rank != 0:
            return
        output_dir = self._resolve_output_dir(_param(params, "logging.output_dir", DEFAULT_SIMULATION_OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tick_metrics.csv"
        self._tick_metrics_file = path.open("w", encoding="utf-8", newline="")
        self._tick_metrics_file.write(",".join(TICK_METRIC_COLUMNS) + "\n")

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
        effects = getattr(
            self.environment,
            "last_committed_effects",
            self.environment.effects
        )
        logger.snapshot_field("extracellular_debris", scalars.extracellular_debris)
        logger.snapshot_field("inflammation_level", scalars.inflammation_level)
        logger.snapshot_field("dopamine_output", scalars.dopamine_output)
        logger.buffer_commit("debris_added", "extracellular_debris", effects.debris_added, "positive", "sn_debris_added_commit")
        logger.buffer_commit("debris_removed", "extracellular_debris", effects.debris_removed, "negative", "sn_debris_removed_commit")
        logger.buffer_commit("inflammation_added", "inflammation_level", effects.inflammation_added, "positive", "sn_inflammation_added_commit")
        logger.buffer_commit("inflammation_removed", "inflammation_level", effects.inflammation_removed, "negative", "sn_inflammation_removed_commit")
        logger.buffer_commit("dopamine_released", "dopamine_output", effects.dopamine_released, "positive", "sn_dopamine_release_commit")

    def _record_tick_metrics_csv(self) -> None:
        """Append one global per-tick metric row for later time-series analysis."""
        if not getattr(self, "tick_metrics_enabled", False):
            return
        local_counts = self._local_tick_metric_counts()
        global_counts = {
            key: int(self._global_sum(local_counts.get(key, 0)))
            for key in TICK_METRIC_COUNT_KEYS
        }
        if self.rank != 0:
            return
        tick_metrics_file = getattr(self, "_tick_metrics_file", None)
        if tick_metrics_file is None:
            return
        scalars = self.environment.scalars
        row = [
            str(getattr(self, "tick_count", 0)),
            f"{scalars.extracellular_debris:.6f}",
            f"{scalars.inflammation_level:.6f}",
            f"{scalars.dopamine_output:.6f}",
            *(str(global_counts[key]) for key in TICK_METRIC_COUNT_KEYS),
        ]
        tick_metrics_file.write(",".join(row) + "\n")
        tick_metrics_file.flush()

    def _local_tick_metric_counts(self) -> dict[str, int]:
        """Count local agent states used by the per-tick CSV."""

        counts = {key: 0 for key in TICK_METRIC_COUNT_KEYS}
        for agent in self.context.agents():
            if isinstance(agent, Neuron):
                state = _state_value(getattr(agent, "state", None))
                if state == NeuronState.HEALTHY.value:
                    counts["neurons_healthy"] += 1
                elif state == NeuronState.COMPROMISED.value:
                    counts["neurons_compromised"] += 1
                elif state == NeuronState.APOPTOTIC.value:
                    counts["neurons_apoptotic"] += 1
                elif state == NeuronState.RUPTURED.value:
                    counts["neurons_ruptures"] += 1
                self._count_tick_alpha_agents(getattr(agent, "grid", None), counts)
        self._count_tick_alpha_agents(getattr(self.environment, "grid", None), counts)
        return counts

    def _count_tick_alpha_agents(self, grid, counts: dict[str, int]) -> None:
        """Count alpha proteins and aggregate agents from one grid registry."""

        for agent in getattr(grid, "agent_registry", []):
            if isinstance(agent, AlphaSynuclein) and agent.aggregate_id is None:
                if getattr(agent, "state", None) != AlphaSynucleinState.CLEARED:
                    counts["free_alpha"] += 1
            elif isinstance(agent, AlphaAggregate):
                counts["alpha_aggregate"] += 1

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

    def _log_startup(self) -> None:
        """Print one simulation startup line from rank 0."""
        if not self._progress_stdout_enabled():
            return
        local_agents = len(list(self.context.agents()))
        total_agents = int(self._global_sum(local_agents))
        print(
            f"[progress] starting run: ranks={self._comm_size()}, "
            f"ticks={self.stop_at}, agents={total_agents}, seed={self.seed}",
            flush=True
        )

    def _log_progress_tick(self) -> None:
        """Print periodic progress and scalar status from rank 0."""
        if not self._progress_stdout_enabled():
            return
        tick = getattr(self, "tick_count", 0)
        if tick <= 0:
            return
        if tick % self.progress_interval != 0 and tick < self.stop_at:
            return
        scalars = self.environment.scalars
        progress = min(100.0, 100.0 * tick / max(1, self.stop_at))
        print(
            f"[progress] tick {tick}/{self.stop_at} ({progress:.1f}%) | "
            f"debris={scalars.extracellular_debris:.3f}, "
            f"inflammation={scalars.inflammation_level:.3f}, "
            f"dopamine={scalars.dopamine_output:.3f}",
            flush=True
        )

    def _log_completion(self) -> None:
        """Print final status after all ranks have joined summary collectives."""
        summary_enabled = bool(_param(getattr(self, "params", {}), "logging.summary_stdout", True))
        metrics = self._final_metrics() if summary_enabled else None
        transition_details = self._final_neuron_transition_details() if summary_enabled else []
        if self.rank != 0:
            return
        scalars = self.environment.scalars
        if bool(_param(getattr(self, "params", {}), "logging.progress_stdout", True)):
            print(
                f"[progress] completed at tick {getattr(self, 'tick_count', 0)} | "
                f"debris={scalars.extracellular_debris:.3f}, "
                f"inflammation={scalars.inflammation_level:.3f}, "
                f"dopamine={scalars.dopamine_output:.3f}",
                flush=True
            )
        if summary_enabled and metrics is not None:
            print(
                "[summary] neurons="
                f"Healthy:{metrics['neurons.Healthy']} "
                f"Compromised:{metrics['neurons.Compromised']} "
                f"Apoptotic:{metrics['neurons.Apoptotic']} "
                f"Ruptured:{metrics['neurons.Ruptured']} | "
                "astrocytes="
                f"Supportive:{metrics['astrocytes.Supportive']} "
                f"Reactive:{metrics['astrocytes.Reactive']} | "
                "microglia="
                f"Resting:{metrics['microglia.Resting']} "
                f"Clearing:{metrics['microglia.Clearing']} "
                f"Activated:{metrics['microglia.Activated']}",
                flush=True
            )
            print(
                "[summary] alpha="
                f"free_monomer:{metrics['alpha.free.Monomer']} "
                f"free_misfolded:{metrics['alpha.free.Misfolded']} "
                f"free_oligomer:{metrics['alpha.free.Oligomer']} "
                f"free_lewy:{metrics['alpha.free.LewyBody']} "
                f"extracellular_free:{metrics['alpha.extracellular.free.total']} "
                f"extracellular_members:{metrics['alpha.extracellular.members']} "
                f"intracellular_free:{metrics['alpha.intracellular.free.total']} "
                f"intracellular_members:{metrics['alpha.intracellular.members']} "
                f"members:{metrics['alpha.members']} "
                f"oligomer_members:{metrics['alpha.members.Oligomer']} "
                f"lewy_members:{metrics['alpha.members.LewyBody']} "
                f"cleared_members:{metrics['alpha.members.Cleared']} "
                f"orphan_lewy:{metrics['alpha.orphan_lewy']} | "
                "aggregates="
                f"total:{metrics['aggregates.total']} "
                f"oligomer:{metrics['aggregates.Oligomer']} "
                f"lewy:{metrics['aggregates.LewyBody']} "
                f"avg_size:{metrics['aggregates.avg_size']:.2f} "
                f"max_size:{metrics['aggregates.max_size']} "
                f"intracellular_total:{metrics['aggregates.intracellular.total']} "
                f"extracellular_total:{metrics['aggregates.extracellular.total']} "
                f"extracellular_lewy:{metrics['aggregates.extracellular.LewyBody']} "
                f"invariant_failures:{metrics['aggregates.invariant_failures']} "
                f"intracellular_invariant_failures:{metrics['aggregates.intracellular.invariant_failures']} "
                f"extracellular_invariant_failures:{metrics['aggregates.extracellular.invariant_failures']}",
                flush=True
            )
            print(
                "[summary] neuron_progression="
                f"h2c:{metrics['neurons.transitions.healthy_to_compromised.count']} "
                f"c2a:{metrics['neurons.transitions.compromised_to_apoptotic.count']} "
                f"a2r:{metrics['neurons.transitions.apoptotic_to_ruptured.count']} "
                f"avg_compromised_ticks:{metrics['neurons.state_time.compromised_avg_ticks']:.2f} "
                f"avg_apoptotic_ticks:{metrics['neurons.state_time.apoptotic_avg_ticks']:.2f} "
                f"compromised_recoveries:{metrics['neurons.recoveries.compromised_to_healthy']} "
                f"number_of_neurons_ever_compromised:{metrics['neurons.ever_compromised']} "
                f"number_of_neurons_ever_apoptotic:{metrics['neurons.ever_apoptotic']} "
                f"number_of_neurons_ever_recovered:{metrics['neurons.ever_recovered']} "
                f"blocked_by_min_ticks_compromised:{metrics['neurons.blocks.min_ticks_compromised']} "
                f"blocked_by_apoptotic_internal_damage_threshold:{metrics['neurons.blocks.apoptotic_internal_damage_threshold']} "
                f"final_by_rank:{self._format_final_state_by_rank(transition_details)}",
                flush=True
            )
            print(
                "[summary] neuron_transition_ticks="
                f"h2c:{self._format_transition_ticks(transition_details, 'first_compromised_tick')} "
                f"c2a:{self._format_transition_ticks(transition_details, 'first_apoptotic_tick')} "
                f"a2r:{self._format_transition_ticks(transition_details, 'first_ruptured_tick')}",
                flush=True
            )

    def _progress_stdout_enabled(self) -> bool:
        """Return whether this rank should print progress information."""
        if self.rank != 0:
            return False
        return bool(_param(getattr(self, "params", {}), "logging.progress_stdout", True))

    def _final_neuron_transition_details(self) -> list[dict[str, Any]]:
        """Gather per-neuron transition timing details for the final summary."""
        local_details = [
            {
                "uid": _uid_text(neuron),
                "rank": self.rank,
                "final_state": _state_value(getattr(neuron, "state", None)),
                "first_compromised_tick": getattr(neuron, "first_compromised_tick", None),
                "first_apoptotic_tick": getattr(neuron, "first_apoptotic_tick", None),
                "first_ruptured_tick": getattr(neuron, "first_ruptured_tick", None),
                "compromised_ticks_total": getattr(neuron, "compromised_ticks_total", 0),
                "apoptotic_ticks_total": getattr(neuron, "apoptotic_ticks_total", 0),
                "compromised_recoveries": getattr(neuron, "compromised_recoveries", 0)
            }
            for neuron in self.context.agents()
            if isinstance(neuron, Neuron)
        ]
        allgather = getattr(getattr(self, "comm", None), "allgather", None)
        if not callable(allgather):
            return local_details
        gathered = allgather(local_details)
        details = []
        for rank_details in gathered:
            details.extend(rank_details)
        return details

    def _format_transition_ticks(self, details: list[dict[str, Any]], field: str) -> str:
        """Format per-neuron first transition ticks compactly."""
        values = [
            f"{detail['uid']}:{detail[field]}"
            for detail in sorted(details, key=lambda item: item["uid"])
            if detail.get(field) is not None
        ]
        return ",".join(values) if values else "none"

    def _format_final_state_by_rank(self, details: list[dict[str, Any]]) -> str:
        """Format final neuron state distributions grouped by rank."""
        grouped: dict[int, Counter] = {}
        for detail in details:
            rank = int(detail.get("rank", 0))
            grouped.setdefault(rank, Counter())[detail.get("final_state", "Unknown")] += 1
        if not grouped:
            return "none"
        parts = []
        for rank in sorted(grouped):
            counts = grouped[rank]
            state_counts = "/".join(
                f"{state}:{counts.get(state, 0)}"
                for state in ("Healthy", "Compromised", "Apoptotic", "Ruptured", "Unknown")
                if counts.get(state, 0) > 0
            )
            parts.append(f"rank{rank}={state_counts or 'none'}")
        return ";".join(parts)

    def _final_metrics(self) -> dict[str, float]:
        """Return global end-of-run counts useful for pathology tuning."""
        local = self._local_final_metrics()
        global_metrics = {
            key: self._global_sum(local.get(key, 0.0))
            for key in FINAL_SUM_METRIC_KEYS
        }
        for key in FINAL_MAX_METRIC_KEYS:
            global_metrics[key] = self._global_max(local.get(key, 0.0))
        aggregate_total = global_metrics["aggregates.total"]
        if aggregate_total > 0:
            global_metrics["aggregates.avg_size"] = global_metrics["aggregates.size_total"] / aggregate_total
        else:
            global_metrics["aggregates.avg_size"] = 0.0
        for compartment in ("intracellular", "extracellular"):
            compartment_total = global_metrics[f"aggregates.{compartment}.total"]
            if compartment_total > 0:
                global_metrics[f"aggregates.{compartment}.avg_size"] = global_metrics[f"aggregates.{compartment}.size_total"] / compartment_total
            else:
                global_metrics[f"aggregates.{compartment}.avg_size"] = 0.0
        compromised_count = global_metrics["neurons.state_time.compromised_neuron_count"]
        apoptotic_count = global_metrics["neurons.state_time.apoptotic_neuron_count"]
        global_metrics["neurons.state_time.compromised_avg_ticks"] = (
            global_metrics["neurons.state_time.compromised_ticks_total"] / compromised_count
            if compromised_count > 0
            else 0.0
        )
        global_metrics["neurons.state_time.apoptotic_avg_ticks"] = (
            global_metrics["neurons.state_time.apoptotic_ticks_total"] / apoptotic_count
            if apoptotic_count > 0
            else 0.0
        )
        return global_metrics

    def _local_final_metrics(self) -> dict[str, float]:
        """Collect final local agent-state and aggregate-consistency metrics."""
        metrics = self._empty_final_metrics()
        for agent in self.context.agents():
            if isinstance(agent, Neuron):
                state = _metric_state_value(getattr(agent, "state", None), NEURON_METRIC_STATES)
                metrics[f"neurons.{state}"] += 1
                self._collect_neuron_transition_metrics(agent, metrics)
                self._collect_neuron_internal_metrics(agent, metrics)
            elif isinstance(agent, Astrocyte):
                state = _metric_state_value(getattr(agent, "state", None), ASTROCYTE_METRIC_STATES)
                metrics[f"astrocytes.{state}"] += 1
            elif isinstance(agent, Microglia):
                state = _metric_state_value(getattr(agent, "state", None), MICROGLIA_METRIC_STATES)
                metrics[f"microglia.{state}"] += 1
        self._collect_extracellular_alpha_metrics(metrics)
        return metrics

    def _empty_final_metrics(self) -> dict[str, float]:
        """Create the fixed summary schema every rank reduces in the same order."""
        metrics = {key: 0 for key in FINAL_SUM_METRIC_KEYS}
        metrics.update({key: 0 for key in FINAL_MAX_METRIC_KEYS})
        return metrics

    def _collect_neuron_transition_metrics(self, neuron: Neuron, metrics: dict[str, float]) -> None:
        """Collect local neuron progression timing counters."""
        if getattr(neuron, "first_compromised_tick", None) is not None:
            metrics["neurons.transitions.healthy_to_compromised.count"] += 1
            metrics["neurons.ever_compromised"] += 1
        if getattr(neuron, "first_apoptotic_tick", None) is not None:
            metrics["neurons.transitions.compromised_to_apoptotic.count"] += 1
            metrics["neurons.ever_apoptotic"] += 1
        if getattr(neuron, "first_ruptured_tick", None) is not None:
            metrics["neurons.transitions.apoptotic_to_ruptured.count"] += 1
        compromised_ticks = getattr(neuron, "compromised_ticks_total", 0)
        apoptotic_ticks = getattr(neuron, "apoptotic_ticks_total", 0)
        metrics["neurons.state_time.compromised_ticks_total"] += compromised_ticks
        metrics["neurons.state_time.apoptotic_ticks_total"] += apoptotic_ticks
        if compromised_ticks > 0:
            metrics["neurons.state_time.compromised_neuron_count"] += 1
        if apoptotic_ticks > 0:
            metrics["neurons.state_time.apoptotic_neuron_count"] += 1
        metrics["neurons.recoveries.compromised_to_healthy"] += getattr(neuron, "compromised_recoveries", 0)
        metrics["neurons.blocks.min_ticks_compromised"] += getattr(neuron, "blocked_by_min_ticks_compromised", 0)
        metrics["neurons.blocks.apoptotic_internal_damage_threshold"] += getattr(neuron, "blocked_by_apoptotic_internal_damage_threshold", 0)
        if getattr(neuron, "compromised_recoveries", 0) > 0:
            metrics["neurons.ever_recovered"] += 1

    def _collect_neuron_internal_metrics(self, neuron: Neuron, metrics: dict[str, float]) -> None:
        """Collect intracellular alpha and aggregate metrics for one neuron."""
        try:
            neuron.aggregate_registry.validate_invariants(neuron)
        except RuntimeError:
            metrics["aggregates.invariant_failures"] += 1
            metrics["aggregates.intracellular.invariant_failures"] += 1
        for aggregate in neuron.aggregate_registry.aggregates(neuron):
            self._collect_aggregate_metrics(
                aggregate,
                metrics,
                compartment="intracellular",
                members=neuron.aggregate_registry.members(aggregate.aggregate_id)
            )
        for internal_agent in neuron.grid.agent_registry:
            if not isinstance(internal_agent, AlphaSynuclein):
                continue
            state = _metric_state_value(internal_agent.state, ALPHA_METRIC_STATES)
            if state == AlphaSynucleinState.LEWY_BODY.value and internal_agent.aggregate_id is None:
                metrics["alpha.orphan_lewy"] += 1
            if internal_agent.aggregate_id is None:
                metrics[f"alpha.free.{state}"] += 1
                metrics["alpha.intracellular.free.total"] += 1
                metrics[f"alpha.intracellular.free.{state}"] += 1

    def _collect_extracellular_alpha_metrics(self, metrics: dict[str, float]) -> None:
        """Collect alpha and aggregate metrics from the shared extracellular grid."""
        environment = getattr(self, "environment", None)
        grid = getattr(environment, "grid", None)
        registry = getattr(environment, "aggregate_registry", None)
        agents = getattr(grid, "agent_registry", [])
        for agent in agents:
            if isinstance(agent, AlphaSynuclein):
                state = _metric_state_value(agent.state, ALPHA_METRIC_STATES)
                if agent.aggregate_id is None:
                    metrics[f"alpha.free.{state}"] += 1
                    metrics["alpha.extracellular.free.total"] += 1
                    metrics[f"alpha.extracellular.free.{state}"] += 1
            elif isinstance(agent, AlphaAggregate):
                if not self._extracellular_aggregate_invariants_hold(agent, agents, registry):
                    metrics["aggregates.invariant_failures"] += 1
                    metrics["aggregates.extracellular.invariant_failures"] += 1
                self._collect_aggregate_metrics(
                    agent,
                    metrics,
                    compartment="extracellular"
                )

    def _collect_aggregate_metrics(self, aggregate: AlphaAggregate, metrics: dict[str, float], compartment: str, members=None) -> None:
        """Collect aggregate-agent and represented-protein counts."""
        state = _metric_state_value(aggregate.state, AGGREGATE_METRIC_STATES)
        size = aggregate.size
        metrics["aggregates.total"] += 1
        metrics[f"aggregates.{state}"] += 1
        metrics["aggregates.size_total"] += size
        metrics["aggregates.max_size"] = max(metrics["aggregates.max_size"], size)
        metrics[f"aggregates.{compartment}.total"] += 1
        metrics[f"aggregates.{compartment}.{state}"] += 1
        metrics[f"aggregates.{compartment}.size_total"] += size
        metrics[f"aggregates.{compartment}.max_size"] = max(
            metrics[f"aggregates.{compartment}.max_size"],
            size
        )

        member_agents = set(members) if members is not None else set(aggregate.member_agents)
        if member_agents:
            for member in member_agents:
                member_state = _state_value(getattr(member, "state", None))
                self._count_aggregate_member(metrics, compartment, member_state)
            return

        inferred_member_state = _alpha_state_for_aggregate(aggregate.state)
        for _ in range(size):
            self._count_aggregate_member(metrics, compartment, inferred_member_state)

    def _count_aggregate_member(self, metrics: dict[str, float], compartment: str, member_state: str) -> None:
        """Count one alpha-synuclein protein represented by an aggregate."""
        member_state = _metric_state_value(member_state, ALPHA_METRIC_STATES)
        metrics["alpha.members"] += 1
        metrics[f"alpha.{compartment}.members"] += 1
        metrics[f"alpha.members.{member_state}"] += 1

    def _extracellular_aggregate_invariants_hold(self, aggregate: AlphaAggregate, agents, registry=None) -> bool:
        """Return whether an extracellular aggregate is self-consistent."""
        if aggregate.owner_neuron is not None:
            return False
        if registry is not None and registry.aggregate_for(aggregate.aggregate_id) is not aggregate:
            return False
        if aggregate.size <= 0 or not aggregate.member_ids:
            return False
        member_agents = set(getattr(aggregate, "member_agents", set()))
        if not member_agents:
            return False
        if registry is not None:
            registered_members = registry.members(aggregate.aggregate_id)
            if registered_members and registered_members != member_agents:
                return False
        member_ids = {_alpha_member_id(member) for member in member_agents}
        if member_ids != set(aggregate.member_ids):
            return False
        active_agents = set(agents)
        aggregate_state = _metric_state_value(aggregate.state, AGGREGATE_METRIC_STATES)
        expected_state = AlphaSynucleinState.LEWY_BODY if aggregate_state == AggregateState.LEWY_BODY.value else AlphaSynucleinState.OLIGOMER
        for member in member_agents:
            if not isinstance(member, AlphaSynuclein):
                return False
            if member in active_agents:
                return False
            if member.aggregate_id != aggregate.aggregate_id:
                return False
            if member.state != expected_state:
                return False
            if member.compartment != AlphaSynucleinCompartment.EXTRACELLULAR:
                return False
            if member.owner_neuron is not None:
                return False
        return True

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

def _state_value(state) -> str:
    """Return enum values as stable strings for summary metrics."""
    return getattr(state, "value", str(state))

def _metric_state_value(state, allowed_states: tuple[str, ...]) -> str:
    """Return a state value that is guaranteed to have a summary metric key."""
    value = _state_value(state)
    if value in allowed_states:
        return value
    return "Unknown"

def _alpha_member_id(alpha: AlphaSynuclein):
    """Return the membership id used by AlphaAggregate.member_ids."""
    return getattr(alpha, "uid", id(alpha))

def _uid_text(agent) -> str:
    """Return a compact stable uid string for final summaries."""
    uid = getattr(agent, "uid", None)
    if uid is None:
        return str(id(agent))
    if isinstance(uid, tuple):
        return ":".join(str(item) for item in uid)
    return str(uid)

def _alpha_state_for_aggregate(state) -> str:
    """Infer member alpha state when only aggregate member ids are available."""
    if _state_value(state) == AggregateState.LEWY_BODY.value:
        return AlphaSynucleinState.LEWY_BODY.value
    return AlphaSynucleinState.OLIGOMER.value

if __name__ == "__main__":
    run()
