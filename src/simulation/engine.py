# Standard libraries
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional, Callable


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    project_root_path = str(project_root)
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
if __package__ in (None, ""):
    _ensure_project_root_on_path()

# Third-party libraries (MPI, Repast4Py)
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import random
from repast4py import schedule
from repast4py import space
from repast4py.space import DiscretePoint

# Simulation agents
from src.simulation.agents.aggregate import AlphaAggregate
from src.simulation.agents.alphasynuclein import AlphaSynuclein, AlphaSynucleinCompartment
from src.simulation.agents.astrocyte import Astrocyte
from src.simulation.agents.lysosome import Lysosome
from src.simulation.agents.microglia import Microglia
from src.simulation.agents.mitochondrion import Mitochondrion
from src.simulation.agents.neuron import Neuron
from src.simulation.agents.structure.agenttypes import AgentType
from src.simulation.agents.structure.states import AggregateState, AlphaSynucleinState, NeuronState

# Runtime / Environment
from src.simulation.substantia_nigra import SubstantiaNigra
from src.simulation.runtime.mpi import MpiHelper

# Config / Utils
from src.simulation.utils.param import get_param
from src.simulation.utils import Params, RNG
from src.simulation.utils.config_factory import ConfigFactory
from src.simulation.runtime.agent_factory import AgentFactory

# Logging
from src.simulation.runtime.reporter import RuntimeReporter

# Metrics
from src.simulation.metrics.final_metrics import FinalMetricsCollector, collect_local_final_metrics, extracellular_aggregate_invariants_hold, state_value as _state_value
from src.simulation.metrics.tick_metrics import collect_tick_counts, count_tick_alpha_agents


class SimulationCompleted(Exception):
    pass

_param = get_param

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
        self.mpi = MpiHelper(comm)
        self.rank = comm.Get_rank()
        self.params = params
        self.seed = int(get_param(params, "random.seed", 42))
        self.stop_at = int(get_param(params, "stop.at", 100))
        self.progress_interval = max(
            1,
            int(get_param(params, "logging.progress_interval", 25))
        )
        self._init_repast_rng(self.seed)
        self.config_rng = RNG(self.seed + self.rank)
        self.rng = random.default_rng
        self.neuron_params = Params("neuron")
        self.neuron_param_values = self.neuron_params.as_dict()
        self.tick_count = 0
        self._finalized = False
        self.context = ctx.SharedContext(comm)
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(self.stop_at)
        self.grid = self._create_grid(params)
        self.context.add_projection(self.grid)
        self.environment = SubstantiaNigra(grid=self.grid, config=ConfigFactory.build_substantia_nigra_config())
        self.agent_factory = AgentFactory(
            rank=self.rank,
            agent_type=AgentType,
            config_rng=self.config_rng,
            neuron_params=self.neuron_params,
            environment=self.environment,
            new_id=self._new_id
        )
        self._next_local_id = 0
        self.reporter = RuntimeReporter(self, params)
        self._tick_metrics_file = self.reporter._tick_metrics_file
        self.causal_logger = self.reporter.causal_logger
        self.initialization_logger = self.reporter.initialization_logger
        self._create_agents(params)
        self.initial_dopamine_capacity = self._global_sum(self._local_dopamine_capacity())
        self.reporter.close_initialization_logger()
        self.reporter.log_startup()


    # Initialization

    def _create_grid(self, params: dict[str, Any]) -> space.SharedGrid:
        """Create the shared extracellular grid.
        world.width and world.height are the biological space size.
        world.buffer_size is the number of neighboring cells mirrored from adjacent ranks.
        It is not a biological buffer and it is not related to neuron / lysosome target buffers."""
        width = int(get_param(params, "world.width", 50))
        height = int(get_param(params, "world.height", 50))
        buffer_size = int(get_param(params, "world.buffer_size", 2))
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
        self._create_neurons(params)
        self._create_microglia(params)
        self._create_astrocytes(params)
        self._create_extracellular_alpha(params)

    def _create_neurons(self, params: dict[str, Any]) -> None:
        self._create_population(param_key="external.population.neurons", default_count=10, create_agent=self.agent_factory.create_neuron, after_add=self._populate_neuron)

    def _create_microglia(self, params: dict[str, Any]) -> None:
        self._create_population(param_key="external.population.microglia", default_count=5, create_agent=self.agent_factory.create_microglia)

    def _create_astrocytes(self, params: dict[str, Any]) -> None:
        self._create_population(param_key="external.population.astrocytes", default_count=5, create_agent=self.agent_factory.create_astrocyte)

    def _create_extracellular_alpha(self, params: dict[str, Any]) -> None:
        self._create_population(param_key="external.population.alpha", default_count=0, create_agent=self.agent_factory.create_extracellular_alpha)

    def _populate_neuron(self, neuron: Neuron) -> None:
        alpha_count = self._intracellular_count("alpha")
        mitochondria_count = self._intracellular_count("mitochondria")
        lysosome_count = self._intracellular_count("lysosomes")
        for index in range(alpha_count):
            alpha = self.agent_factory.create_intracellular_alpha(neuron)
            self._add_internal_agent(neuron, alpha, self._internal_point(neuron, index))
        for index in range(mitochondria_count):
            mitochondrion = self.agent_factory.create_mitochondrion(neuron)
            self._add_internal_agent(neuron, mitochondrion, self._internal_point(neuron, alpha_count + index))
        for index in range(lysosome_count):
            lysosome = self.agent_factory.create_lysosome(neuron)
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
        reporter = getattr(self, "reporter", None)
        if reporter is not None:
            reporter.begin_tick(self.tick_count)
        self.environment.begin_tick()
        # Use a list snapshot because agents can release or absorb alpha during
        # a step. Iterating over the live context while it mutates would be
        # brittle and can skip agents.
        for agent in list(self.context.agents()):
            if hasattr(agent, "step"):
                agent.step(self)
        self._synchronize_environment_effects()
        self.environment.commit_effects(max_possible_dopamine=self._max_possible_dopamine())
        if reporter is not None:
            reporter.record_tick()
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
        point = self.grid.get_random_local_pt(self.rng)
        self.context.add(agent)
        agent.pt = point
        self.environment.add_agent(agent, point)
        self.reporter.initialization_logger.record_agent(agent, position=point, raw_details={"habitat": "SubstantiaNigra"})
        self.reporter.record_initial_g0_alpha(agent)

    def _add_internal_agent(self, neuron: Neuron, agent, point: DiscretePoint) -> None:
        neuron.add_agent(agent, point)
        self.reporter.initialization_logger.record_agent(agent, position=point, owner=neuron, target=neuron, raw_details={"habitat": "Neuron"})
        self.reporter.record_initial_g0_alpha(agent, owner=neuron)

    def _internal_point(self, neuron: Neuron, index: int) -> DiscretePoint:
        """Return a deterministic initial point inside a neuron's local grid."""
        width = max(1, neuron.internal_cfg.width)
        height = max(1, neuron.internal_cfg.height)
        cell = index % (width * height)
        return DiscretePoint(cell % width, cell // width)

    def _intracellular_count(self, neuron_key: str) -> int:
        """Read one initial intracellular population count from neuron.yaml."""
        return int(get_param(self.neuron_param_values, f"intracellular.population.{neuron_key}", 0))

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
        """Return MPI world size through the runtime helper."""
        return self._mpi_helper().size

    def _global_sum(self, value: float) -> float:
        """Sum a scalar across MPI ranks through the runtime helper."""
        return self._mpi_helper().sum(value)

    def _global_max(self, value: float) -> float:
        """Return a maximum across MPI ranks through the runtime helper."""
        return self._mpi_helper().max(value)

    def _mpi_helper(self) -> MpiHelper:
        """Return the configured MPI helper, creating one for bare test models."""

        helper = getattr(self, "mpi", None)
        if helper is None:
            helper = MpiHelper(getattr(self, "comm", None))
            self.mpi = helper
        return helper

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
        if getattr(self, "_finalized", False):
            return
        self._finalized = True
        self._log_completion()
        reporter = getattr(self, "reporter", None)
        if reporter is not None:
            reporter.close()

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

    def _local_tick_metric_counts(self) -> dict[str, int]:
        """Count local agent states used by the per-tick CSV."""
        return collect_tick_counts(self.context, self.environment)

    def _count_tick_alpha_agents(self, grid, counts: dict[str, int]) -> None:
        """Count free proteins and aggregate member proteins from one grid."""
        count_tick_alpha_agents(grid, counts)

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

    def _final_metrics(self) -> dict[str, float]:
        collector = getattr(self, "final_metrics", None)
        if collector is None:
            collector = FinalMetricsCollector(global_sum=self._global_sum, global_max=self._global_max)
        return collector.reduce(self._local_final_metrics())

    def _local_final_metrics(self) -> dict[str, float]:
        """Collect final metrics for this rank.

        Kept as a thin compatibility wrapper around the metrics module so older
        tests and notebooks can still inspect local metrics through the model.
        """

        return collect_local_final_metrics(self.context, self.environment)

    def _log_completion(self) -> None:
        """Emit final progress and summary logs through the runtime reporter."""

        reporter = getattr(self, "reporter", None)
        if reporter is not None:
            reporter.log_completion()
            return
        summary_enabled = bool(get_param(getattr(self, "params", {}), "logging.summary_stdout", True))
        metrics = self._final_metrics() if summary_enabled else None
        transition_details = self._final_neuron_transition_details() if summary_enabled else []
        if getattr(self, "rank", 0) != 0:
            return
        scalars = self.environment.scalars
        if bool(get_param(getattr(self, "params", {}), "logging.progress_stdout", True)):
            print(
                f"[progress] completed at tick {getattr(self, 'tick_count', 0)} | "
                f"debris={scalars.extracellular_debris:.3f}, "
                f"inflammation={scalars.inflammation_level:.3f}, "
                f"dopamine={scalars.dopamine_output:.3f}",
                flush=True
            )
        if summary_enabled and metrics is not None:
            from src.simulation.metrics.summary_formatter import format_summary_lines
            for line in format_summary_lines(metrics, transition_details=transition_details):
                print(line, flush=True)

    def _extracellular_aggregate_invariants_hold(self, aggregate: AlphaAggregate, agents, registry=None) -> bool:
        """Return whether an extracellular aggregate is self-consistent."""
        return extracellular_aggregate_invariants_hold(aggregate, agents, registry)

    def _create_population(self, *, param_key: str, default_count: int, create_agent: Callable[[], Any], after_add: Callable[[Any], None] | None = None) -> None:
        global_count = int(get_param(self.params, param_key, default_count))
        local_count = self._local_population_count(global_count)
        for _ in range(local_count):
            agent = create_agent()
            self._add_agent_randomly(agent)
            if after_add is not None:
                after_add(agent)

def run(params: Optional[dict[str, Any]] = None) -> None:
    if params is None:
        params = Params("system").as_dict()
    model = ParkinsonModel(MPI.COMM_WORLD, params)
    model.start()

def _uid_text(agent) -> str:
    uid = getattr(agent, "uid", None)
    if uid is None:
        return str(id(agent))
    if isinstance(uid, tuple):
        return ":".join(str(item) for item in uid)
    return str(uid)

if __name__ == "__main__":
    run()
