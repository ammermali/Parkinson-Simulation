from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from src.simulation.agents.alphasynuclein import AlphaSynuclein
from src.simulation.agents.neuron import Neuron
from src.simulation.logger import InitializationLogger, CausalTraceLogger
from src.simulation.metrics.final_metrics import FinalMetricsCollector
from src.simulation.agents.structure.agenttypes import AgentType
from src.simulation.metrics.summary_formatter import format_summary_lines
from src.simulation.metrics.tick_metrics import TickMetricsRecorder
from src.simulation.utils.param import get_param

if TYPE_CHECKING:
    from src.simulation.engine import ParkinsonModel

DEFAULT_SIMULATION_OUTPUT_DIR = "output/simulation/logs"

class RuntimeReporter:
    def __init__(self, model: ParkinsonModel, params: dict[str, Any]):
        self.model = model
        self.params = params
        self.output_dir = self._resolve_output_dir(get_param(params, "logging.output_dir", DEFAULT_SIMULATION_OUTPUT_DIR))
        self.run_id = str(get_param(params, "logging.run_id", f"run_seed_{model.seed}"))
        self.causal_logger, self.initialization_logger = self._create_loggers(params)
        self.final_metrics = FinalMetricsCollector(global_sum=model._global_sum, global_max=model._global_max)
        self.tick_metrics = None
        self._tick_metrics_file = None
        self._create_tick_metrics_csv(params)

    def _create_loggers(self, params: dict[str, Any]) -> tuple[CausalTraceLogger, InitializationLogger]:
        """Create separated causal and initialization loggers."""
        output_dir = self._resolve_output_dir(get_param(params, "logging.output_dir", DEFAULT_SIMULATION_OUTPUT_DIR))
        run_id = str(get_param(params, "logging.run_id", f"run_seed_{self.model.seed}"))
        agent_type_map = {
            AgentType.NEURON: "Neuron",
            AgentType.MICROGLIA: "Microglia",
            AgentType.ASTROCYTE: "Astrocyte",
            AgentType.ALPHA: "AlphaSynuclein",
            AgentType.MITOCHONDRION: "Mitochondrion",
            AgentType.LYSOSOME: "Lysosome"
        }
        causal_logger = CausalTraceLogger(
            run_id=run_id,
            comm=self.model.comm,
            rank=self.model.rank,
            output_dir=output_dir,
            enabled=bool(get_param(params, "logging.causal.enabled", get_param(params, "logging.enabled", False))),
            agent_type_map=agent_type_map,
            params=params
        )
        initialization_logger = InitializationLogger(
            run_id=run_id,
            comm=self.model.comm,
            rank=self.model.rank,
            output_dir=output_dir,
            enabled=bool(get_param(params, "logging.initialization.enabled", get_param(params, "logging.enabled", False)))
        )
        return causal_logger, initialization_logger

    def _resolve_output_dir(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[3] / path

    def _create_tick_metrics_csv(self, params: dict[str, Any]) -> None:
        """Create the rank-aware tick metrics recorder."""

        default_enabled = bool(get_param(params, "logging.enabled", True))
        self.tick_metrics_enabled = bool(get_param(params, "logging.tick_metrics_csv", default_enabled))
        output_dir = self._resolve_output_dir(get_param(params, "logging.output_dir", DEFAULT_SIMULATION_OUTPUT_DIR))
        self.tick_metrics = TickMetricsRecorder(
            enabled=self.tick_metrics_enabled,
            rank=self.model.rank,
            output_dir=output_dir,
            global_sum=self.model._global_sum
        )
        self._tick_metrics_file = self.tick_metrics.file

    # Initialization logging
    def close_initialization_logger(self) -> None:
        self.initialization_logger.close()

    def record_initial_agent(self, agent, *, position, owner=None, target=None, raw_details: dict[str, Any] | None = None) -> None:
        self.initialization_logger.record_agent(agent, position=position, owner=owner, target=target, raw_details=raw_details)

    def record_initial_g0_alpha(self, agent, owner: Optional[Neuron] = None) -> None:
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

    # Per-tick reporting
    def begin_tick(self, tick: int) -> None:
        logger = self.causal_logger
        if logger is not None:
            logger.set_tick(tick)

    def record_scalar_tick(self) -> None:
        logger = getattr(self, "causal_logger", None)
        if logger is None:
            return
        scalars = self.model.environment.scalars
        effects = getattr(self.model.environment, "last_committed_effects", self.model.environment.effects)
        logger.snapshot_field("extracellular_debris", scalars.extracellular_debris)
        logger.snapshot_field("inflammation_level", scalars.inflammation_level)
        logger.snapshot_field("dopamine_output", scalars.dopamine_output)
        logger.buffer_commit("debris_added", "extracellular_debris", effects.debris_added, "positive", "sn_debris_added_commit")
        logger.buffer_commit("debris_removed", "extracellular_debris", effects.debris_removed, "negative", "sn_debris_removed_commit")
        logger.buffer_commit("inflammation_added", "inflammation_level", effects.inflammation_added, "positive", "sn_inflammation_added_commit")
        logger.buffer_commit("inflammation_removed", "inflammation_level", effects.inflammation_removed, "negative", "sn_inflammation_removed_commit")
        logger.buffer_commit("dopamine_released", "dopamine_output", effects.dopamine_released, "positive", "sn_dopamine_release_commit")

    def record_tick_metrics_csv(self) -> None:
        tick_metrics = getattr(self, "tick_metrics", None)
        if tick_metrics is None:
            return
        tick_metrics.record(tick=getattr(self.model, "tick_count", 0), context=self.model.context, environment=self.model.environment)

    def log_tick(self) -> None:
        if not bool(get_param(getattr(self, "params", {}), "logging.scalar_stdout", True)):
            return
        if self.model.rank != 0:
            return
        scalars = self.model.environment.scalars
        print(
            f"debris={scalars.extracellular_debris:.3f}, "
            f"inflammation={scalars.inflammation_level:.3f}, "
            f"dopamine={scalars.dopamine_output:.3f}"
        )

    def _log_progress_tick(self) -> None:
        if not self.progress_stdout_enabled():
            return
        tick = getattr(self.model, "tick_count", 0)
        if tick <= 0:
            return
        if tick % self.model.progress_interval != 0 and tick < self.model.stop_at:
            return
        scalars = self.model.environment.scalars
        progress = min(100.0, 100.0 * tick / max(1, self.model.stop_at))
        print(
            f"[progress] tick {tick}/{self.model.stop_at} ({progress:.1f}%) | "
            f"debris={scalars.extracellular_debris:.3f}, "
            f"inflammation={scalars.inflammation_level:.3f}, "
            f"dopamine={scalars.dopamine_output:.3f}",
            flush=True
        )

    def record_tick(self) -> None:
        self.record_scalar_tick()
        self.record_tick_metrics_csv()
        self.log_tick()
        self._log_progress_tick()

    # Startup / completion
    def log_startup(self) -> None:
        if not self.progress_stdout_enabled():
            return
        local_agents = len(list(self.model.context.agents()))
        total_agents = int(self.model._global_sum(local_agents))
        print(
            f"[progress] starting run: ranks={self.model._comm_size()}, "
            f"ticks={self.model.stop_at}, agents={total_agents}, seed={self.model.seed}",
            flush=True
        )

    def log_completion(self) -> None:
        summary_enabled = bool(get_param(getattr(self, "params", {}), "logging.summary_stdout", True))
        metrics = self.model._final_metrics() if summary_enabled else None
        transition_details = self.model._final_neuron_transition_details() if summary_enabled else []
        if self.model.rank != 0:
            return
        scalars = self.model.environment.scalars
        if bool(get_param(getattr(self, "params", {}), "logging.progress_stdout", True)):
            print(
                f"[progress] completed at tick {getattr(self.model, 'tick_count', 0)} | "
                f"debris={scalars.extracellular_debris:.3f}, "
                f"inflammation={scalars.inflammation_level:.3f}, "
                f"dopamine={scalars.dopamine_output:.3f}",
                flush=True
            )
        if summary_enabled and metrics is not None:
            for line in format_summary_lines(metrics, transition_details=transition_details):
                print(line, flush=True)

    def progress_stdout_enabled(self) -> bool:
        if self.model.rank != 0:
            return False
        return bool(get_param(getattr(self, "params", {}), "logging.progress_stdout", True))

    # Shutdown

    def close(self) -> None:
        tick_metrics = getattr(self, "tick_metrics", None)
        if tick_metrics is not None:
            tick_metrics.close()
            self._tick_metrics_file = None
        else:
            tick_metrics_file = getattr(self, "_tick_metrics_file", None)
            if tick_metrics_file is not None:
                tick_metrics_file.close()
                self._tick_metrics_file = None
        causal_logger = getattr(self, "causal_logger", None)
        if causal_logger is not None:
            causal_logger.close()

