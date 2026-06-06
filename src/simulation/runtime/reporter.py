from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from src.simulation.agents.aggregate import AlphaAggregate
from src.simulation.agents.alphasynuclein import AlphaSynuclein
from src.simulation.agents.astrocyte import Astrocyte
from src.simulation.agents.microglia import Microglia
from src.simulation.agents.neuron import Neuron
from src.simulation.logger.event_logger import EventLogger
from src.simulation.logger.initialization_logger import InitializationLogger
from src.simulation.logger.snapshot_logger import SnapshotLogger
from src.simulation.metrics.final_metrics import FinalMetricsCollector
from src.simulation.metrics.summary_formatter import format_summary_lines
from src.simulation.metrics.tick_metrics import TickMetricsRecorder
from src.simulation.utils.param import get_param

if TYPE_CHECKING:
    from src.simulation.engine import ParkinsonModel

DEFAULT_OUTPUT_ROOT = "output"
SPATIAL_AGENT_TYPES = (Neuron, AlphaSynuclein, AlphaAggregate, Microglia, Astrocyte)

class RuntimeReporter:
    def __init__(self, model: ParkinsonModel, params: dict[str, Any]):
        self.model = model
        self.params = params
        self.output_root = self._resolve_output_dir(get_param(params, "logging.output_dir", DEFAULT_OUTPUT_ROOT))
        self.run_log_dir = self._resolve_layout_dir(params, "logging.run_log_dir", "run_logs")
        self.rank_log_dir = self._resolve_layout_dir(params, "logging.rank_log_dir", "logs_per_rank")
        self.initialization_log_dir = self._resolve_layout_dir(params, "logging.initialization_log_dir", "initialization_logs")
        self.metrics_dir = self._resolve_layout_dir(params, "logging.metrics_dir", "metrics")
        self.output_dir = self.run_log_dir
        self.run_id = str(get_param(params, "logging.run_id", f"run_seed_{model.seed}"))
        self.event_logger, self.initialization_logger = self._create_loggers(params)
        self.causal_logger = self.event_logger
        self.snapshot_logger = self._create_spatial_logger(params)
        self.spatial_logger = self.snapshot_logger
        self.final_metrics = FinalMetricsCollector(global_sum=model._global_sum, global_max=model._global_max)
        self.tick_metrics = None
        self._tick_metrics_file = None
        self._create_tick_metrics_csv(params)

    def _create_loggers(self, params: dict[str, Any]) -> tuple[EventLogger, InitializationLogger]:
        """Create canonical event and initialization loggers."""
        run_id = str(get_param(params, "logging.run_id", f"run_seed_{self.model.seed}"))
        events_enabled = bool(get_param(params, "logging.events.enabled", get_param(params, "logging.causal.enabled", get_param(params, "logging.enabled", False))))
        event_logger = EventLogger(comm=self.model.comm, rank=self.model.rank, output_dir=self.run_log_dir, rank_output_dir=self.rank_log_dir, enabled=events_enabled, run_id=run_id)
        initialization_logger = InitializationLogger(comm=self.model.comm, rank=self.model.rank, output_dir=self.initialization_log_dir, enabled=bool(get_param(params, "logging.initialization.enabled", get_param(params, "logging.enabled", False))))
        return event_logger, initialization_logger

    def _create_spatial_logger(self, params: dict[str, Any]) -> SnapshotLogger:
        enabled = bool(get_param(params, "logging.spatial.enabled", get_param(params, "logging.enabled", False)))
        return SnapshotLogger(comm=self.model.comm, rank=self.model.rank, output_dir=self.run_log_dir, rank_output_dir=self.rank_log_dir, enabled=enabled, run_id=self.run_id)

    def _resolve_output_dir(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[3] / path

    def _resolve_layout_dir(self, params: dict[str, Any], key: str, default_name: str) -> Path:
        value = get_param(params, key, None)
        if value is not None:
            return self._resolve_output_dir(value)
        return self.output_root / default_name

    def _create_tick_metrics_csv(self, params: dict[str, Any]) -> None:
        """Create the rank-aware tick metrics recorder."""
        default_enabled = bool(get_param(params, "logging.enabled", True))
        self.tick_metrics_enabled = bool(get_param(params, "logging.tick_metrics_csv", default_enabled))
        self.tick_metrics = TickMetricsRecorder(enabled=self.tick_metrics_enabled, rank=self.model.rank, output_dir=self.metrics_dir, global_sum=self.model._global_sum)
        self._tick_metrics_file = self.tick_metrics.file

    # Initialization logging
    def close_initialization_logger(self) -> None:
        self.initialization_logger.close()

    def record_initial_agent(self, agent, *, position, owner=None, target=None, raw_details: dict[str, Any] | None = None) -> None:
        self.initialization_logger.record_agent(agent, position=position, owner=owner, target=target, raw_details=raw_details)

    def record_initial_g0_alpha(self, agent, owner: Optional[Neuron] = None) -> None:
        return

    # Per-tick reporting
    def begin_tick(self, tick: int) -> None:
        logger = self.event_logger
        if logger is not None:
            logger.set_tick(tick)

    def record_scalar_tick(self) -> None:
        return

    def record_tick_metrics_csv(self) -> None:
        tick_metrics = getattr(self, "tick_metrics", None)
        if tick_metrics is None:
            return
        tick_metrics.record(tick=getattr(self.model, "tick_count", 0), context=self.model.context, environment=self.model.environment)

    def record_spatial_tick(self) -> None:
        """Record final external positions for this tick."""
        logger = getattr(self, "spatial_logger", None)
        if logger is None:
            return
        tick = getattr(self.model, "tick_count", 0)
        env = self.model.environment
        for agent in list(env.grid.agent_registry):
            if not isinstance(agent, SPATIAL_AGENT_TYPES):
                continue
            position = env.position_of(agent)
            logger.record_agent(tick, agent, position)

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
        self.record_spatial_tick()
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
        event_logger = getattr(self, "event_logger", None)
        if event_logger is not None:
            event_logger.close()
        snapshot_logger = getattr(self, "snapshot_logger", None)
        if snapshot_logger is not None:
            snapshot_logger.close()
