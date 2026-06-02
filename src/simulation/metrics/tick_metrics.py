from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

from src.simulation.agents.aggregate import AlphaAggregate
from src.simulation.agents.alphasynuclein import AlphaSynuclein
from src.simulation.agents.neuron import Neuron
from src.simulation.agents.structure.states import AlphaSynucleinState, NeuronState
from src.simulation.metrics.key import TICK_METRIC_COLUMNS, TICK_METRIC_COUNT_KEYS


class TickMetricsRecorder:
    def __init__(self, *, enabled: bool, rank: int, output_dir: Path, global_sum: Callable[[float], float]) -> None:
        self.enabled = enabled
        self.rank = rank
        self.output_dir = Path(output_dir)
        self.global_sum = global_sum
        self.path = None
        self.file = None
        if not self.enabled or self.rank != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "tick_metrics.csv"
        self.path.write_text(",".join(TICK_METRIC_COLUMNS) + "\n", encoding="utf-8")

    def record(self, *, tick: int, context: Any, environment: Any) -> None:
        if not self.enabled:
            return
        local_counts = collect_tick_counts(context, environment)
        global_counts = {
            key: int(self.global_sum(local_counts.get(key, 0)))
            for key in TICK_METRIC_COUNT_KEYS
        }
        if self.rank != 0 or self.path is None:
            return
        scalars = environment.scalars
        row = [
            str(tick),
            f"{scalars.extracellular_debris:.6f}",
            f"{scalars.inflammation_level:.6f}",
            f"{scalars.dopamine_output:.6f}",
            *(str(global_counts[key]) for key in TICK_METRIC_COUNT_KEYS)
        ]
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(",".join(row) + "\n")

    def close(self) -> None:
        if self.file is None:
            return
        self.file.close()
        self.file = None


def collect_tick_counts(context: Any, environment: Any) -> dict[str, int]:
    counts = {key: 0 for key in TICK_METRIC_COUNT_KEYS}
    for agent in context.agents():
        if isinstance(agent, Neuron):
            state = state_value(getattr(agent, "state", None))
            if state == NeuronState.HEALTHY.value:
                counts["neurons_healthy"] += 1
            elif state == NeuronState.COMPROMISED.value:
                counts["neurons_compromised"] += 1
            elif state == NeuronState.APOPTOTIC.value:
                counts["neurons_apoptotic"] += 1
            elif state == NeuronState.RUPTURED.value:
                counts["neurons_ruptures"] += 1
            count_tick_alpha_agents(getattr(agent, "grid", None), counts)
    count_tick_alpha_agents(getattr(environment, "grid", None), counts)
    return counts


def count_tick_alpha_agents(grid: Any, counts: dict[str, int]) -> None:
    for agent in getattr(grid, "agent_registry", []):
        if isinstance(agent, AlphaSynuclein) and agent.aggregate_id is None:
            if getattr(agent, "state", None) != AlphaSynucleinState.CLEARED:
                counts["free_alpha"] += 1
        elif isinstance(agent, AlphaAggregate):
            counts["alpha_aggregate"] += agent.size


def state_value(state: Any) -> str:
    return getattr(state, "value", str(state))
