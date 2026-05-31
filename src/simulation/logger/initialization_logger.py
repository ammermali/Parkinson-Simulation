from __future__ import annotations
import json
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from src.simulation.logger.causal_trace_logger import uid_of, value_of


@dataclass(frozen=True)
class InitializationAgentRecord:
    run_id: str
    rank: int
    uid: str
    local_id: Optional[int]
    type_id: Optional[int]
    agent_class: str
    initial_state: Optional[str]
    compartment: Optional[str]
    owner_uid: Optional[str]
    target_uid: Optional[str]
    target_class: Optional[str]
    position: Optional[dict[str, int]]
    aggregate_id: Optional[int]
    config: Optional[dict[str, Any]]
    initial_scalars: Optional[dict[str, Any]]
    initial_internal_scalars: Optional[dict[str, Any]]
    initial_buffers: Optional[dict[str, Any]]
    raw_details: Optional[dict[str, Any]]
    display: dict[str, Any]


class InitializationLogger:
    """Exhaustive JSON logger for initial conditions and visualization.
    Unlike CausalTraceLogger, this logger intentionally stores position, full
    configs, owners, buffers and scalar snapshots. It runs only during setup and
    should not be used as a causal edge source. It's purpose is only for visualization of initial conditions."""

    schema_version = "1.0-initialization-json"
    def __init__(self, run_id: str, rank: int, comm=None, output_dir: Path | str = "src/simulation/output/logs", enabled: bool = False):
        self.run_id = run_id
        self.rank = rank
        self.comm = comm
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self._records: list[InitializationAgentRecord] = []
        self._counts_by_class: dict[str, int] = {}
        self._counts_by_state: dict[str, dict[str, int]] = {}
        self._counts_by_rank: dict[str, dict[str, int]] = {}
        self._neurons: dict[str, dict[str, Any]] = {}
        self._extracellular_agents: dict[str, list[str]] = {}
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            self.agents_path.write_text("", encoding="utf-8")
        self._barrier()

    @property
    def agents_path(self) -> Path:
        return self.output_dir / "initialization_agents.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "initialization_manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "initialization_summary.json"

    def record_agent(self, agent, position=None, owner=None, target=None, raw_details: Optional[dict[str, Any]] = None) -> None:
        """Write one initialized agent with full documentary details."""

        if not self.enabled:
            return
        agent_class = type(agent).__name__
        owner_uid = uid_of(owner or getattr(agent, "owner_neuron", None))
        target_uid = uid_of(target)
        record = InitializationAgentRecord(
            run_id=self.run_id,
            rank=self.rank,
            uid=uid_of(agent) or "",
            local_id=_local_id(agent),
            type_id=getattr(agent, "ptype", getattr(agent, "type_id", None)),
            agent_class=agent_class,
            initial_state=value_of(getattr(agent, "state", None)),
            compartment=value_of(getattr(agent, "compartment", None)),
            owner_uid=owner_uid,
            target_uid=target_uid,
            target_class=type(target).__name__ if target is not None else None,
            position=point_dict(position),
            aggregate_id=getattr(agent, "aggregate_id", None),
            config=safe_serialize(getattr(agent, "cfg", None)),
            initial_scalars=safe_serialize(getattr(agent, "scalars", None)),
            initial_internal_scalars=safe_serialize(getattr(agent, "internal_scalars", None)),
            initial_buffers=safe_serialize(getattr(agent, "effects", getattr(agent, "internal_effects", None))),
            raw_details=safe_serialize(raw_details or {}),
            display={
                "label": f"{agent_class} {_local_id(agent)}",
                "group": agent_class,
                "owner_label": f"Neuron {owner_uid}" if owner_uid else None,
                "visual_level": "intracellular" if owner_uid else "extracellular"
            }
        )
        self._records.append(record)
        self._update_manifest(record)

    def close(self) -> None:
        """Write manifest and compact summary."""

        if not self.enabled:
            return
        records = [
            record
            for rank_records in self._allgather([safe_serialize(record) for record in self._records])
            for record in rank_records
        ]
        if self.rank != 0:
            return
        counts_by_class = Counter()
        counts_by_rank: dict[str, dict[str, int]] = {}
        counts_by_state: dict[str, dict[str, int]] = {}
        neurons: dict[str, dict[str, Any]] = {}
        extracellular_agents: dict[str, list[str]] = {}
        for record in records:
            agent_class = record["agent_class"]
            rank = str(record["rank"])
            state = record.get("initial_state") or "None"
            counts_by_class[agent_class] += 1
            counts_by_rank.setdefault(rank, {})
            counts_by_rank[rank][agent_class] = counts_by_rank[rank].get(agent_class, 0) + 1
            counts_by_state.setdefault(agent_class, {})
            counts_by_state[agent_class][state] = counts_by_state[agent_class].get(state, 0) + 1
            if agent_class == "Neuron":
                neurons[record["uid"]] = {
                    "uid": record["uid"],
                    "rank": record["rank"],
                    "initial_state": record["initial_state"],
                    "position": record["position"],
                    "internal_agents": {}
                }
                extracellular_agents.setdefault(agent_class, []).append(record["uid"])
            elif record.get("owner_uid") and record["owner_uid"] in neurons:
                neurons[record["owner_uid"]].setdefault("internal_agents", {}).setdefault(agent_class, []).append(record["uid"])
            elif not record.get("owner_uid"):
                extracellular_agents.setdefault(agent_class, []).append(record["uid"])
        with self.agents_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(safe_serialize(record), ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "run_id": self.run_id,
            "logger_schema_version": self.schema_version,
            "counts": {
                "total_agents": len(records),
                "by_class": dict(counts_by_class),
                "by_rank": counts_by_rank,
                "by_initial_state": counts_by_state
            },
            "neurons": neurons,
            "extracellular_agents": extracellular_agents,
            "config_summary": {
                "description": "Full per-agent configs are stored in initialization_agents.jsonl."
            },
        }
        summary = {
            "run_id": self.run_id,
            "total_agents": len(records),
            "counts_by_class": dict(counts_by_class),
            "counts_by_rank": counts_by_rank,
            "counts_by_initial_state": counts_by_state,
            "counts_by_compartment": _count_records(records, lambda record: record.get("compartment") or "None"),
            "counts_by_owner_neuron": _count_records(records, lambda record: record.get("owner_uid") or "None")
        }
        self.manifest_path.write_text(json.dumps(safe_serialize(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        self.summary_path.write_text(json.dumps(safe_serialize(summary), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _update_manifest(self, record: InitializationAgentRecord) -> None:
        self._counts_by_class[record.agent_class] = self._counts_by_class.get(record.agent_class, 0) + 1
        self._counts_by_rank.setdefault(str(record.rank), {})
        self._counts_by_rank[str(record.rank)][record.agent_class] = self._counts_by_rank[str(record.rank)].get(record.agent_class, 0) + 1
        state = record.initial_state or "None"
        self._counts_by_state.setdefault(record.agent_class, {})
        self._counts_by_state[record.agent_class][state] = self._counts_by_state[record.agent_class].get(state, 0) + 1
        if record.agent_class == "Neuron":
            self._neurons[record.uid] = {
                "uid": record.uid,
                "rank": record.rank,
                "initial_state": record.initial_state,
                "position": record.position,
                "internal_agents": {}
            }
            self._extracellular_agents.setdefault(record.agent_class, []).append(record.uid)
        elif record.owner_uid and record.owner_uid in self._neurons:
            internal = self._neurons[record.owner_uid].setdefault("internal_agents", {})
            internal.setdefault(record.agent_class, []).append(record.uid)
        elif not record.owner_uid:
            self._extracellular_agents.setdefault(record.agent_class, []).append(record.uid)

    def _count_by(self, key_fn) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records:
            key = str(key_fn(record))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _barrier(self) -> None:
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier):
            barrier()

    def _allgather(self, records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        allgather = getattr(self.comm, "allgather", None)
        if callable(allgather):
            return allgather(records)
        return [records]


def safe_serialize(value):
    """Serialize dataclasses and Repast objects without recursion."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {field.name: safe_serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(safe_serialize(key)): safe_serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_serialize(item) for item in value]
    point = point_dict(value)
    if point is not None:
        return point
    if hasattr(value, "__dict__") and not hasattr(value, "uid"):
        return {key: safe_serialize(item) for key, item in vars(value).items() if not key.startswith("_")}
    if hasattr(value, "uid"):
        return {
            "uid": uid_of(value),
            "class": type(value).__name__,
            "state": value_of(getattr(value, "state", None)),
        }
    return str(value)


def point_dict(point) -> Optional[dict[str, int]]:
    if point is None:
        return None
    if hasattr(point, "x") and hasattr(point, "y"):
        return {
            "x": int(getattr(point, "x")),
            "y": int(getattr(point, "y")),
            "z": int(getattr(point, "z", 0))
        }
    return None


def _count_records(records: list[dict[str, Any]], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(key_fn(record))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _local_id(agent) -> Optional[int]:
    uid = getattr(agent, "uid", None)
    if isinstance(uid, tuple) and uid:
        return uid[0]
    local_id = getattr(agent, "local_id", None)
    return local_id if isinstance(local_id, int) else None
