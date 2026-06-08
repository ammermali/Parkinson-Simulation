from __future__ import annotations
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


EVENT_TYPES = {"state_transition", "threshold_trigger", "field_change", "aggregation", "degradation"}


@dataclass(frozen=True)
class Event:
    event_id: str
    tick: int
    rank: int
    event_type: str
    mechanism: str
    actor: Optional[dict[str, Any]]
    target: Optional[dict[str, Any]]
    effects: list[dict[str, Any]]
    stochastic: Optional[dict[str, Any]]
    outcome: Optional[str]
    context: dict[str, Any]

class EventLogger:
    def __init__(self, rank: int, comm=None, output_dir: Path | str = "output/run_logs", rank_output_dir: Path | str | None = None, enabled: bool = False, model_version: Optional[str] = None, run_id: Optional[str] = None):
        self.rank = rank
        self.comm = comm
        self.output_dir = Path(output_dir)
        self.rank_output_dir = resolve_rank_output_dir(self.output_dir, rank_output_dir)
        self.enabled = enabled
        self.model_version = model_version
        self.run_id = run_id
        self.current_tick = 0
        self._event_index = 0
        self.path = self.rank_output_dir / f"events_rank{self.rank}.jsonl"
        self.merged_path = self.output_dir / "events.jsonl"
        self.metadata_path = self.output_dir / "event_log_metadata.json"
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank_output_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        if self.rank == 0:
            self.merged_path.write_text("", encoding="utf-8")
            self.metadata_path.write_text(
                json.dumps(
                    compact_dict(
                        {
                            "run_id": self.run_id,
                            "model_version": self.model_version,
                            "format": "semantic_events_jsonl",
                            "event_types": sorted(EVENT_TYPES),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        self._barrier()

    def set_tick(self, tick: int) -> None:
        self.current_tick = int(tick)

    def state_transition(
        self,
        agent,
        from_state: Any,
        to_state: Any,
        mechanism: str,
        rule_id: Optional[str] = None,
        probability: Optional[float] = None,
        rng_value: Optional[float] = None,
        outcome: Optional[str] = "transitioned",
        owner=None,
        compartment=None
    ) -> None:
        if value_of(from_state) == value_of(to_state):
            return
        self.record_event(
            event_type="state_transition",
            mechanism=mechanism,
            actor=agent_ref(agent, state_before=from_state, state_after=to_state, owner=owner, compartment=compartment),
            stochastic=stochastic_info(probability=probability, rng_value=rng_value, outcome=outcome),
            outcome=outcome,
            context=compact_dict({"rule_id": rule_id})
        )

    def threshold_trigger(
        self,
        source,
        target_agent,
        target_state: Any,
        mechanism: str,
        rule_id: Optional[str] = None,
        description: Optional[str] = None,
        owner=None,
        compartment=None
    ) -> None:
        self.record_event(
            event_type="threshold_trigger",
            mechanism=mechanism,
            actor=field_ref(source),
            target=agent_ref(target_agent, state_after=target_state, owner=owner, compartment=compartment),
            outcome="triggered",
            context=compact_dict({"rule_id": rule_id, "description": description})
        )

    def internal_field_node(self, owner, field: str, stage: str, value: Any, **context: Any) -> dict[str, Any]:
        return compact_dict(
            {
                "uid": f"{uid_of(owner) or 'unknown'}:{field}:{stage}",
                "type": "InternalField",
                "field": field,
                "value": value,
                "scope": "agent_internal",
                "owner_uid": uid_of(owner),
                "compartment": "Intracellular",
                **context,
            }
        )

    def aggregate_snapshot(self, aggregate, aggregate_id: Optional[int] = None, owner=None, **context: Any) -> None:
        self.record_event(
            event_type="aggregation",
            mechanism="aggregate_snapshot",
            actor=agent_ref(
                aggregate,
                state=getattr(aggregate, "state", None),
                owner=owner,
                compartment="Intracellular" if owner is not None else getattr(aggregate, "compartment", None),
                extra=compact_dict(
                    {
                        "aggregate_id": aggregate_id or getattr(aggregate, "aggregate_id", None),
                        "size": getattr(aggregate, "size", None),
                    }
                ),
            ),
            outcome="snapshot",
            context=compact_dict(context)
        )

    def field_effect(
        self,
        agent,
        action,
        field: str,
        effect_value: float,
        mechanism: str,
        unit: Optional[str] = None
    ) -> None:
        self.field_change(
            agent,
            field=field,
            delta=effect_value,
            mechanism=mechanism,
            unit=unit,
            action=action,
            scope="environment"
        )

    def internal_field_effect(
        self,
        agent,
        owner,
        field: str,
        effect_value: float,
        mechanism: str,
        action=None,
        unit: Optional[str] = None
    ) -> None:
        self.field_change(
            agent,
            field=field,
            delta=effect_value,
            mechanism=mechanism,
            unit=unit,
            action=action or getattr(agent, "pending_action", None),
            scope="agent_internal",
            owner=owner,
            compartment="Intracellular"
        )

    def field_change(
        self,
        agent,
        *,
        field: str,
        delta: float,
        mechanism: str,
        unit: Optional[str] = None,
        action: Any = None,
        scope: str = "environment",
        owner=None,
        compartment: Any = None,
    ) -> None:
        if delta == 0:
            return
        self.record_event(
            event_type="field_change",
            mechanism=mechanism,
            actor=agent_ref(agent, state=getattr(agent, "state", None), owner=owner, compartment=compartment),
            effects=[
                compact_dict(
                    {
                        "kind": "field_delta",
                        "scope": scope,
                        "field": canonical_field_name(field),
                        "delta": delta,
                        "unit": unit
                    }
                )
            ],
            outcome="applied",
            context=compact_dict({"action": value_of(action)})
        )

    def aggregation(
        self,
        source_agent,
        aggregate_agent,
        mechanism: str,
        aggregate_id: Optional[int] = None,
        owner=None,
        outcome: Optional[str] = "aggregated",
    ) -> None:
        self.record_event(
            event_type="aggregation",
            mechanism=mechanism,
            actor=agent_ref(
                source_agent,
                state=getattr(source_agent, "state", None),
                owner=owner,
                compartment="Intracellular" if owner is not None else getattr(source_agent, "compartment", None)
            ),
            target=agent_ref(
                aggregate_agent,
                state=getattr(aggregate_agent, "state", None),
                owner=owner,
                compartment="Intracellular" if owner is not None else getattr(aggregate_agent, "compartment", None),
                extra=compact_dict(
                    {
                        "aggregate_id": aggregate_id or getattr(aggregate_agent, "aggregate_id", None),
                        "size": getattr(aggregate_agent, "size", None)
                    }
                )
            ),
            outcome=outcome
        )

    def degradation(
        self,
        lysosome,
        target_agent,
        mechanism: str,
        outcome: str,
        probability: Optional[float] = None,
        rng_value: Optional[float] = None,
        owner=None,
    ) -> None:
        self.record_event(
            event_type="degradation",
            mechanism=mechanism,
            actor=agent_ref(lysosome, state=getattr(lysosome, "state", None), owner=owner, compartment="Intracellular" if owner is not None else getattr(lysosome, "compartment", None)),
            target=agent_ref(target_agent, state=getattr(target_agent, "state", None), owner=owner, compartment="Intracellular" if owner is not None else getattr(target_agent, "compartment", None)),
            stochastic=stochastic_info(probability=probability, rng_value=rng_value, outcome=outcome),
            outcome=outcome
        )

    def record_event(
        self,
        *,
        event_type: str,
        mechanism: str,
        actor: Optional[dict[str, Any]] = None,
        target: Optional[dict[str, Any]] = None,
        effects: Optional[list[dict[str, Any]]] = None,
        stochastic: Optional[dict[str, Any]] = None,
        outcome: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[Event]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Invalid event type: {event_type}")
        event = Event(
            event_id=self.event_id(),
            tick=self.current_tick,
            rank=self.rank,
            event_type=event_type,
            mechanism=mechanism,
            actor=actor,
            target=target,
            effects=effects or [],
            stochastic=stochastic,
            outcome=outcome,
            context=context or {}
        )
        self._write_event(event)
        return event

    def event_id(self) -> str:
        self._event_index += 1
        return f"event_{self.rank}_{self.current_tick}_{self._event_index}"

    def close(self) -> None:
        if not self.enabled:
            return
        self._barrier()
        if self.rank == 0:
            self._merge_rank_files("events_rank*.jsonl", self.merged_path, source_dir=self.rank_output_dir)
        self._barrier()

    def _write_event(self, event: Event) -> None:
        if not self.enabled:
            return
        row = compact_dict(event.__dict__)
        self._append_jsonl(self.path, row)
        if self.rank == 0:
            self._append_jsonl(self.merged_path, row)

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _merge_rank_files(self, pattern: str, destination: Path, *, source_dir: Path) -> None:
        with destination.open("w", encoding="utf-8") as output:
            for path in sorted(source_dir.glob(pattern), key=_rank_file_sort_key):
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        if line.strip():
                            output.write(line if line.endswith("\n") else line + "\n")

    def _barrier(self) -> None:
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier):
            barrier()


def agent_ref(agent, *, state: Any = None, state_before: Any = None, state_after: Any = None, owner=None, compartment: Any = None, extra: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    if agent is None:
        return None
    row = {
        "uid": uid_of(agent),
        "type": type_name(agent),
        "state": value_of(state),
        "state_before": value_of(state_before),
        "state_after": value_of(state_after),
        "owner_uid": uid_of(owner),
        "compartment": value_of(compartment or getattr(agent, "compartment", None))
    }
    if extra:
        row.update(extra)
    return compact_dict(row)


def field_ref(source) -> Optional[dict[str, Any]]:
    if source is None:
        return None
    if isinstance(source, dict):
        return compact_dict(
            {
                "uid": source.get("uid"),
                "type": source.get("type"),
                "field": canonical_field_name(source.get("field")),
                "value": source.get("value"),
                "scope": source.get("scope"),
                "owner_uid": source.get("owner_uid"),
                "compartment": source.get("compartment")
            }
        )
    return compact_dict(
        {
            "uid": getattr(source, "uid", None),
            "type": getattr(source, "agent_type", None),
            "field": canonical_field_name(getattr(source, "field", None)),
            "value": getattr(source, "value", None),
            "scope": "environment" if getattr(source, "uid", None) == "SN" else "agent_internal",
            "owner_uid": getattr(source, "owner_uid", None),
            "compartment": getattr(source, "compartment", None)
        }
    )


def stochastic_info(*, probability: Optional[float] = None, rng_value: Optional[float] = None, outcome: Optional[str] = None) -> Optional[dict[str, Any]]:
    row = compact_dict({"probability": probability, "rng_value": rng_value, "outcome": outcome})
    return row or None


def compact_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: compact_value(value)
        for key, value in row.items()
        if value is not None and value != {} and value != []
    }


def compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return compact_dict(value)
    if isinstance(value, list):
        return [compact_value(item) for item in value if item is not None]
    return value_of(value) if isinstance(value, Enum) else value


def canonical_field_name(field: Any) -> Optional[str]:
    value = value_of(field)
    if value is None:
        return None
    return value.removesuffix("_buffer")


def uid_of(agent) -> Optional[str]:
    if agent is None:
        return None
    uid = getattr(agent, "uid", None)
    if uid is not None:
        if isinstance(uid, tuple):
            return ":".join(str(item) for item in uid)
        return str(uid)
    local_id = getattr(agent, "local_id", getattr(agent, "id", ""))
    ptype = getattr(agent, "ptype", getattr(agent, "type_id", ""))
    rank = getattr(agent, "rank", "")
    if local_id == "" and ptype == "" and rank == "":
        return None
    return f"{local_id}:{ptype}:{rank}"


def type_name(agent) -> Optional[str]:
    if agent is None:
        return None
    return type(agent).__name__


def value_of(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _rank_file_sort_key(path: Path) -> tuple[int, str]:
    marker = "_rank"
    if marker not in path.stem:
        return (0, path.name)
    suffix = path.stem.rsplit(marker, 1)[1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (0, path.name)


def resolve_rank_output_dir(output_dir: Path, rank_output_dir: Path | str | None) -> Path:
    if rank_output_dir is not None:
        return Path(rank_output_dir)
    if output_dir.name == "run_logs":
        return output_dir.parent / "logs_per_rank"
    return output_dir / "logs_per_rank"
