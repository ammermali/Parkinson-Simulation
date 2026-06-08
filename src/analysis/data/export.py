from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
from src.analysis.data.run_data import RunData

DEFAULT_FORMATS = ("jsonl", "csv")
EVENT_COLUMNS = ("schema_version", "run_id", "event_id", "tick", "phase",
    "rank", "event_type", "mechanism", "rule_id", "actor_uid", "actor_type",
    "actor_state", "actor_state_before", "actor_state_after", "target_uid",
    "target_type", "target_state", "target_state_before", "target_state_after",
    "effect_count", "outcome", "actor", "target", "effects", "stochastic", "context")
EVENT_EFFECT_COLUMNS = ("run_id", "event_id", "tick", "phase", "rank", "event_type",
    "mechanism", "rule_id", "actor_uid", "actor_type", "target_uid", "target_type", "effect_index",
    "field", "scope", "delta", "unit", "value", "owner_uid", "compartment", "outcome")
SPATIAL_COLUMNS = ("run_id", "tick", "rank", "uid", "agent_class",
    "state", "x", "y", "z", "compartment", "owner_uid", "aggregate_id")
COUNT_COLUMNS = ("tick", "agent_type", "state", "count")
AGENT_COUNT_COLUMNS = {
    "neurons_healthy": ("Neuron", "Healthy"),
    "neurons_compromised": ("Neuron", "Compromised"),
    "neurons_apoptotic": ("Neuron", "Apoptotic"),
    "neurons_ruptures": ("Neuron", "Ruptured"),
    "free_alpha": ("AlphaSynuclein", "Free"),
    "alpha_aggregate": ("AlphaAggregate", "MemberProteins")}


def export_run_tables(log_dir: Path | str, *, output_dir: Path | str | None = None, formats: Sequence[str] = DEFAULT_FORMATS) -> dict[str, Path]:
    run_data = RunData.resolve(log_dir, required_stems=("events",))
    destination = Path(output_dir) if output_dir is not None else run_data.default_table_dir
    destination.mkdir(parents=True, exist_ok=True)
    events = list(run_data.iter_events())
    event_rows = [event_table_row(event) for event in events]
    event_effect_rows = list(event_effect_table_rows(events))
    tick_metrics = run_data.tick_metrics_rows()
    tables: dict[str, tuple[list[dict[str, Any]], tuple[str, ...] | None]] = {
        "events": ([select_columns(row, EVENT_COLUMNS) for row in event_rows], EVENT_COLUMNS),
        "event_effects": ([select_columns(row, EVENT_EFFECT_COLUMNS) for row in event_effect_rows], EVENT_EFFECT_COLUMNS),
        "spatial_snapshots": ([select_columns(row, SPATIAL_COLUMNS) for row in run_data.iter_spatial_snapshots()], SPATIAL_COLUMNS),
        "tick_metrics": (tick_metrics, None),
        "agent_counts_by_tick": (agent_counts_by_tick(tick_metrics), COUNT_COLUMNS)
    }
    produced: dict[str, Path] = {}
    requested = {item.lower() for item in formats}
    unknown = requested - set(DEFAULT_FORMATS)
    if unknown:
        raise ValueError(f"Unsupported export format(s): {', '.join(sorted(unknown))}")
    for table_name, (rows, columns) in tables.items():
        if "jsonl" in requested:
            path = destination / f"{table_name}.jsonl"
            write_jsonl(path, rows)
            produced[f"{table_name}.jsonl"] = path
        if "csv" in requested:
            path = destination / f"{table_name}.csv"
            write_csv(path, rows, columns)
            produced[f"{table_name}.csv"] = path
    return produced


def event_table_row(event: dict[str, Any]) -> dict[str, Any]:
    context = as_dict(event.get("context"))
    actor = as_dict(event.get("actor"))
    target = as_dict(event.get("target"))
    effects = as_list(event.get("effects"))
    return {
        **event,
        "rule_id": event.get("rule_id") or context.get("rule_id"),
        "actor_uid": actor.get("uid"),
        "actor_type": actor.get("type"),
        "actor_state": actor.get("state") or actor.get("state_after") or actor.get("state_before"),
        "actor_state_before": actor.get("state_before"),
        "actor_state_after": actor.get("state_after"),
        "target_uid": target.get("uid"),
        "target_type": target.get("type"),
        "target_state": target.get("state") or target.get("state_after") or target.get("state_before"),
        "target_state_before": target.get("state_before"),
        "target_state_after": target.get("state_after"),
        "effect_count": len([effect for effect in effects if isinstance(effect, dict)])
    }


def event_effect_table_rows(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for event in events:
        context = as_dict(event.get("context"))
        actor = as_dict(event.get("actor"))
        target = as_dict(event.get("target"))
        for index, raw_effect in enumerate(as_list(event.get("effects"))):
            if not isinstance(raw_effect, dict):
                continue
            yield {
                "run_id": event.get("run_id"),
                "event_id": event.get("event_id"),
                "tick": event.get("tick"),
                "phase": event.get("phase"),
                "rank": event.get("rank"),
                "event_type": event.get("event_type"),
                "mechanism": event.get("mechanism"),
                "rule_id": event.get("rule_id") or context.get("rule_id"),
                "actor_uid": actor.get("uid"),
                "actor_type": actor.get("type"),
                "target_uid": target.get("uid"),
                "target_type": target.get("type"),
                "effect_index": index,
                "field": raw_effect.get("field"),
                "scope": raw_effect.get("scope"),
                "delta": raw_effect.get("delta"),
                "unit": raw_effect.get("unit"),
                "value": raw_effect.get("value"),
                "owner_uid": raw_effect.get("owner_uid") or actor.get("owner_uid") or target.get("owner_uid"),
                "compartment": raw_effect.get("compartment") or actor.get("compartment") or target.get("compartment"),
                "outcome": event.get("outcome") or as_dict(event.get("stochastic")).get("outcome")
            }


def agent_counts_by_tick(tick_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tick_row in tick_rows:
        tick = tick_row.get("tick")
        for column, (agent_type, state) in AGENT_COUNT_COLUMNS.items():
            rows.append({"tick": tick,
                        "agent_type": agent_type,
                        "state": state,
                        "count": tick_row.get(column)})
    return rows


def select_columns(row: dict[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return {column: row.get(column) for column in columns}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns is not None else infer_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key)) for key in fieldnames})


def infer_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
