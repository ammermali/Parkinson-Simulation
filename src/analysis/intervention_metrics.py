from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ENV_FIELDS = ("inflammation_level", "extracellular_debris", "dopamine_output")
STATE_AGENTS = ("Neuron", "Astrocyte", "Microglia", "AlphaSynuclein", "AlphaAggregate")
EVENTS = (
    "alpha_misfolding",
    "alpha_added_to_aggregate",
    "aggregate_matures_to_lewy_body",
    "lysosome_degradation_success",
    "lysosome_degradation_failure",
    "lysosome_overwhelmed_by_target",
    "neuron_dump_debris",
    "microglia_inflammation_release",
    "astrocyte_reactive_inflammation_release"
)

DEFAULT_SIMULATION_LOG_DIR = Path("output/simulation/logs")
DEFAULT_ANALYSIS_OUTPUT = Path("output/analysis/intervention_metrics_latest.json")
TICK_METRIC_ENV_FIELDS = {"inflammation": "inflammation_level", "debris": "extracellular_debris", "dopamine": "dopamine_output"}
TICK_METRIC_NEURON_STATES = {
    "neurons_healthy": "Healthy",
    "neurons_compromised": "Compromised",
    "neurons_apoptotic": "Apoptotic",
    "neurons_ruptures": "Ruptured",
    "neurons_ruptured": "Ruptured"
}

def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists() or is_git_lfs_pointer(path):
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def first_crossing(values: dict[int, float], threshold: float, direction: str) -> int | None:
    for tick in sorted(values):
        value = values[tick]
        if direction == "above" and value >= threshold:
            return tick
        if direction == "below" and value <= threshold:
            return tick
    return None


def summarize_run(output_dir: Path) -> dict:
    requested_output_dir = Path(output_dir)
    output_dir, path_warnings = resolve_run_dir(requested_output_dir)
    fields: dict[str, dict[int, list[float]]] = {
        field: defaultdict(list)
        for field in ENV_FIELDS
    }
    state_counts: dict[str, dict[int, Counter]] = {
        agent: defaultdict(Counter)
        for agent in STATE_AGENTS
    }
    event_counts = Counter()
    node_paths = _log_paths(output_dir, "g0_nodes")
    edge_paths = _log_paths(output_dir, "g0_edges")
    tick_metrics_path = find_tick_metrics_path(output_dir, requested_output_dir)
    warnings = list(path_warnings)
    if not node_paths:
        warnings.append(
            "No readable G0 node JSONL files were found. Environment and state metrics will use tick_metrics.csv when available."
        )
    if not edge_paths:
        warnings.append("No readable G0 edge JSONL files were found. Event counts may be empty.")
    for node in _iter_many(node_paths):
        tick = int(node.get("tick") or 0)
        if node.get("kind") == "env_field" and node.get("phase") == "5_commit":
            field = node.get("field")
            value = node.get("value")
            if field in fields and isinstance(value, (int, float)):
                fields[field][tick].append(float(value))
        if node.get("kind") == "agent_state" and node.get("agent_type") in state_counts:
            state_counts[node["agent_type"]][tick][node.get("state")] += 1
    for edge in _iter_many(edge_paths):
        mechanism = edge.get("mechanism")
        if mechanism in EVENTS:
            event_counts[mechanism] += 1
    tick_rows = read_tick_metrics(tick_metrics_path)
    if tick_rows:
        _merge_tick_metric_fields(fields, tick_rows)
        _merge_tick_metric_state_counts(state_counts, tick_rows)
    elif not node_paths:
        warnings.append("tick_metrics.csv was not found, so scalar and compact state metrics could not be recovered.")
    field_means = {
        field: {
            tick: sum(values) / len(values)
            for tick, values in by_tick.items()
            if values
        }
        for field, by_tick in fields.items()
    }
    return {
        "output_dir": str(output_dir),
        "requested_output_dir": str(requested_output_dir),
        "input_status": {
            "g0_node_files": [str(path) for path in node_paths],
            "g0_edge_files": [str(path) for path in edge_paths],
            "tick_metrics_csv": str(tick_metrics_path) if tick_metrics_path is not None else None,
            "warnings": warnings
        },
        "first_crossings": {
            "inflammation_ge_0_5": first_crossing(field_means["inflammation_level"], 0.5, "above"),
            "inflammation_ge_0_9": first_crossing(field_means["inflammation_level"], 0.9, "above"),
            "debris_ge_0_5": first_crossing(field_means["extracellular_debris"], 0.5, "above"),
            "debris_ge_0_9": first_crossing(field_means["extracellular_debris"], 0.9, "above"),
            "dopamine_le_0_5": first_crossing(field_means["dopamine_output"], 0.5, "below"),
            "dopamine_le_0_25": first_crossing(field_means["dopamine_output"], 0.25, "below")
        },
        "final_environment": {
            field: field_means[field][max(field_means[field])] if field_means[field] else None
            for field in ENV_FIELDS
        },
        "event_counts": dict(event_counts),
        "state_counts_by_tick": {
            agent: {
                str(tick): dict(counter)
                for tick, counter in sorted(by_tick.items())
            }
            for agent, by_tick in state_counts.items()
        }
    }

def _log_paths(output_dir: Path, stem: str) -> list[Path]:
    merged = output_dir / f"{stem}.jsonl"
    if looks_like_jsonl(merged):
        return [merged]
    return [
        path
        for path in sorted(output_dir.glob(f"{stem}_rank*.jsonl"))
        if looks_like_jsonl(path)
    ]


def _iter_many(paths: list[Path]) -> Iterable[dict]:
    for path in paths:
        yield from iter_jsonl(path)


def resolve_run_dir(requested: Path) -> tuple[Path, list[str]]:
    requested = Path(requested)
    candidates = _run_dir_candidates(requested)
    for candidate in candidates:
        if _has_readable_g0(candidate):
            if candidate != requested:
                return candidate, [f"Requested output dir {requested} did not contain readable G0 logs; using {candidate}."]
            return candidate, []
    for candidate in candidates:
        if find_tick_metrics_path(candidate, requested) is not None:
            if candidate != requested:
                return candidate, [f"Requested output dir {requested} had no readable G0 logs; using tick metrics near {candidate}."]
            return candidate, []
    return requested, [f"No readable G0 logs or tick_metrics.csv were found near {requested}."]


def _run_dir_candidates(requested: Path) -> list[Path]:
    candidates = [requested]
    if requested.name == "logs":
        candidates.append(requested.with_name("log"))
    elif requested.name == "log":
        candidates.append(requested.with_name("logs"))
    else:
        candidates.extend([requested / "logs", requested / "log"])
    return _unique_paths(candidates)


def _has_readable_g0(path: Path) -> bool:
    return bool(_log_paths(path, "g0_nodes") or _log_paths(path, "g0_edges"))


def looks_like_jsonl(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0 or is_git_lfs_pointer(path):
        return False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError:
                return False
            return True
    return False


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        first_line = path.open("r", encoding="utf-8", errors="replace").readline().strip()
    except OSError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def find_tick_metrics_path(*bases: Path) -> Path | None:
    candidates: list[Path] = []
    for base in bases:
        if base is None:
            continue
        base = Path(base)
        candidates.append(base / "tick_metrics.csv")
        if base.name in {"log", "logs"}:
            candidates.append(base.parent / "tick_metrics.csv")
        candidates.append(base / "logs" / "tick_metrics.csv")
        candidates.append(base / "log" / "tick_metrics.csv")
    for candidate in _unique_paths(candidates):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def read_tick_metrics(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _merge_tick_metric_fields(fields: dict[str, dict[int, list[float]]], rows: list[dict[str, str]]) -> None:
    for row in rows:
        tick = _row_tick(row)
        if tick is None:
            continue
        for column, field in TICK_METRIC_ENV_FIELDS.items():
            value = _float_or_none(row.get(column))
            if value is not None:
                fields[field][tick].append(value)


def _merge_tick_metric_state_counts(state_counts: dict[str, dict[int, Counter]], rows: list[dict[str, str]]) -> None:
    for row in rows:
        tick = _row_tick(row)
        if tick is None:
            continue
        for column, state in TICK_METRIC_NEURON_STATES.items():
            value = _int_or_none(row.get(column))
            if value is not None:
                state_counts["Neuron"][tick][state] += value
        free_alpha = _int_or_none(row.get("free_alpha"))
        if free_alpha is not None:
            state_counts["AlphaSynuclein"][tick]["Free"] += free_alpha
        alpha_aggregate = _int_or_none(row.get("alpha_aggregate"))
        if alpha_aggregate is not None:
            state_counts["AlphaAggregate"][tick]["MemberProteins"] += alpha_aggregate


def _row_tick(row: dict[str, str]) -> int | None:
    try:
        return int(float(row.get("tick", "")))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    parsed = _float_or_none(value)
    return None if parsed is None else int(parsed)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize simulation logs for intervention analysis.")
    parser.add_argument("output_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT, help="JSON destination for the report.")
    parser.add_argument("--stdout", action="store_true", help="Print the report instead of writing output/analysis.")
    args = parser.parse_args()
    report = {
        "runs": [
            summarize_run(output_dir)
            for output_dir in args.output_dirs
        ]
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.stdout:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
