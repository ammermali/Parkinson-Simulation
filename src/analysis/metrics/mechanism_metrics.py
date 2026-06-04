from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

LYSOSOME_DEGRADATION_MECHANISMS = {"lysosome_degradation_success", "lysosome_degradation_failure", "lysosome_overwhelmed_by_target"}

ALPHA_AGGREGATION_MECHANISMS = {"alpha_misfolding", "alpha_oligomerization_intention", "alpha_added_to_aggregate", "aggregate_merge", "aggregate_matures_to_lewy_body"}

DEFAULT_SIMULATION_LOG_DIR = Path("output/simulation/logs") #TODO
DEFAULT_ANALYSIS_OUTPUT = Path("output/analysis/mechanism_metrics_latest.json") #TODO


class NumericSummary:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None

    def add(self, value) -> None:
        if not isinstance(value, (int, float)):
            return
        number = float(value)
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def as_dict(self) -> dict:
        if self.count == 0:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {"count": self.count, "mean": self.total / self.count, "min": self.minimum, "max": self.maximum}


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def summarize_mechanisms(output_dir: Path, include_by_tick: bool = True) -> dict:
    node_paths = _log_paths(output_dir, "g0_nodes")
    edge_paths = _log_paths(output_dir, "g0_edges")
    metrics = _new_metrics(include_by_tick)
    for node in _iter_many(node_paths):
        _count_general_node(metrics, node)
    for edge in _iter_many(edge_paths):
        _count_general_edge(metrics, edge)
        _count_alpha_mechanism(metrics, edge)
        _count_lysosome_mechanism(metrics, edge)
        _count_mitochondrion_mechanism(metrics, edge)
        _count_neuron_mechanism(metrics, edge)
        _count_glial_mechanism(metrics, edge)
    return _finalize_metrics(output_dir, edge_paths, metrics, include_by_tick)


def _new_metrics(include_by_tick: bool) -> dict:
    return {
        "total_edges": 0,
        "mechanism_counts": Counter(),
        "relation_counts": Counter(),
        "outcomes_by_mechanism": defaultdict(Counter),
        "target_types_by_mechanism": defaultdict(Counter),
        "source_types_by_mechanism": defaultdict(Counter),
        "source_target_pairs_by_mechanism": defaultdict(Counter),
        "probabilities_by_mechanism": defaultdict(NumericSummary),
        "rng_values_by_mechanism": defaultdict(NumericSummary),
        "first_tick_by_mechanism": {},
        "last_tick_by_mechanism": {},
        "by_tick": defaultdict(Counter) if include_by_tick else None,
        "alpha": {
            "aggregation_groups": set(),
            "aggregate_targets_touched": set(),
            "member_additions_by_group": defaultdict(int),
            "members_added_by_aggregate": Counter()
        },
        "lysosome": {
            "degradation_attempts": Counter(),
            "degradation_by_target_type": defaultdict(Counter),
            "degradation_by_outcome": Counter(),
            "target_registrations": Counter(),
            "target_assignments": Counter()
        },
        "mitochondrion": {"lifecycle_transitions": Counter()},
        "neuron": {"state_transitions": Counter(), "actions": Counter()},
        "glia": {"microglia_actions": Counter(), "astrocyte_actions": Counter()},
        "nodes": {
            "initial_alpha_states": Counter(),
            "aggregate_state_observations": Counter(),
            "aggregate_uids": set(),
            "lewy_body_aggregate_uids": set(),
            "aggregate_max_size_by_uid": {},
            "lewy_body_max_size_by_uid": {}
        }
    }


def _count_general_node(metrics: dict, node: dict) -> None:
    if node.get("kind") == "aggregate" and node.get("agent_type") == "AlphaAggregate":
        state = node.get("state") or "unknown"
        uid = node.get("uid") or node.get("g1_key") or "unknown"
        size = _numeric_value(node.get("value"))
        nodes = metrics["nodes"]
        nodes["aggregate_state_observations"][state] += 1
        nodes["aggregate_uids"].add(uid)
        if size is not None:
            previous = nodes["aggregate_max_size_by_uid"].get(uid, 0.0)
            nodes["aggregate_max_size_by_uid"][uid] = max(previous, size)
        if state == "LewyBody":
            nodes["lewy_body_aggregate_uids"].add(uid)
            if size is not None:
                previous = nodes["lewy_body_max_size_by_uid"].get(uid, 0.0)
                nodes["lewy_body_max_size_by_uid"][uid] = max(previous, size)
        return
    if node.get("kind") != "agent_state":
        return
    if node.get("agent_type") != "AlphaSynuclein":
        return
    if _tick(node) != 0:
        return
    metrics["nodes"]["initial_alpha_states"][node.get("state") or "unknown"] += 1


def _count_general_edge(metrics: dict, edge: dict) -> None:
    mechanism = edge.get("mechanism") or "unknown"
    relation = edge.get("relation") or "unknown"
    target_type = edge.get("target_type") or "unknown"
    source_type = edge.get("source_type") or "unknown"
    outcome = edge.get("outcome") or "unknown"
    tick = _tick(edge)
    metrics["total_edges"] += 1
    metrics["mechanism_counts"][mechanism] += 1
    metrics["relation_counts"][relation] += 1
    metrics["outcomes_by_mechanism"][mechanism][outcome] += 1
    metrics["target_types_by_mechanism"][mechanism][target_type] += 1
    metrics["source_types_by_mechanism"][mechanism][source_type] += 1
    metrics["source_target_pairs_by_mechanism"][mechanism][f"{source_type}->{target_type}"] += 1
    metrics["probabilities_by_mechanism"][mechanism].add(edge.get("probability"))
    metrics["rng_values_by_mechanism"][mechanism].add(edge.get("rng_value"))
    metrics["first_tick_by_mechanism"].setdefault(mechanism, tick)
    metrics["last_tick_by_mechanism"][mechanism] = tick
    if metrics["by_tick"] is not None:
        metrics["by_tick"][str(tick)][mechanism] += 1


def _count_alpha_mechanism(metrics: dict, edge: dict) -> None:
    mechanism = edge.get("mechanism")
    alpha = metrics["alpha"]
    if mechanism == "alpha_added_to_aggregate":
        group_key = _aggregation_group_key(edge)
        alpha["aggregation_groups"].add(group_key)
        alpha["aggregate_targets_touched"].add(edge.get("target_uid"))
        alpha["member_additions_by_group"][group_key] += 1
        alpha["members_added_by_aggregate"][edge.get("target_uid") or "unknown"] += 1
    elif mechanism in ALPHA_AGGREGATION_MECHANISMS:
        return


def _count_lysosome_mechanism(metrics: dict, edge: dict) -> None:
    mechanism = edge.get("mechanism")
    target_type = edge.get("target_type") or "unknown"
    target_state = edge.get("target_state") or "unknown"
    lysosome = metrics["lysosome"]
    if mechanism == "neuron_registers_degradation_target":
        lysosome["target_registrations"][target_type] += 1
        return

    if mechanism == "lysosome_selects_degradation_target":
        lysosome["target_assignments"][target_type] += 1
        return

    if edge.get("relation") != "degradation":
        return

    if mechanism not in LYSOSOME_DEGRADATION_MECHANISMS:
        return

    if mechanism == "lysosome_degradation_success":
        result = "success"
    elif mechanism == "lysosome_degradation_failure":
        result = "failure"
    else:
        result = "overwhelmed"

    lysosome["degradation_attempts"][result] += 1
    lysosome["degradation_by_target_type"][target_type][result] += 1
    lysosome["degradation_by_outcome"][edge.get("outcome") or "unknown"] += 1
    if mechanism == "lysosome_overwhelmed_by_target" and target_state == "LewyBody":
        lysosome["degradation_attempts"]["overwhelmed_by_lewy_body"] += 1


def _count_mitochondrion_mechanism(metrics: dict, edge: dict) -> None:
    if edge.get("source_type") != "Mitochondrion":
        return
    if edge.get("relation") != "state_transition":
        return
    source_state = edge.get("source_state") or "unknown"
    target_state = edge.get("target_state") or "unknown"
    metrics["mitochondrion"]["lifecycle_transitions"][f"{source_state}->{target_state}"] += 1


def _count_neuron_mechanism(metrics: dict, edge: dict) -> None:
    if edge.get("target_type") != "Neuron":
        return
    if edge.get("relation") == "state_transition":
        source_state = edge.get("source_state") or "unknown"
        target_state = edge.get("target_state") or "unknown"
        metrics["neuron"]["state_transitions"][f"{source_state}->{target_state}"] += 1
    elif edge.get("mechanism") == "neuron_state_action_policy":
        action = edge.get("target_state") or "unknown"
        metrics["neuron"]["actions"][action] += 1


def _count_glial_mechanism(metrics: dict, edge: dict) -> None:
    if edge.get("mechanism") == "microglia_state_action_policy":
        metrics["glia"]["microglia_actions"][edge.get("target_state") or "unknown"] += 1
    elif edge.get("mechanism") == "astrocyte_state_action_policy":
        metrics["glia"]["astrocyte_actions"][edge.get("target_state") or "unknown"] += 1


def _finalize_metrics(output_dir: Path, edge_paths: list[Path], metrics: dict, include_by_tick: bool) -> dict:
    alpha = metrics["alpha"]
    lysosome = metrics["lysosome"]
    nodes = metrics["nodes"]
    aggregate_sizes = list(nodes["aggregate_max_size_by_uid"].values())
    lewy_body_sizes = list(nodes["lewy_body_max_size_by_uid"].values())
    degradation_total = sum(
        lysosome["degradation_attempts"][key]
        for key in ("success", "failure", "overwhelmed")
    )
    success_count = lysosome["degradation_attempts"]["success"]
    overwhelm_count = lysosome["degradation_attempts"]["overwhelmed"]

    report = {
        "output_dir": str(output_dir),
        "input_edge_files": [str(path) for path in edge_paths],
        "total_edges": metrics["total_edges"],
        "selected_mechanisms": {
            "alpha_synuclein": {
                "initial_state_nodes": {
                    "total": sum(nodes["initial_alpha_states"].values()),
                    "by_state": _counter_dict(nodes["initial_alpha_states"]),
                    "free_monomer_misfolded": (
                        nodes["initial_alpha_states"]["Monomer"]
                        + nodes["initial_alpha_states"]["Misfolded"]
                    ),
                    "cleared": nodes["initial_alpha_states"]["Cleared"],
                    "lewy_body_members": nodes["initial_alpha_states"]["LewyBody"]
                },
                "aggregate_nodes": {
                    "unique_aggregates_observed": len(nodes["aggregate_uids"]),
                    "unique_lewy_body_aggregates_observed": len(nodes["lewy_body_aggregate_uids"]),
                    "state_observations": _counter_dict(nodes["aggregate_state_observations"]),
                    "mean_max_size": _mean(aggregate_sizes),
                    "lewy_body_size_summary": _number_list_summary(lewy_body_sizes)
                },
                "misfolding_events": metrics["mechanism_counts"]["alpha_misfolding"],
                "oligomerization_intentions": metrics["mechanism_counts"]["alpha_oligomerization_intention"],
                "aggregate_member_additions": metrics["mechanism_counts"]["alpha_added_to_aggregate"],
                "aggregation_events_inferred": len(alpha["aggregation_groups"]),
                "aggregate_targets_touched": len([uid for uid in alpha["aggregate_targets_touched"] if uid is not None]),
                "members_added_by_aggregate": _counter_dict(alpha["members_added_by_aggregate"]),
                "aggregate_merges": metrics["mechanism_counts"]["aggregate_merge"],
                "lewy_body_maturations": metrics["mechanism_counts"]["aggregate_matures_to_lewy_body"],
            },
            "lysosome": {
                "targets_registered": {
                    "total": sum(lysosome["target_registrations"].values()),
                    "by_target_type": _counter_dict(lysosome["target_registrations"]),
                },
                "successful_target_claims": {
                    "total": sum(lysosome["target_assignments"].values()),
                    "by_target_type": _counter_dict(lysosome["target_assignments"]),
                    "note": "Counts lysosome_selects_degradation_target only; neuron_assigns_degradation_target is the same claim seen from the neuron buffer.",
                },
                "degradation_attempts": {
                    "total": degradation_total,
                    "success": success_count,
                    "failure": lysosome["degradation_attempts"]["failure"],
                    "overwhelmed": overwhelm_count,
                    "overwhelmed_by_lewy_body": lysosome["degradation_attempts"]["overwhelmed_by_lewy_body"],
                    "success_rate": _ratio(success_count, degradation_total),
                    "overwhelm_rate": _ratio(overwhelm_count, degradation_total),
                    "by_target_type": _nested_counter_dict(lysosome["degradation_by_target_type"]),
                    "by_outcome": _counter_dict(lysosome["degradation_by_outcome"])
                }
            },
            "mitochondrion": {
                "lifecycle_transitions": _counter_dict(metrics["mitochondrion"]["lifecycle_transitions"]),
                "fusion_repairs": metrics["mechanism_counts"]["mitochondrion_fusion_repair"],
                "lysosome_repairs": metrics["mechanism_counts"]["mitochondrion_lysosome_repair"],
                "stress_releases": metrics["mechanism_counts"]["mitochondrion_stress_release"],
                "damaged_stress_releases": metrics["mechanism_counts"]["damaged_mitochondrion_stress_release"],
                "damaged_debris_releases": metrics["mechanism_counts"]["damaged_mitochondrion_debris_release"]
            },
            "neuron": {
                "state_transitions": _counter_dict(metrics["neuron"]["state_transitions"]),
                "actions": _counter_dict(metrics["neuron"]["actions"]),
                "debris_dumps": metrics["mechanism_counts"]["neuron_dump_debris"],
                "dopamine_releases": metrics["mechanism_counts"]["neuron_dopamine_release"],
                "stress_inflammation_releases": metrics["mechanism_counts"]["neuron_stress_inflammation_release"]
            },
            "glia": {
                "microglia_actions": _counter_dict(metrics["glia"]["microglia_actions"]),
                "astrocyte_actions": _counter_dict(metrics["glia"]["astrocyte_actions"]),
                "microglia_debris_clearance": metrics["mechanism_counts"]["microglia_debris_clearance"],
                "microglia_inflammation_release": metrics["mechanism_counts"]["microglia_inflammation_release"],
                "astrocyte_support_inflammation_reduction": metrics["mechanism_counts"]["astrocyte_support_inflammation_reduction"],
                "astrocyte_reactive_inflammation_release": metrics["mechanism_counts"]["astrocyte_reactive_inflammation_release"],
            },
        },
        "all_mechanisms": {
            "counts": _counter_dict(metrics["mechanism_counts"]),
            "first_tick": dict(sorted(metrics["first_tick_by_mechanism"].items())),
            "last_tick": dict(sorted(metrics["last_tick_by_mechanism"].items())),
            "outcomes": _nested_counter_dict(metrics["outcomes_by_mechanism"]),
            "source_types": _nested_counter_dict(metrics["source_types_by_mechanism"]),
            "target_types": _nested_counter_dict(metrics["target_types_by_mechanism"]),
            "source_target_pairs": _nested_counter_dict(metrics["source_target_pairs_by_mechanism"]),
            "probability_summary": _numeric_summary_dict(metrics["probabilities_by_mechanism"]),
            "rng_summary": _numeric_summary_dict(metrics["rng_values_by_mechanism"]),
        },
        "coverage_notes": [
            "Lewy body formation, lysosome success/failure/overwhelming and target claims are counted from explicit causal edges.",
            "aggregation_events_inferred groups alpha_added_to_aggregate edges by tick, owner and aggregate target; member additions remain the exact logged count.",
            "Actual alpha release and absorption are currently visible mainly through neuron action-selection edges unless dedicated transfer edges are added to the runtime logger.",
        ]
    }
    if include_by_tick:
        report["by_tick"] = {
            tick: _counter_dict(counter)
            for tick, counter in sorted(metrics["by_tick"].items(), key=lambda item: int(item[0]))
        }
    return report


def _aggregation_group_key(edge: dict) -> tuple:
    return (_tick(edge), edge.get("owner_uid"), edge.get("target_uid"))

def _log_paths(output_dir: Path, stem: str) -> list[Path]:
    merged = output_dir / f"{stem}.jsonl"
    if merged.exists() and merged.stat().st_size > 0:
        return [merged]
    return sorted(output_dir.glob(f"{stem}_rank*.jsonl"), key=_rank_file_sort_key)

def _iter_many(paths: list[Path]) -> Iterable[dict]:
    for path in paths:
        yield from iter_jsonl(path)


def _rank_file_sort_key(path: Path) -> tuple[int, str]:
    marker = "_rank"
    if marker not in path.stem:
        return (0, path.name)
    suffix = path.stem.split(marker, 1)[1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (0, path.name)


def _tick(edge: dict) -> int:
    try:
        return int(edge.get("tick") or 0)
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _number_list_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "values": []}
    ordered = sorted(values)
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered), "min": ordered[0], "max": ordered[-1], "values": ordered}


def _numeric_value(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _counter_dict(counter: Counter) -> dict:
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _nested_counter_dict(counters: dict[str, Counter]) -> dict:
    return {
        str(key): _counter_dict(counter)
        for key, counter in sorted(counters.items(), key=lambda item: str(item[0]))
    }


def _numeric_summary_dict(summaries: dict[str, NumericSummary]) -> dict:
    return {
        str(key): summary.as_dict()
        for key, summary in sorted(summaries.items(), key=lambda item: str(item[0]))
        if summary.count > 0
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count biological mechanisms from G0 causal trace logs.")
    parser.add_argument("output_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    parser.add_argument("--no-by-tick", action="store_true", help="Omit per-tick mechanism counts from the report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT, help="JSON destination for the report.")
    parser.add_argument("--stdout", action="store_true", help="Print the report instead of writing output/analysis.")
    args = parser.parse_args()
    report = {
        "runs": [
            summarize_mechanisms(output_dir, include_by_tick=not args.no_by_tick)
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
