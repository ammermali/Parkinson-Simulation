from __future__ import annotations
import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from src.analysis.data.run_data import RunData

DEFAULT_SIMULATION_LOG_DIR = Path("output/run_logs")
DEFAULT_GRAPH_OUTPUT_DIR = Path("output/graphs")
DEFAULT_GRAPH_OUTPUT = Path("output/graphs/g0.gexf")
DEFAULT_GRAPH_LITE_OUTPUT = Path("output/graphs/g0.lite.gext")
MAX_GEPHI_ATTRIBUTE_CHARS = 4096
GEPHI_DROPPED_PROVENANCE_ATTRS = {"source_log_node_ids", "node_origins", "edge_ids", "relations", "mechanisms","outcomes"}

FIELD_KINDS = {"env_field", "internal_field", "buffer"}
AGENT_KINDS = {"agent_state", "aggregate"}
ALPHA_AGENT_TYPES = {"AlphaSynuclein", "AlphaAggregate"}


@dataclass
class G0ProjectionStats:
    event_count: int = 0
    projected_event_count: int = 0
    skipped_event_count: int = 0
    event_type_counts: dict[str, int] = field(default_factory=dict)
    projected_relation_counts: dict[str, int] = field(default_factory=dict)
    skipped_event_type_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class G0SnapshotStats:
    row_count: int = 0
    projected_node_count: int = 0
    skipped_row_count: int = 0


@dataclass
class G0BuildReport:
    log_dir: str
    event_paths: list[str]
    snapshot_paths: list[str]
    event_count: int
    event_type_counts: dict[str, int]
    projected_event_count: int
    skipped_event_count: int
    skipped_event_type_counts: dict[str, int]
    projected_relation_counts: dict[str, int]
    snapshot_row_count: int
    projected_snapshot_node_rows: int
    skipped_snapshot_rows: int
    projected_node_rows: int
    projected_edge_rows: int
    node_rows_in_range: int = 0
    node_rows_skipped_by_tick: int = 0
    node_rows_skipped_invalid_endpoint: int = 0
    node_rows_skipped_by_lifecycle: int = 0
    node_rows_merged: int = 0
    edge_rows_in_range: int = 0
    edge_rows_skipped_by_tick: int = 0
    edge_rows_skipped_invalid_endpoint: int = 0
    edge_rows_skipped_by_lifecycle: int = 0
    edge_rows_skipped_disallowed: int = 0
    edge_rows_skipped_invalid: int = 0
    causal_edge_rows_merged: int = 0
    continuity_edges_added: int = 0
    node_count: int = 0
    edge_count: int = 0
    add_continuity: bool = True
    include_field_continuity: bool = True
    include_snapshot_nodes: bool = True
    rupture_grace_ticks: int = 2
    start_tick: Optional[int] = None
    end_tick: Optional[int] = None
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_dir": self.log_dir,
            "event_paths": self.event_paths,
            "snapshot_paths": self.snapshot_paths,
            "event_count": self.event_count,
            "event_type_counts": dict(sorted(self.event_type_counts.items())),
            "projected_event_count": self.projected_event_count,
            "skipped_event_count": self.skipped_event_count,
            "skipped_event_type_counts": dict(sorted(self.skipped_event_type_counts.items())),
            "projected_relation_counts": dict(sorted(self.projected_relation_counts.items())),
            "snapshot_row_count": self.snapshot_row_count,
            "projected_snapshot_node_rows": self.projected_snapshot_node_rows,
            "skipped_snapshot_rows": self.skipped_snapshot_rows,
            "projected_node_rows": self.projected_node_rows,
            "projected_edge_rows": self.projected_edge_rows,
            "node_rows_in_range": self.node_rows_in_range,
            "node_rows_skipped_by_tick": self.node_rows_skipped_by_tick,
            "node_rows_skipped_invalid_endpoint": self.node_rows_skipped_invalid_endpoint,
            "node_rows_skipped_by_lifecycle": self.node_rows_skipped_by_lifecycle,
            "node_rows_merged": self.node_rows_merged,
            "edge_rows_in_range": self.edge_rows_in_range,
            "edge_rows_skipped_by_tick": self.edge_rows_skipped_by_tick,
            "edge_rows_skipped_invalid_endpoint": self.edge_rows_skipped_invalid_endpoint,
            "edge_rows_skipped_by_lifecycle": self.edge_rows_skipped_by_lifecycle,
            "edge_rows_skipped_disallowed": self.edge_rows_skipped_disallowed,
            "edge_rows_skipped_invalid": self.edge_rows_skipped_invalid,
            "causal_edge_rows_merged": self.causal_edge_rows_merged,
            "continuity_edges_added": self.continuity_edges_added,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "add_continuity": self.add_continuity,
            "include_field_continuity": self.include_field_continuity,
            "include_snapshot_nodes": self.include_snapshot_nodes,
            "rupture_grace_ticks": self.rupture_grace_ticks,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "output_paths": dict(sorted(self.output_paths.items()))}


@dataclass
class G0BuildResult:
    graph: Any
    report: G0BuildReport
    output_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class LifecycleFilters:
    ruptured_cutoff_by_neuron: dict[str, int] = field(default_factory=dict)
    cleared_cutoff_by_alpha: dict[str, int] = field(default_factory=dict)

    def is_after_rupture_grace(self, owner_uid: Optional[str], tick: Optional[int]) -> bool:
        if owner_uid is None or tick is None:
            return False
        cutoff = self.ruptured_cutoff_by_neuron.get(owner_uid)
        return cutoff is not None and tick > cutoff

    def is_after_alpha_clearance(self, uid: Optional[str], tick: Optional[int]) -> bool:
        if uid is None or tick is None:
            return False
        cutoff = self.cleared_cutoff_by_alpha.get(uid)
        return cutoff is not None and tick > cutoff


def build_g0(log_dir: Path | str = DEFAULT_SIMULATION_LOG_DIR, *, add_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, strict: bool = False, start_tick: Optional[int] = None, end_tick: Optional[int] = None):
    return build_g0_with_report(log_dir,
        add_continuity=add_continuity, include_field_continuity=include_field_continuity, include_snapshot_nodes=include_snapshot_nodes,
        rupture_grace_ticks=rupture_grace_ticks, strict=strict, start_tick=start_tick, end_tick=end_tick).graph

def build_g0_with_report(log_dir: Path | str = DEFAULT_SIMULATION_LOG_DIR, *, add_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, strict: bool = False, start_tick: Optional[int] = None, end_tick: Optional[int] = None, verbose: bool = False, logger: Optional[Callable[[str], None]] = None) -> G0BuildResult:
    nx = require_networkx()
    log = logger or print
    if verbose:
        log(f"[G0] Resolving event logs from {log_dir}")
    run_data = RunData.resolve(log_dir, required_stems=("events",))
    log_dir = run_data.log_dir
    event_paths = run_data.event_paths
    if verbose:
        log(f"[G0] Event files: {', '.join(str(path) for path in event_paths)}")
        log("[G0] Loading semantic events")
    events = list(run_data.iter_events(strict=strict))
    snapshot_paths = run_data.spatial_snapshot_paths
    snapshots = list(run_data.iter_spatial_snapshots(strict=strict)) if include_snapshot_nodes else []
    if verbose:
        log(f"[G0] Loaded {len(events)} events")
        if include_snapshot_nodes:
            log(f"[G0] Loaded {len(snapshots)} spatial snapshot rows")
    event_node_rows, edge_rows, projection = project_events_to_g0_rows(events)
    snapshot_node_rows, snapshot_stats = snapshots_to_g0_node_rows(snapshots)
    node_rows = [*snapshot_node_rows, *event_node_rows]
    report = G0BuildReport(
        log_dir=str(log_dir),
        event_paths=[str(path) for path in event_paths],
        snapshot_paths=[str(path) for path in snapshot_paths] if include_snapshot_nodes else [],
        event_count=projection.event_count,
        event_type_counts=projection.event_type_counts,
        projected_event_count=projection.projected_event_count,
        skipped_event_count=projection.skipped_event_count,
        skipped_event_type_counts=projection.skipped_event_type_counts,
        projected_relation_counts=projection.projected_relation_counts,
        snapshot_row_count=snapshot_stats.row_count,
        projected_snapshot_node_rows=snapshot_stats.projected_node_count,
        skipped_snapshot_rows=snapshot_stats.skipped_row_count,
        projected_node_rows=len(node_rows),
        projected_edge_rows=len(edge_rows),
        add_continuity=add_continuity,
        include_field_continuity=include_field_continuity,
        include_snapshot_nodes=include_snapshot_nodes,
        rupture_grace_ticks=rupture_grace_ticks,
        start_tick=start_tick,
        end_tick=end_tick)
    if verbose:
        log(f"[G0] Projection: {report.projected_event_count} events -> {report.projected_node_rows} node rows, {report.projected_edge_rows} edge rows")
    filters = collect_lifecycle_filters_from_rows(node_rows, edge_rows, rupture_grace_ticks=rupture_grace_ticks)
    graph = nx.DiGraph(
        name="G0",
        level="G0",
        source_log_dir=str(log_dir),
        built_from="events+snapshots" if include_snapshot_nodes else "events",
        event_log_files=[str(path) for path in event_paths],
        snapshot_log_files=[str(path) for path in snapshot_paths] if include_snapshot_nodes else [],
        event_count=len(events),
        snapshot_row_count=len(snapshots),
        rupture_grace_ticks=rupture_grace_ticks,
        start_tick=start_tick if start_tick is not None else "",
        end_tick=end_tick if end_tick is not None else "")
    if verbose:
        log("[G0] Merging nodes")
    for row in node_rows:
        if not row_tick_in_range(row, start_tick, end_tick):
            report.node_rows_skipped_by_tick += 1
            continue
        report.node_rows_in_range += 1
        endpoint = endpoint_from_node_row(row)
        if endpoint is None:
            report.node_rows_skipped_invalid_endpoint += 1
            continue
        node_id, attrs = endpoint
        if attrs.get("node_origin") != "snapshot" and should_skip_endpoint(attrs, filters):
            report.node_rows_skipped_by_lifecycle += 1
            continue
        report.node_rows_merged += 1
        merge_node(graph, node_id, attrs)
    if verbose:
        log(f"[G0] Nodes after merge: {graph.number_of_nodes()}")
        log("[G0] Merging causal edges")
    for row in edge_rows:
        if not row_tick_in_range(row, start_tick, end_tick):
            report.edge_rows_skipped_by_tick += 1
            continue
        report.edge_rows_in_range += 1
        status = merge_causal_edge(graph, row, filters)
        if status == "merged":
            report.causal_edge_rows_merged += 1
        elif status == "invalid_row":
            report.edge_rows_skipped_invalid += 1
        elif status == "invalid_endpoint":
            report.edge_rows_skipped_invalid_endpoint += 1
        elif status == "lifecycle_filtered":
            report.edge_rows_skipped_by_lifecycle += 1
        elif status == "disallowed":
            report.edge_rows_skipped_disallowed += 1
    if add_continuity:
        if verbose:
            log("[G0] Adding continuity edges")
        report.continuity_edges_added = add_continuity_edges(graph, include_fields=include_field_continuity)
    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["edge_count"] = graph.number_of_edges()
    graph.graph["continuity_enabled"] = add_continuity
    graph.graph["field_continuity_enabled"] = include_field_continuity
    graph.graph["snapshot_nodes_enabled"] = include_snapshot_nodes
    graph.graph["projected_event_count"] = report.projected_event_count
    graph.graph["projected_snapshot_node_rows"] = report.projected_snapshot_node_rows
    graph.graph["skipped_event_count"] = report.skipped_event_count
    graph.graph["causal_edge_rows_merged"] = report.causal_edge_rows_merged
    graph.graph["continuity_edges_added"] = report.continuity_edges_added
    report.node_count = graph.number_of_nodes()
    report.edge_count = graph.number_of_edges()
    if verbose:
        log(f"[G0] Built graph: nodes={report.node_count} edges={report.edge_count}")
    return G0BuildResult(graph=graph, report=report)


def build_g0_exports(log_dir: Path | str = DEFAULT_SIMULATION_LOG_DIR, *, output_dir: Path | str = DEFAULT_GRAPH_OUTPUT_DIR, add_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, strict: bool = False, start_tick: Optional[int] = None, end_tick: Optional[int] = None, verbose: bool = False, logger: Optional[Callable[[str], None]] = None) -> G0BuildResult:
    log = logger or print
    output_dir = Path(output_dir)
    result = build_g0_with_report(log_dir,
        add_continuity=add_continuity,
        include_field_continuity=include_field_continuity,
        include_snapshot_nodes=include_snapshot_nodes,
        rupture_grace_ticks=rupture_grace_ticks,
        strict=strict, start_tick=start_tick, end_tick=end_tick,
        verbose=verbose, logger=logger)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.output_paths["g0_gephi"] = write_g0_gexf(result.graph, output_dir / "g0.gexf")
    result.output_paths["g0_lite_gephi"] = write_g0_lite_gexf(result.graph, output_dir / "g0.lite.gext")
    result.report.output_paths = {key: str(path) for key, path in result.output_paths.items()}
    if verbose:
        for key, path in result.output_paths.items():
            log(f"[G0] Wrote {key}: {path}")
    return result

def build_g0_graph(*args, **kwargs):
    return build_g0(*args, **kwargs)


def require_networkx():
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("G0 graph construction requires `networkx`. Install project dependencies with `pip install -r requirements.txt`.") from exc
    return nx


def events_to_g0_rows(events: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes, edges, _ = project_events_to_g0_rows(events)
    return nodes, edges


def project_events_to_g0_rows(events: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], G0ProjectionStats]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    stats = G0ProjectionStats()
    for event in events:
        stats.event_count += 1
        event_type = str(event.get("event_type") or "unknown")
        increment_count(stats.event_type_counts, event_type)
        projected = project_event(event)
        if projected is None:
            stats.skipped_event_count += 1
            increment_count(stats.skipped_event_type_counts, event_type)
            continue
        source, target, edge = projected
        nodes[source["node_id"]] = source
        nodes[target["node_id"]] = target
        edges.append(edge)
        stats.projected_event_count += 1
        increment_count(stats.projected_relation_counts, edge.get("relation") or "unknown")
    return list(nodes.values()), edges, stats


def snapshots_to_g0_node_rows(snapshots: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], G0SnapshotStats]:
    nodes: dict[str, dict[str, Any]] = {}
    stats = G0SnapshotStats()
    for snapshot in snapshots:
        stats.row_count += 1
        row = snapshot_to_g0_node_row(snapshot)
        if row is None:
            stats.skipped_row_count += 1
            continue
        nodes[row["node_id"]] = row
        stats.projected_node_count += 1
    return list(nodes.values()), stats


def snapshot_to_g0_node_row(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    tick = int_or_none(snapshot.get("tick"))
    uid = value_of(snapshot.get("uid"))
    agent_type = value_of(snapshot.get("agent_class"))
    if tick is None or not uid or not agent_type:
        return None
    state = value_of(snapshot.get("state")) or "unknown"
    node_id = agent_node_id(agent_type, uid, state, tick)
    return {
        "run_id": snapshot.get("run_id"),
        "tick": tick,
        "rank": snapshot.get("rank"),
        "node_id": node_id,
        "kind": "agent_state",
        "uid": uid,
        "agent_type": agent_type,
        "state": state,
        "field": None,
        "value": None,
        "level": snapshot_level(snapshot),
        "owner_uid": snapshot.get("owner_uid"),
        "compartment": snapshot.get("compartment"),
        "x": int_or_none(snapshot.get("x")),
        "y": int_or_none(snapshot.get("y")),
        "z": int_or_none(snapshot.get("z")),
        "aggregate_id": snapshot.get("aggregate_id"),
        "source_log_node_id": f"snapshot:{uid}:{tick}",
        "node_origin": "snapshot"
    }


def project_event(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    event_type = event.get("event_type")
    if event_type == "state_transition":
        actor = event.get("actor") or {}
        source = projected_agent_node(event, actor, state=actor.get("state_before"))
        target = projected_agent_node(event, actor, state=actor.get("state_after"))
        return source, target, projected_edge_row(event, source, target, relation="state_transition")
    if event_type == "threshold_trigger":
        source = projected_field_node(event, event.get("actor") or {})
        target = projected_agent_node(event, event.get("target") or {}, state=(event.get("target") or {}).get("state_after"))
        return source, target, projected_edge_row(event, source, target, relation="threshold_trigger")
    if event_type == "field_change":
        actor = event.get("actor") or {}
        effect = first_effect(event)
        source = projected_agent_node(event, actor, state=actor.get("state"))
        target = projected_field_node(event, effect, owner_uid=actor.get("owner_uid"))
        relation = "field_effect" if effect.get("scope") == "environment" else "internal_field_effect"
        return source, target, projected_edge_row(event, source, target, relation=relation, effect=effect)
    if event_type == "aggregation":
        source = projected_agent_node(event, event.get("actor") or {})
        target_ref = event.get("target") or event.get("actor") or {}
        target = projected_aggregate_node(event, target_ref)
        return source, target, projected_edge_row(event, source, target, relation="aggregation")
    if event_type == "degradation":
        source = projected_agent_node(event, event.get("actor") or {})
        target = projected_agent_node(event, event.get("target") or {})
        return source, target, projected_edge_row(event, source, target, relation="degradation")
    return None


def projected_agent_node(event: dict[str, Any], ref: dict[str, Any], *, state: Any = None) -> dict[str, Any]:
    agent_type = ref.get("type")
    uid = ref.get("uid")
    state_value = value_of(state if state is not None else ref.get("state"))
    node_id = projected_node_id_for(kind="agent_state", uid=uid, agent_type=agent_type, state=state_value, field=None, event=event)
    return projected_base_node(event, node_id=node_id, kind="agent_state", ref=ref, state=state_value)


def projected_aggregate_node(event: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    state_value = value_of(ref.get("state") or ref.get("state_after"))
    node_id = projected_node_id_for(kind="aggregate", uid=ref.get("uid"), agent_type=ref.get("type"), state=state_value, field=None, event=event)
    row = projected_base_node(event, node_id=node_id, kind="aggregate", ref=ref, state=state_value)
    row["value"] = ref.get("size")
    return row


def projected_field_node(event: dict[str, Any], ref: dict[str, Any], *, owner_uid: Any = None) -> dict[str, Any]:
    field = ref.get("field")
    scope = ref.get("scope")
    uid = "SN" if scope == "environment" else ref.get("uid") or owner_uid
    agent_type = "SubstantiaNigra" if uid == "SN" else "Neuron"
    kind = "env_field" if uid == "SN" else "internal_field"
    node_id = projected_node_id_for(kind=kind, uid=uid, agent_type=agent_type, state=None, field=field, event=event)
    row = projected_base_node(
        event,
        node_id=node_id,
        kind=kind,
        ref={"uid": uid, "type": agent_type, "owner_uid": owner_uid or ref.get("owner_uid"), "compartment": ref.get("compartment")},
        state=None
    )
    row["field"] = field
    row["value"] = ref.get("delta", ref.get("value"))
    row["level"] = "environment" if uid == "SN" else "macro"
    return row


def projected_base_node(event: dict[str, Any], *, node_id: str, kind: str, ref: dict[str, Any], state: Any) -> dict[str, Any]:
    return {
        "run_id": event.get("run_id"),
        "tick": int_or_none(event.get("tick")) or 0,
        "rank": event.get("rank"),
        "node_id": node_id,
        "kind": kind,
        "uid": ref.get("uid"),
        "agent_type": ref.get("type"),
        "state": value_of(state),
        "field": None,
        "value": None,
        "level": "macro" if ref.get("type") == "Neuron" else "intracellular" if ref.get("owner_uid") else "extracellular",
        "owner_uid": ref.get("owner_uid"),
        "compartment": ref.get("compartment"),
        "node_origin": "event"
    }


def projected_edge_row(event: dict[str, Any], source: dict[str, Any], target: dict[str, Any], *, relation: str, effect: dict[str, Any] | None = None) -> dict[str, Any]:
    effect = effect or {}
    stochastic = event.get("stochastic") or {}
    return {
        "run_id": event.get("run_id"),
        "tick": int_or_none(event.get("tick")) or 0,
        "rank": event.get("rank"),
        "edge_id": event.get("event_id"),
        "source_node_id": source.get("node_id"),
        "target_node_id": target.get("node_id"),
        "source_kind": source.get("kind"),
        "target_kind": target.get("kind"),
        "source_uid": source.get("uid"),
        "target_uid": target.get("uid"),
        "source_type": source.get("agent_type"),
        "target_type": target.get("agent_type"),
        "source_state": source.get("state"),
        "target_state": target.get("state"),
        "source_field": source.get("field"),
        "target_field": target.get("field"),
        "relation": relation,
        "mechanism": event.get("mechanism"),
        "rule_id": event.get("rule_id") or (event.get("context") or {}).get("rule_id"),
        "effect_value": effect.get("delta"),
        "effect_unit": effect.get("unit"),
        "predicate": (event.get("context") or {}).get("predicate"),
        "probability": stochastic.get("probability"),
        "rng_value": stochastic.get("rng_value"),
        "outcome": event.get("outcome") or stochastic.get("outcome"),
        "compartment": source.get("compartment") or target.get("compartment"),
        "owner_uid": source.get("owner_uid") or target.get("owner_uid"),
        "valid": True
    }


def projected_node_id_for(*, kind: str, uid: Any, agent_type: Any, state: Any, field: Any, event: dict[str, Any]) -> str:
    if kind in {"env_field", "internal_field"}:
        base = f"{agent_type or 'Field'}_{uid or 'unknown'}.{field or 'field'}"
    else:
        base = f"{agent_type or 'Agent'}_{uid or 'unknown'}.{state or 'unknown'}"
    return f"{base}@{int_or_none(event.get('tick')) or 0}"


def first_effect(event: dict[str, Any]) -> dict[str, Any]:
    effects = event.get("effects") or []
    first = effects[0] if effects else {}
    return first if isinstance(first, dict) else {}


def write_g0_gexf(graph, output_path: Path | str = DEFAULT_GRAPH_OUTPUT) -> Path:
    nx = require_networkx()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(serializable_copy(graph), output_path)
    return output_path


def write_g0_lite_gexf(graph, output_path: Path | str = DEFAULT_GRAPH_LITE_OUTPUT) -> Path:
    nx = require_networkx()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(lite_g0_copy(graph), output_path)
    return output_path


def lite_g0_copy(graph):
    nx = require_networkx()
    copy = nx.DiGraph(name="G0 lite", level="G0", built_from="g0")
    for node_id in graph.nodes():
        copy.add_node(node_id, label=str(node_id))
    for source, target, attrs in graph.edges(data=True):
        relation = str(attrs.get("relation") or attrs.get("causal_kind") or "unknown")
        copy.add_edge(source, target, relation=relation, label=relation)
    return copy


def collect_lifecycle_filters_from_rows(node_rows: Iterable[dict[str, Any]], edge_rows: Iterable[dict[str, Any]], *, rupture_grace_ticks: int) -> LifecycleFilters:
    ruptured_first_tick: dict[str, int] = {}
    cleared_first_tick: dict[str, int] = {}
    def observe(uid: Optional[str], agent_type: Optional[str], state: Optional[str], tick: Optional[int]) -> None:
        if uid is None or tick is None:
            return
        if agent_type == "Neuron" and state == "Ruptured":
            ruptured_first_tick[uid] = min(tick, ruptured_first_tick.get(uid, tick))
        if agent_type == "AlphaSynuclein" and state == "Cleared":
            cleared_first_tick[uid] = min(tick, cleared_first_tick.get(uid, tick))
    for row in node_rows:
        observe(row.get("uid"), row.get("agent_type"), value_of(row.get("state")), int_or_none(row.get("tick")))
    for row in edge_rows:
        observe(row.get("source_uid"), row.get("source_type"), value_of(row.get("source_state")), int_or_none(row.get("tick")))
        observe(row.get("target_uid"), row.get("target_type"), value_of(row.get("target_state")), int_or_none(row.get("tick")))
    return LifecycleFilters(
        ruptured_cutoff_by_neuron={
            uid: first_tick + rupture_grace_ticks
            for uid, first_tick in ruptured_first_tick.items()
        }, cleared_cutoff_by_alpha=cleared_first_tick)


def merge_causal_edge(graph, row: dict[str, Any], filters: LifecycleFilters) -> str:
    if not row.get("valid", True):
        return "invalid_row"
    source = endpoint_from_edge_side(row, "source")
    target = endpoint_from_edge_side(row, "target")
    if source is None or target is None:
        return "invalid_endpoint"
    source_id, source_attrs = source
    target_id, target_attrs = target
    if should_skip_endpoint(source_attrs, filters) or should_skip_endpoint(target_attrs, filters):
        return "lifecycle_filtered"
    if not edge_allowed(row, source_attrs, target_attrs):
        return "disallowed"
    merge_node(graph, source_id, source_attrs)
    merge_node(graph, target_id, target_attrs)
    merge_edge(graph, source_id, target_id, edge_attrs_from_row(row, source_attrs, target_attrs))
    return "merged"


def endpoint_from_node_row(row: dict[str, Any]) -> Optional[tuple[str, dict[str, Any]]]:
    kind = row.get("kind")
    tick = int_or_none(row.get("tick"))
    if tick is None:
        return None
    if kind in FIELD_KINDS:
        node_id = field_node_id(
            field=row.get("field"),
            uid=row.get("uid"),
            agent_type=row.get("agent_type"),
            owner_uid=row.get("owner_uid"),
            tick=tick
        )
        attrs = base_node_attrs(row, tick)
        attrs.update(
            {
                "formal_id": node_id,
                "display_id": node_id,
                "semantic_kind": "environment_field",
                "entity_key": field_entity_key(
                    field=row.get("field"),
                    uid=row.get("uid"),
                    agent_type=row.get("agent_type"),
                    owner_uid=row.get("owner_uid")
                )
            }
        )
        return node_id, attrs

    if kind in AGENT_KINDS:
        node_id = agent_node_id(
            agent_type=row.get("agent_type"),
            uid=row.get("uid"),
            state=value_of(row.get("state")),
            tick=tick
        )
        attrs = base_node_attrs(row, tick)
        attrs.update(
            {
                "formal_id": node_id,
                "display_id": node_id,
                "semantic_kind": "agent_state",
                "entity_key": agent_entity_key(row.get("agent_type"), row.get("uid"))
            }
        )
        return node_id, attrs
    return None


def endpoint_from_edge_side(row: dict[str, Any], side: str) -> Optional[tuple[str, dict[str, Any]]]:
    kind = row.get(f"{side}_kind")
    tick = int_or_none(row.get("tick"))
    if tick is None:
        return None
    if kind in FIELD_KINDS:
        endpoint_row = {
            "agent_type": row.get(f"{side}_type"),
            "uid": row.get(f"{side}_uid"),
            "state": row.get(f"{side}_state"),
            "field": row.get(f"{side}_field"),
            "tick": tick,
            "kind": kind,
            "level": "environment" if row.get(f"{side}_uid") == "SN" else "macro",
            "owner_uid": row.get("owner_uid") or row.get(f"{side}_uid"),
            "compartment": row.get("compartment"),
            "node_id": row.get(f"{side}_node_id"),
            "rank": row.get("rank"),
            "run_id": row.get("run_id"),
            "value": row.get("effect_value") if kind == "buffer" else None,
            "node_origin": "event"
        }
        return endpoint_from_node_row(endpoint_row)
    if kind in AGENT_KINDS:
        endpoint_row = {
            "agent_type": row.get(f"{side}_type"),
            "uid": row.get(f"{side}_uid"),
            "state": row.get(f"{side}_state"),
            "tick": tick,
            "kind": "aggregate" if kind == "aggregate" else "agent_state",
            "level": row.get("compartment") or "unknown",
            "owner_uid": row.get("owner_uid"),
            "compartment": row.get("compartment"),
            "node_id": row.get(f"{side}_node_id"),
            "rank": row.get("rank"),
            "run_id": row.get("run_id"),
            "value": None,
            "node_origin": "event"
        }
        return endpoint_from_node_row(endpoint_row)
    return None


def should_skip_endpoint(attrs: dict[str, Any], filters: LifecycleFilters) -> bool:
    tick = int_or_none(attrs.get("tick"))
    agent_type = attrs.get("agent_type")
    uid = attrs.get("uid")
    owner_uid = attrs.get("owner_uid")
    semantic_kind = attrs.get("semantic_kind")
    if agent_type == "AlphaSynuclein" and filters.is_after_alpha_clearance(uid, tick):
        return True
    if agent_type == "Neuron" and semantic_kind == "agent_state":
        return filters.is_after_rupture_grace(uid, tick)
    if filters.is_after_rupture_grace(owner_uid, tick) and agent_type not in ALPHA_AGENT_TYPES:
        return True
    return False


def edge_allowed(row: dict[str, Any], source_attrs: dict[str, Any], target_attrs: dict[str, Any]) -> bool:
    relation = row.get("relation")
    source_agent = source_attrs.get("semantic_kind") == "agent_state"
    target_agent = target_attrs.get("semantic_kind") == "agent_state"
    if source_agent and target_agent:
        if relation == "degradation":
            return source_attrs.get("agent_type") == "Lysosome"
        if relation == "aggregation":
            return source_attrs.get("agent_type") in ALPHA_AGENT_TYPES and target_attrs.get("agent_type") == "AlphaAggregate"
        if relation == "state_transition":
            return True
        return False
    return True


def add_continuity_edges(graph, *, include_fields: bool) -> int:
    edge_count_before = graph.number_of_edges()
    buckets: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for node_id, attrs in graph.nodes(data=True):
        semantic_kind = attrs.get("semantic_kind")
        if semantic_kind == "environment_field" and not include_fields:
            continue
        if semantic_kind not in {"agent_state", "environment_field"}:
            continue
        tick = int_or_none(attrs.get("tick"))
        entity_key = attrs.get("entity_key")
        if tick is None or entity_key is None:
            continue
        buckets[entity_key][tick].append(node_id)
    for entity_key, by_tick in buckets.items():
        for tick in sorted(by_tick):
            next_tick = tick + 1
            if next_tick not in by_tick:
                continue
            source = continuity_source_at_tick(by_tick[tick])
            target = continuity_target_at_tick(by_tick[next_tick])
            if source == target:
                continue
            merge_edge(
                graph,
                source,
                target,
                {
                    "edge_id": f"continuity:{entity_key}:{tick}->{next_tick}",
                    "relation": "continuity",
                    "mechanism": "temporal_identity",
                    "causal_kind": "continuity",
                    "tick": tick,
                    "first_tick": tick,
                    "last_tick": next_tick,
                    "effect_value": 0.0,
                    "source_kind": graph.nodes[source].get("semantic_kind"),
                    "target_kind": graph.nodes[target].get("semantic_kind")
                }
            )
    return graph.number_of_edges() - edge_count_before

def merge_node(graph, node_id: str, attrs: dict[str, Any]) -> None:
    if not graph.has_node(node_id):
        attrs = dict(attrs)
        attrs["source_log_node_ids"] = as_list(attrs.pop("source_log_node_id", None))
        attrs["node_origins"] = as_list(attrs.pop("node_origin", None))
        attrs["node_origin"] = compact_label(attrs["node_origins"])
        graph.add_node(node_id, **attrs)
        return
    current = graph.nodes[node_id]
    append_unique(current["source_log_node_ids"], attrs.get("source_log_node_id"))
    append_unique(current["node_origins"], attrs.get("node_origin"))
    current["node_origin"] = compact_label(current["node_origins"])
    if current.get("value") is None and attrs.get("value") is not None:
        current["value"] = attrs.get("value")

def merge_edge(graph, source: str, target: str, attrs: dict[str, Any]) -> None:
    effect = number_or_zero(attrs.get("effect_value"))
    tick = int_or_none(attrs.get("tick")) or 0
    first_tick = int_or_none(attrs.get("first_tick")) or tick
    last_tick = int_or_none(attrs.get("last_tick")) or tick
    payload = dict(attrs)
    for key in ("count", "total_effect", "mean_effect", "first_tick", "last_tick", "edge_ids", "relations", "mechanisms", "outcomes"):
        payload.pop(key, None)
    if not graph.has_edge(source, target):
        graph.add_edge(
            source,
            target,
            **payload,
            count=1,
            total_effect=effect,
            mean_effect=effect,
            first_tick=first_tick,
            last_tick=last_tick,
            edge_ids=as_list(attrs.get("edge_id")),
            relations=as_list(attrs.get("relation")),
            mechanisms=as_list(attrs.get("mechanism")),
            outcomes=as_list(attrs.get("outcome"))
        )
        return
    current = graph.edges[source, target]
    current["count"] += 1
    current["total_effect"] += effect
    current["mean_effect"] = current["total_effect"] / current["count"]
    current["first_tick"] = min(current.get("first_tick", first_tick), first_tick)
    current["last_tick"] = max(current.get("last_tick", last_tick), last_tick)
    append_unique(current["edge_ids"], attrs.get("edge_id"))
    append_unique(current["relations"], attrs.get("relation"))
    append_unique(current["mechanisms"], attrs.get("mechanism"))
    append_unique(current["outcomes"], attrs.get("outcome"))
    current["relation"] = compact_label(current["relations"])
    current["mechanism"] = compact_label(current["mechanisms"])
    current["outcome"] = compact_label(current["outcomes"])


def edge_attrs_from_row(row: dict[str, Any], source_attrs: dict[str, Any], target_attrs: dict[str, Any]) -> dict[str, Any]:
    relation = row.get("relation") or "unknown"
    attrs = {
        "edge_id": row.get("edge_id"),
        "relation": relation,
        "mechanism": row.get("mechanism"),
        "causal_kind": causal_kind(row, source_attrs, target_attrs),
        "tick": int_or_none(row.get("tick")),
        "rank": row.get("rank"),
        "run_id": row.get("run_id"),
        "rule_id": row.get("rule_id"),
        "predicate": row.get("predicate"),
        "effect_value": row.get("effect_value"),
        "effect_unit": row.get("effect_unit"),
        "probability": row.get("probability"),
        "rng_value": row.get("rng_value"),
        "outcome": row.get("outcome"),
        "source_log_node_id": row.get("source_node_id"),
        "target_log_node_id": row.get("target_node_id"),
        "raw_source_kind": row.get("source_kind"),
        "raw_target_kind": row.get("target_kind"),
        "raw_source_state": row.get("source_state"),
        "raw_target_state": row.get("target_state"),
        "raw_source_field": row.get("source_field"),
        "raw_target_field": row.get("target_field"),
        "source_kind": source_attrs.get("semantic_kind"),
        "target_kind": target_attrs.get("semantic_kind"),
        "source_type": source_attrs.get("agent_type"),
        "target_type": target_attrs.get("agent_type"),
        "source_uid": source_attrs.get("uid"),
        "target_uid": target_attrs.get("uid"),
        "owner_uid": row.get("owner_uid"),
        "compartment": row.get("compartment"),
        "valid": row.get("valid", True)
    }
    return attrs


def causal_kind(row: dict[str, Any], source_attrs: dict[str, Any], target_attrs: dict[str, Any]) -> str:
    relation = row.get("relation")
    if relation == "threshold_trigger":
        return "threshold_trigger"
    if relation in {"field_effect", "internal_field_effect"}:
        return relation
    if relation == "state_transition":
        return "state_transition"
    if relation == "degradation":
        return "degradation"
    if relation == "aggregation":
        return "aggregation"
    if source_attrs.get("semantic_kind") == "agent_state" and target_attrs.get("semantic_kind") == "agent_state":
        return "agent_relation"
    return relation or "unknown"


def base_node_attrs(row: dict[str, Any], tick: int) -> dict[str, Any]:
    return {
        "source_log_node_id": row.get("node_id"),
        "kind": row.get("kind"),
        "agent_type": row.get("agent_type"),
        "uid": row.get("uid"),
        "state": value_of(row.get("state")),
        "field": row.get("field"),
        "value": row.get("value"),
        "tick": tick,
        "rank": row.get("rank"),
        "run_id": row.get("run_id"),
        "level": row.get("level"),
        "owner_uid": row.get("owner_uid"),
        "compartment": row.get("compartment"),
        "x": row.get("x"),
        "y": row.get("y"),
        "z": row.get("z"),
        "aggregate_id": row.get("aggregate_id"),
        "node_origin": row.get("node_origin")
    }


def agent_node_id(agent_type: Optional[str], uid: Optional[str], state: Optional[str], tick: int) -> str:
    name = clean_token(agent_type or "Agent")
    identifier = clean_token(identity_uid(uid))
    state_name = clean_token(value_of(state) or "unknown")
    return f"{name}_{identifier}_{state_name}@{tick}"


def field_node_id(field: Optional[str], uid: Optional[str], agent_type: Optional[str], owner_uid: Optional[str], tick: int) -> str:
    field_name = clean_token(field or "field")
    if uid == "SN" or agent_type == "SubstantiaNigra":
        return f"SN_{field_name}@{tick}"
    owner = clean_token(identity_uid(owner_uid or uid))
    return f"Neuron_{owner}_{field_name}@{tick}"


def agent_entity_key(agent_type: Optional[str], uid: Optional[str]) -> str:
    return f"agent:{agent_type or 'Agent'}:{uid or 'unknown'}"


def field_entity_key(field: Optional[str], uid: Optional[str], agent_type: Optional[str], owner_uid: Optional[str]) -> str:
    if uid == "SN" or agent_type == "SubstantiaNigra":
        return f"field:SN:{field or 'field'}"
    return f"field:Neuron:{owner_uid or uid or 'unknown'}:{field or 'field'}"


def snapshot_level(snapshot: dict[str, Any]) -> str:
    compartment = value_of(snapshot.get("compartment"))
    if compartment:
        return compartment.lower()
    if snapshot.get("owner_uid"):
        return "intracellular"
    return "extracellular"


def continuity_source_at_tick(node_ids: list[str]) -> str:
    return sorted(node_ids, key=str)[-1]


def continuity_target_at_tick(node_ids: list[str]) -> str:
    return sorted(node_ids, key=str)[0]


def serializable_copy(graph):
    nx = require_networkx()
    copy = nx.DiGraph(**{key: scalar_attr(value) for key, value in graph.graph.items()})
    for node_id, attrs in graph.nodes(data=True):
        copy.add_node(node_id, **export_attrs(attrs))
    for source, target, attrs in graph.edges(data=True):
        copy.add_edge(source, target, **export_attrs(attrs))
    return copy


def export_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in attrs.items():
        if key in GEPHI_DROPPED_PROVENANCE_ATTRS:
            safe[f"{key}_count"] = provenance_count(value)
            continue
        scalar = export_scalar_attr(key, value)
        if scalar == "":
            continue
        safe[key] = scalar
    return safe


def export_scalar_attr(key: str, value: Any) -> Any:
    scalar = scalar_attr(value)
    if isinstance(scalar, str) and len(scalar) > MAX_GEPHI_ATTRIBUTE_CHARS:
        return scalar[:MAX_GEPHI_ATTRIBUTE_CHARS] + "...[truncated]"
    return scalar


def scalar_attr(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value if item is not None)
    return str(value)


def provenance_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        return 0 if not value else len(value.split("|"))
    return 1


def value_of(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def row_tick_in_range(row: dict[str, Any], start_tick: Optional[int], end_tick: Optional[int]) -> bool:
    if start_tick is None and end_tick is None:
        return True
    tick = int_or_none(row.get("tick"))
    if tick is None:
        return False
    if start_tick is not None and tick < start_tick:
        return False
    if end_tick is not None and tick > end_tick:
        return False
    return True


def number_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def identity_uid(uid: Optional[str]) -> str:
    if uid is None:
        return "unknown"
    return str(uid)


def clean_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value))
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"

def as_list(value: Any) -> list[Any]:
    return [] if value is None else [value]

def append_unique(values: list[Any], value: Any) -> None:
    if value is None or value in values:
        return
    values.append(value)


def increment_count(counts: dict[str, int], key: Any) -> None:
    label = str(key or "unknown")
    counts[label] = counts.get(label, 0) + 1


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def format_tick_range(start_tick: Optional[int], end_tick: Optional[int]) -> str:
    if start_tick is None and end_tick is None:
        return "full trace"
    start = str(start_tick) if start_tick is not None else "beginning"
    end = str(end_tick) if end_tick is not None else "end"
    return f"{start}..{end}"


def compact_label(values: list[Any]) -> Optional[str]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return str(cleaned[0])
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G0 from simulation semantic event JSONL logs.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_OUTPUT_DIR)
    parser.add_argument("--no-continuity", action="store_true")
    parser.add_argument("--no-field-continuity", action="store_true")
    parser.add_argument("--no-snapshot-nodes", action="store_true")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2)
    parser.add_argument("--start-tick", type=int)
    parser.add_argument("--end-tick", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        result = build_g0_exports(
            args.log_dir,
            output_dir=args.output_dir,
            add_continuity=not args.no_continuity,
            include_field_continuity=not args.no_field_continuity,
            include_snapshot_nodes=not args.no_snapshot_nodes,
            rupture_grace_ticks=args.rupture_grace_ticks,
            strict=args.strict,
            start_tick=args.start_tick,
            end_tick=args.end_tick,
            verbose=False
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc))

    for path in result.output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
