from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import networkx as nx
    from networkx.readwrite import json_graph
except ModuleNotFoundError:
    nx = None
    json_graph = None


DEFAULT_LOG_DIR = Path("output/simulation/logs")
DEFAULT_GRAPH_DIR = Path("output/analysis/graphs")
DEFAULT_OUTPUT = DEFAULT_GRAPH_DIR / "g0.gexf"

PHASE_INDEX = {
    "0_pre_state": 0,
    "1_perception": 1,
    "2_state_update": 2,
    "3_action_selection": 3,
    "4_effect_buffer": 4,
    "5_commit": 5,
}

FIELD_KINDS = {"env_field", "internal_field", "buffer"}
AGENT_KINDS = {"agent_state", "aggregate"}
ACTION_KIND = "action"


class G0TraceError(RuntimeError):
    """Raised when a strict G0 parse finds malformed trace data."""


def build_g0_graph(
    log_dir: Path | str = DEFAULT_LOG_DIR,
    *,
    collapse_actions: bool = True,
    add_continuity: bool = True,
    multigraph: bool = True,
    strict: bool = False,
) -> nx.MultiDiGraph | nx.DiGraph:
    """Build the G0 causal graph from g0_nodes and g0_edges JSONL logs.

    G0 nodes are temporal biological entities: agent/aggregate states and
    environment or neuron fields. Runtime action nodes produced by the logger
    are collapsed by default so an action becomes an edge attribute rather than
    an entity in the graph.
    """

    require_networkx()
    resolved_log_dir = Path(log_dir)
    graph: nx.MultiDiGraph | nx.DiGraph
    graph = nx.MultiDiGraph() if multigraph else nx.DiGraph()
    graph.graph.update({
        "level": "G0",
        "log_dir": str(resolved_log_dir),
        "collapse_actions": collapse_actions,
        "continuity_edges": add_continuity,
        "node_semantics": "AgentNameID_State@Time or Environment_Field@Time",
    })

    node_paths = log_paths(resolved_log_dir, "g0_nodes")
    edge_paths = log_paths(resolved_log_dir, "g0_edges")
    action_context = collect_action_context(edge_paths, strict=strict) if collapse_actions else {}

    for row in iter_many_jsonl(node_paths, strict=strict):
        attrs = normalize_node_attrs(row, inferred=False)
        if collapse_actions and attrs["kind"] == ACTION_KIND:
            continue
        add_or_update_node(graph, attrs)

    for index, row in enumerate(iter_many_jsonl(edge_paths, strict=strict), 1):
        add_edge_row(
            graph,
            row,
            edge_index=index,
            action_context=action_context,
            collapse_actions=collapse_actions,
        )

    if add_continuity:
        add_continuity_edges(graph)

    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["edge_count"] = graph.number_of_edges()
    return graph


def log_paths(log_dir: Path, stem: str) -> list[Path]:
    """Prefer merged JSONL files and fall back to rank-local files."""

    merged = log_dir / f"{stem}.jsonl"
    if merged.exists() and _has_json_rows(merged):
        return [merged]
    return [
        path
        for path in sorted(log_dir.glob(f"{stem}_rank*.jsonl"), key=rank_file_sort_key)
        if _has_json_rows(path)
    ]


def load_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load one JSONL file into memory."""

    return list(iter_jsonl(path, strict=strict))


def load_many_jsonl(paths: Iterable[Path], *, strict: bool = False) -> list[dict[str, Any]]:
    """Load several JSONL files into memory."""

    return list(iter_many_jsonl(paths, strict=strict))


def load_g0_trace(log_dir: Path | str = DEFAULT_LOG_DIR, *, strict: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return raw G0 node and edge rows from a log directory."""

    resolved = Path(log_dir)
    return (
        load_many_jsonl(log_paths(resolved, "g0_nodes"), strict=strict),
        load_many_jsonl(log_paths(resolved, "g0_edges"), strict=strict),
    )


def iter_many_jsonl(paths: Iterable[Path], *, strict: bool = False) -> Iterable[dict[str, Any]]:
    """Yield JSON rows from all selected files."""

    for path in paths:
        yield from iter_jsonl(path, strict=strict)


def iter_jsonl(path: Path, *, strict: bool = False) -> Iterable[dict[str, Any]]:
    """Yield dictionary rows from one JSONL file.

    Non-strict mode skips malformed lines, including Git LFS pointer files.
    Strict mode raises G0TraceError.
    """

    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if strict:
                    raise G0TraceError(f"Invalid JSON in {path}:{line_no}") from exc
                continue
            if not isinstance(row, dict):
                if strict:
                    raise G0TraceError(f"JSONL row is not an object in {path}:{line_no}")
                continue
            yield row


def collect_action_context(edge_paths: list[Path], *, strict: bool = False) -> dict[str, dict[str, Any]]:
    """Map logger action nodes to the agent-state node that selected them."""

    context: dict[str, dict[str, Any]] = {}
    for row in iter_many_jsonl(edge_paths, strict=strict):
        if row.get("relation") != "action_selection":
            continue
        target_id = pick(row, "target_node_id")
        source_id = pick(row, "source_node_id")
        if target_id is None or source_id is None:
            continue
        context[str(target_id)] = {
            "source_node_id": str(source_id),
            "action_node_id": str(target_id),
            "action": pick(row, "target_state"),
            "action_edge_id": pick(row, "edge_id"),
            "source_attrs": endpoint_attrs(row, "source"),
            "action_attrs": endpoint_attrs(row, "target"),
        }
    return context


def add_edge_row(
    graph: nx.MultiDiGraph | nx.DiGraph,
    row: dict[str, Any],
    *,
    edge_index: int,
    action_context: dict[str, dict[str, Any]],
    collapse_actions: bool,
) -> None:
    """Normalize one logged edge and add it to the graph."""

    relation = pick(row, "relation", default="unknown")
    source_attrs = endpoint_attrs(row, "source")
    target_attrs = endpoint_attrs(row, "target")
    source_id = source_attrs["node_id"]
    target_id = target_attrs["node_id"]
    action_details: dict[str, Any] = {}

    if collapse_actions:
        if relation == "action_selection" and target_attrs["kind"] == ACTION_KIND:
            add_or_update_node(graph, source_attrs)
            _append_node_action(graph, source_id, pick(row, "target_state"))
            return
        if source_attrs["kind"] == ACTION_KIND:
            context = action_context.get(source_id)
            if context is not None:
                source_attrs = context["source_attrs"]
                action_details = {
                    "collapsed_action_node_id": context["action_node_id"],
                    "action": context["action"],
                    "action_selection_edge_id": context["action_edge_id"],
                }
                source_id = source_attrs["node_id"]
        if target_attrs["kind"] == ACTION_KIND:
            return

    add_or_update_node(graph, source_attrs)
    add_or_update_node(graph, target_attrs)

    attrs = normalize_edge_attrs(row, edge_index)
    attrs.update(action_details)
    attrs["source_log_node_id"] = pick(row, "source_node_id")
    attrs["target_log_node_id"] = pick(row, "target_node_id")
    attrs["causal_kind"] = causal_kind(attrs)
    add_graph_edge(graph, source_id, target_id, attrs)


def add_or_update_node(graph: nx.MultiDiGraph | nx.DiGraph, attrs: dict[str, Any]) -> None:
    """Add a node, merging later details into inferred placeholders."""

    node_id = attrs["node_id"]
    if graph.has_node(node_id):
        current = graph.nodes[node_id]
        current["inferred"] = bool(current.get("inferred")) and bool(attrs.get("inferred"))
        for key, value in attrs.items():
            if value is not None and current.get(key) is None:
                current[key] = value
        return
    graph.add_node(node_id, **attrs)


def add_graph_edge(graph: nx.MultiDiGraph | nx.DiGraph, source: str, target: str, attrs: dict[str, Any]) -> None:
    """Add one edge, preserving parallel relations when the graph supports them."""

    edge_id = str(attrs.get("edge_id") or attrs.get("g0_edge_id") or f"edge_{graph.number_of_edges() + 1}")
    if graph.is_multigraph():
        graph.add_edge(source, target, key=edge_id, **attrs)
        return
    if graph.has_edge(source, target):
        existing = graph.edges[source, target]
        existing.setdefault("parallel_edge_ids", []).append(edge_id)
        existing.setdefault("parallel_relations", []).append(attrs.get("relation"))
        return
    graph.add_edge(source, target, **attrs)


def add_continuity_edges(graph: nx.MultiDiGraph | nx.DiGraph) -> None:
    """Connect consecutive temporal observations of the same entity or field."""

    by_entity: dict[str, list[tuple[tuple[int, int], str]]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("semantic_kind") not in {"agent_state", "environment_field"}:
            continue
        entity_key = attrs.get("entity_key")
        if not entity_key:
            continue
        by_entity[entity_key].append((time_sort_key(attrs), node_id))

    edge_index = 0
    for entity_key, nodes in by_entity.items():
        ordered = []
        seen = set()
        for _, node_id in sorted(nodes):
            if node_id in seen:
                continue
            ordered.append(node_id)
            seen.add(node_id)
        for source, target in zip(ordered, ordered[1:]):
            if source == target:
                continue
            source_attrs = graph.nodes[source]
            target_attrs = graph.nodes[target]
            if time_sort_key(target_attrs) <= time_sort_key(source_attrs):
                continue
            edge_index += 1
            attrs = {
                "edge_id": f"continuity_{edge_index}",
                "g0_edge_id": f"continuity_{edge_index}",
                "relation": "continuity",
                "causal_kind": "continuity",
                "mechanism": "temporal_continuity",
                "entity_key": entity_key,
                "tick": target_attrs.get("tick"),
                "tick_from": source_attrs.get("tick"),
                "tick_to": target_attrs.get("tick"),
                "time_from": source_attrs.get("time"),
                "time_to": target_attrs.get("time"),
                "state_preserving": source_attrs.get("state") == target_attrs.get("state"),
                "field_preserving": source_attrs.get("field") == target_attrs.get("field"),
                "level": "G0",
            }
            add_graph_edge(graph, source, target, attrs)


def normalize_node_attrs(row: dict[str, Any], *, inferred: bool) -> dict[str, Any]:
    """Normalize a logged or inferred node row into G0 node attributes."""

    node_id = str(pick(row, "node_id", "g0_node_id", default=semantic_node_id(row)))
    kind = str(pick(row, "kind", default="unknown"))
    tick = to_int(pick(row, "tick", "time"))
    phase = pick(row, "phase", "simulation_phase")
    phase_index = PHASE_INDEX.get(str(phase), None)
    uid = pick(row, "uid", "agent_uid")
    agent_type = pick(row, "agent_type", "type")
    state = pick(row, "state")
    field = pick(row, "field")
    owner_uid = pick(row, "owner_uid")
    compartment = pick(row, "compartment")
    value = pick(row, "value")
    semantic_kind = semantic_kind_for(kind)
    entity_key = entity_key_for(kind, uid, agent_type, field, owner_uid)
    time = time_value(tick, phase_index)
    display_id = display_node_id(kind, uid, agent_type, state, field, owner_uid, time)

    attrs = dict(row)
    attrs.update({
        "node_id": node_id,
        "g0_id": node_id,
        "kind": kind,
        "semantic_kind": semantic_kind,
        "entity_key": entity_key,
        "display_id": display_id,
        "tick": tick,
        "phase": phase,
        "phase_index": phase_index,
        "time": time,
        "uid": uid,
        "agent_id": uid,
        "agent_uid": uid,
        "agent_type": agent_type,
        "state": state,
        "field": field,
        "value": value,
        "owner_uid": owner_uid,
        "compartment": compartment,
        "level": pick(row, "level", default=infer_level(kind, uid, agent_type, owner_uid, compartment)),
        "g1_key": pick(row, "g1_key", "g1_node_key", default=entity_key),
        "g2_key": pick(row, "g2_key", "g2_node_key", default=g2_key_for(kind, agent_type, state, field)),
        "inferred": inferred,
    })
    return attrs


def endpoint_attrs(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Build a node row from source_* or target_* fields in an edge row."""

    node_id = pick(row, f"{prefix}_node_id")
    kind = pick(row, f"{prefix}_kind")
    uid = pick(row, f"{prefix}_uid")
    agent_type = pick(row, f"{prefix}_type")
    state = pick(row, f"{prefix}_state")
    field = pick(row, f"{prefix}_field")
    phase = pick(row, f"phase_{'from' if prefix == 'source' else 'to'}")
    g1_key = pick(row, f"g1_{prefix}_key", f"{prefix}_g1_key")
    g2_key = pick(row, f"g2_{prefix}_key", f"{prefix}_g2_key")
    return normalize_node_attrs(
        {
            "node_id": node_id,
            "kind": kind,
            "uid": uid,
            "agent_type": agent_type,
            "state": state,
            "field": field,
            "tick": pick(row, "tick"),
            "phase": phase,
            "rank": pick(row, "rank"),
            "owner_uid": pick(row, "owner_uid"),
            "compartment": pick(row, "compartment"),
            "g1_key": g1_key,
            "g2_key": g2_key,
            "run_id": pick(row, "run_id"),
        },
        inferred=True,
    )


def normalize_edge_attrs(row: dict[str, Any], edge_index: int) -> dict[str, Any]:
    """Normalize one logged edge into G0 edge attributes."""

    effect = pick(row, "effect_value", "effect", "delta")
    attrs = dict(row)
    attrs.update({
        "edge_id": str(pick(row, "edge_id", "id", default=f"g0_edge_{edge_index}")),
        "g0_edge_id": str(pick(row, "edge_id", "id", default=f"g0_edge_{edge_index}")),
        "relation": pick(row, "relation", "type", "kind", default="unknown"),
        "mechanism": pick(row, "mechanism", default="unknown"),
        "phase_from": pick(row, "phase_from", "source_phase"),
        "phase_to": pick(row, "phase_to", "target_phase"),
        "tick": to_int(pick(row, "tick")),
        "effect": effect,
        "sign": pick(row, "effect_sign", "sign", default=effect_sign(effect)),
        "level": "G0",
    })
    return attrs


def causal_kind(edge_attrs: dict[str, Any]) -> str:
    """Map logger relations into analysis-level G0 edge classes."""

    relation = edge_attrs.get("relation")
    source_kind = edge_attrs.get("source_kind")
    target_kind = edge_attrs.get("target_kind")
    mechanism = edge_attrs.get("mechanism") or ""
    if relation == "continuity":
        return "continuity"
    if relation == "threshold_trigger" or (source_kind in FIELD_KINDS and target_kind in AGENT_KINDS):
        return "perception"
    if relation in {"field_effect", "internal_field_effect", "buffer_commit"}:
        return "action"
    if relation in {"target_assignment", "degradation", "agent_to_agent"}:
        return "agent_relation"
    if relation == "aggregation" or "aggregate" in mechanism:
        return "aggregation"
    if relation == "state_transition":
        return "transition"
    if relation == "action_selection":
        return "action_selection"
    return "causal"


def semantic_kind_for(kind: str) -> str:
    """Return the G0 semantic category for a logger node kind."""

    if kind in AGENT_KINDS:
        return "agent_state"
    if kind in FIELD_KINDS:
        return "environment_field"
    if kind == ACTION_KIND:
        return "action"
    return "unknown"


def entity_key_for(kind: str, uid: Optional[str], agent_type: Optional[str], field: Optional[str], owner_uid: Optional[str]) -> str:
    """Return a stable entity key used by continuity and contractions."""

    if kind in AGENT_KINDS or kind == ACTION_KIND:
        return f"{agent_type or 'Agent'}:{uid or 'unknown'}"
    if kind in FIELD_KINDS:
        if uid == "SN" or agent_type == "SubstantiaNigra":
            return f"SN:{field or 'field'}"
        return f"Neuron:{owner_uid or uid or 'unknown'}:{field or 'field'}"
    return f"unknown:{uid or field or 'node'}"


def g2_key_for(kind: str, agent_type: Optional[str], state: Optional[str], field: Optional[str]) -> str:
    """Return a coarse key suitable for second-level contraction."""

    if kind in AGENT_KINDS or kind == ACTION_KIND:
        return f"{agent_type or 'Agent'}.{state or 'unknown'}"
    if kind in FIELD_KINDS:
        return field or "field"
    return "unknown"


def infer_level(kind: str, uid: Optional[str], agent_type: Optional[str], owner_uid: Optional[str], compartment: Optional[str]) -> str:
    """Infer the biological level of an inferred node."""

    if kind in FIELD_KINDS and (uid == "SN" or agent_type == "SubstantiaNigra"):
        return "environment"
    if kind in FIELD_KINDS:
        return "macro"
    if agent_type == "Neuron":
        return "macro"
    if owner_uid is not None or compartment == "Intracellular":
        return "intracellular"
    return "extracellular"


def display_node_id(kind: str, uid: Optional[str], agent_type: Optional[str], state: Optional[str], field: Optional[str], owner_uid: Optional[str], time: str) -> str:
    """Build the human-readable G0 label requested by the model spec."""

    if kind in AGENT_KINDS:
        return f"{agent_type or 'Agent'}_{uid or 'unknown'}_{state or 'unknown'}@{time}"
    if kind in FIELD_KINDS:
        if uid == "SN" or agent_type == "SubstantiaNigra":
            return f"SN_{field or 'field'}@{time}"
        return f"Neuron_{owner_uid or uid or 'unknown'}_{field or 'field'}@{time}"
    if kind == ACTION_KIND:
        return f"{agent_type or 'Agent'}_{uid or 'unknown'}_{state or 'action'}@{time}"
    return f"Node_{uid or field or 'unknown'}@{time}"


def semantic_node_id(row: dict[str, Any]) -> str:
    """Build an inferred node id when a log row lacks an explicit id."""

    tick = to_int(pick(row, "tick")) or 0
    phase_index = PHASE_INDEX.get(str(pick(row, "phase")), 0)
    time = time_value(tick, phase_index)
    return display_node_id(
        str(pick(row, "kind", default="unknown")),
        pick(row, "uid", "agent_uid"),
        pick(row, "agent_type", "type"),
        pick(row, "state"),
        pick(row, "field"),
        pick(row, "owner_uid"),
        time,
    )


def time_sort_key(attrs: dict[str, Any]) -> tuple[int, int]:
    """Return a sortable tick/phase tuple."""

    tick = attrs.get("tick")
    phase_index = attrs.get("phase_index")
    return (
        tick if isinstance(tick, int) else -1,
        phase_index if isinstance(phase_index, int) else -1,
    )


def time_value(tick: Optional[int], phase_index: Optional[int]) -> str:
    """Return the compact time value used in G0 labels."""

    if tick is None:
        return "unknown"
    if phase_index is None:
        return str(tick)
    return f"{tick}.{phase_index}"


def pick(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present non-null field from a JSON row."""

    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def to_int(value) -> Optional[int]:
    """Coerce tick values to int when possible."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def effect_sign(value) -> Optional[str]:
    """Return positive, negative, neutral or None for effect values."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "neutral"


def rank_file_sort_key(path: Path) -> tuple[int, str]:
    """Sort rank-local files by numeric rank."""

    marker = "_rank"
    if marker not in path.stem:
        return (0, path.name)
    suffix = path.stem.split(marker, 1)[1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (0, path.name)


def _append_node_action(graph: nx.MultiDiGraph | nx.DiGraph, node_id: str, action: Optional[str]) -> None:
    """Store action-selection labels on the agent-state node when collapsed."""

    if not graph.has_node(node_id) or action is None:
        return
    actions = graph.nodes[node_id].setdefault("selected_actions", [])
    if action not in actions:
        actions.append(action)


def _has_json_rows(path: Path) -> bool:
    """Return True when a file has at least one JSON object row."""

    try:
        next(iter_jsonl(path, strict=False))
        return True
    except StopIteration:
        return False


def graph_summary(graph: nx.MultiDiGraph | nx.DiGraph) -> dict[str, Any]:
    """Return small counts useful for CLI sanity checks."""

    node_kinds = defaultdict(int)
    edge_kinds = defaultdict(int)
    for _, attrs in graph.nodes(data=True):
        node_kinds[str(attrs.get("semantic_kind"))] += 1
    for _, _, attrs in graph.edges(data=True):
        edge_kinds[str(attrs.get("causal_kind"))] += 1
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_kinds": dict(sorted(node_kinds.items())),
        "edge_kinds": dict(sorted(edge_kinds.items())),
        "is_multigraph": graph.is_multigraph(),
    }


def write_graph(graph: nx.MultiDiGraph | nx.DiGraph, path: Path) -> None:
    """Write a NetworkX graph using a suffix-based format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json_graph.node_link_data(graph, edges="edges")
        except TypeError:
            payload = json_graph.node_link_data(graph)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    elif suffix == ".graphml":
        nx.write_graphml(plain_attribute_graph(graph), path)
    elif suffix == ".gexf":
        nx.write_gexf(plain_attribute_graph(graph), path)
    elif suffix in {".pkl", ".pickle"}:
        with path.open("wb") as stream:
            pickle.dump(graph, stream)
    else:
        raise G0TraceError(f"Unsupported graph output suffix: {suffix}")


def plain_attribute_graph(graph: nx.MultiDiGraph | nx.DiGraph) -> nx.MultiDiGraph | nx.DiGraph:
    """Return a graph copy whose attrs are safe for GraphML/GEXF writers."""

    converted = graph.__class__()
    converted.graph.update({
        key: plain_value(value)
        for key, value in graph.graph.items()
    })
    for node_id, attrs in graph.nodes(data=True):
        converted.add_node(
            node_id,
            **{
                key: plain_value(value)
                for key, value in attrs.items()
            },
        )
    if graph.is_multigraph():
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            converted.add_edge(
                source,
                target,
                key=key,
                **{
                    attr_key: plain_value(value)
                    for attr_key, value in attrs.items()
                },
            )
    else:
        for source, target, attrs in graph.edges(data=True):
            converted.add_edge(
                source,
                target,
                **{
                    key: plain_value(value)
                    for key, value in attrs.items()
                },
            )
    return converted


def plain_value(value):
    """Convert complex values to strings for graph exchange formats."""

    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def require_networkx() -> None:
    """Raise a clear error when NetworkX is unavailable."""

    if nx is None:
        raise G0TraceError("networkx is required to build G0 graphs")


def main() -> None:
    """Command-line entry point for building and exporting G0."""

    parser = argparse.ArgumentParser(description="Build the G0 NetworkX causal graph from simulation logs.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Graph destination: .json, .graphml, .gexf, .pkl.")
    parser.add_argument("--keep-actions", action="store_true", help="Keep runtime action nodes instead of collapsing them into action edges.")
    parser.add_argument("--no-continuity", action="store_true", help="Do not add temporal continuity edges.")
    parser.add_argument("--digraph", action="store_true", help="Use DiGraph instead of MultiDiGraph, aggregating parallel edges.")
    parser.add_argument("--strict", action="store_true", help="Raise on malformed JSONL rows.")
    parser.add_argument("--summary", action="store_true", help="Print graph counts after building.")
    args = parser.parse_args()

    graph = build_g0_graph(
        args.log_dir,
        collapse_actions=not args.keep_actions,
        add_continuity=not args.no_continuity,
        multigraph=not args.digraph,
        strict=args.strict,
    )
    write_graph(graph, args.output)
    if args.summary:
        print(json.dumps(graph_summary(graph), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
