from __future__ import annotations
import re
from collections.abc import Iterable
from typing import Any, Optional
import networkx as nx

def iter_edges(graph) -> Iterable[tuple[Any, Any, dict[str, Any]]]:
    if graph.is_multigraph():
        for source, target, _key, attrs in graph.edges(keys=True, data=True):
            yield source, target, attrs
        return
    for source, target, attrs in graph.edges(data=True):
        yield source, target, attrs

def merge_summary_edge(graph, source: str, target: str, attrs: dict[str, Any]) -> None:
    count = int_or_default(attrs.get("count"), 1)
    total_effect = numeric(attrs.get("total_effect", attrs.get("effect_value", attrs.get("effect"))))
    if "total_effect" not in attrs:
        total_effect *= count
    mean_effect = numeric(attrs.get("mean_effect", total_effect / count if count else 0.0))
    tick = int_or_none(attrs.get("tick"))
    first_seen = int_or_none(attrs.get("first_seen")) or tick or 0
    last_seen = int_or_none(attrs.get("last_seen")) or tick or first_seen
    relation = attrs.get("relation")
    mechanism = attrs.get("mechanism")
    causal_kind = attrs.get("causal_kind")
    outcome = attrs.get("outcome")
    sign = edge_sign(attrs, total_effect)
    if not graph.has_edge(source, target):
        graph.add_edge(source, target, count=0, lower_edge_count=0, total_effect=0.0, mean_effect=0.0, mean_of_mean_effect=0.0, first_seen=first_seen, last_seen=last_seen, sign=sign, relation=relation, mechanism=mechanism, causal_kind=causal_kind, outcome=outcome, relations=[], mechanisms=[], causal_kinds=[], outcomes=[], source_edge_ids=[])
    edge = graph.edges[source, target]
    edge["count"] += count
    edge["lower_edge_count"] += 1
    edge["total_effect"] += total_effect
    edge["mean_effect"] = edge["total_effect"] / edge["count"] if edge["count"] else 0.0
    previous_mean_total = edge["mean_of_mean_effect"] * (edge["lower_edge_count"] - 1)
    edge["mean_of_mean_effect"] = (previous_mean_total + mean_effect) / edge["lower_edge_count"]
    edge["first_seen"] = min(edge["first_seen"], first_seen)
    edge["last_seen"] = max(edge["last_seen"], last_seen)
    edge["sign"] = combine_signs(edge.get("sign"), sign)
    append_many(edge["relations"], attrs.get("relations", relation))
    append_many(edge["mechanisms"], attrs.get("mechanisms", mechanism))
    append_many(edge["causal_kinds"], attrs.get("causal_kinds", causal_kind))
    append_many(edge["outcomes"], attrs.get("outcomes", outcome))
    append_many(edge["source_edge_ids"], attrs.get("source_edge_ids", attrs.get("edge_ids", attrs.get("edge_id"))))
    edge["relation"] = compact_label(edge["relations"])
    edge["mechanism"] = compact_label(edge["mechanisms"])
    edge["causal_kind"] = compact_label(edge["causal_kinds"])
    edge["outcome"] = compact_label(edge["outcomes"])
    edge["weight"] = edge["count"]
    edge["label"] = edge_label(edge)


def summarize_nodes(nodes: Iterable[tuple[Any, dict[str, Any]]], *, key: str, contraction: str, level: str) -> dict[str, Any]:
    node_rows = list(nodes)
    attrs = [row for _, row in node_rows]
    ticks = [
        value
        for attr in attrs
        for value in (int_or_none(attr.get("start_tick")), int_or_none(attr.get("tick")), int_or_none(attr.get("end_tick")))
        if value is not None
    ]
    original_node_ids: list[Any] = []
    for node_id, attr in node_rows:
        append_many(original_node_ids, attr.get("original_node_ids", node_id))
    observation_count = sum(int_or_default(attr.get("observation_count"), 1) for attr in attrs)
    absorbed_node_count = sum(int_or_default(attr.get("absorbed_node_count"), 1) for attr in attrs)
    semantic_kind = common_value(attr.get("semantic_kind") for attr in attrs)
    agent_type = common_value(attr.get("agent_type") for attr in attrs)
    state = common_value(attr.get("state") for attr in attrs)
    field = common_value(attr.get("field") for attr in attrs)
    uid = common_value(attr.get("uid") for attr in attrs)

    return {
        "label": key,
        "contraction": contraction,
        "level": level,
        "semantic_kind": semantic_kind,
        "agent_type": agent_type,
        "uid": uid,
        "state": state,
        "field": field,
        "entity_key": common_value(attr.get("entity_key") for attr in attrs),
        "member_count": len(node_rows),
        "observation_count": observation_count,
        "absorbed_node_count": absorbed_node_count,
        "absorbed_edge_count": sum(int_or_default(attr.get("absorbed_edge_count"), 0) for attr in attrs),
        "first_seen": min(ticks) if ticks else None,
        "last_seen": max(ticks) if ticks else None,
        "original_node_ids": original_node_ids}


def summarize_supernodes(supernodes: Iterable[Any], *, key: str, contraction: str, level: str) -> dict[str, Any]:
    return summarize_nodes(
        ((node.key, node.attr) for node in supernodes),
        key=key,
        contraction=contraction,
        level=level)


def summarize_superedges(superedges: Iterable[Any]) -> dict[str, Any]:
    summary_graph = make_digraph()
    source = "__source__"
    target = "__target__"
    summary_graph.add_node(source)
    summary_graph.add_node(target)
    for edge in superedges:
        merge_summary_edge(summary_graph, source, target, edge.attr)
    if not summary_graph.has_edge(source, target):
        return {}
    return dict(summary_graph.edges[source, target])


def time_group_key(node_id: Any, attrs: dict[str, Any], window_size: Optional[int] = None) -> str:
    semantic_kind = attrs.get("semantic_kind")
    if semantic_kind == "agent_state":
        base = "_".join(
            clean_token(part)
            for part in (
                attrs.get("agent_type") or "Agent",
                identity_uid(attrs.get("uid")),
                attrs.get("state") or "unknown",
            )
        )
    elif semantic_kind == "environment_field":
        base = field_key(attrs, node_id)
    else:
        base = strip_tick(node_id)

    if window_size is None:
        return base
    tick = int_or_none(attrs.get("tick"))
    if tick is None:
        raise ValueError(f"Cannot time-contract node {node_id!r}: missing 'tick' attribute.")
    bucket = tick // window_size
    start = bucket * window_size
    end = start + window_size - 1
    return f"{base}_t{start}_{end}"


def agent_cluster_key(node_id: Any, attrs: dict[str, Any]) -> str:
    semantic_kind = attrs.get("semantic_kind")
    if semantic_kind == "agent_state":
        return "_".join(
            clean_token(part)
            for part in (
                attrs.get("agent_type") or "Agent",
                attrs.get("state") or "unknown",
            )
        )
    if semantic_kind == "environment_field" and attrs.get("agent_type") == "Neuron":
        return "Neuron_internal_environment"
    return str(node_id)


def field_key(attrs: dict[str, Any], node_id: Any) -> str:
    field = attrs.get("field") or strip_tick(node_id)
    if attrs.get("uid") == "SN" or attrs.get("agent_type") == "SubstantiaNigra":
        return f"SN_{clean_token(field)}"
    owner = identity_uid(attrs.get("owner_uid") or attrs.get("uid"))
    return f"Neuron_{clean_token(owner)}_{clean_token(field)}"


def edge_label(attrs: dict[str, Any]) -> str:
    relation = attrs.get("relation") or attrs.get("causal_kind") or "edge"
    return f"{relation} n={attrs.get('count', 0)} mean={numeric(attrs.get('mean_effect')):.4f}"


def edge_sign(attrs: dict[str, Any], effect: float) -> str:
    raw = attrs.get("sign")
    if raw in {"positive", "+"}:
        return "+"
    if raw in {"negative", "-"}:
        return "-"
    if raw in {"state", "structural"}:
        return raw
    if attrs.get("causal_kind") in {"transition", "state_transition"} or attrs.get("relation") == "state_transition":
        return "state"
    if attrs.get("causal_kind") == "continuity" or attrs.get("relation") == "continuity":
        return "structural"
    if effect > 0:
        return "+"
    if effect < 0:
        return "-"
    return "structural"


def combine_signs(current: str | None, incoming: str) -> str:
    if current in (None, incoming):
        return incoming
    if "state" in {current, incoming}:
        return "state"
    if "structural" in {current, incoming}:
        return incoming if current == "structural" else current
    return "mixed"


def common_value(values: Iterable[Any]) -> Any:
    cleaned = [value for value in values if value not in (None, "")]
    if not cleaned:
        return None
    first = cleaned[0]
    if all(value == first for value in cleaned):
        return first
    return "mixed"


def append_many(target: list[Any], values: Any) -> None:
    if values is None:
        return
    if isinstance(values, (set, tuple, list)):
        for value in values:
            append_many(target, value)
        return
    if values not in target:
        target.append(values)


def compact_label(values: list[Any]) -> Any:
    cleaned = [value for value in values if value not in (None, "")]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return "mixed"


def strip_tick(node_id: Any) -> str:
    return str(node_id).split("@", 1)[0]


def clean_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value))
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def identity_uid(uid: Any) -> str:
    if uid is None:
        return "unknown"
    return str(uid)


def display_uid(uid: Any) -> str:
    return identity_uid(uid)


def numeric(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def int_or_default(value: Any, default: int) -> int:
    parsed = int_or_none(value)
    return default if parsed is None else parsed


def make_digraph():
    return nx.DiGraph()
