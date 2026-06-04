from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import networkx as nx

DEFAULT_SIMULATION_LOG_DIR = Path("output/simulation/logs")
DEFAULT_GRAPH_OUTPUT = Path("output/analysis/graphs/g0.gexf")

FIELD_KINDS = {"env_field", "internal_field", "buffer"}
AGENT_KINDS = {"agent_state", "aggregate"}
ALPHA_AGENT_TYPES = {"AlphaSynuclein", "AlphaAggregate"}

PHASE_INDEX = {"0_pre_state": 0, "1_perception": 1, "2_state_update": 2, "3_action_selection": 3, "4_effect_buffer": 4, "5_commit": 5}

@dataclass(frozen=True)
class ActionResolution:
    uid: Optional[str]
    agent_type: Optional[str]
    state: Optional[str]
    owner_uid: Optional[str]
    compartment: Optional[str]
    action: Optional[str]
    tick: Optional[int]


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


def build_g0(log_dir: Path | str = DEFAULT_SIMULATION_LOG_DIR, *, add_continuity: bool = True, include_field_continuity: bool = True, rupture_grace_ticks: int = 2, strict: bool = False, start_tick: Optional[int] = None, end_tick: Optional[int] = None):
    log_dir = Path(log_dir)
    node_paths = log_paths(log_dir, "g0_nodes")
    edge_paths = log_paths(log_dir, "g0_edges")
    filters = collect_lifecycle_filters(node_paths, edge_paths, rupture_grace_ticks=rupture_grace_ticks, strict=strict)
    state_index = collect_agent_state_index(node_paths, filters, strict=strict, start_tick=start_tick, end_tick=end_tick)
    action_lookup = collect_action_resolutions(edge_paths, state_index, filters, strict=strict, start_tick=start_tick, end_tick=end_tick)
    graph = nx.DiGraph(name="G0", level="G0", source_log_dir=str(log_dir), node_log_files=[str(path) for path in node_paths], edge_log_files=[str(path) for path in edge_paths], rupture_grace_ticks=rupture_grace_ticks, start_tick=start_tick if start_tick is not None else "", end_tick=end_tick if end_tick is not None else "")
    for row in iter_many_jsonl(node_paths, strict=strict):
        if not row_tick_in_range(row, start_tick, end_tick):
            continue
        endpoint = endpoint_from_node_row(row)
        if endpoint is None:
            continue
        node_id, attrs = endpoint
        if should_skip_endpoint(attrs, filters):
            continue
        merge_node(graph, node_id, attrs)
    for row in iter_many_jsonl(edge_paths, strict=strict):
        if not row_tick_in_range(row, start_tick, end_tick):
            continue
        merge_causal_edge(graph, row, action_lookup, state_index, filters)
    if add_continuity:
        add_continuity_edges(graph, include_fields=include_field_continuity)
    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["edge_count"] = graph.number_of_edges()
    graph.graph["continuity_enabled"] = add_continuity
    graph.graph["field_continuity_enabled"] = include_field_continuity
    return graph

def build_g0_graph(*args, **kwargs):
    return build_g0(*args, **kwargs)

def write_g0_gexf(graph, output_path: Path | str = DEFAULT_GRAPH_OUTPUT) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(serializable_copy(graph), output_path)
    return output_path

def collect_lifecycle_filters(node_paths: Iterable[Path], edge_paths: Iterable[Path], *, rupture_grace_ticks: int, strict: bool) -> LifecycleFilters:
    ruptured_first_tick: dict[str, int] = {}
    cleared_first_tick: dict[str, int] = {}
    def observe(uid: Optional[str], agent_type: Optional[str], state: Optional[str], tick: Optional[int]) -> None:
        if uid is None or tick is None:
            return
        if agent_type == "Neuron" and state == "Ruptured":
            ruptured_first_tick[uid] = min(tick, ruptured_first_tick.get(uid, tick))
        if agent_type == "AlphaSynuclein" and state == "Cleared":
            cleared_first_tick[uid] = min(tick, cleared_first_tick.get(uid, tick))
    for row in iter_many_jsonl(node_paths, strict=strict):
        observe(row.get("uid"), row.get("agent_type"), value_of(row.get("state")), int_or_none(row.get("tick")))
    for row in iter_many_jsonl(edge_paths, strict=strict):
        observe(row.get("source_uid"), row.get("source_type"), value_of(row.get("source_state")), int_or_none(row.get("tick")))
        observe(row.get("target_uid"), row.get("target_type"), value_of(row.get("target_state")), int_or_none(row.get("tick")))
    return LifecycleFilters(
        ruptured_cutoff_by_neuron={
            uid: first_tick + rupture_grace_ticks
            for uid, first_tick in ruptured_first_tick.items()
        }, cleared_cutoff_by_alpha=cleared_first_tick)


def collect_agent_state_index(node_paths: Iterable[Path], filters: LifecycleFilters, *, strict: bool, start_tick: Optional[int] = None, end_tick: Optional[int] = None) -> dict[tuple[Optional[str], Optional[str], int], list[dict[str, Any]]]:
    state_index: dict[tuple[Optional[str], Optional[str], int], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_many_jsonl(node_paths, strict=strict):
        if not row_tick_in_range(row, start_tick, end_tick):
            continue
        if row.get("kind") not in AGENT_KINDS:
            continue
        endpoint = endpoint_from_node_row(row)
        if endpoint is None:
            continue
        _, attrs = endpoint
        if should_skip_endpoint(attrs, filters):
            continue
        tick = attrs.get("tick")
        key = (attrs.get("agent_type"), attrs.get("uid"), tick)
        state_index[key].append(attrs)
    for rows in state_index.values():
        rows.sort(key=lambda item: phase_order(item.get("phase")))
    return state_index


def collect_action_resolutions(edge_paths: Iterable[Path], state_index: dict[tuple[Optional[str], Optional[str], int], list[dict[str, Any]]], filters: LifecycleFilters, *, strict: bool, start_tick: Optional[int] = None, end_tick: Optional[int] = None) -> dict[str, ActionResolution]:
    action_lookup: dict[str, ActionResolution] = {}
    for row in iter_many_jsonl(edge_paths, strict=strict):
        if not row_tick_in_range(row, start_tick, end_tick):
            continue
        if row.get("relation") != "action_selection":
            continue
        if row.get("target_kind") != "action":
            continue
        tick = int_or_none(row.get("tick"))
        if row.get("source_kind") == "agent_state":
            state = value_of(row.get("source_state"))
            owner_uid = row.get("owner_uid")
        else:
            state = resolve_agent_state(state_index, row.get("target_type"), row.get("target_uid"), tick)
            owner_uid = row.get("owner_uid")
        attrs = {
            "kind": "agent_state",
            "semantic_kind": "agent_state",
            "agent_type": row.get("target_type"),
            "uid": row.get("target_uid"),
            "state": state,
            "tick": tick,
            "owner_uid": owner_uid,
            "compartment": row.get("compartment"),
        }
        if should_skip_endpoint(attrs, filters):
            continue
        action_lookup[row.get("target_node_id")] = ActionResolution(
            uid=row.get("target_uid"),
            agent_type=row.get("target_type"),
            state=state,
            owner_uid=owner_uid,
            compartment=row.get("compartment"),
            action=value_of(row.get("target_state")),
            tick=tick
        )
    return action_lookup


def merge_causal_edge(graph, row: dict[str, Any], action_lookup: dict[str, ActionResolution], state_index: dict[tuple[Optional[str], Optional[str], int], list[dict[str, Any]]], filters: LifecycleFilters) -> None:
    if not row.get("valid", True):
        return
    relation = row.get("relation") or "unknown"
    if relation == "action_selection" and row.get("source_kind") not in FIELD_KINDS:
        return
    if relation == "target_assignment":
        return
    source = endpoint_from_edge_side(row, "source", action_lookup, state_index)
    target = endpoint_from_edge_side(row, "target", action_lookup, state_index)
    if source is None or target is None:
        return
    source_id, source_attrs = source
    target_id, target_attrs = target
    if should_skip_endpoint(source_attrs, filters) or should_skip_endpoint(target_attrs, filters):
        return
    if not edge_allowed(row, source_attrs, target_attrs):
        return
    merge_node(graph, source_id, source_attrs)
    merge_node(graph, target_id, target_attrs)
    merge_edge(graph, source_id, target_id, edge_attrs_from_row(row, source_attrs, target_attrs))


def endpoint_from_node_row(row: dict[str, Any]) -> Optional[tuple[str, dict[str, Any]]]:
    kind = row.get("kind")
    if kind == "action":
        return None
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


def endpoint_from_edge_side(row: dict[str, Any], side: str, action_lookup: dict[str, ActionResolution], state_index: dict[tuple[Optional[str], Optional[str], int], list[dict[str, Any]]]) -> Optional[tuple[str, dict[str, Any]]]:
    kind = row.get(f"{side}_kind")
    tick = int_or_none(row.get("tick"))
    if tick is None:
        return None
    if kind == "action":
        action_id = row.get(f"{side}_node_id")
        resolution = action_lookup.get(action_id)
        if resolution is not None:
            state = resolution.state
            owner_uid = resolution.owner_uid
            compartment = resolution.compartment
            action = resolution.action
        else:
            state = resolve_agent_state(
                state_index,
                row.get(f"{side}_type"),
                row.get(f"{side}_uid"),
                tick
            )
            owner_uid = row.get("owner_uid")
            compartment = row.get("compartment")
            action = value_of(row.get(f"{side}_state"))
        endpoint_row = {
            "agent_type": row.get(f"{side}_type"),
            "uid": row.get(f"{side}_uid"),
            "state": state,
            "tick": tick,
            "kind": "agent_state",
            "level": row.get("compartment") or "unknown",
            "owner_uid": owner_uid,
            "compartment": compartment,
            "node_id": action_id,
            "phase": row.get(f"phase_{'from' if side == 'source' else 'to'}"),
            "rank": row.get("rank"),
            "run_id": row.get("run_id"),
            "value": None,
            "collapsed_action": action
        }
        node_id, attrs = endpoint_from_node_row(endpoint_row)
        attrs["collapsed_action"] = action
        attrs["inferred_from_action_node"] = True
        return node_id, attrs
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
            "phase": row.get(f"phase_{'from' if side == 'source' else 'to'}"),
            "rank": row.get("rank"),
            "run_id": row.get("run_id"),
            "value": row.get("effect_value") if kind == "buffer" else None
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
            "phase": row.get(f"phase_{'from' if side == 'source' else 'to'}"),
            "rank": row.get("rank"),
            "run_id": row.get("run_id"),
            "value": None
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


def add_continuity_edges(graph, *, include_fields: bool) -> None:
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
            source = latest_node_at_tick(graph, by_tick[tick])
            target = earliest_node_at_tick(graph, by_tick[next_tick])
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
                    "effect_sign": "structural",
                    "source_kind": graph.nodes[source].get("semantic_kind"),
                    "target_kind": graph.nodes[target].get("semantic_kind")
                }
            )

def merge_node(graph, node_id: str, attrs: dict[str, Any]) -> None:
    if not graph.has_node(node_id):
        attrs = dict(attrs)
        attrs["source_log_node_ids"] = as_list(attrs.pop("source_log_node_id", None))
        attrs["phases"] = as_list(attrs.get("phase"))
        attrs["min_phase"] = attrs.get("phase")
        attrs["max_phase"] = attrs.get("phase")
        graph.add_node(node_id, **attrs)
        return
    current = graph.nodes[node_id]
    append_unique(current["source_log_node_ids"], attrs.get("source_log_node_id"))
    append_unique(current["phases"], attrs.get("phase"))
    current["min_phase"] = min_phase(current.get("min_phase"), attrs.get("phase"))
    current["max_phase"] = max_phase(current.get("max_phase"), attrs.get("phase"))
    if current.get("value") is None and attrs.get("value") is not None:
        current["value"] = attrs.get("value")

def merge_edge(graph, source: str, target: str, attrs: dict[str, Any]) -> None:
    effect = number_or_zero(attrs.get("effect_value"))
    tick = int_or_none(attrs.get("tick")) or 0
    first_tick = int_or_none(attrs.get("first_tick")) or tick
    last_tick = int_or_none(attrs.get("last_tick")) or tick
    payload = dict(attrs)
    for key in ("count", "total_effect", "mean_effect", "first_tick", "last_tick", "edge_ids", "relations", "mechanisms", "actions", "outcomes"):
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
            actions=as_list(attrs.get("action")),
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
    append_unique(current["actions"], attrs.get("action"))
    append_unique(current["outcomes"], attrs.get("outcome"))
    current["relation"] = compact_label(current["relations"])
    current["mechanism"] = compact_label(current["mechanisms"])
    current["outcome"] = compact_label(current["outcomes"])


def edge_attrs_from_row(row: dict[str, Any], source_attrs: dict[str, Any], target_attrs: dict[str, Any]) -> dict[str, Any]:
    relation = row.get("relation") or "unknown"
    action = source_attrs.get("collapsed_action") or target_attrs.get("collapsed_action")
    attrs = {
        "edge_id": row.get("edge_id"),
        "relation": relation,
        "mechanism": row.get("mechanism"),
        "causal_kind": causal_kind(row, source_attrs, target_attrs),
        "tick": int_or_none(row.get("tick")),
        "phase_from": row.get("phase_from"),
        "phase_to": row.get("phase_to"),
        "rank": row.get("rank"),
        "run_id": row.get("run_id"),
        "rule_id": row.get("rule_id"),
        "predicate": row.get("predicate"),
        "effect_value": row.get("effect_value"),
        "effect_sign": normalized_sign(row.get("effect_sign"), row.get("effect_value"), relation),
        "effect_unit": row.get("effect_unit"),
        "probability": row.get("probability"),
        "rng_value": row.get("rng_value"),
        "outcome": row.get("outcome"),
        "action": action,
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
        "g1_source_key": row.get("g1_source_key"),
        "g1_target_key": row.get("g1_target_key"),
        "g2_source_key": row.get("g2_source_key"),
        "g2_target_key": row.get("g2_target_key"),
        "valid": row.get("valid", True)
    }
    return attrs


def causal_kind(row: dict[str, Any], source_attrs: dict[str, Any], target_attrs: dict[str, Any]) -> str:
    relation = row.get("relation")
    if relation in {"threshold_trigger", "action_selection"}:
        return "perception"
    if relation in {"field_effect", "internal_field_effect"}:
        return "action"
    if relation == "buffer_commit":
        return "field_commit"
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
        "phase": row.get("phase"),
        "rank": row.get("rank"),
        "run_id": row.get("run_id"),
        "level": row.get("level"),
        "owner_uid": row.get("owner_uid"),
        "compartment": row.get("compartment"),
        "g1_key": row.get("g1_key"),
        "g2_key": row.get("g2_key")
    }


def log_paths(directory: Path, stem: str) -> list[Path]:
    merged = directory / f"{stem}.jsonl"
    if has_jsonl_rows(merged):
        return [merged]
    return [
        path
        for path in sorted(directory.glob(f"{stem}_rank*.jsonl"), key=rank_path_key)
        if has_jsonl_rows(path)
    ]


def iter_many_jsonl(paths: Iterable[Path], *, strict: bool = False) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield from iter_jsonl(path, strict=strict)


def iter_jsonl(path: Path, *, strict: bool = False) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise ValueError(f"Malformed JSONL in {path} at line {line_no}.")


def has_jsonl_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return any(line.strip() for line in stream)


def agent_node_id(agent_type: Optional[str], uid: Optional[str], state: Optional[str], tick: int) -> str:
    name = clean_token(agent_type or "Agent")
    identifier = clean_token(identity_uid(uid))
    state_name = clean_token(value_of(state) or "unknown")
    return f"{name}_{identifier}_{state_name}@{tick}"


def field_node_id(
    field: Optional[str],
    uid: Optional[str],
    agent_type: Optional[str],
    owner_uid: Optional[str],
    tick: int,
) -> str:
    field_name = clean_token(field or "field")
    if uid == "SN" or agent_type == "SubstantiaNigra":
        return f"SN_{field_name}@{tick}"
    owner = clean_token(identity_uid(owner_uid or uid))
    return f"Neuron_{owner}_{field_name}@{tick}"


def agent_entity_key(agent_type: Optional[str], uid: Optional[str]) -> str:
    return f"agent:{agent_type or 'Agent'}:{uid or 'unknown'}"


def field_entity_key(
    field: Optional[str],
    uid: Optional[str],
    agent_type: Optional[str],
    owner_uid: Optional[str],
) -> str:
    if uid == "SN" or agent_type == "SubstantiaNigra":
        return f"field:SN:{field or 'field'}"
    return f"field:Neuron:{owner_uid or uid or 'unknown'}:{field or 'field'}"


def resolve_agent_state(
    state_index: dict[tuple[Optional[str], Optional[str], int], list[dict[str, Any]]],
    agent_type: Optional[str],
    uid: Optional[str],
    tick: Optional[int],
) -> Optional[str]:
    if tick is None:
        return None
    rows = state_index.get((agent_type, uid, tick))
    if not rows:
        return None
    return rows[-1].get("state")


def latest_node_at_tick(graph, node_ids: list[str]) -> str:
    return max(node_ids, key=lambda node_id: phase_order(graph.nodes[node_id].get("max_phase")))


def earliest_node_at_tick(graph, node_ids: list[str]) -> str:
    return min(node_ids, key=lambda node_id: phase_order(graph.nodes[node_id].get("min_phase")))


def min_phase(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return current if phase_order(current) <= phase_order(incoming) else incoming


def max_phase(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return current if phase_order(current) >= phase_order(incoming) else incoming


def phase_order(phase: Optional[str]) -> int:
    return PHASE_INDEX.get(phase or "", -1)


def normalized_sign(raw_sign: Optional[str], effect_value: Any, relation: Optional[str]) -> str:
    if raw_sign == "positive":
        return "+"
    if raw_sign == "negative":
        return "-"
    if raw_sign in {"+", "-", "state", "structural"}:
        return raw_sign
    if relation == "state_transition":
        return "state"
    effect = number_or_zero(effect_value)
    if effect > 0:
        return "+"
    if effect < 0:
        return "-"
    return "structural"


def serializable_copy(graph):
    copy = nx.DiGraph(**{key: scalar_attr(value) for key, value in graph.graph.items()})
    for node_id, attrs in graph.nodes(data=True):
        copy.add_node(node_id, **{key: scalar_attr(value) for key, value in attrs.items()})
    for source, target, attrs in graph.edges(data=True):
        copy.add_edge(source, target, **{key: scalar_attr(value) for key, value in attrs.items()})
    return copy


def scalar_attr(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value if item is not None)
    return str(value)


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

def compact_label(values: list[Any]) -> Optional[str]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return str(cleaned[0])
    return "mixed"


def rank_path_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_rank(\d+)\.jsonl$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G0 from simulation causal JSONL logs.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_GRAPH_OUTPUT)
    parser.add_argument("--no-continuity", action="store_true")
    parser.add_argument("--no-field-continuity", action="store_true")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2)
    args = parser.parse_args()
    graph = build_g0(args.log_dir, add_continuity=not args.no_continuity, include_field_continuity=not args.no_field_continuity, rupture_grace_ticks=args.rupture_grace_ticks)
    output = write_g0_gexf(graph, args.output)
    print(f"G0 written to {output} | nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")


if __name__ == "__main__":
    main()
