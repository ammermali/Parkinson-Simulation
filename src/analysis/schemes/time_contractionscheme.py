from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import networkx as nx
from src.analysis.g0_lexer import DEFAULT_LOG_DIR, build_g0_graph, graph_summary, write_graph

DEFAULT_GRAPH_DIR = Path("output/analysis/graphs")
DEFAULT_G1_OUTPUT = DEFAULT_GRAPH_DIR / "g1_time.gexf"

AGENT_STATE_KIND = "agent_state"
ENVIRONMENT_FIELD_KIND = "environment_field"


class TimeContraptionError(RuntimeError):
    """Raised when G1 construction cannot proceed."""


@dataclass(frozen=True)
class TimeContraptionConfig:
    keep_internal_edge_details: bool = True
    graph_level: str = "G1"


class TimeContraption:
    def __init__(self, config: Optional[TimeContraptionConfig] = None):
        self.config = config or TimeContraptionConfig()

    def contraction_name(self) -> str:
        return "time_contraption_state_preserving"

    def clone(self) -> "TimeContraption":
        return TimeContraption(self.config)

    def contract(self, g0_graph) -> "nx.DiGraph":
        require_networkx()
        g1_graph = nx.DiGraph()
        g1_graph.graph.update({
            "level": self.config.graph_level,
            "source_level": g0_graph.graph.get("level", "G0"),
            "contraction": self.contraction_name(),
            "node_semantics": "AgentNameID_State or Environment_Field",
            "edge_semantics": "time-compacted causal interaction"
        })

        node_to_supernode: dict[str, str] = {}
        for node_id, attrs in g0_graph.nodes(data=True):
            supernode_id = self.supernode_id(node_id, attrs)
            node_to_supernode[node_id] = supernode_id
            self._add_or_merge_supernode(g1_graph, supernode_id, node_id, attrs)

        for edge in self._iter_edges(g0_graph):
            source_id, target_id, attrs = edge
            source_supernode = node_to_supernode.get(source_id)
            target_supernode = node_to_supernode.get(target_id)
            if source_supernode is None or target_supernode is None:
                continue
            if source_supernode == target_supernode:
                self._absorb_internal_edge(g1_graph, source_supernode, attrs)
                continue
            self._add_or_merge_superedge(g1_graph, source_supernode, target_supernode, attrs)
        self._finalize_graph(g1_graph)
        return g1_graph

    def contract_from_logs(self, log_dir: Path | str = DEFAULT_LOG_DIR, **g0_options) -> "nx.DiGraph":
        g0_graph = build_g0_graph(log_dir, **g0_options)
        return self.contract(g0_graph)

    def supernode_id(self, node_id: str, attrs: dict[str, Any]) -> str:
        semantic_kind = attrs.get("semantic_kind")
        if semantic_kind == AGENT_STATE_KIND:
            return agent_state_supernode_id(attrs)
        if semantic_kind == ENVIRONMENT_FIELD_KIND:
            return environment_field_supernode_id(attrs)
        return generic_supernode_id(node_id, attrs)

    def _add_or_merge_supernode(self, graph, supernode_id: str, g0_node_id: str, attrs: dict[str, Any]) -> None:
        tick = to_int(attrs.get("tick"))
        if not graph.has_node(supernode_id):
            graph.add_node(
                supernode_id,
                node_id=supernode_id,
                label=supernode_id,
                level=self.config.graph_level,
                source_level="G0",
                semantic_kind=attrs.get("semantic_kind"),
                entity_key=attrs.get("entity_key"),
                agent_type=attrs.get("agent_type"),
                agent_uid=attrs.get("agent_uid") or attrs.get("uid"),
                state=attrs.get("state"),
                field=attrs.get("field"),
                owner_uid=attrs.get("owner_uid"),
                compartment=attrs.get("compartment"),
                first_seen=tick,
                last_seen=tick,
                observation_count=0,
                absorbed_edge_count=0,
                absorbed_total_effect=0.0,
                absorbed_relations=[],
                g0_node_ids=[]
            )
        node_attrs = graph.nodes[supernode_id]
        node_attrs["observation_count"] += 1
        append_unique(node_attrs["g0_node_ids"], g0_node_id)
        node_attrs["first_seen"] = min_tick(node_attrs.get("first_seen"), tick)
        node_attrs["last_seen"] = max_tick(node_attrs.get("last_seen"), tick)
        merge_first_non_empty(node_attrs, "semantic_kind", attrs.get("semantic_kind"))
        merge_first_non_empty(node_attrs, "entity_key", attrs.get("entity_key"))
        merge_first_non_empty(node_attrs, "agent_type", attrs.get("agent_type"))
        merge_first_non_empty(node_attrs, "agent_uid", attrs.get("agent_uid") or attrs.get("uid"))
        merge_first_non_empty(node_attrs, "state", attrs.get("state"))
        merge_first_non_empty(node_attrs, "field", attrs.get("field"))
        merge_first_non_empty(node_attrs, "owner_uid", attrs.get("owner_uid"))
        merge_first_non_empty(node_attrs, "compartment", attrs.get("compartment"))

    def _absorb_internal_edge(self, graph, supernode_id: str, attrs: dict[str, Any]) -> None:
        node_attrs = graph.nodes[supernode_id]
        effect = numeric_effect(attrs)
        node_attrs["absorbed_edge_count"] += 1
        node_attrs["absorbed_total_effect"] += effect
        append_unique(node_attrs["absorbed_relations"], attrs.get("relation"))
        if self.config.keep_internal_edge_details:
            details = node_attrs.setdefault("absorbed_edge_ids", [])
            append_unique(details, attrs.get("edge_id") or attrs.get("g0_edge_id"))

    def _add_or_merge_superedge(self, graph, source: str, target: str, attrs: dict[str, Any]) -> None:
        effect = numeric_effect(attrs)
        tick = to_int(attrs.get("tick"))
        if not graph.has_edge(source, target):
            graph.add_edge(
                source,
                target,
                edge_id=f"{source}->{target}",
                label=f"{source}->{target}",
                level=self.config.graph_level,
                count=0,
                total_effect=0.0,
                mean_effect=0.0,
                first_seen=tick,
                last_seen=tick,
                sign="structural",
                signs=[],
                relations=[],
                mechanisms=[],
                causal_kinds=[],
                actions=[],
                g0_edge_ids=[]
            )
        edge_attrs = graph.edges[source, target]
        edge_attrs["count"] += 1
        edge_attrs["total_effect"] += effect
        edge_attrs["first_seen"] = min_tick(edge_attrs.get("first_seen"), tick)
        edge_attrs["last_seen"] = max_tick(edge_attrs.get("last_seen"), tick)
        append_unique(edge_attrs["relations"], attrs.get("relation"))
        append_unique(edge_attrs["mechanisms"], attrs.get("mechanism"))
        append_unique(edge_attrs["causal_kinds"], attrs.get("causal_kind"))
        append_unique(edge_attrs["actions"], attrs.get("action"))
        append_unique(edge_attrs["g0_edge_ids"], attrs.get("edge_id") or attrs.get("g0_edge_id"))
        append_unique(edge_attrs["signs"], normalize_sign(attrs))

    def _finalize_graph(self, graph) -> None:
        for _, node_attrs in graph.nodes(data=True):
            count = node_attrs.get("absorbed_edge_count", 0)
            total = node_attrs.get("absorbed_total_effect", 0.0)
            node_attrs["absorbed_mean_effect"] = total / count if count else 0.0
        for _, _, edge_attrs in graph.edges(data=True):
            count = edge_attrs.get("count", 0)
            total = edge_attrs.get("total_effect", 0.0)
            edge_attrs["mean_effect"] = total / count if count else 0.0
            edge_attrs["sign"] = compacted_sign(edge_attrs)
            edge_attrs["weight"] = count
        graph.graph["node_count"] = graph.number_of_nodes()
        graph.graph["edge_count"] = graph.number_of_edges()

    def _iter_edges(self, graph) -> Iterable[tuple[str, str, dict[str, Any]]]:
        if graph.is_multigraph():
            for source, target, _, attrs in graph.edges(keys=True, data=True):
                yield source, target, attrs
            return
        for source, target, attrs in graph.edges(data=True):
            yield source, target, attrs


def build_time_contracted_graph(g0_graph) -> "nx.DiGraph":
    return TimeContraption().contract(g0_graph)


TimeContractionScheme = TimeContraption


def export_time_contracted_graph(log_dir: Path | str = DEFAULT_LOG_DIR, output: Path | str = DEFAULT_G1_OUTPUT, *, g0_multigraph: bool = True, strict: bool = False) -> "nx.DiGraph":
    g1_graph = TimeContraption().contract_from_logs(
        log_dir,
        multigraph=g0_multigraph,
        strict=strict,
    )
    write_graph(g1_graph, Path(output))
    return g1_graph

def agent_state_supernode_id(attrs: dict[str, Any]) -> str:
    agent_type = attrs.get("agent_type") or "Agent"
    uid = attrs.get("agent_uid") or attrs.get("uid") or "unknown"
    state = attrs.get("state") or "unknown"
    return f"{agent_type}_{uid}_{state}"


def environment_field_supernode_id(attrs: dict[str, Any]) -> str:
    field = attrs.get("field") or "field"
    uid = attrs.get("uid")
    agent_type = attrs.get("agent_type")
    if uid == "SN" or agent_type == "SubstantiaNigra":
        return f"SN_{field}"
    owner_uid = attrs.get("owner_uid") or uid or "unknown"
    return f"Neuron_{owner_uid}_{field}"


def generic_supernode_id(node_id: str, attrs: dict[str, Any]) -> str:
    display_id = attrs.get("display_id")
    if isinstance(display_id, str) and "@" in display_id:
        return display_id.rsplit("@", 1)[0]
    return str(node_id).rsplit("@", 1)[0]


def numeric_effect(attrs: dict[str, Any]) -> float:
    value = attrs.get("effect")
    if value is None:
        value = attrs.get("effect_value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_sign(attrs: dict[str, Any]) -> str:
    relation = attrs.get("relation")
    if relation == "state_transition":
        return "state"
    sign = attrs.get("sign") or attrs.get("effect_sign")
    if sign in {"+", "positive"}:
        return "+"
    if sign in {"-", "negative"}:
        return "-"
    return "structural"


def compacted_sign(edge_attrs: dict[str, Any]) -> str:
    total_effect = edge_attrs.get("total_effect", 0.0)
    if total_effect > 0:
        return "+"
    if total_effect < 0:
        return "-"
    signs = set(edge_attrs.get("signs") or [])
    if "state" in signs:
        return "state"
    return "structural"


def append_unique(values: list[Any], value: Any) -> None:
    if value is None or value == "":
        return
    if value not in values:
        values.append(value)


def merge_first_non_empty(attrs: dict[str, Any], key: str, value: Any) -> None:
    if attrs.get(key) in {None, ""} and value not in {None, ""}:
        attrs[key] = value


def min_tick(left: Optional[int], right: Optional[int]) -> Optional[int]:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def max_tick(left: Optional[int], right: Optional[int]) -> Optional[int]:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def require_networkx() -> None:
    if nx is None:
        raise TimeContraptionError("networkx is required to build time-contracted G1 graphs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G1 by time-contracting the G0 causal graph.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_G1_OUTPUT, help="G1 destination: .gexf, .graphml, .json, .pkl.")
    parser.add_argument("--strict", action="store_true", help="Raise on malformed JSONL rows while building G0.")
    parser.add_argument("--digraph-g0", action="store_true", help="Build G0 as DiGraph instead of MultiDiGraph before contraction.")
    parser.add_argument("--summary", action="store_true", help="Print G1 counts after building.")
    args = parser.parse_args()
    graph = export_time_contracted_graph(
        args.log_dir,
        args.output,
        g0_multigraph=not args.digraph_g0,
        strict=args.strict
    )
    if args.summary:
        print(json.dumps(graph_summary(graph), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# TODO furtherly prepare this scheme for multilevel graph integration
