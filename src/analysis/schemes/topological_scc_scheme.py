from __future__ import annotations
import hashlib
from collections import Counter
from typing import Any
import networkx as nx
from multilevelgraphs.contraction_schemes import CompTable, ComponentSet, EdgeBasedContractionScheme
from src.analysis.schemes.contraction_utils import append_many, clean_token, iter_edges, make_digraph, merge_summary_edge, numeric, summarize_nodes, summarize_superedges, summarize_supernodes

class TopologicalSCCContractionScheme(EdgeBasedContractionScheme):
    def __init__(self):
        super().__init__(supernode_attr_function=self._supernode_attr, superedge_attr_function=self._superedge_attr, c_set_attr_function=self._component_set_attr)

    def contraction_name(self) -> str:
        return "topological_scc"

    def clone(self):
        return TopologicalSCCContractionScheme()

    def contract(self, graph):
        if hasattr(graph, "nodes") and hasattr(graph, "edges") and not hasattr(graph, "V"):
            return self.contract_networkx(graph)
        return super().contract(graph)

    def contract_networkx(self, graph):
        g3 = make_digraph()
        g3.graph.update(level="G3", contraction="topological_scc")
        components = sorted(nx.strongly_connected_components(graph), key=lambda nodes: sorted(str(node) for node in nodes)[0])
        node_to_component: dict[Any, str] = {}
        for nodes in components:
            key = component_key(nodes)
            for node in nodes:
                node_to_component[node] = key
            g3.add_node(key, **topological_component_attrs(graph, key, nodes))
        for source, target, attrs in iter_edges(graph):
            source_key = node_to_component[source]
            target_key = node_to_component[target]
            if source_key == target_key:
                g3.nodes[source_key]["absorbed_edge_count"] += int(attrs.get("count") or 1)
                continue
            merge_summary_edge(g3, source_key, target_key, attrs)
        for node_id in g3.nodes:
            g3.nodes[node_id]["boundary_in_superedge_count"] = g3.in_degree(node_id)
            g3.nodes[node_id]["boundary_out_superedge_count"] = g3.out_degree(node_id)
        return g3

    def contraction_function(self, dec_graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(dec_graph.nodes())
        for edge in dec_graph.edges():
            graph.add_edge(edge.tail, edge.head)
        return CompTable(ComponentSet(self._get_component_set_id(), set(nodes), **self._component_set_attr(set(nodes))) for nodes in nx.strongly_connected_components(graph))

    def _component_set_attr(self, nodes: set) -> dict[str, Any]:
        key = component_key(node.key for node in nodes)
        attrs = summarize_supernodes(nodes, key=key, contraction="topological_scc", level="G3")
        attrs["pattern_kind"] = "feedback_component" if len(nodes) > 1 else "singleton_process"
        attrs["component_size"] = len(nodes)
        return attrs

    def _supernode_attr(self, supernode) -> dict[str, Any]:
        key = supernode.attr.get("label") or str(supernode.key)
        attrs = summarize_supernodes(supernode.dec.nodes(), key=key, contraction="topological_scc", level="G3")
        attrs["pattern_kind"] = "feedback_component" if len(supernode.dec.nodes()) > 1 else "singleton_process"
        attrs["component_size"] = len(supernode.dec.nodes())
        return attrs

    def _superedge_attr(self, superedge) -> dict[str, Any]:
        return summarize_superedges(superedge.dec)

    def _update_added_edge(self, superedge) -> None:
        return None

    def _update_removed_edge(self, superedge) -> None:
        return None

def topological_component_attrs(graph, key: str, nodes: set[Any]) -> dict[str, Any]:
    ordered_nodes = sorted(nodes, key=str)
    internal_edges = [
        (source, target, attrs)
        for source, target, attrs in iter_edges(graph)
        if source in nodes and target in nodes
    ]
    incoming_edges = [
        (source, target, attrs)
        for source, target, attrs in iter_edges(graph)
        if source not in nodes and target in nodes
    ]
    outgoing_edges = [
        (source, target, attrs)
        for source, target, attrs in iter_edges(graph)
        if source in nodes and target not in nodes
    ]
    attrs = summarize_nodes(
        ((node, graph.nodes[node]) for node in ordered_nodes),
        key=key,
        contraction="topological_scc",
        level="G3",
    )
    self_loop = len(nodes) == 1 and any(source == target for source, target, _ in internal_edges)
    pattern_kind = "feedback_component" if len(nodes) > 1 else "self_feedback" if self_loop else "singleton_process"
    attrs.update(
        {
            "pattern_kind": pattern_kind,
            "component_size": len(nodes),
            "is_feedback_pattern": pattern_kind in {"feedback_component", "self_feedback"},
            "internal_edge_count": len(internal_edges),
            "boundary_in_edge_count": len(incoming_edges),
            "boundary_out_edge_count": len(outgoing_edges),
            "internal_event_count": edge_event_count(internal_edges),
            "boundary_in_event_count": edge_event_count(incoming_edges),
            "boundary_out_event_count": edge_event_count(outgoing_edges),
            "total_internal_effect": edge_total_effect(internal_edges),
            "mean_internal_effect": edge_mean_effect(internal_edges),
            "total_boundary_in_effect": edge_total_effect(incoming_edges),
            "mean_boundary_in_effect": edge_mean_effect(incoming_edges),
            "total_boundary_out_effect": edge_total_effect(outgoing_edges),
            "mean_boundary_out_effect": edge_mean_effect(outgoing_edges),
            "dominant_internal_relation": dominant_edge_attr(internal_edges, "relation"),
            "dominant_internal_mechanism": dominant_edge_attr(internal_edges, "mechanism"),
            "dominant_internal_causal_kind": dominant_edge_attr(internal_edges, "causal_kind"),
            "dominant_internal_sign": dominant_edge_attr(internal_edges, "sign"),
            "dominant_boundary_in_relation": dominant_edge_attr(incoming_edges, "relation"),
            "dominant_boundary_out_relation": dominant_edge_attr(outgoing_edges, "relation"),
            "node_signature": node_signature(graph, ordered_nodes),
            "internal_edge_signature": edge_signature(internal_edges),
            "boundary_in_signature": edge_signature(incoming_edges),
            "boundary_out_signature": edge_signature(outgoing_edges),
            "topological_signature": topological_signature(graph, ordered_nodes, internal_edges),
            "component_node_ids": list(ordered_nodes)
        })
    attrs["label"] = pattern_label(attrs)
    return attrs


def component_key(nodes) -> str:
    ordered = sorted(str(node) for node in nodes)
    if len(ordered) == 1:
        return f"G3_{clean_token(ordered[0])}"
    digest = hashlib.sha1("|".join(ordered).encode("utf-8")).hexdigest()[:12]
    return f"G3_feedback_{digest}"


def pattern_label(attrs: dict[str, Any]) -> str:
    kind = attrs.get("pattern_kind")
    size = attrs.get("component_size")
    signature = attrs.get("topological_signature") or attrs.get("node_signature")
    return f"{kind} size={size} | {signature}"


def node_signature(graph, nodes: list[Any]) -> str:
    counter = Counter(node_role(graph.nodes[node]) for node in nodes)
    return format_counter(counter)


def edge_signature(edges: list[tuple[Any, Any, dict[str, Any]]]) -> str:
    counter = Counter(edge_role(attrs) for _, _, attrs in edges)
    return format_counter(counter)


def topological_signature(graph, nodes: list[Any], internal_edges: list[tuple[Any, Any, dict[str, Any]]]) -> str:
    node_part = node_signature(graph, nodes)
    edge_part = edge_signature(internal_edges)
    return f"{node_part} :: {edge_part or 'no_internal_edges'}"


def node_role(attrs: dict[str, Any]) -> str:
    if attrs.get("semantic_kind") == "environment_field":
        field = attrs.get("field") or "field"
        if attrs.get("agent_type") == "SubstantiaNigra" or attrs.get("uid") == "SN":
            return f"SNField:{field}"
        if attrs.get("agent_type") == "Neuron":
            return "NeuronField"
        return f"Field:{field}"
    agent_type = attrs.get("agent_type") or "Agent"
    state = attrs.get("state") or "unknown"
    return f"{agent_type}:{state}"


def edge_role(attrs: dict[str, Any]) -> str:
    relation = attrs.get("relation") or attrs.get("causal_kind") or "edge"
    mechanism = attrs.get("mechanism")
    if mechanism and mechanism != "mixed":
        return f"{relation}:{mechanism}"
    return str(relation)


def format_counter(counter: Counter) -> str:
    if not counter:
        return ""
    return "|".join(f"{key}={value}" for key, value in sorted(counter.items()))


def dominant_edge_attr(edges: list[tuple[Any, Any, dict[str, Any]]], attr_name: str) -> Any:
    counter: Counter = Counter()
    for _, _, attrs in edges:
        values: list[Any] = []
        append_many(values, attrs.get(f"{attr_name}s", attrs.get(attr_name)))
        if not values:
            continue
        weight = int(attrs.get("count") or 1)
        for value in values:
            counter[value] += weight
    if not counter:
        return None
    if len(counter) == 1:
        return next(iter(counter))
    most_common = counter.most_common(2)
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "mixed"
    return most_common[0][0]


def edge_event_count(edges: list[tuple[Any, Any, dict[str, Any]]]) -> int:
    return sum(int(attrs.get("count") or 1) for _, _, attrs in edges)


def edge_total_effect(edges: list[tuple[Any, Any, dict[str, Any]]]) -> float:
    return float(sum(numeric(attrs.get("total_effect", attrs.get("effect_value"))) for _, _, attrs in edges))


def edge_mean_effect(edges: list[tuple[Any, Any, dict[str, Any]]]) -> float:
    event_count = edge_event_count(edges)
    return edge_total_effect(edges) / event_count if event_count else 0.0
