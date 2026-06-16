from __future__ import annotations
from collections import defaultdict
from typing import Any
from multilevelgraphs.contraction_schemes import CompTable, ComponentSet, ContractionScheme
from src.analysis.schemes.contraction_utils import agent_cluster_key, iter_edges, make_digraph, merge_summary_edge, summarize_nodes, summarize_superedges, summarize_supernodes

class AgentClusteringScheme(ContractionScheme):
    def __init__(self):
        super().__init__(supernode_attr_function=self._supernode_attr, superedge_attr_function=self._superedge_attr, c_set_attr_function=self._component_set_attr)

    def contraction_name(self) -> str:
        return "agent_cluster"

    def clone(self):
        return AgentClusteringScheme()

    def contract(self, graph):
        if hasattr(graph, "nodes") and hasattr(graph, "edges") and not hasattr(graph, "V"):
            return self.contract_networkx(graph)
        return super().contract(graph)

    # Fallback method
    def contract_networkx(self, graph):
        g2 = make_digraph()
        g2.graph.update(level="G2", contraction="agent_clustering")
        buckets: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
        node_to_supernode: dict[Any, str] = {}
        for node_id, attrs in graph.nodes(data=True):
            key = agent_cluster_key(node_id, attrs)
            buckets[key].append((node_id, attrs))
            node_to_supernode[node_id] = key
        for key, nodes in sorted(buckets.items()):
            attrs = summarize_nodes(nodes, key=key, contraction="agent_clustering", level="G2")
            attrs["cluster_kind"] = "agent_state_cluster" if attrs.get("semantic_kind") == "agent_state" else "singleton"
            g2.add_node(key, **attrs)
        for source, target, attrs in iter_edges(graph):
            source_key = node_to_supernode[source]
            target_key = node_to_supernode[target]
            if source_key == target_key:
                g2.nodes[source_key]["absorbed_edge_count"] += int(attrs.get("count") or 1)
                continue
            merge_summary_edge(g2, source_key, target_key, attrs)
        return g2

    def contraction_function(self, dec_graph):
        grouped = defaultdict(set)
        for node in dec_graph.nodes():
            grouped[agent_cluster_key(node.key, node.attr)].add(node)
        return CompTable(
            ComponentSet(self._get_component_set_id(), nodes, **self._component_set_attr(nodes))
            for nodes in grouped.values())

    def _component_set_attr(self, nodes: set) -> dict[str, Any]:
        key = agent_cluster_key(next(iter(nodes)).key, next(iter(nodes)).attr)
        attrs = summarize_supernodes(nodes, key=key, contraction="agent_clustering", level="G2")
        attrs["cluster_kind"] = "agent_state_cluster" if attrs.get("semantic_kind") == "agent_state" else "singleton"
        return attrs

    def _supernode_attr(self, supernode) -> dict[str, Any]:
        key = supernode.attr.get("label") or str(supernode.key)
        attrs = summarize_supernodes(supernode.dec.nodes(), key=key, contraction="agent_clustering", level="G2")
        attrs["cluster_kind"] = "agent_state_cluster" if attrs.get("semantic_kind") == "agent_state" else "singleton"
        return attrs

    def _superedge_attr(self, superedge) -> dict[str, Any]:
        return summarize_superedges(superedge.dec)

    def _update_added_node(self, supernode) -> None:
        return None

    def _update_removed_node(self, supernode) -> None:
        return None

    def _update_added_edge(self, superedge) -> None:
        return None

    def _update_removed_edge(self, superedge) -> None:
        return None
