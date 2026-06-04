from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from multilevelgraphs.contraction_schemes import CompTable, ComponentSet, ContractionScheme

from src.analysis.schemes.contraction_utils import (
    iter_edges,
    make_digraph,
    merge_summary_edge,
    summarize_nodes,
    summarize_superedges,
    summarize_supernodes,
    time_group_key,
)


DEFAULT_GRAPH_DIR = Path("output/analysis/graphs")


class TimeContractionScheme(ContractionScheme):
    def __init__(self, window_size: Optional[int] = None):
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be a positive integer or None")
        self.window_size = window_size
        super().__init__(
            supernode_attr_function=self._supernode_attr,
            superedge_attr_function=self._superedge_attr,
            c_set_attr_function=self._component_set_attr,
        )

    def contraction_name(self) -> str:
        if self.window_size is None:
            return "time"
        return f"time_w{self.window_size}"

    def clone(self):
        return TimeContractionScheme(window_size=self.window_size)

    def contract(self, graph):
        """Contract either a NetworkX graph or a multilevelgraphs DecGraph."""

        if hasattr(graph, "nodes") and hasattr(graph, "edges") and not hasattr(graph, "V"):
            return self.contract_networkx(graph)
        return super().contract(graph)

    def contract_networkx(self, g0_graph):
        """Return a NetworkX G1 graph with time removed from G0 nodes."""
        g1 = make_digraph()
        g1.graph.update(level="G1", contraction="time", window_size=self.window_size)
        buckets: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
        node_to_supernode: dict[Any, str] = {}

        for node_id, attrs in g0_graph.nodes(data=True):
            key = time_group_key(node_id, attrs, self.window_size)
            buckets[key].append((node_id, attrs))
            node_to_supernode[node_id] = key

        for key, nodes in sorted(buckets.items()):
            g1.add_node(
                key,
                **summarize_nodes(nodes, key=key, contraction="time", level="G1"),
            )

        for source, target, attrs in iter_edges(g0_graph):
            source_key = node_to_supernode[source]
            target_key = node_to_supernode[target]
            if source_key == target_key:
                g1.nodes[source_key]["absorbed_edge_count"] += int(attrs.get("count") or 1)
                continue
            merge_summary_edge(g1, source_key, target_key, attrs)

        return g1

    def contraction_function(self, dec_graph):
        """Return component sets grouping lower nodes by entity-state identity."""
        grouped = defaultdict(set)
        for node in dec_graph.nodes():
            grouped[time_group_key(node.key, node.attr, self.window_size)].add(node)
        return CompTable(
            ComponentSet(
                self._get_component_set_id(),
                nodes,
                **self._component_set_attr(nodes),
            )
            for nodes in grouped.values()
        )

    def _component_set_attr(self, nodes: set) -> dict[str, Any]:
        key = time_group_key(next(iter(nodes)).key, next(iter(nodes)).attr, self.window_size)
        return summarize_supernodes(nodes, key=key, contraction="time", level="G1")

    def _supernode_attr(self, supernode) -> dict[str, Any]:
        key = supernode.attr.get("label") or str(supernode.key)
        return summarize_supernodes(supernode.dec.nodes(), key=key, contraction="time", level="G1")

    def _superedge_attr(self, superedge) -> dict[str, Any]:
        return summarize_superedges(superedge.dec)

    def _update_added_node(self, node):  # pragma: no cover - dynamic updates are not used in this pipeline.
        raise NotImplementedError("Dynamic updates are not supported by TimeContractionScheme.")

    def _update_removed_node(self, node):  # pragma: no cover
        raise NotImplementedError("Dynamic updates are not supported by TimeContractionScheme.")

    def _update_added_edge(self, edge):  # pragma: no cover
        raise NotImplementedError("Dynamic updates are not supported by TimeContractionScheme.")

    def _update_removed_edge(self, edge):  # pragma: no cover
        raise NotImplementedError("Dynamic updates are not supported by TimeContractionScheme.")
