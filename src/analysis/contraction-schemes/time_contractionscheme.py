from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Set, Tuple
from multilevelgraphs.contraction_schemes import ContractionScheme, CompTable, ComponentSet
from multilevelgraphs.dec_graphs import DecGraph, Supernode, Superedge

def _attr(node: Supernode, key: str, default: None):
    return node.attr.get(key, default)

def _edge_attr(edge: Superedge, key: str, default: None):
    return edge.attr.get(key, default)

def _leaf_nodes(supernode: Supernode) -> list[Supernode]:
    """Returns the represented lower-level nodes."""
    if len(supernode.dec.nodes()) == 0:
        return [supernode]
    leaves: list[Supernode] = []
    for child in supernode.dec.nodes():
        leaves.extend(_leaf_nodes(child))
    return leaves

def _leaf_edges(superedge: Superedge) -> list[Superedge]:
    """Returns the represented lower-level edges."""
    if len(superedge.dec.edges()) == 0:
        return [superedge]
    edges: list[Superedge] = []
    for child in superedge.dec.edges():
        edges.extend(_leaf_edges(child))
    return edges

def aggregate_supernode_attrs(supernode: Supernode) -> Dict[str, Any]:
    leaves = _leaf_nodes(supernode)
    ticks = [
        _attr(n, "tick")
        for n in leaves
        if _attr(n, "tick") is not None
    ]
    agent_ids = {
        _attr(n, "agent_id")
        for n in leaves
        if _attr(n, "agent_id") is not None
    }
    agent_types = {
        _attr(n, "agent_type")
        for n in leaves
        if _attr(n, "agent_type") is not None
    }
    return {
        "size": len(leaves),
        "first_seen": min(ticks) if ticks else None,
        "last_seen": max(ticks) if ticks else None,
        "support_agent_ids": sorted(agent_ids),
        "support": len(agent_ids),
        "agent_types": sorted(agent_types)
    }

def aggregate_superedge_attrs(superedge: Superedge) -> Dict[str, Any]:
    leaves = _leaf_edges(superedge)
    effects = [
        float(_edge_attr(e, "effect", 0.0) or 0.0)
        for e in leaves
    ]
    ticks = [
        _edge_attr(e, "tick")
        for e in leaves
        if _edge_attr(e, "tick") is not None
    ]
    signs = {
        _edge_attr(e, "sign")
        for e in leaves
        if _edge_attr(e, "sign") is not None
    }
    total_effect = sum(effects)
    count = len(leaves)
    return {
        "count": count,
        "total_effect": total_effect,
        "mean_effect": total_effect / count if count else 0.0,
        "first_seen": min(ticks) if ticks else None,
        "last_seen": max(ticks) if ticks else None,
        "signs": sorted(signs),
        "decomposes_to_count": count
    }
# TODO complete
