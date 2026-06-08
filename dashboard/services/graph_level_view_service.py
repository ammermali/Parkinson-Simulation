from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import networkx as nx

from dashboard.services.g0_view_service import Direction, G0Entity, G0ViewError, G0ViewService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATHS = {
    "G1": PROJECT_ROOT / "output" / "graphs" / "g1.gexf",
    "G2": PROJECT_ROOT / "output" / "graphs" / "g2.gexf",
    "G3": PROJECT_ROOT / "output" / "graphs" / "g3.gexf"}


@dataclass(frozen=True)
class GraphLevelViewResult:
    graph: nx.DiGraph
    level: str
    entity: G0Entity | None = None
    radius: int = 0
    direction: Direction = "both"
    include_continuity: bool = True

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()


class GraphLevelViewService(G0ViewService):
    def __init__(self, level: str, graph_path: Path | str | None = None) -> None:
        self.level = level.upper()
        if self.level not in DEFAULT_GRAPH_PATHS:
            raise ValueError(f"Unsupported graph level: {level}")
        self.graph_path = Path(graph_path) if graph_path is not None else DEFAULT_GRAPH_PATHS[self.level]

    def load_graph(self) -> nx.DiGraph:
        if not self.has_graph():
            raise FileNotFoundError(f"{self.level} graph not found: {self.graph_path}")
        try:
            loaded = nx.read_gexf(self.graph_path, node_type=str)
        except Exception as exc:
            if not self._is_empty_numeric_gexf_error(exc):
                raise G0ViewError(f"Unable to read {self.level} from `{self.graph_path}`: {exc}") from exc
            loaded = self._load_graph_without_empty_attvalues(exc)
        graph = loaded if isinstance(loaded, nx.DiGraph) else nx.DiGraph(loaded)
        self._normalize_graph_attributes(graph)
        self._normalize_contracted_attributes(graph)
        return graph

    def build_entity_view(self, graph: nx.DiGraph, *, entity: G0Entity, radius: int = 1, direction: Direction = "both", include_continuity: bool = True, max_nodes: int | None = 2_000) -> GraphLevelViewResult:
        if radius < 0:
            raise ValueError("radius cannot be negative.")
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError(f"Unsupported direction: {direction}")
        working_graph = graph.copy()
        if not include_continuity:
            self._remove_continuity_edges(working_graph)
        seed_nodes = {
            node_id
            for node_id, attributes in working_graph.nodes(data=True)
            if self._node_entity_key(node_id, attributes) == entity.key}
        if not seed_nodes:
            raise G0ViewError("The selected entity has no nodes in this contracted graph.")
        selected_nodes = self._expand_neighborhood(graph=working_graph, seeds=seed_nodes, radius=radius, direction=direction, max_nodes=max_nodes)
        local_view = working_graph.subgraph(selected_nodes).copy()
        local_view.graph.update(
            name=f"{self.level} local view",
            level=self.level,
            source_graph=str(self.graph_path),
            entity_key=entity.key,
            entity_label=entity.label,
            radius=radius,
            direction=direction,
            continuity_included=include_continuity,
            node_count=local_view.number_of_nodes(),
            edge_count=local_view.number_of_edges())
        return GraphLevelViewResult(
            graph=local_view,
            level=self.level,
            entity=entity,
            radius=radius,
            direction=direction,
            include_continuity=include_continuity)

    def build_full_view(self, graph: nx.DiGraph) -> GraphLevelViewResult:
        full_view = graph.copy()
        full_view.graph.update(
            name=f"{self.level} full view",
            level=self.level,
            source_graph=str(self.graph_path),
            node_count=full_view.number_of_nodes(),
            edge_count=full_view.number_of_edges())
        return GraphLevelViewResult(graph=full_view, level=self.level)

    def _load_graph_without_empty_attvalues(self, original_error: Exception) -> nx.Graph:
        try:
            tree = ElementTree.parse(self.graph_path)
            root = tree.getroot()
            namespace = self._xml_namespace(root)
            for attvalues in root.iter(self._xml_tag(namespace, "attvalues")):
                for attvalue in list(attvalues):
                    if self._xml_local_name(attvalue.tag) == "attvalue" and attvalue.get("value") == "":
                        attvalues.remove(attvalue)
            stream = StringIO(ElementTree.tostring(root, encoding="unicode"))
            return nx.read_gexf(stream, node_type=str)
        except Exception as exc:
            raise G0ViewError(f"Unable to read {self.level} from `{self.graph_path}`: {original_error}") from exc

    @classmethod
    def _normalize_contracted_attributes(cls, graph: nx.DiGraph) -> None:
        node_int_keys = (
            "first_seen",
            "last_seen",
            "member_count",
            "observation_count",
            "absorbed_node_count",
            "absorbed_edge_count",
            "component_size",
            "internal_edge_count",
            "boundary_in_edge_count",
            "boundary_out_edge_count")
        edge_int_keys = ("first_seen", "last_seen", "lower_edge_count", "weight")
        for _, attributes in graph.nodes(data=True):
            for key in node_int_keys:
                converted = cls._as_int(attributes.get(key))
                if converted is not None:
                    attributes[key] = converted
        for _, _, attributes in graph.edges(data=True):
            for key in edge_int_keys:
                converted = cls._as_int(attributes.get(key))
                if converted is not None:
                    attributes[key] = converted