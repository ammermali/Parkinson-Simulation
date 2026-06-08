from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G0_FULL_PATH = PROJECT_ROOT / "output" / "graphs" / "g0.gexf"
DEFAULT_G0_LITE_PATH = PROJECT_ROOT / "output" / "graphs" / "g0.lite.gext"
DEFAULT_G0_PATH = DEFAULT_G0_LITE_PATH
Direction = Literal["incoming", "outgoing", "both"]

@dataclass(frozen=True)
class G0Entity:
    key: str
    label: str
    kind: str
    agent_type: str | None
    uid: str | None
    field: str | None
    owner_uid: str | None

    @property
    def category(self) -> str:
        if self.kind == "environment_field":
            if self.uid == "SN":
                return "Environmental field"
            return "Internal field"
        return self.agent_type or "Agent"

@dataclass(frozen=True)
class G0ViewResult:
    graph: nx.DiGraph
    entity: G0Entity
    start_tick: int
    end_tick: int
    radius: int
    direction: Direction
    include_continuity: bool

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    @property
    def tick_count(self) -> int:
        return self.end_tick - self.start_tick + 1


class G0ViewError(RuntimeError):
    pass

class G0ViewService:
    CONTINUITY_RELATIONS = {"continuity"}

    CONTINUITY_MECHANISMS = {"temporal_identity"}

    CONTINUITY_CAUSAL_KINDS = {"continuity"}

    def __init__(self, graph_path: Path | str | None = None) -> None:
        self.graph_path = Path(graph_path) if graph_path is not None else self.default_graph_path()

    def has_graph(self) -> bool:
        try:
            return self.graph_path.exists() and self.graph_path.is_file() and self.graph_path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def default_graph_path() -> Path:
        try:
            if DEFAULT_G0_LITE_PATH.exists() and DEFAULT_G0_LITE_PATH.stat().st_size > 0:
                return DEFAULT_G0_LITE_PATH
        except OSError:
            pass
        return DEFAULT_G0_FULL_PATH

    def load_graph(self) -> nx.DiGraph:
        if not self.has_graph():
            raise FileNotFoundError(f"G0 graph not found: {self.graph_path}")
        try:
            loaded = nx.read_gexf(self.graph_path, node_type=str)
        except Exception as exc:
            if not self._is_empty_numeric_gexf_error(exc):
                raise G0ViewError(f"Unable to read G0 from `{self.graph_path}`: {exc}") from exc
            loaded = self._load_graph_without_empty_attvalues(exc)
        graph = loaded if isinstance(loaded, nx.DiGraph) else nx.DiGraph(loaded)
        self._normalize_graph_attributes(graph)
        return graph

    def available_ticks(self, graph: nx.DiGraph) -> list[int]:
        ticks: set[int] = set()
        for node_id, attributes in graph.nodes(data=True):
            tick = self._node_tick(node_id, attributes)
            if tick is not None:
                ticks.add(tick)

        return sorted(ticks)

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
            raise G0ViewError(f"Unable to read G0 from `{self.graph_path}`: {original_error}") from exc

    @staticmethod
    def _is_empty_numeric_gexf_error(error: Exception) -> bool:
        return "could not convert string to float" in str(error)

    @staticmethod
    def _xml_namespace(root: ElementTree.Element) -> str | None:
        if root.tag.startswith("{") and "}" in root.tag:
            return root.tag[1:].split("}", 1)[0]
        return None

    @staticmethod
    def _xml_tag(namespace: str | None, local_name: str) -> str:
        if namespace is None:
            return local_name
        return f"{{{namespace}}}{local_name}"

    @staticmethod
    def _xml_local_name(tag: str) -> str:
        if tag.startswith("{") and "}" in tag:
            return tag.rsplit("}", 1)[1]
        return tag

    def available_entities(self, graph: nx.DiGraph) -> list[G0Entity]:
        entities: dict[str, G0Entity] = {}
        for node_id, attributes in graph.nodes(data=True):
            key = self._node_entity_key(node_id, attributes)
            if not key or key in entities:
                continue
            entities[key] = self._entity_from_attributes(key, attributes, node_id=node_id)
        return sorted(entities.values(), key=lambda entity: (entity.category.lower(), entity.label.lower(), entity.key,))

    def available_entity_categories(self, entities: list[G0Entity]) -> list[str]:
        return sorted({entity.category for entity in entities})

    def filter_entities(self, entities: list[G0Entity], *, category: str | None = None, search: str = "") -> list[G0Entity]:
        filtered = entities
        if category and category != "All":
            filtered = [entity for entity in filtered if entity.category == category]
        query = search.strip().lower()
        if query:
            filtered = [
                entity
                for entity in filtered
                if query in entity.label.lower() or query in entity.key.lower() or query in (entity.uid or "").lower() or query in (entity.field or "").lower() or query in (entity.agent_type or "").lower()
            ]
        return filtered

    def build_view(self, graph: nx.DiGraph, *, entity: G0Entity, start_tick: int, end_tick: int, radius: int = 1, direction: Direction = "both", include_continuity: bool = True, max_nodes: int | None = 2_000) -> G0ViewResult:
        self._validate_view_arguments(start_tick=start_tick, end_tick=end_tick, radius=radius, direction=direction)

        temporal_nodes = self._nodes_in_tick_range(graph, start_tick=start_tick, end_tick=end_tick)
        if not temporal_nodes:
            raise G0ViewError("No G0 nodes exist in the selected tick range.")
        temporal_view = graph.subgraph(temporal_nodes).copy()
        if not include_continuity:
            self._remove_continuity_edges(temporal_view)
        seed_nodes = {
            node_id
            for node_id, attributes in temporal_view.nodes(data=True)
            if self._node_entity_key(node_id, attributes) == entity.key
        }
        if not seed_nodes:
            raise G0ViewError("The selected entity has no nodes in the selected tick range.")
        selected_nodes = self._expand_neighborhood(graph=temporal_view, seeds=seed_nodes, radius=radius, direction=direction, max_nodes=max_nodes)
        local_view = temporal_view.subgraph(selected_nodes).copy()
        local_view.graph.update({
                "name": "G0 local view",
                "level": "G0",
                "source_graph": str(self.graph_path),
                "entity_key": entity.key,
                "entity_label": entity.label,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "radius": radius,
                "direction": direction,
                "continuity_included": include_continuity,
                "node_count": local_view.number_of_nodes(),
                "edge_count": local_view.number_of_edges()})

        return G0ViewResult(graph=local_view, entity=entity, start_tick=start_tick, end_tick=end_tick, radius=radius, direction=direction, include_continuity=include_continuity)

    def nodes_frame_rows(self, graph: nx.DiGraph) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node_id, attributes in graph.nodes(data=True):
            rows.append({"node_id": node_id, **attributes})
        return sorted(
            rows,
            key=lambda row: (
                self._as_int(row.get("tick"))
                if self._as_int(row.get("tick")) is not None
                else -1,
                str(row.get("entity_key", "")),
                str(row.get("node_id", ""))
            ))

    def edges_frame_rows(self, graph: nx.DiGraph) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source, target, attributes in graph.edges(data=True):
            rows.append({"source": source, "target": target, **attributes})
        return sorted(
            rows,
            key=lambda row: (
                self._as_int(row.get("tick"))
                if self._as_int(row.get("tick")) is not None
                else -1,
                str(row.get("relation", "")),
                str(row.get("source", "")),
                str(row.get("target", ""))
            )
        )

    @classmethod
    def _nodes_in_tick_range(cls, graph: nx.DiGraph, *, start_tick: int, end_tick: int) -> set[str]:
        return {
            node_id
            for node_id, attributes in graph.nodes(data=True)
            if cls._node_in_tick_range(node_id, attributes, start_tick=start_tick, end_tick=end_tick)
        }

    @classmethod
    def _node_in_tick_range(cls, node_id: str, attributes: dict[str, Any], *, start_tick: int, end_tick: int) -> bool:
        tick = cls._node_tick(node_id, attributes)
        return tick is not None and start_tick <= tick <= end_tick

    @classmethod
    def _remove_continuity_edges(cls, graph: nx.DiGraph) -> None:
        edges_to_remove = [
            (source, target)
            for source, target, attributes in graph.edges(data=True)
            if cls._is_continuity_edge(attributes)
        ]
        graph.remove_edges_from(edges_to_remove)

    @classmethod
    def _is_continuity_edge(cls, attributes: dict[str, Any]) -> bool:
        relation_values = cls._attribute_tokens(attributes.get("relation"))
        mechanism_values = cls._attribute_tokens(attributes.get("mechanism"))
        causal_kind_values = cls._attribute_tokens(attributes.get("causal_kind"))
        return bool(
            relation_values & cls.CONTINUITY_RELATIONS
            or mechanism_values & cls.CONTINUITY_MECHANISMS
            or causal_kind_values & cls.CONTINUITY_CAUSAL_KINDS)

    @staticmethod
    def _expand_neighborhood(*, graph: nx.DiGraph, seeds: set[str], radius: int, direction: Direction, max_nodes: int | None) -> set[str]:
        selected = set(seeds)
        frontier = set(seeds)
        if max_nodes is not None and len(selected) > max_nodes:
            raise G0ViewError(f"The selected entity already exceeds the maximum view size of {max_nodes} nodes.")
        for _ in range(radius):
            next_frontier: set[str] = set()
            for node_id in frontier:
                if direction in {"outgoing", "both"}:
                    next_frontier.update(graph.successors(node_id))
                if direction in {"incoming", "both"}:
                    next_frontier.update(graph.predecessors(node_id))
            next_frontier -= selected
            if not next_frontier:
                break
            if max_nodes is not None and len(selected) + len(next_frontier) > max_nodes:
                raise G0ViewError(f"The requested view would exceed {max_nodes} nodes. Reduce the tick range or causal radius.")
            selected.update(next_frontier)
            frontier = next_frontier
        return selected

    @classmethod
    def _normalize_graph_attributes(cls, graph: nx.DiGraph) -> None:
        for node_id, attributes in graph.nodes(data=True):
            tick = cls._node_tick(node_id, attributes)
            if tick is not None:
                attributes["tick"] = tick
            if not cls._optional_string(attributes.get("entity_key")):
                attributes["entity_key"] = cls._node_entity_key(node_id, attributes)
            rank = cls._as_int(attributes.get("rank"))
            if rank is not None:
                attributes["rank"] = rank
            value = cls._as_float(attributes.get("value"))
            if value is not None:
                attributes["value"] = value
        for _, _, attributes in graph.edges(data=True):
            for key in ("tick", "first_tick", "last_tick", "count", "rank"):
                converted = cls._as_int(attributes.get(key))
                if converted is not None:
                    attributes[key] = converted
            for key in ("effect_value", "total_effect", "mean_effect", "probability", "rng_value"):
                converted = cls._as_float(attributes.get(key))
                if converted is not None:
                    attributes[key] = converted

    @staticmethod
    def _entity_from_attributes(key: str, attributes: dict[str, Any], *, node_id: str) -> G0Entity:
        kind = str(attributes.get("semantic_kind") or attributes.get("kind") or "unknown")
        agent_type = G0ViewService._optional_string(attributes.get("agent_type"))
        uid = G0ViewService._optional_string(attributes.get("uid"))
        field = G0ViewService._optional_string(attributes.get("field"))
        owner_uid = G0ViewService._optional_string(attributes.get("owner_uid"))
        if kind == "unknown" and not any((agent_type, uid, field, owner_uid)):
            label = key or node_id
        elif kind == "environment_field":
            if uid == "SN":
                label = f"Environmental field · {field or 'unknown'}"
            else:
                label = (f"Internal field · {field or 'unknown'} · owner {owner_uid or uid or 'unknown'}")
        else:
            label = f"{agent_type or 'Agent'} · {uid or 'unknown'}"
        return G0Entity(key=key, label=label,kind=kind,agent_type=agent_type,uid=uid,field=field,owner_uid=owner_uid)

    @classmethod
    def _node_tick(cls, node_id: Any, attributes: dict[str, Any]) -> int | None:
        tick = cls._as_int(attributes.get("tick"))
        if tick is not None:
            return tick
        return cls._tick_from_node_id(node_id)

    @classmethod
    def _node_entity_key(cls, node_id: Any, attributes: dict[str, Any]) -> str:
        raw_key = cls._optional_string(attributes.get("entity_key"))
        if raw_key:
            return raw_key
        text = str(node_id)
        if "@" in text:
            return text.rsplit("@", 1)[0]
        return text

    @staticmethod
    def _tick_from_node_id(node_id: Any) -> int | None:
        text = str(node_id)
        if "@" not in text:
            return None
        return G0ViewService._as_int(text.rsplit("@", 1)[1])

    @staticmethod
    def _validate_view_arguments(*, start_tick: int, end_tick: int, radius: int, direction: Direction) -> None:
        if start_tick > end_tick:
            raise ValueError("start_tick cannot be greater than end_tick.")
        if radius < 0:
            raise ValueError("radius cannot be negative.")
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError(f"Unsupported direction: {direction}")

    @staticmethod
    def _attribute_tokens(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if item is not None}
        return {token.strip() for token in str(value).split("|") if token.strip()}

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
