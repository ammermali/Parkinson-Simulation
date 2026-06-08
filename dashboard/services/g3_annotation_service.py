from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_PATH = PROJECT_ROOT / "output" / "graphs" / "g3_annotations.json"

@dataclass(frozen=True)
class G3NodeAnnotation:
    node_id: str
    biological_label: str = ""
    semantic_note: str = ""
    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.node_id, "biological_label": self.biological_label, "semantic_note": self.semantic_note}

class G3AnnotationService:
    def __init__(self, annotation_path: Path | str = DEFAULT_ANNOTATION_PATH) -> None:
        self.annotation_path = Path(annotation_path)
    def load_annotations(self) -> dict[str, G3NodeAnnotation]:
        if not self.annotation_path.exists():
            return {}
        try:
            payload = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw_items = payload.get("nodes", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw_items, dict):
            return {}
        annotations: dict[str, G3NodeAnnotation] = {}
        for node_id, raw in raw_items.items():
            if not isinstance(raw, dict):
                continue
            annotations[str(node_id)] = G3NodeAnnotation(
                node_id=str(node_id),
                biological_label=str(raw.get("biological_label") or ""),
                semantic_note=str(raw.get("semantic_note") or ""))
        return annotations

    def save_annotation(self, annotation: G3NodeAnnotation) -> None:
        annotations = self.load_annotations()
        if annotation.biological_label.strip() or annotation.semantic_note.strip():
            annotations[annotation.node_id] = annotation
        else:
            annotations.pop(annotation.node_id, None)
        self.save_annotations(annotations)

    def save_annotations(self, annotations: dict[str, G3NodeAnnotation]) -> None:
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "nodes": {node_id: annotation.to_dict() for node_id, annotation in sorted(annotations.items())}}
        temporary = self.annotation_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.annotation_path)

    def apply_annotations(self, graph: nx.DiGraph) -> nx.DiGraph:
        annotations = self.load_annotations()
        for _, attributes in graph.nodes(data=True):
            attributes.pop("biological_label", None)
            attributes.pop("semantic_note", None)
            attributes.pop("display_label", None)
        for node_id, annotation in annotations.items():
            if not graph.has_node(node_id):
                continue
            attributes = graph.nodes[node_id]
            attributes["biological_label"] = annotation.biological_label
            attributes["semantic_note"] = annotation.semantic_note
            if annotation.biological_label:
                attributes["display_label"] = annotation.biological_label
        graph.graph["g3_annotation_file"] = str(self.annotation_path)
        graph.graph["g3_annotation_count"] = len(annotations)
        return graph

    def supernode_rows(self, graph: nx.DiGraph) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        annotations = self.load_annotations()
        for node_id, attributes in graph.nodes(data=True):
            annotation = annotations.get(str(node_id))
            rows.append({"node_id": node_id,
                    "biological_label": annotation.biological_label if annotation else "",
                    "semantic_note": annotation.semantic_note if annotation else "",
                    "label": attributes.get("label"),
                    "pattern_kind": attributes.get("pattern_kind"),
                    "component_size": attributes.get("component_size"),
                    "member_count": attributes.get("member_count"),
                    "semantic_kind": attributes.get("semantic_kind"),
                    "agent_type": attributes.get("agent_type"),
                    "state": attributes.get("state"),
                    "dominant_internal_relation": attributes.get("dominant_internal_relation"),
                    "dominant_internal_mechanism": attributes.get("dominant_internal_mechanism"),
                    "dominant_boundary_in_relation": attributes.get("dominant_boundary_in_relation"),
                    "dominant_boundary_out_relation": attributes.get("dominant_boundary_out_relation"),
                    "node_signature": attributes.get("node_signature"),
                    "internal_edge_signature": attributes.get("internal_edge_signature"),
                    "topological_signature": attributes.get("topological_signature")})
        return sorted(rows,
            key=lambda row: (
                str(row.get("biological_label") or ""),
                str(row.get("pattern_kind") or ""),
                str(row.get("node_id") or "")
            )
        )
