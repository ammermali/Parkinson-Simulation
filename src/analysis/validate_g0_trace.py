from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

VALID_RELATIONS = {
    "state_transition",
    "threshold_trigger",
    "action_selection",
    "field_effect",
    "internal_field_effect",
    "agent_to_agent",
    "aggregation",
    "degradation",
    "target_assignment",
    "buffer_commit",
    "lifecycle",
    "structural"
}

PHASE_INDEX = {
    "0_pre_state": 0,
    "1_perception": 1,
    "2_state_update": 2,
    "3_action_selection": 3,
    "4_effect_buffer": 4,
    "5_commit": 5
}

FORBIDDEN_RUNTIME_FIELDS = {"position", "x", "y", "z", "config", "raw_perception", "details_json"}


def load_jsonl(path: Path) -> tuple[list[dict], dict]:
    stats = {"valid": 0, "malformed": 0, "blank": 0, "examples": []}
    rows: list[dict] = []
    if not path.exists():
        return rows, stats
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                stats["blank"] += 1
                continue
            try:
                rows.append(json.loads(line))
                stats["valid"] += 1
            except json.JSONDecodeError:
                stats["malformed"] += 1
                if len(stats["examples"]) < 10:
                    stats["examples"].append({"line": line_no, "text": line[:160].rstrip()})
    return rows, stats


def validate_trace(output_dir: Path) -> dict:
    nodes, node_stats = load_jsonl(output_dir / "g0_nodes.jsonl")
    edges, edge_stats = load_jsonl(output_dir / "g0_edges.jsonl")
    node_rank_files = sorted(output_dir.glob("g0_nodes_rank*.jsonl"))
    edge_rank_files = sorted(output_dir.glob("g0_edges_rank*.jsonl"))
    if not nodes and node_rank_files:
        nodes, node_stats = load_many_jsonl(node_rank_files)
    if not edges and edge_rank_files:
        edges, edge_stats = load_many_jsonl(edge_rank_files)
    relation_counts = Counter(edge.get("relation") for edge in edges)
    invalid_relation_count = sum(1 for edge in edges if edge.get("relation") not in VALID_RELATIONS)
    missing_source_or_target = sum(1 for edge in edges if not edge.get("source_node_id") or not edge.get("target_node_id"))
    missing_agent_targets = sum(
        1
        for edge in edges
        if edge.get("relation") in {"agent_to_agent", "degradation", "target_assignment", "aggregation"}
        and not edge.get("target_uid")
    )
    phase_violations = sum(
        1
        for edge in edges
        if PHASE_INDEX.get(edge.get("phase_from"), -1) > PHASE_INDEX.get(edge.get("phase_to"), -1)
    )
    forbidden_field_rows = [
        edge.get("edge_id")
        for edge in edges
        if any(field in edge for field in FORBIDDEN_RUNTIME_FIELDS)
    ]
    field_effect_totals = defaultdict(float)
    for edge in edges:
        if edge.get("relation") in {"field_effect", "internal_field_effect", "buffer_commit"}:
            field_effect_totals[edge.get("target_field")] += abs(edge.get("effect_value") or 0.0)
    return {
        "node_jsonl": node_stats,
        "edge_jsonl": edge_stats,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "rank_files": {
            "nodes": [path.name for path in node_rank_files],
            "edges": [path.name for path in edge_rank_files]
        },
        "edges_by_relation": dict(relation_counts),
        "missing_source_or_target": missing_source_or_target,
        "agent_edges_missing_target_uid": missing_agent_targets,
        "invalid_relation_count": invalid_relation_count,
        "phase_order_violations": phase_violations,
        "forbidden_runtime_field_rows": forbidden_field_rows[:20],
        "top_field_effects": sorted(field_effect_totals.items(), key=lambda item: item[1], reverse=True)[:10],
        "g1_preview": [
            (edge.get("g1_source_key"), edge.get("g1_target_key"), edge.get("relation"))
            for edge in edges[:10]
        ],
        "g2_preview": [
            (edge.get("g2_source_key"), edge.get("g2_target_key"), edge.get("relation"))
            for edge in edges[:10]
        ]
    }


def load_many_jsonl(paths: list[Path]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    total = {"valid": 0, "malformed": 0, "blank": 0, "examples": []}
    for path in paths:
        path_rows, stats = load_jsonl(path)
        rows.extend(path_rows)
        for key in ("valid", "malformed", "blank"):
            total[key] += stats[key]
        for example in stats["examples"]:
            if len(total["examples"]) < 10:
                total["examples"].append({"file": path.name, **example})
    return rows, total


def main() -> None:
    output_dir = Path("src/simulation/output/logs")
    report = validate_trace(output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
