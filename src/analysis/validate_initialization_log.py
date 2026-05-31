from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


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


def validate_initialization(output_dir: Path) -> dict:
    agents, jsonl_stats = load_jsonl(output_dir / "initialization_agents.jsonl")
    manifest_path = output_dir / "initialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    by_class = Counter(agent.get("agent_class") for agent in agents)
    by_rank = Counter(str(agent.get("rank")) for agent in agents)
    by_state = Counter((agent.get("agent_class"), agent.get("initial_state")) for agent in agents)
    intracellular_missing_owner = sum(
        1
        for agent in agents
        if agent.get("display", {}).get("visual_level") == "intracellular"
        and not agent.get("owner_uid")
    )
    missing_config = sum(
        1
        for agent in agents
        if agent.get("agent_class") in {"Neuron", "AlphaSynuclein", "Mitochondrion", "Lysosome", "Microglia", "Astrocyte"}
        and agent.get("config") is None
    )
    malformed_positions = sum(
        1
        for agent in agents
        if agent.get("position") is not None
        and not {"x", "y", "z"}.issubset(agent["position"])
    )
    return {
        "jsonl": jsonl_stats,
        "total_initialized_agents": len(agents),
        "counts_by_class": dict(by_class),
        "counts_by_rank": dict(by_rank),
        "counts_by_initial_state": {f"{key[0]}:{key[1]}": value for key, value in by_state.items()},
        "agents_missing_uid": sum(1 for agent in agents if not agent.get("uid")),
        "agents_missing_initial_state": sum(1 for agent in agents if not agent.get("initial_state")),
        "intracellular_agents_missing_owner_uid": intracellular_missing_owner,
        "agents_missing_config": missing_config,
        "malformed_positions": malformed_positions,
        "neuron_containment_summary": manifest.get("neurons", {})
    }


def main() -> None:
    output_dir = Path("src/simulation/output/logs")
    report = validate_initialization(output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
