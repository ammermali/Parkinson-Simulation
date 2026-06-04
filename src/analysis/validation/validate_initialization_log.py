from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_SIMULATION_LOG_DIR = Path("output/simulation/logs")
DEFAULT_ANALYSIS_OUTPUT = Path("output/analysis/initialization_validation_latest.json")

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
    parser = argparse.ArgumentParser(description="Validate initialization logs.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT, help="JSON destination for the validation report.")
    parser.add_argument("--stdout", action="store_true", help="Print the report instead of writing output/analysis.")
    args = parser.parse_args()

    report = validate_initialization(args.output_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.stdout:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
