import json
from types import SimpleNamespace

from src.simulation.logger.causal_trace_logger import CausalTraceLogger
from src.simulation.logger.initialization_logger import InitializationLogger


class TestCausalTraceLogger:
    def test_writes_json_nodes_edges_and_metadata_without_positions(self, tmp_path):
        logger = CausalTraceLogger(
            run_id="test_run",
            rank=0,
            output_dir=tmp_path,
            enabled=True,
            agent_type_map={1: "Microglia"},
            params={"random.seed": 1}
        )
        logger.set_tick(4)
        microglia = SimpleNamespace(uid=(3, 1, 0), ptype=1, state="Activated")
        source = logger.env_field_node("SN.inflammation_level", "inflammation_level", "1_perception", 0.9)
        logger.threshold_trigger(
            source,
            microglia,
            "Activated",
            "microglia_activation_by_inflammation",
            "MICROGLIA_ACTIVATION_INFLAMMATION_HIGH",
            "inflammation_level >= inflammation_high_threshold"
        )
        logger.close()
        edge = json.loads((tmp_path / "g0_edges.jsonl").read_text(encoding="utf-8").splitlines()[0])
        metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
        assert (tmp_path / "g0_edges_rank0.jsonl").exists()
        assert (tmp_path / "g0_nodes_rank0.jsonl").exists()
        assert edge["relation"] == "threshold_trigger"
        assert edge["source_node_id"]
        assert edge["target_node_id"]
        assert edge["g1_source_key"]
        assert edge["g2_target_key"] == "SimpleNamespace.Activated"
        assert "position" not in edge
        assert metadata["logger_schema_version"] == "2.0-json"


class TestInitializationLogger:
    def test_writes_full_initial_agent_record_and_manifest(self, tmp_path):
        logger = InitializationLogger(run_id="test_run", rank=0, output_dir=tmp_path, enabled=True)
        point = SimpleNamespace(x=2, y=3, z=0)
        agent = SimpleNamespace(
            uid=(4, 1, 0),
            ptype=1,
            state="Monomer",
            compartment="Intracellular",
            cfg=SimpleNamespace(move_probability=0.5)
        )
        owner = SimpleNamespace(uid=(1, 0, 0), ptype=0, state="Healthy")
        logger.record_agent(agent, position=point, owner=owner)
        logger.close()
        row = json.loads((tmp_path / "initialization_agents.jsonl").read_text(encoding="utf-8").splitlines()[0])
        manifest = json.loads((tmp_path / "initialization_manifest.json").read_text(encoding="utf-8"))
        assert row["uid"] == "4:1:0"
        assert row["position"] == {"x": 2, "y": 3, "z": 0}
        assert row["config"]["move_probability"] == 0.5
        assert row["owner_uid"] == "1:0:0"
        assert manifest["counts"]["total_agents"] == 1
