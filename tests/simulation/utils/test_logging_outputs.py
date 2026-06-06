import json
from types import SimpleNamespace

from src.simulation.logger.event_logger import EventLogger
from src.simulation.logger.initialization_logger import InitializationLogger
from src.simulation.logger.snapshot_logger import SnapshotLogger


class TestEventLogger:
    def test_writes_semantic_events_without_legacy_g0_files(self, tmp_path):
        logger = EventLogger(run_id="test_run", rank=0, output_dir=tmp_path, enabled=True)
        logger.set_tick(7)
        agent = SimpleNamespace(uid=(3, 1, 0), state="Supportive")
        target = SimpleNamespace(uid=(4, 2, 0), state="Misfolded")
        logger.action_selection(agent, "support", "astrocyte_state_action_policy")
        logger.field_effect(agent, "support", "inflammation_level", -0.05, "astrocyte_support")
        logger.target_assignment(agent, target, "legacy_target_assignment")
        logger.buffer_commit("inflammation_removed", "inflammation_level", -0.05, "legacy_commit")
        logger.close()
        events = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [event["event_type"] for event in events] == ["action_selected", "field_change"]
        assert events[1]["effects"][0]["field"] == "inflammation_level"
        assert not (tmp_path / "g0_nodes.jsonl").exists()
        assert not (tmp_path / "g0_edges.jsonl").exists()


class TestSnapshotLogger:
    def test_writes_spatial_snapshots_with_run_identity(self, tmp_path):
        logger = SnapshotLogger(run_id="test_run", rank=0, output_dir=tmp_path, enabled=True)
        owner = SimpleNamespace(uid=(1, 0, 0))
        agent = SimpleNamespace(
            uid=(4, 3, 0),
            state="LewyBody",
            compartment="Intracellular",
            owner_neuron=owner,
            aggregate_id=8
        )
        logger.record_agent(3, agent, SimpleNamespace(x=2, y=5, z=0))
        logger.close()

        row = json.loads((tmp_path / "spatial_snapshots.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["run_id"] == "test_run"
        assert row["uid"] == "4:3:0"
        assert row["owner_uid"] == "1:0:0"
        assert row["aggregate_id"] == 8

class TestInitializationLogger:
    def test_writes_full_initial_agent_record_and_manifest(self, tmp_path):
        logger = InitializationLogger(rank=0, output_dir=tmp_path, enabled=True)
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
