import json

import pytest

nx = pytest.importorskip("networkx")
pytest.importorskip("multilevelgraphs")

from dashboard.services.graph_compute_service import GraphComputeService
from dashboard.services.g0_view_service import DEFAULT_G0_LITE_PATH
from src.analysis.graph import write_g0_lite_gexf
from src.analysis.graph.g0_builder import DEFAULT_GRAPH_LITE_OUTPUT, build_g0_exports
from src.analysis.graph.multilevel_builder import build_multilevel_graphs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def transition_event(tick=1):
    return {
        "event_id": f"event_0_{tick}_1",
        "tick": tick,
        "rank": 0,
        "event_type": "state_transition",
        "mechanism": "test_transition",
        "actor": {
            "uid": "n0:0:0",
            "type": "Neuron",
            "state_before": "Healthy",
            "state_after": "Compromised",
        },
        "outcome": "transitioned",
        "context": {"rule_id": "TEST_RULE"},
    }


def make_log_dir(tmp_path):
    log_dir = tmp_path / "run_logs"
    write_jsonl(log_dir / "events.jsonl", [transition_event()])
    return log_dir


class TestGraphExportArtifacts:
    def test_g0_exports_write_canonical_lite_gexf(self, tmp_path):
        log_dir = make_log_dir(tmp_path)
        output_dir = tmp_path / "graphs"

        result = build_g0_exports(log_dir, output_dir=output_dir, include_snapshot_nodes=False, add_continuity=False)

        assert DEFAULT_GRAPH_LITE_OUTPUT.name == "g0.lite.gexf"
        assert result.output_paths["g0_gephi"] == output_dir / "g0.gexf"
        assert result.output_paths["g0_lite_gephi"] == output_dir / "g0.lite.gexf"
        assert (output_dir / "g0.gexf").exists()
        assert (output_dir / "g0.lite.gexf").exists()
        assert not (output_dir / "g0.lite.gext").exists()

    def test_multilevel_exports_write_canonical_lite_gexf(self, tmp_path):
        log_dir = make_log_dir(tmp_path)
        output_dir = tmp_path / "graphs"

        result = build_multilevel_graphs(log_dir, output_dir=output_dir, include_snapshot_nodes=False, add_g0_continuity=False)

        assert result.output_paths["g0_lite_gephi"] == output_dir / "g0.lite.gexf"
        assert (output_dir / "g0.lite.gexf").exists()
        assert (output_dir / "g1.gexf").exists()
        assert (output_dir / "g2.gexf").exists()
        assert (output_dir / "g3.gexf").exists()


class TestGraphPublicApiAndDashboardPaths:
    def test_graph_package_exports_lite_writer_without_legacy_typo(self):
        import src.analysis.graph as graph

        assert graph.write_g0_lite_gexf is write_g0_lite_gexf
        assert not hasattr(graph, "write_gx0_lite_gexf")

    def test_dashboard_g0_lite_path_uses_canonical_gexf_extension(self):
        assert DEFAULT_G0_LITE_PATH.name == "g0.lite.gexf"

    def test_graph_compute_service_commands_use_graph_entrypoints(self, tmp_path):
        service = GraphComputeService(project_root=tmp_path, log_dir=tmp_path / "run_logs", graph_dir=tmp_path / "graphs")

        assert service.command_for("G0")[1:3] == ["main.py", "graph-g0"]
        assert service.command_for("G1")[1:3] == ["main.py", "graph-g1"]
        assert service.command_for("G2")[1:3] == ["main.py", "graph-g2"]
        assert service.command_for("G3")[1:3] == ["main.py", "graph-g3"]
