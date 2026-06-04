import json
import pytest
pytest.importorskip("networkx")
from src.analysis.graph.g0_builder import build_g0_graph

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def edge(edge_id, source_id, target_id, relation, **overrides):
    row = {
        "edge_id": edge_id,
        "tick": 1,
        "rank": 0,
        "run_id": "test",
        "source_node_id": source_id,
        "target_node_id": target_id,
        "source_kind": "agent_state",
        "target_kind": "agent_state",
        "source_uid": "a:1",
        "target_uid": "a:1",
        "source_type": "AlphaSynuclein",
        "target_type": "AlphaSynuclein",
        "source_state": "Monomer",
        "target_state": "Misfolded",
        "source_field": None,
        "target_field": None,
        "phase_from": "0_pre_state",
        "phase_to": "2_state_update",
        "relation": relation,
        "mechanism": "test_mechanism",
        "outcome": "ok"
    }
    row.update(overrides)
    return row


class TestG0Builder:
    def test_builds_state_transition_graph_and_continuity_edges(self, tmp_path):
        write_jsonl(
            tmp_path / "g0_edges.jsonl",
            [
                edge("e1", "AlphaSynuclein_a:1.Monomer@1.0", "AlphaSynuclein_a:1.Misfolded@1.2", "state_transition"),
                edge(
                    "e2",
                    "AlphaSynuclein_a:1.Misfolded@1.2",
                    "AlphaSynuclein_a:1.Misfolded@2.2",
                    "threshold_trigger",
                    tick=2,
                    source_state="Misfolded",
                    target_state="Misfolded"
                )
            ]
        )

        graph = build_g0_graph(tmp_path)
        assert graph.number_of_nodes() == 3
        assert any(data["relation"] == "continuity" for _, _, data in graph.edges(data=True))
        assert graph.nodes["AlphaSynuclein_a_1_Misfolded@1"]["display_id"] == "AlphaSynuclein_a_1_Misfolded@1"

    def test_collapses_action_node_into_field_effect_edge(self, tmp_path):
        write_jsonl(
            tmp_path / "g0_edges.jsonl",
            [
                edge(
                    "select",
                    "Neuron_n:1.Healthy@1.2",
                    "Neuron_n:1.release_dopamine@1.3",
                    "action_selection",
                    source_uid="n:1",
                    target_uid="n:1",
                    source_type="Neuron",
                    target_type="Neuron",
                    source_state="Healthy",
                    target_state="release_dopamine",
                    target_kind="action",
                    phase_from="2_state_update",
                    phase_to="3_action_selection",
                    mechanism="neuron_state_action_policy"
                ),
                edge(
                    "effect",
                    "Neuron_n:1.release_dopamine@1.3",
                    "SN.dopamine_output_buffer@1.4",
                    "field_effect",
                    source_kind="action",
                    target_kind="buffer",
                    source_uid="n:1",
                    target_uid="SN",
                    source_type="Neuron",
                    target_type="SubstantiaNigra",
                    source_state="release_dopamine",
                    target_state=None,
                    target_field="dopamine_output_buffer",
                    phase_from="3_action_selection",
                    phase_to="4_effect_buffer",
                    mechanism="neuron_dopamine_release"
                )
            ]
        )

        graph = build_g0_graph(tmp_path, add_continuity=False)

        assert "Neuron_n_1_release_dopamine@1" not in graph
        assert graph.has_edge("Neuron_n_1_Healthy@1", "SN_dopamine_output_buffer@1")
        edge_data = graph.get_edge_data("Neuron_n_1_Healthy@1", "SN_dopamine_output_buffer@1")
        assert edge_data["causal_kind"] == "action"
        assert edge_data["action"] == "release_dopamine"

    def test_keeps_lysosome_targeting_and_alpha_aggregation_edges(self, tmp_path):
        write_jsonl(
            tmp_path / "g0_edges.jsonl",
            [
                edge(
                    "target",
                    "Lysosome_l:1.Active@3.3",
                    "AlphaSynuclein_a:1.Misfolded@3.4",
                    "target_assignment",
                    source_uid="l:1",
                    target_uid="a:1",
                    source_type="Lysosome",
                    target_type="AlphaSynuclein",
                    source_state="Active",
                    target_state="Misfolded",
                    mechanism="lysosome_selects_degradation_target"
                ),
                edge(
                    "aggregation",
                    "AlphaSynuclein_a:1.Oligomer@4.2",
                    "n:1::Aggregate_1.Oligomer@4.4",
                    "aggregation",
                    source_uid="a:1",
                    target_uid="n:1::Aggregate_1",
                    source_type="AlphaSynuclein",
                    target_type="AlphaAggregate",
                    source_state="Oligomer",
                    target_state="Oligomer",
                    target_kind="aggregate",
                    mechanism="alpha_added_to_aggregate"
                )
            ]
        )

        graph = build_g0_graph(tmp_path, add_continuity=False)
        edge_kinds = {
            data["causal_kind"]
            for _, _, data in graph.edges(data=True)
        }
        assert "aggregation" in edge_kinds
        assert graph.nodes["AlphaAggregate_n_1_Aggregate_1_Oligomer@4"]["semantic_kind"] == "agent_state"
