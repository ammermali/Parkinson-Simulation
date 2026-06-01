import json

from src.analysis.mechanism_metrics import summarize_mechanisms


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def edge(tick, mechanism, relation, **overrides):
    row = {
        "tick": tick,
        "mechanism": mechanism,
        "relation": relation,
        "source_type": "unknown",
        "source_state": None,
        "source_uid": None,
        "target_type": "unknown",
        "target_state": None,
        "target_uid": None,
        "owner_uid": None,
        "outcome": None,
        "probability": None,
        "rng_value": None
    }
    row.update(overrides)
    return row


def node(tick, **overrides):
    row = {
        "tick": tick,
        "kind": "agent_state",
        "agent_type": "AlphaSynuclein",
        "state": "Monomer"
    }
    row.update(overrides)
    return row


class TestMechanismMetrics:
    def test_counts_core_alpha_and_lysosome_mechanisms_without_double_counting_target_claims(self, tmp_path):
        write_jsonl(
            tmp_path / "g0_nodes.jsonl",
            [
                node(0, state="Monomer"),
                node(0, state="Monomer"),
                node(0, state="Misfolded"),
                node(
                    2,
                    kind="aggregate",
                    agent_type="AlphaAggregate",
                    state="Oligomer",
                    uid="agg1",
                    value=2,
                ),
                node(
                    3,
                    kind="aggregate",
                    agent_type="AlphaAggregate",
                    state="LewyBody",
                    uid="agg1",
                    value=3,
                ),
                node(1, state="Monomer")
            ]
        )
        write_jsonl(
            tmp_path / "g0_edges.jsonl",
            [
                edge(
                    1,
                    "alpha_misfolding",
                    "state_transition",
                    source_type="AlphaSynuclein",
                    source_state="Monomer",
                    target_type="AlphaSynuclein",
                    target_state="Misfolded",
                    target_uid="a1",
                    owner_uid="n1",
                    outcome="transitioned",
                    probability=0.2,
                    rng_value=0.1
                ),
                edge(
                    2,
                    "alpha_added_to_aggregate",
                    "aggregation",
                    source_type="AlphaSynuclein",
                    target_type="AlphaAggregate",
                    target_state="Oligomer",
                    target_uid="agg1",
                    owner_uid="n1",
                    outcome="member_added"
                ),
                edge(
                    2,
                    "alpha_added_to_aggregate",
                    "aggregation",
                    source_type="AlphaSynuclein",
                    target_type="AlphaAggregate",
                    target_state="Oligomer",
                    target_uid="agg1",
                    owner_uid="n1",
                    outcome="member_added"
                ),
                edge(
                    3,
                    "aggregate_matures_to_lewy_body",
                    "state_transition",
                    source_type="AlphaAggregate",
                    source_state="Oligomer",
                    target_type="AlphaAggregate",
                    target_state="LewyBody",
                    target_uid="agg1",
                    owner_uid="n1",
                    outcome="transitioned"
                ),
                edge(
                    4,
                    "neuron_registers_degradation_target",
                    "target_assignment",
                    source_type="Neuron",
                    target_type="AlphaSynuclein",
                    target_state="Misfolded",
                    target_uid="a1",
                    owner_uid="n1",
                    outcome="registered"
                ),
                edge(
                    5,
                    "neuron_assigns_degradation_target",
                    "target_assignment",
                    source_type="Lysosome",
                    target_type="AlphaSynuclein",
                    target_state="Misfolded",
                    target_uid="a1",
                    owner_uid="n1",
                    outcome="assigned"
                ),
                edge(
                    5,
                    "lysosome_selects_degradation_target",
                    "target_assignment",
                    source_type="Lysosome",
                    target_type="AlphaSynuclein",
                    target_state="Misfolded",
                    target_uid="a1",
                    owner_uid="n1",
                    outcome="assigned"
                ),
                edge(
                    6,
                    "lysosome_degradation_success",
                    "degradation",
                    source_type="Lysosome",
                    source_state="degrade",
                    target_type="AlphaSynuclein",
                    target_state="Cleared",
                    target_uid="a1",
                    owner_uid="n1",
                    outcome="protein_cleared",
                    probability=0.6,
                    rng_value=0.4
                ),
                edge(
                    7,
                    "lysosome_degradation_failure",
                    "degradation",
                    source_type="Lysosome",
                    source_state="degrade",
                    target_type="AlphaAggregate",
                    target_state="Oligomer",
                    target_uid="agg2",
                    owner_uid="n1",
                    outcome="failed_requeued",
                    probability=0.3,
                    rng_value=0.9,
                ),
                edge(
                    8,
                    "lysosome_overwhelmed_by_target",
                    "degradation",
                    source_type="Lysosome",
                    source_state="degrade",
                    target_type="AlphaAggregate",
                    target_state="LewyBody",
                    target_uid="agg3",
                    owner_uid="n1",
                    outcome="overwhelmed"
                ),
                edge(
                    8,
                    "lysosome_overwhelmed_by_target",
                    "state_transition",
                    source_type="Lysosome",
                    source_state="Active",
                    target_type="Lysosome",
                    target_state="Overwhelmed",
                    target_uid="l1",
                    owner_uid="n1",
                    outcome="transitioned"
                ),
            ],
        )

        report = summarize_mechanisms(tmp_path, include_by_tick=True)

        alpha = report["selected_mechanisms"]["alpha_synuclein"]
        assert alpha["initial_state_nodes"] == {
            "total": 3,
            "by_state": {"Misfolded": 1, "Monomer": 2},
            "free_monomer_misfolded": 3,
            "cleared": 0,
            "lewy_body_members": 0,
        }
        assert alpha["aggregate_nodes"]["unique_aggregates_observed"] == 1
        assert alpha["aggregate_nodes"]["unique_lewy_body_aggregates_observed"] == 1
        assert alpha["aggregate_nodes"]["lewy_body_size_summary"]["max"] == 3
        assert alpha["misfolding_events"] == 1
        assert alpha["aggregate_member_additions"] == 2
        assert alpha["aggregation_events_inferred"] == 1
        assert alpha["aggregate_targets_touched"] == 1
        assert alpha["lewy_body_maturations"] == 1
        lysosome = report["selected_mechanisms"]["lysosome"]
        assert lysosome["targets_registered"]["total"] == 1
        assert lysosome["successful_target_claims"]["total"] == 1
        assert lysosome["degradation_attempts"]["total"] == 3
        assert lysosome["degradation_attempts"]["success"] == 1
        assert lysosome["degradation_attempts"]["failure"] == 1
        assert lysosome["degradation_attempts"]["overwhelmed"] == 1
        assert lysosome["degradation_attempts"]["overwhelmed_by_lewy_body"] == 1
        assert lysosome["degradation_attempts"]["success_rate"] == 1 / 3
        assert report["by_tick"]["5"]["neuron_assigns_degradation_target"] == 1
        assert report["by_tick"]["5"]["lysosome_selects_degradation_target"] == 1

    def test_falls_back_to_rank_local_edges_when_merged_log_is_missing(self, tmp_path):
        write_jsonl(
            tmp_path / "g0_edges_rank1.jsonl",
            [
                edge(
                    2,
                    "microglia_debris_clearance",
                    "field_effect",
                    source_type="Microglia",
                    target_type="SubstantiaNigra",
                    outcome="buffered"
                )
            ]
        )
        write_jsonl(
            tmp_path / "g0_edges_rank0.jsonl",
            [
                edge(
                    1,
                    "astrocyte_reactive_inflammation_release",
                    "field_effect",
                    source_type="Astrocyte",
                    target_type="SubstantiaNigra",
                    outcome="buffered"
                )
            ]
        )
        report = summarize_mechanisms(tmp_path, include_by_tick=False)
        glia = report["selected_mechanisms"]["glia"]
        assert glia["microglia_debris_clearance"] == 1
        assert glia["astrocyte_reactive_inflammation_release"] == 1
        assert [path.rsplit("\\", 1)[-1] for path in report["input_edge_files"]] == [
            str(tmp_path / "g0_edges_rank0.jsonl").rsplit("\\", 1)[-1],
            str(tmp_path / "g0_edges_rank1.jsonl").rsplit("\\", 1)[-1]
        ]
        assert "by_tick" not in report

    def test_probability_summary_is_grouped_by_mechanism(self, tmp_path):
        write_jsonl(
            tmp_path / "g0_edges.jsonl",
            [
                edge(
                    1,
                    "alpha_misfolding",
                    "state_transition",
                    source_type="AlphaSynuclein",
                    target_type="AlphaSynuclein",
                    outcome="transitioned",
                    probability=0.2,
                    rng_value=0.1
                ),
                edge(
                    2,
                    "alpha_misfolding",
                    "state_transition",
                    source_type="AlphaSynuclein",
                    target_type="AlphaSynuclein",
                    outcome="transitioned",
                    probability=0.6,
                    rng_value=0.3
                )
            ]
        )
        report = summarize_mechanisms(tmp_path, include_by_tick=False)
        probability = report["all_mechanisms"]["probability_summary"]["alpha_misfolding"]
        rng = report["all_mechanisms"]["rng_summary"]["alpha_misfolding"]
        assert probability["count"] == 2
        assert probability["mean"] == 0.4
        assert probability["min"] == 0.2
        assert probability["max"] == 0.6
        assert rng["mean"] == 0.2
