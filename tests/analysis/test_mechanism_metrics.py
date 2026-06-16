import json

from src.analysis.metrics.mechanism_metrics import summarize_mechanisms


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def event(tick, mechanism, event_type, **overrides):
    row = {
        "schema_version": "1.0",
        "run_id": "test-run",
        "event_id": f"{tick}:{mechanism}:{event_type}",
        "tick": tick,
        "phase": "2_state_update",
        "rank": 0,
        "event_type": event_type,
        "mechanism": mechanism,
        "actor": {},
        "target": {},
        "effects": [],
        "stochastic": {},
        "context": {},
    }
    row.update(overrides)
    return row


def agent(uid, agent_type, state=None, **overrides):
    row = {"uid": uid, "type": agent_type}
    if state is not None:
        row["state"] = state
    row.update(overrides)
    return row


def transition(tick, mechanism, uid, agent_type, before, after, **overrides):
    return event(
        tick,
        mechanism,
        "state_transition",
        actor=agent(uid, agent_type, state_before=before, state_after=after, **overrides.pop("actor_overrides", {})),
        **overrides,
    )


class TestMechanismMetrics:
    def test_counts_core_alpha_and_lysosome_mechanisms_from_events(self, tmp_path):
        write_jsonl(
            tmp_path / "events.jsonl",
            [
                transition(0, "initial_alpha_state", "a0", "AlphaSynuclein", None, "Monomer"),
                transition(0, "initial_alpha_state", "a1", "AlphaSynuclein", None, "Monomer"),
                transition(0, "initial_alpha_state", "a2", "AlphaSynuclein", None, "Misfolded"),
                transition(
                    1,
                    "alpha_misfolding",
                    "a1",
                    "AlphaSynuclein",
                    "Monomer",
                    "Misfolded",
                    outcome="transitioned",
                    stochastic={"probability": 0.2, "rng_value": 0.1},
                ),
                event(
                    2,
                    "alpha_added_to_aggregate",
                    "aggregation",
                    actor=agent("a1", "AlphaSynuclein", "Misfolded", owner_uid="n1"),
                    target=agent("agg1", "AlphaAggregate", "Oligomer", size=2),
                    outcome="member_added",
                ),
                event(
                    2,
                    "alpha_added_to_aggregate",
                    "aggregation",
                    actor=agent("a2", "AlphaSynuclein", "Misfolded", owner_uid="n1"),
                    target=agent("agg1", "AlphaAggregate", "Oligomer", size=2),
                    outcome="member_added",
                ),
                transition(
                    3,
                    "aggregate_matures_to_lewy_body",
                    "agg1",
                    "AlphaAggregate",
                    "Oligomer",
                    "LewyBody",
                    outcome="transitioned",
                    actor_overrides={"size": 3},
                ),
                event(
                    6,
                    "lysosome_degradation_success",
                    "degradation",
                    actor=agent("l1", "Lysosome", "degrade"),
                    target=agent("a1", "AlphaSynuclein", "Cleared"),
                    outcome="protein_cleared",
                    stochastic={"probability": 0.6, "rng_value": 0.4},
                ),
                event(
                    7,
                    "lysosome_degradation_failure",
                    "degradation",
                    actor=agent("l1", "Lysosome", "degrade"),
                    target=agent("agg2", "AlphaAggregate", "Oligomer"),
                    outcome="failed_requeued",
                    stochastic={"probability": 0.3, "rng_value": 0.9},
                ),
                event(
                    8,
                    "lysosome_overwhelmed_by_target",
                    "degradation",
                    actor=agent("l1", "Lysosome", "degrade"),
                    target=agent("agg3", "AlphaAggregate", "LewyBody"),
                    outcome="overwhelmed",
                ),
                transition(
                    8,
                    "lysosome_overwhelmed_by_target",
                    "l1",
                    "Lysosome",
                    "Active",
                    "Overwhelmed",
                    outcome="transitioned",
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
        assert alpha["aggregate_nodes"]["unique_aggregates_observed"] == 3
        assert alpha["aggregate_nodes"]["unique_lewy_body_aggregates_observed"] == 2
        assert alpha["aggregate_nodes"]["lewy_body_size_summary"]["max"] == 3
        assert alpha["misfolding_events"] == 1
        assert alpha["aggregate_member_additions"] == 2
        assert alpha["aggregation_events_inferred"] == 1
        assert alpha["aggregate_targets_touched"] == 1
        assert alpha["lewy_body_maturations"] == 1
        lysosome = report["selected_mechanisms"]["lysosome"]
        assert lysosome["degradation_attempts"]["total"] == 3
        assert lysosome["degradation_attempts"]["success"] == 1
        assert lysosome["degradation_attempts"]["failure"] == 1
        assert lysosome["degradation_attempts"]["overwhelmed"] == 1
        assert lysosome["degradation_attempts"]["overwhelmed_by_lewy_body"] == 1
        assert lysosome["degradation_attempts"]["success_rate"] == 1 / 3

    def test_reads_rank_local_events_when_merged_log_is_missing(self, tmp_path):
        write_jsonl(
            tmp_path / "events_rank1.jsonl",
            [
                event(
                    2,
                    "microglia_debris_clearance",
                    "field_change",
                    actor=agent("m1", "Microglia"),
                    target=agent("SN", "SubstantiaNigra"),
                    effects=[{"field": "debris", "scope": "environment", "delta": -1}],
                    outcome="buffered",
                )
            ],
        )
        write_jsonl(
            tmp_path / "events_rank0.jsonl",
            [
                event(
                    1,
                    "astrocyte_reactive_inflammation_release",
                    "field_change",
                    actor=agent("a1", "Astrocyte"),
                    target=agent("SN", "SubstantiaNigra"),
                    effects=[{"field": "inflammation", "scope": "environment", "delta": 1}],
                    outcome="buffered",
                )
            ],
        )

        report = summarize_mechanisms(tmp_path, include_by_tick=False)

        glia = report["selected_mechanisms"]["glia"]
        assert glia["microglia_debris_clearance"] == 1
        assert glia["astrocyte_reactive_inflammation_release"] == 1
        assert [path.rsplit("\\", 1)[-1] for path in report["input_event_files"]] == [
            str(tmp_path / "events_rank0.jsonl").rsplit("\\", 1)[-1],
            str(tmp_path / "events_rank1.jsonl").rsplit("\\", 1)[-1],
        ]
        assert "by_tick" not in report

    def test_probability_summary_is_grouped_by_mechanism(self, tmp_path):
        write_jsonl(
            tmp_path / "events.jsonl",
            [
                transition(
                    1,
                    "alpha_misfolding",
                    "a1",
                    "AlphaSynuclein",
                    "Monomer",
                    "Misfolded",
                    outcome="transitioned",
                    stochastic={"probability": 0.2, "rng_value": 0.1},
                ),
                transition(
                    2,
                    "alpha_misfolding",
                    "a2",
                    "AlphaSynuclein",
                    "Monomer",
                    "Misfolded",
                    outcome="transitioned",
                    stochastic={"probability": 0.6, "rng_value": 0.3},
                ),
            ],
        )

        report = summarize_mechanisms(tmp_path, include_by_tick=False)

        probability = report["all_mechanisms"]["probability_summary"]["alpha_misfolding"]
        rng = report["all_mechanisms"]["rng_summary"]["alpha_misfolding"]
        assert probability["count"] == 2
        assert probability["mean"] == 0.4
        assert probability["min"] == 0.2
        assert probability["max"] == 0.6
        assert rng["mean"] == 0.2
