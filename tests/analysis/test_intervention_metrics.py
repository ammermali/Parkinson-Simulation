import json

from src.analysis.intervention_metrics import summarize_run


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class TestInterventionMetrics:
    def test_falls_back_to_readable_log_dir_and_tick_metrics(self, tmp_path):
        simulation_dir = tmp_path / "output" / "simulation"
        logs_dir = simulation_dir / "logs"
        legacy_log_dir = simulation_dir / "log"
        logs_dir.mkdir(parents=True)
        legacy_log_dir.mkdir(parents=True)
        (legacy_log_dir / "g0_nodes.jsonl").write_text(
            "version https://git-lfs.github.com/spec/v1\n",
            encoding="utf-8",
        )
        write_jsonl(
            legacy_log_dir / "g0_edges.jsonl",
            [
                {
                    "tick": 1,
                    "mechanism": "alpha_misfolding",
                },
                {
                    "tick": 2,
                    "mechanism": "lysosome_degradation_success",
                },
            ],
        )
        (simulation_dir / "tick_metrics.csv").write_text(
            "\n".join(
                [
                    "tick,debris,inflammation,dopamine,neurons_healthy,neurons_compromised,neurons_apoptotic,neurons_ruptures,free_alpha,alpha_aggregate",
                    "1,0.20,0.10,0.90,2,0,0,0,5,0",
                    "2,0.30,0.20,0.80,1,1,0,0,4,2",
                ]
            ),
            encoding="utf-8",
        )

        report = summarize_run(logs_dir)

        assert report["output_dir"] == str(legacy_log_dir)
        assert report["event_counts"] == {
            "alpha_misfolding": 1,
            "lysosome_degradation_success": 1,
        }
        assert report["final_environment"] == {
            "inflammation_level": 0.2,
            "extracellular_debris": 0.3,
            "dopamine_output": 0.8,
        }
        assert report["state_counts_by_tick"]["Neuron"]["2"] == {
            "Healthy": 1,
            "Compromised": 1,
            "Apoptotic": 0,
            "Ruptured": 0,
        }
        assert report["state_counts_by_tick"]["AlphaSynuclein"]["2"] == {"Free": 4}
        assert report["state_counts_by_tick"]["AlphaAggregate"]["2"] == {"MemberProteins": 2}
        assert report["input_status"]["g0_node_files"] == []
        assert report["input_status"]["tick_metrics_csv"] == str(simulation_dir / "tick_metrics.csv")
