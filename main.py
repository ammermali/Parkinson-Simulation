from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_SIMULATION_LOG_DIR = Path("output/simulation/logs")


def main(argv: Optional[list[str]] = None) -> int:
    """Central CLI entry point for simulation, validation and analysis tasks."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without importing heavy runtime modules."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Central command line interface for the Parkinson simulation project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run the Repast4Py simulation.")
    simulate.add_argument("--mode", choices=("rule",), default="rule", help="Simulation mode. Currently only rule mode is implemented.")
    simulate.add_argument("--params", default="system", help="System params name, YAML filename, or explicit YAML path.")
    simulate.set_defaults(handler=command_simulate)

    validate_g0 = subparsers.add_parser("validate-g0", help="Validate G0 causal trace logs.")
    validate_g0.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    validate_g0.add_argument("--output", type=Path, default=Path("output/analysis/g0_trace_validation_latest.json"))
    validate_g0.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    validate_g0.set_defaults(handler=command_validate_g0)

    validate_init = subparsers.add_parser("validate-init", help="Validate initialization logs.")
    validate_init.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    validate_init.add_argument("--output", type=Path, default=Path("output/analysis/initialization_validation_latest.json"))
    validate_init.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    validate_init.set_defaults(handler=command_validate_initialization)

    mechanisms = subparsers.add_parser("mechanisms", help="Summarize biological mechanisms from G0 logs.")
    mechanisms.add_argument("output_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    mechanisms.add_argument("--no-by-tick", action="store_true", help="Omit per-tick mechanism counts.")
    mechanisms.add_argument("--output", type=Path, default=Path("output/analysis/mechanism_metrics_latest.json"))
    mechanisms.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    mechanisms.set_defaults(handler=command_mechanisms)

    intervention = subparsers.add_parser("intervention", help="Summarize logs for controlled intervention comparisons.")
    intervention.add_argument("output_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    intervention.add_argument("--output", type=Path, default=Path("output/analysis/intervention_metrics_latest.json"))
    intervention.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    intervention.set_defaults(handler=command_intervention)

    graphs = subparsers.add_parser("graphs", help="Build G0 and G1 graph exports.")
    graphs.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    graphs.add_argument("--output-dir", type=Path, default=Path("output/analysis/graphs"))
    graphs.add_argument("--formats", nargs="+", default=["gexf", "graphml", "json"], help="Graph formats: gexf graphml json pkl.")
    graphs.add_argument("--strict", action="store_true", help="Raise on malformed JSONL rows.")
    graphs.set_defaults(handler=command_graphs)

    plot = subparsers.add_parser("plot", help="Create tick_metrics.csv plots.")
    plot.add_argument("group", choices=("neurons", "alpha", "alpha-free", "alpha-aggregate", "sn", "all"))
    plot.add_argument("--metrics", type=Path, help="Path to tick_metrics.csv. Defaults to output/simulation/logs/tick_metrics.csv.")
    plot.add_argument("--output", type=Path, help="Single plot destination. Not valid for 'alpha' or 'all'.")
    plot.add_argument("--output-dir", type=Path, default=Path("output/plots"), help="Plot directory for grouped outputs.")
    plot.set_defaults(handler=command_plot)

    return parser


def command_simulate(args: argparse.Namespace) -> int:
    """Run the simulation through the existing engine module."""

    from src.simulation.engine import run
    from src.simulation.utils import Params

    params = Params(args.params).as_dict()
    run(params)
    return 0


def command_validate_g0(args: argparse.Namespace) -> int:
    """Validate G0 logs and write or print a JSON report."""

    from src.analysis.validate_g0_trace import validate_trace

    report = validate_trace(args.output_dir)
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_validate_initialization(args: argparse.Namespace) -> int:
    """Validate initialization logs and write or print a JSON report."""

    from src.analysis.validate_initialization_log import validate_initialization

    report = validate_initialization(args.output_dir)
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_mechanisms(args: argparse.Namespace) -> int:
    """Summarize mechanism counts for one or more simulation outputs."""

    from src.analysis.mechanism_metrics import summarize_mechanisms

    report = {
        "runs": [
            summarize_mechanisms(output_dir, include_by_tick=not args.no_by_tick)
            for output_dir in args.output_dirs
        ]
    }
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_intervention(args: argparse.Namespace) -> int:
    """Summarize scalar and event behavior for intervention analysis."""

    from src.analysis.intervention_metrics import summarize_run

    report = {
        "runs": [
            summarize_run(output_dir)
            for output_dir in args.output_dirs
        ]
    }
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_graphs(args: argparse.Namespace) -> int:
    """Build and export G0/G1 graphs for Gephi and downstream analysis."""

    from src.analysis.build_multilevel_graphs import build_and_export_multilevel_graphs

    outputs = build_and_export_multilevel_graphs(
        args.log_dir,
        args.output_dir,
        formats=args.formats,
        strict=args.strict,
    )
    for key, path in sorted(outputs.items()):
        print(f"{key}: {path}")
    return 0


def command_plot(args: argparse.Namespace) -> int:
    """Generate one or more tick-metrics plots."""

    from src.visualization.tick_metrics_plotter import SERIES_GROUPS, plot_alpha_metrics, plot_group

    if args.output is not None and args.group in {"alpha", "all"}:
        raise SystemExit("--output can only be used with single-figure plot groups")

    if args.group == "all":
        paths = [
            plot_group("neurons", args.metrics, args.output_dir / SERIES_GROUPS["neurons"]["filename"]),
            *plot_alpha_metrics(args.metrics, args.output_dir),
            plot_group("substantia_nigra", args.metrics, args.output_dir / SERIES_GROUPS["substantia_nigra"]["filename"]),
        ]
    elif args.group == "alpha":
        paths = plot_alpha_metrics(args.metrics, args.output_dir)
    else:
        group = {"alpha-free": "alpha_free", "alpha-aggregate": "alpha_aggregate", "sn": "substantia_nigra"}.get(args.group, args.group)
        output = args.output or args.output_dir / SERIES_GROUPS[group]["filename"]
        paths = [plot_group(group, args.metrics, output)]
    for path in paths:
        print(path)
    return 0


def write_or_print_report(report: dict[str, Any], output: Path, to_stdout: bool) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if to_stdout:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    raise SystemExit(main())
