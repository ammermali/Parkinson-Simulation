from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SIMULATION_LOG_DIR = PROJECT_ROOT / "output" / "run_logs"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "output" / "metrics"
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "output" / "validation_reports"
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "output" / "graphs"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "output" / "plots"
DEFAULT_PARAM_DIR = PROJECT_ROOT / "src" / "configuration" / "param"


def main(argv: Optional[list[str]] = None) -> int:
    """Run the central command line interface for the project suite."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def build_parser() -> argparse.ArgumentParser:
    """Build the root parser without importing heavy simulation dependencies."""

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Central CLI for simulation, validation, analysis, graph export and plotting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser(
        "simulate",
        aliases=["run"],
        help="Run the Repast4Py simulation.",
    )
    simulate.add_argument("--mode", choices=("rule",), default="rule", help="Simulation mode. Currently only rule mode is implemented.")
    simulate.add_argument("--params", default="system", help="System params name, YAML filename, or explicit YAML path.")
    simulate.set_defaults(handler=command_simulate)

    validate_g0 = subparsers.add_parser(
        "validate-g0",
        help="Validate semantic traces used for G0 projection.",
    )
    validate_g0.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    validate_g0.add_argument("--output", type=Path, default=DEFAULT_VALIDATION_DIR / "g0_trace_validation_latest.json")
    validate_g0.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    validate_g0.set_defaults(handler=command_validate_g0)

    validate_init = subparsers.add_parser(
        "validate-init",
        help="Validate initialization logs.",
    )
    validate_init.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    validate_init.add_argument("--output", type=Path, default=DEFAULT_VALIDATION_DIR / "initialization_validation_latest.json")
    validate_init.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    validate_init.set_defaults(handler=command_validate_initialization)

    mechanisms = subparsers.add_parser(
        "mechanisms",
        help="Summarize biological mechanisms from event logs.",
    )
    mechanisms.add_argument("log_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    mechanisms.add_argument("--no-by-tick", action="store_true", help="Omit per-tick mechanism counts.")
    mechanisms.add_argument("--output", type=Path, default=DEFAULT_METRICS_DIR / "mechanism_metrics_latest.json")
    mechanisms.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    mechanisms.set_defaults(handler=command_mechanisms)

    intervention = subparsers.add_parser(
        "intervention",
        aliases=["interventions"],
        help="Summarize completed runs for intervention comparison.",
    )
    intervention.add_argument("log_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    intervention.add_argument("--output", type=Path, default=DEFAULT_METRICS_DIR / "intervention_metrics_latest.json")
    intervention.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    intervention.set_defaults(handler=command_intervention)

    export_data = subparsers.add_parser(
        "export-data",
        aliases=["export-tables"],
        help="Export normalized post-run JSONL/CSV tables for analysis and visualization.",
    )
    export_data.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    export_data.add_argument("--output-dir", type=Path, help="Destination directory. Defaults to the run tables directory.")
    export_data.add_argument("--formats", nargs="+", choices=("jsonl", "csv"), default=("jsonl", "csv"))
    export_data.set_defaults(handler=command_export_data)

    graphs = subparsers.add_parser(
        "graphs",
        help="Build G0/G1/G2/G3 GEXF graph exports.",
    )
    add_graph_arguments(graphs)
    graphs.set_defaults(handler=command_graphs)

    graph_g0 = subparsers.add_parser(
        "graph-g0",
        help="Build only the G0 graph export from semantic event logs.",
    )
    graph_g0.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    graph_g0.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    graph_g0.add_argument("--no-continuity", action="store_true", help="Do not add temporal continuity edges to G0.")
    graph_g0.add_argument("--no-field-continuity", action="store_true", help="Do not add continuity edges for environment or neuron fields.")
    graph_g0.add_argument("--no-snapshot-nodes", action="store_true", help="Build event-only G0 without full snapshot existence nodes.")
    graph_g0.add_argument("--rupture-grace-ticks", type=int, default=2, help="Ticks retained after a neuron becomes ruptured.")
    graph_g0.add_argument("--start-tick", type=int, help="Optional first tick to include.")
    graph_g0.add_argument("--end-tick", type=int, help="Optional last tick to include.")
    graph_g0.add_argument("--strict", action="store_true", help="Fail on malformed JSONL rows instead of skipping them.")
    graph_g0.set_defaults(handler=command_graph_g0)

    graph_g1 = subparsers.add_parser(
        "graph-g1",
        help="Build only the G1 time-contracted graph export.",
    )
    add_intermediate_graph_arguments(graph_g1)
    graph_g1.set_defaults(handler=command_graph_g1)

    graph_g2 = subparsers.add_parser(
        "graph-g2",
        help="Build only the G2 agent/state-clustered graph export.",
    )
    add_intermediate_graph_arguments(graph_g2)
    graph_g2.set_defaults(handler=command_graph_g2)

    graph_g3 = subparsers.add_parser(
        "graph-g3",
        help="Apply the G3 topological contraction to an existing g2.gexf export.",
    )
    graph_g3.add_argument("--g2", type=Path, default=DEFAULT_GRAPH_DIR / "g2.gexf", help="Path to an existing G2 GEXF file.")
    graph_g3.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    graph_g3.set_defaults(handler=command_graph_g3)

    plot = subparsers.add_parser(
        "plot",
        help="Create tick_metrics.csv plots.",
    )
    plot.add_argument("group", choices=("neurons", "alpha", "alpha-free", "alpha-aggregate", "sn", "all"))
    plot.add_argument("--metrics", type=Path, help="Path to tick_metrics.csv.")
    plot.add_argument("--output", type=Path, help="Single plot destination. Not valid for 'alpha' or 'all'.")
    plot.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR, help="Plot directory for grouped outputs.")
    plot.set_defaults(handler=command_plot)

    params = subparsers.add_parser(
        "params",
        help="Inspect or edit YAML parameter files.",
    )
    params.add_argument("--param-dir", type=Path, default=DEFAULT_PARAM_DIR)
    params_subparsers = params.add_subparsers(dest="params_command", required=True)
    params_list = params_subparsers.add_parser("list", help="List available parameter files.")
    params_list.set_defaults(handler=command_params_list)
    params_get = params_subparsers.add_parser("get", help="Read one parameter value.")
    params_get.add_argument("file", help="Parameter file name, filename or path.")
    params_get.add_argument("key", help="Dot-separated key.")
    params_get.set_defaults(handler=command_params_get)
    params_set = params_subparsers.add_parser("set", help="Set one parameter value.")
    params_set.add_argument("file", help="Parameter file name, filename or path.")
    params_set.add_argument("key", help="Dot-separated key.")
    params_set.add_argument("value", help="YAML-parsed value.")
    params_set.add_argument("--create", action="store_true", help="Create missing nested keys.")
    params_set.add_argument("--dry-run", action="store_true", help="Print updated YAML without writing.")
    params_set.set_defaults(handler=command_params_set)

    postprocess = subparsers.add_parser(
        "postprocess",
        aliases=["analyze"],
        help="Run the standard post-simulation analysis pipeline.",
    )
    postprocess.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    postprocess.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    postprocess.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    postprocess.add_argument("--analysis-dir", type=Path, default=DEFAULT_METRICS_DIR, dest="metrics_dir", help=argparse.SUPPRESS)
    postprocess.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    postprocess.add_argument("--no-by-tick", action="store_true", help="Omit per-tick mechanism counts.")
    postprocess.add_argument("--skip-validation", action="store_true", help="Do not run validate-g0 and validate-init.")
    postprocess.add_argument("--skip-metrics", action="store_true", help="Do not run mechanisms and intervention metrics.")
    postprocess.add_argument("--export-data", action="store_true", help="Also export normalized post-run tables.")
    postprocess.add_argument("--table-dir", type=Path, help="Destination for normalized post-run tables.")
    postprocess.add_argument("--plots", action="store_true", help="Also generate all tick metric plots.")
    postprocess.add_argument("--graphs", action="store_true", help="Also build G0/G1/G2/G3 graph exports.")
    add_graph_arguments(postprocess, include_log_dir=False)
    postprocess.set_defaults(handler=command_postprocess)

    return parser


def add_graph_arguments(parser: argparse.ArgumentParser, *, include_log_dir: bool = True) -> None:
    """Attach graph-build options shared by `graphs` and `postprocess`."""

    if include_log_dir:
        parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--skip-g0", action="store_true", help="Do not export G0 files. G1/G2/G3 are still built from G0.")
    parser.add_argument("--no-g0-continuity", action="store_true", help="Do not add temporal continuity edges to G0 before contraction.")
    parser.add_argument("--no-field-continuity", action="store_true", help="Do not add continuity edges for environment or neuron fields.")
    parser.add_argument("--no-snapshot-nodes", action="store_true", help="Build event-only G0 before contraction.")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2, help="Ticks retained after a neuron becomes ruptured.")
    parser.add_argument("--time-window-size", type=int, default=None, help="Optional fixed time window size for G1.")
    parser.add_argument(
        "--external-multilevelgraph",
        action="store_true",
        help="Also instantiate the external MultilevelGraph object. This can be slow on large traces.",
    )


def add_intermediate_graph_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach options shared by the G1 and G2 single-level entry points."""

    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--no-g0-continuity", action="store_true", help="Do not add temporal continuity edges to G0 before contraction.")
    parser.add_argument("--no-field-continuity", action="store_true", help="Do not add continuity edges for environment or neuron fields.")
    parser.add_argument("--no-snapshot-nodes", action="store_true", help="Build event-only G0 before contraction.")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2, help="Ticks retained after a neuron becomes ruptured.")
    parser.add_argument("--time-window-size", type=int, default=None, help="Optional fixed time window size for G1.")


def command_simulate(args: argparse.Namespace) -> int:
    """Load system params and run the simulation engine."""

    from src.simulation.engine import run
    from src.simulation.utils import Params

    if args.mode != "rule":
        raise SystemExit(f"Unsupported simulation mode: {args.mode}")
    run(Params(args.params).as_dict())
    return 0


def command_validate_g0(args: argparse.Namespace) -> int:
    """Validate semantic traces used for G0 projection and write a JSON report."""

    from src.analysis.validation.validate_g0_trace import validate_trace

    report = validate_trace(args.log_dir)
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_validate_initialization(args: argparse.Namespace) -> int:
    """Validate initialization logs and write a JSON report."""
    from src.analysis.validation.validate_initialization_log import validate_initialization
    report = validate_initialization(args.log_dir)
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_mechanisms(args: argparse.Namespace) -> int:
    """Summarize mechanism counts for one or more simulation outputs."""

    from src.analysis.metrics.mechanism_metrics import summarize_mechanisms

    report = {
        "runs": [
            summarize_mechanisms(log_dir, include_by_tick=not args.no_by_tick)
            for log_dir in args.log_dirs
        ]
    }
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_intervention(args: argparse.Namespace) -> int:
    """Summarize one or more runs for intervention comparison."""

    from src.analysis.metrics.mechanism_metrics import summarize_mechanisms

    report = {
        "runs": [
            summarize_mechanisms(log_dir, include_by_tick=False)
            for log_dir in args.log_dirs
        ]
    }
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_export_data(args: argparse.Namespace) -> int:
    """Export normalized post-run tables from existing simulation logs."""

    from src.analysis.data.export import export_run_tables

    produced = export_run_tables(args.log_dir, output_dir=args.output_dir, formats=args.formats)
    print_paths(list(produced.values()))
    return 0


def command_graphs(args: argparse.Namespace) -> int:
    """Build and export G0/G1/G2/G3 graphs from simulation logs."""

    paths = build_graph_exports(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        write_g0=should_write_g0(args),
        add_g0_continuity=not args.no_g0_continuity,
        include_field_continuity=not args.no_field_continuity,
        include_snapshot_nodes=not args.no_snapshot_nodes,
        rupture_grace_ticks=args.rupture_grace_ticks,
        time_window_size=args.time_window_size,
        build_external_graph=args.external_multilevelgraph,
    )
    print_paths(paths)
    return 0


def command_graph_g0(args: argparse.Namespace) -> int:
    """Build and export only the G0 graph from simulation semantic events."""

    from src.analysis.graph.g0_builder import build_g0_exports

    try:
        result = build_g0_exports(
            args.log_dir,
            output_dir=args.output_dir,
            add_continuity=not args.no_continuity,
            include_field_continuity=not args.no_field_continuity,
            include_snapshot_nodes=not args.no_snapshot_nodes,
            rupture_grace_ticks=args.rupture_grace_ticks,
            strict=args.strict,
            start_tick=args.start_tick,
            end_tick=args.end_tick,
            verbose=False,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc))

    print_paths(list(result.output_paths.values()))
    return 0


def command_graph_g1(args: argparse.Namespace) -> int:
    """Build and export only the G1 graph from simulation logs."""

    return command_intermediate_graph(args, "g1")


def command_graph_g2(args: argparse.Namespace) -> int:
    """Build and export only the G2 graph from simulation logs."""

    return command_intermediate_graph(args, "g2")


def command_intermediate_graph(args: argparse.Namespace, level: str) -> int:
    """Build and export one contracted intermediate graph level."""

    from src.analysis.graph.multilevel_builder import build_g1_exports, build_g2_exports

    builder = {"g1": build_g1_exports, "g2": build_g2_exports}[level]
    try:
        result = builder(
            args.log_dir,
            output_dir=args.output_dir,
            add_g0_continuity=not args.no_g0_continuity,
            include_field_continuity=not args.no_field_continuity,
            include_snapshot_nodes=not args.no_snapshot_nodes,
            rupture_grace_ticks=args.rupture_grace_ticks,
            time_window_size=args.time_window_size,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc))
    print_paths(list(result.output_paths.values()))
    return 0


def command_graph_g3(args: argparse.Namespace) -> int:
    """Build G3 from an already exported G2 graph."""

    from src.analysis.graph.multilevel_builder import build_g3_from_g2_gexf

    print_paths(list(build_g3_from_g2_gexf(args.g2, output_dir=args.output_dir).values()))
    return 0


def command_plot(args: argparse.Namespace) -> int:
    """Generate one or more tick-metrics plots."""

    paths = build_plots(args.group, metrics_path=args.metrics, output=args.output, output_dir=args.output_dir)
    print_paths(paths)
    return 0


def command_params_list(args: argparse.Namespace) -> int:
    """List YAML parameter files."""

    from src.configuration.param_editing import list_param_files

    for path in list_param_files(args.param_dir):
        print(path.name)
    return 0


def command_params_get(args: argparse.Namespace) -> int:
    """Read a YAML parameter value."""

    from src.configuration.param_editing import get_param_value

    print(json_payload(get_param_value(args.file, args.key, param_dir=args.param_dir)))
    return 0


def command_params_set(args: argparse.Namespace) -> int:
    """Set a YAML parameter value."""

    import yaml
    from src.configuration.param_editing import parse_value, resolve_param_file, set_param_value

    path = resolve_param_file(args.file, args.param_dir)
    updated = set_param_value(path, args.key, parse_value(args.value), create=args.create, dry_run=args.dry_run)
    if args.dry_run:
        print(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True))
    else:
        print(path)
    return 0


def command_postprocess(args: argparse.Namespace) -> int:
    """Run the standard validation, metric, plot and graph pipeline."""

    produced: list[Path] = []

    if not args.skip_validation:
        from src.analysis.validation.validate_g0_trace import validate_trace
        from src.analysis.validation.validate_initialization_log import validate_initialization

        produced.append(write_report(validate_trace(args.log_dir), args.validation_dir / "g0_trace_validation_latest.json"))
        produced.append(write_report(validate_initialization(args.log_dir), args.validation_dir / "initialization_validation_latest.json"))

    if not args.skip_metrics:
        from src.analysis.metrics.mechanism_metrics import summarize_mechanisms

        produced.append(
            write_report(
                {"runs": [summarize_mechanisms(args.log_dir, include_by_tick=not args.no_by_tick)]},
                args.metrics_dir / "mechanism_metrics_latest.json",
            )
        )

    if args.export_data:
        from src.analysis.data.export import export_run_tables

        produced.extend(export_run_tables(args.log_dir, output_dir=args.table_dir).values())

    if args.plots:
        produced.extend(build_plots("all", metrics_path=tick_metrics_for(args.log_dir), output=None, output_dir=args.plot_dir))

    if args.graphs:
        produced.extend(
            build_graph_exports(
                log_dir=args.log_dir,
                output_dir=args.output_dir,
                write_g0=should_write_g0(args),
                add_g0_continuity=not args.no_g0_continuity,
                include_field_continuity=not args.no_field_continuity,
                include_snapshot_nodes=not args.no_snapshot_nodes,
                rupture_grace_ticks=args.rupture_grace_ticks,
                time_window_size=args.time_window_size,
                build_external_graph=args.external_multilevelgraph,
            )
        )

    print_paths(produced)
    return 0


def build_graph_exports(*, log_dir: Path, output_dir: Path, write_g0: bool, add_g0_continuity: bool, include_field_continuity: bool, include_snapshot_nodes: bool, rupture_grace_ticks: int, time_window_size: Optional[int], build_external_graph: bool) -> list[Path]:
    """Build G0/G1/G2/G3 exports through the canonical multilevel graph builder."""
    from src.analysis.graph.multilevel_builder import build_multilevel_graphs

    result = build_multilevel_graphs(
        log_dir,
        output_dir=output_dir,
        add_g0_continuity=add_g0_continuity,
        include_field_continuity=include_field_continuity,
        include_snapshot_nodes=include_snapshot_nodes,
        rupture_grace_ticks=rupture_grace_ticks,
        time_window_size=time_window_size,
        build_external_graph=build_external_graph,
        write_g0=write_g0,
    )
    return list(result.output_paths.values())


def should_write_g0(args: argparse.Namespace) -> bool:
    """Return whether the graph export entry point should write G0 artifacts."""

    return not getattr(args, "skip_g0", False)


def build_plots(group: str, *, metrics_path: Optional[Path], output: Optional[Path], output_dir: Path) -> list[Path]:
    """Build one or more tick-metric plots through the shared plotter module."""

    from src.visualization.plotter.tick_metrics_plotter import SERIES_GROUPS, plot_alpha_metrics, plot_group

    if output is not None and group in {"alpha", "all"}:
        raise SystemExit("--output can only be used with single-figure plot groups")

    if group == "all":
        return [
            plot_group("neurons", metrics_path, output_dir / SERIES_GROUPS["neurons"]["filename"]),
            *plot_alpha_metrics(metrics_path, output_dir),
            plot_group("substantia_nigra", metrics_path, output_dir / SERIES_GROUPS["substantia_nigra"]["filename"]),
        ]
    if group == "alpha":
        return plot_alpha_metrics(metrics_path, output_dir)

    resolved_group = {
        "alpha-free": "alpha_free",
        "alpha-aggregate": "alpha_aggregate",
        "sn": "substantia_nigra",
    }.get(group, group)
    destination = output or output_dir / SERIES_GROUPS[resolved_group]["filename"]
    return [plot_group(resolved_group, metrics_path, destination)]


def tick_metrics_for(log_dir: Path) -> Path:
    """Return the tick_metrics.csv path associated with a simulation output."""

    from src.analysis.data.run_data import find_tick_metrics

    return find_tick_metrics(log_dir)


def write_or_print_report(report: dict[str, Any], output: Path, to_stdout: bool) -> None:
    """Write a JSON report unless the caller requested stdout."""

    if to_stdout:
        print(json_payload(report))
        return
    print(write_report(report, output))


def write_report(report: dict[str, Any], output: Path) -> Path:
    """Write one JSON report and return its path."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_payload(report) + "\n", encoding="utf-8")
    return output


def json_payload(report: dict[str, Any]) -> str:
    """Serialize reports with stable key ordering for easier comparison."""

    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def print_paths(paths: list[Path]) -> None:
    """Print generated paths one per line."""

    for path in paths:
        print(path)


if __name__ == "__main__":
    raise SystemExit(main())
