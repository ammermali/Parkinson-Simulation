from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SIMULATION_LOG_DIR = PROJECT_ROOT / "output" / "simulation" / "logs"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis"
DEFAULT_GRAPH_DIR = DEFAULT_ANALYSIS_DIR / "graphs"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "output" / "plots"
DEFAULT_GIF_OUTPUT = DEFAULT_PLOT_DIR / "simulation_snapshot.gif"
DEFAULT_G0_RANGE_PLOT_OUTPUT = DEFAULT_PLOT_DIR / "g0_tick_range.png"
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
        help="Validate G0 causal trace logs.",
    )
    validate_g0.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    validate_g0.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "g0_trace_validation_latest.json")
    validate_g0.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    validate_g0.set_defaults(handler=command_validate_g0)

    validate_init = subparsers.add_parser(
        "validate-init",
        help="Validate initialization logs.",
    )
    validate_init.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    validate_init.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "initialization_validation_latest.json")
    validate_init.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    validate_init.set_defaults(handler=command_validate_initialization)

    mechanisms = subparsers.add_parser(
        "mechanisms",
        help="Summarize biological mechanisms from G0 logs.",
    )
    mechanisms.add_argument("log_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    mechanisms.add_argument("--no-by-tick", action="store_true", help="Omit per-tick mechanism counts.")
    mechanisms.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "mechanism_metrics_latest.json")
    mechanisms.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    mechanisms.set_defaults(handler=command_mechanisms)

    intervention = subparsers.add_parser(
        "intervention",
        aliases=["interventions"],
        help="Summarize completed runs for intervention comparison.",
    )
    intervention.add_argument("log_dirs", nargs="*", type=Path, default=[DEFAULT_SIMULATION_LOG_DIR])
    intervention.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "intervention_metrics_latest.json")
    intervention.add_argument("--stdout", action="store_true", help="Print the report instead of writing JSON.")
    intervention.set_defaults(handler=command_intervention)

    graphs = subparsers.add_parser(
        "graphs",
        help="Build G1/G2 multilevel graph exports, optionally including G0.",
    )
    add_graph_arguments(graphs)
    graphs.set_defaults(handler=command_graphs)

    plot = subparsers.add_parser(
        "plot",
        help="Create tick_metrics.csv plots.",
    )
    plot.add_argument("group", choices=("neurons", "alpha", "alpha-free", "alpha-aggregate", "sn", "all"))
    plot.add_argument("--metrics", type=Path, help="Path to tick_metrics.csv.")
    plot.add_argument("--output", type=Path, help="Single plot destination. Not valid for 'alpha' or 'all'.")
    plot.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR, help="Plot directory for grouped outputs.")
    plot.set_defaults(handler=command_plot)

    plot_g0 = subparsers.add_parser(
        "plot-g0",
        help="Create a temporally layered PNG of a G0 tick range.",
    )
    plot_g0.add_argument("--log-dir", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    plot_g0.add_argument("--start-tick", type=int, required=True)
    plot_g0.add_argument("--end-tick", type=int, required=True)
    plot_g0.add_argument("--output", type=Path, default=DEFAULT_G0_RANGE_PLOT_OUTPUT)
    plot_g0.add_argument("--width", type=float, help="Figure width in inches.")
    plot_g0.add_argument("--height", type=float, help="Figure height in inches.")
    plot_g0.add_argument("--dpi", type=int, default=170)
    plot_g0.add_argument("--edge-labels", action="store_true", help="Label edges when the selected graph is small.")
    plot_g0.add_argument("--no-continuity", action="store_true", help="Do not include temporal continuity edges.")
    plot_g0.add_argument("--no-field-continuity", action="store_true", help="Do not include field continuity edges.")
    plot_g0.add_argument("--rupture-grace-ticks", type=int, default=2)
    plot_g0.set_defaults(handler=command_plot_g0)

    gif = subparsers.add_parser(
        "gif",
        help="Generate a post-run simulation snapshot GIF from G0 logs.",
    )
    gif.add_argument("environment", choices=("sn", "substantia_nigra", "neuron"), help="Environment grid to visualize.")
    gif.add_argument("--log-dir", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    gif.add_argument("--start-tick", type=int, required=True)
    gif.add_argument("--end-tick", type=int, required=True)
    gif.add_argument("--neuron-uid", help="Neuron uid required when environment is 'neuron'.")
    gif.add_argument("--output", type=Path, default=DEFAULT_GIF_OUTPUT)
    gif.add_argument("--fps", type=int, default=4)
    gif.set_defaults(handler=command_gif)

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
    postprocess.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    postprocess.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    postprocess.add_argument("--no-by-tick", action="store_true", help="Omit per-tick mechanism counts.")
    postprocess.add_argument("--skip-validation", action="store_true", help="Do not run validate-g0 and validate-init.")
    postprocess.add_argument("--skip-metrics", action="store_true", help="Do not run mechanisms and intervention metrics.")
    postprocess.add_argument("--plots", action="store_true", help="Also generate all tick metric plots.")
    postprocess.add_argument("--graphs", action="store_true", help="Also build G1/G2 graph exports.")
    add_graph_arguments(postprocess, include_log_dir=False)
    postprocess.set_defaults(handler=command_postprocess)

    return parser


def add_graph_arguments(parser: argparse.ArgumentParser, *, include_log_dir: bool = True) -> None:
    """Attach graph-build options shared by `graphs` and `postprocess`."""

    if include_log_dir:
        parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_SIMULATION_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--write-g0", action="store_true", help="Also export g0.gexf. This can be very large.")
    parser.add_argument("--no-g0-continuity", action="store_true", help="Do not add temporal continuity edges to G0 before contraction.")
    parser.add_argument("--no-field-continuity", action="store_true", help="Do not add continuity edges for environment or neuron fields.")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2, help="Ticks retained after a neuron becomes ruptured.")
    parser.add_argument("--time-window-size", type=int, default=None, help="Optional fixed time window size for G1.")
    parser.add_argument(
        "--external-multilevelgraph",
        action="store_true",
        help="Also instantiate the external MultilevelGraph object. This can be slow on large traces.",
    )


def command_simulate(args: argparse.Namespace) -> int:
    """Load system params and run the simulation engine."""

    from src.simulation.engine import run
    from src.simulation.utils import Params

    if args.mode != "rule":
        raise SystemExit(f"Unsupported simulation mode: {args.mode}")
    run(Params(args.params).as_dict())
    return 0


def command_validate_g0(args: argparse.Namespace) -> int:
    """Validate G0 causal trace logs and write a JSON report."""

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
    """Summarize scalar and event behavior for intervention analysis."""

    from src.analysis.metrics.intervention_metrics import summarize_run

    report = {
        "runs": [
            summarize_run(log_dir)
            for log_dir in args.log_dirs
        ]
    }
    write_or_print_report(report, args.output, args.stdout)
    return 0


def command_graphs(args: argparse.Namespace) -> int:
    """Build and export Gephi-ready G1/G2 graphs from simulation logs."""

    paths = build_graph_exports(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        write_g0=args.write_g0,
        add_g0_continuity=not args.no_g0_continuity,
        include_field_continuity=not args.no_field_continuity,
        rupture_grace_ticks=args.rupture_grace_ticks,
        time_window_size=args.time_window_size,
        build_external_graph=args.external_multilevelgraph,
    )
    print_paths(paths)
    return 0


def command_plot(args: argparse.Namespace) -> int:
    """Generate one or more tick-metrics plots."""

    paths = build_plots(args.group, metrics_path=args.metrics, output=args.output, output_dir=args.output_dir)
    print_paths(paths)
    return 0


def command_plot_g0(args: argparse.Namespace) -> int:
    """Generate one temporally layered G0 range plot."""

    from src.visualization.plotter.g0_range_plotter import plot_g0_tick_range

    output = plot_g0_tick_range(
        args.log_dir,
        start_tick=args.start_tick,
        end_tick=args.end_tick,
        output=args.output,
        add_continuity=not args.no_continuity,
        include_field_continuity=not args.no_field_continuity,
        rupture_grace_ticks=args.rupture_grace_ticks,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        edge_labels=args.edge_labels,
    )
    print(output)
    return 0


def command_gif(args: argparse.Namespace) -> int:
    """Generate a post-run GIF from G0 logs."""

    from src.visualization.simulation_gif import generate_simulation_gif

    output = generate_simulation_gif(
        args.log_dir,
        environment=args.environment,
        start_tick=args.start_tick,
        end_tick=args.end_tick,
        output=args.output,
        neuron_uid=args.neuron_uid,
        fps=args.fps,
    )
    print(output)
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
    analysis_dir = args.analysis_dir

    if not args.skip_validation:
        from src.analysis.validation.validate_g0_trace import validate_trace
        from src.analysis.validation.validate_initialization_log import validate_initialization

        produced.append(write_report(validate_trace(args.log_dir), analysis_dir / "g0_trace_validation_latest.json"))
        produced.append(write_report(validate_initialization(args.log_dir), analysis_dir / "initialization_validation_latest.json"))

    if not args.skip_metrics:
        from src.analysis.metrics.intervention_metrics import summarize_run
        from src.analysis.metrics.mechanism_metrics import summarize_mechanisms

        produced.append(
            write_report(
                {"runs": [summarize_mechanisms(args.log_dir, include_by_tick=not args.no_by_tick)]},
                analysis_dir / "mechanism_metrics_latest.json",
            )
        )
        produced.append(
            write_report(
                {"runs": [summarize_run(args.log_dir)]},
                analysis_dir / "intervention_metrics_latest.json",
            )
        )

    if args.plots:
        produced.extend(build_plots("all", metrics_path=tick_metrics_for(args.log_dir), output=None, output_dir=args.plot_dir))

    if args.graphs:
        produced.extend(
            build_graph_exports(
                log_dir=args.log_dir,
                output_dir=args.output_dir,
                write_g0=args.write_g0,
                add_g0_continuity=not args.no_g0_continuity,
                include_field_continuity=not args.no_field_continuity,
                rupture_grace_ticks=args.rupture_grace_ticks,
                time_window_size=args.time_window_size,
                build_external_graph=args.external_multilevelgraph,
            )
        )

    print_paths(produced)
    return 0


def build_graph_exports(
    *,
    log_dir: Path,
    output_dir: Path,
    write_g0: bool,
    add_g0_continuity: bool,
    include_field_continuity: bool,
    rupture_grace_ticks: int,
    time_window_size: Optional[int],
    build_external_graph: bool
) -> list[Path]:
    """Build G0/G1/G2 exports through the canonical multilevel graph builder."""
    from src.analysis.graph.multilevel_builder import build_multilevel_graphs

    result = build_multilevel_graphs(
        log_dir,
        output_dir=output_dir,
        add_g0_continuity=add_g0_continuity,
        include_field_continuity=include_field_continuity,
        rupture_grace_ticks=rupture_grace_ticks,
        time_window_size=time_window_size,
        build_external_graph=build_external_graph,
        write_g0=write_g0,
    )
    return list(result.output_paths.values())


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

    candidates = [
        log_dir / "tick_metrics.csv",
        log_dir / "logs" / "tick_metrics.csv",
        log_dir / "log" / "tick_metrics.csv",
    ]
    if log_dir.name in {"logs", "log"}:
        candidates.append(log_dir.parent / "tick_metrics.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return log_dir / "tick_metrics.csv"


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
