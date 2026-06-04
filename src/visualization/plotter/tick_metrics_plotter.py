from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
from typing import Callable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICK_METRICS = PROJECT_ROOT / "output" / "simulation" / "logs" / "tick_metrics.csv"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "output" / "plots"

SERIES_GROUPS = {
    "neurons": {
        "columns": (
            "neurons_healthy",
            "neurons_compromised",
            "neurons_apoptotic",
            "neurons_ruptures"
        ),
        "labels": {
            "neurons_healthy": "Healthy",
            "neurons_compromised": "Compromised",
            "neurons_apoptotic": "Apoptotic",
            "neurons_ruptures": "Ruptured"
        },
        "title": "Neuron State Counts",
        "ylabel": "Neuron count",
        "filename": "neuron_states.png",
    },
    "alpha_free": {
        "columns": (
            "free_alpha",
        ),
        "labels": {
            "free_alpha": "Free alpha-synuclein",
        },
        "title": "Free Alpha-Synuclein Count",
        "ylabel": "Free alpha-synuclein count",
        "filename": "alpha_free.png",
    },
    "alpha_aggregate": {
        "columns": (
            "alpha_aggregate",
        ),
        "labels": {
            "alpha_aggregate": "Aggregated alpha-synuclein proteins",
        },
        "title": "Aggregated Alpha-Synuclein Protein Count",
        "ylabel": "Aggregated protein count",
        "filename": "alpha_aggregate.png",
    },
    "substantia_nigra": {
        "columns": (
            "debris",
            "inflammation",
            "dopamine",
        ),
        "labels": {
            "debris": "Debris",
            "inflammation": "Inflammation",
            "dopamine": "Dopamine",
        },
        "title": "Substantia Nigra Scalars",
        "ylabel": "Normalized scalar",
        "filename": "substantia_nigra_scalars.png",
    },
}


def plot_neuron_states(metrics_path: Path | str | None = None, output_path: Optional[Path | str] = None) -> Path:
    """Plot healthy, compromised, apoptotic and ruptured neuron counts by tick."""

    return plot_group("neurons", metrics_path, output_path)


def plot_alpha_free(metrics_path: Path | str | None = None, output_path: Optional[Path | str] = None) -> Path:
    """Plot free alpha-synuclein count by tick."""

    return plot_group("alpha_free", metrics_path, output_path)


def plot_alpha_aggregate(metrics_path: Path | str | None = None, output_path: Optional[Path | str] = None) -> Path:
    """Plot alpha-synuclein proteins represented inside aggregates by tick."""

    return plot_group("alpha_aggregate", metrics_path, output_path)


def plot_alpha_metrics(metrics_path: Path | str | None = None, output_dir: Optional[Path | str] = None) -> list[Path]:
    """Plot free and aggregate-member alpha-synuclein counts separately."""

    output_base = Path(output_dir) if output_dir is not None else DEFAULT_PLOT_DIR
    return [
        plot_alpha_free(metrics_path, output_base / SERIES_GROUPS["alpha_free"]["filename"]),
        plot_alpha_aggregate(metrics_path, output_base / SERIES_GROUPS["alpha_aggregate"]["filename"]),
    ]


def plot_substantia_nigra_metrics(metrics_path: Path | str | None = None, output_path: Optional[Path | str] = None) -> Path:
    """Plot extracellular debris, inflammation and dopamine by tick."""

    return plot_group("substantia_nigra", metrics_path, output_path)


def plot_group(group: str, metrics_path: Path | str | None = None, output_path: Optional[Path | str] = None) -> Path:
    """Plot one predefined tick-metrics group and return the generated PNG path."""

    if group not in SERIES_GROUPS:
        raise ValueError(f"Unknown plot group: {group}")
    config = SERIES_GROUPS[group]
    metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_TICK_METRICS
    output = Path(output_path) if output_path is not None else DEFAULT_PLOT_DIR / config["filename"]
    rows = read_tick_metrics(metrics_path, config["columns"])
    _render_plot(
        ticks=[row["tick"] for row in rows],
        rows=rows,
        columns=config["columns"],
        labels=config["labels"],
        title=config["title"],
        ylabel=config["ylabel"],
        output_path=output,
    )
    return output


def read_tick_metrics(metrics_path: Path | str | None, required_columns: tuple[str, ...]) -> list[dict[str, float]]:
    """Read tick_metrics.csv and coerce selected columns to floats."""

    metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_TICK_METRICS
    if not metrics_path.exists():
        raise FileNotFoundError(f"Tick metrics CSV not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = [column for column in ("tick", *required_columns) if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {metrics_path}: {', '.join(missing)}")
        rows = []
        for row in reader:
            rows.append({
                "tick": _number(row["tick"]),
                **{
                    column: _number(row[column])
                    for column in required_columns
                },
            })
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_path}")
    return rows


def _render_plot(ticks: list[float], rows: list[dict[str, float]], columns: tuple[str, ...], labels: dict[str, str], title: str, ylabel: str, output_path: Path) -> None:
    """Render a multi-line plot.

    PNG output uses matplotlib by default. SVG remains available explicitly for
    dependency-free output in minimal analysis environments.
    """

    if output_path.suffix.lower() == ".svg":
        _render_svg_plot(ticks, rows, columns, labels, title, ylabel, output_path)
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for non-SVG visualization outputs") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for column in columns:
        axis.plot(ticks, [row[column] for row in rows], label=labels.get(column, column), linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Tick")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _render_svg_plot(ticks: list[float], rows: list[dict[str, float]], columns: tuple[str, ...], labels: dict[str, str], title: str, ylabel: str, output_path: Path) -> None:
    """Render a simple line chart as SVG without external dependencies."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = 560
    left = 82
    right = 28
    top = 56
    bottom = 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_values = [
        row[column]
        for row in rows
        for column in columns
    ]
    x_min = min(ticks)
    x_max = max(ticks)
    y_min = 0.0
    y_max = max(1.0, max(y_values))
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")

    def x_coord(value: float) -> float:
        """Map a tick value into SVG x coordinates."""

        if x_max == x_min:
            return left + plot_width / 2
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        """Map a series value into SVG y coordinates."""

        if y_max == y_min:
            return top + plot_height
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    grid_lines = []
    for index in range(6):
        ratio = index / 5
        y = top + plot_height - ratio * plot_height
        value = y_min + ratio * (y_max - y_min)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" />'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">{value:.2f}</text>'
        )

    series = []
    legend = []
    for index, column in enumerate(columns):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{x_coord(row['tick']):.2f},{y_coord(row[column]):.2f}"
            for row in rows
        )
        series.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round" />')
        legend_x = left + index * 180
        legend_y = height - 28
        legend.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3" />'
            f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="13" fill="#111827">{escape(labels.get(column, column))}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>
<text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#374151">Tick</text>
<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#374151">{escape(ylabel)}</text>
{''.join(grid_lines)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" />
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" />
<text x="{left}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12" fill="#4b5563">{x_min:.0f}</text>
<text x="{left + plot_width}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12" fill="#4b5563">{x_max:.0f}</text>
{''.join(series)}
{''.join(legend)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def _number(value: str) -> float:
    """Coerce CSV numeric strings into floats."""

    return float(value)


def main(default_group: Optional[str] = None) -> None:
    """CLI entry point used by the generic script and the fixed plot wrappers."""

    parser = argparse.ArgumentParser(description="Plot simulation tick_metrics.csv time series.")
    if default_group is None:
        parser.add_argument("group", choices=sorted(SERIES_GROUPS), help="Metric group to plot.")
    parser.add_argument("--metrics", type=Path, help="Path to tick_metrics.csv.")
    parser.add_argument("--output", type=Path, help="Plot destination. Defaults to output/plots.")
    args = parser.parse_args()

    group = default_group or args.group
    output = plot_group(group, args.metrics, args.output)
    print(output)


def run_fixed_plotter(plotter: Callable[[Path | str, Optional[Path | str]], Path]) -> None:
    """Run one fixed plotter from a thin wrapper module."""

    parser = argparse.ArgumentParser(description=plotter.__doc__)
    parser.add_argument("--metrics", type=Path, help="Path to tick_metrics.csv.")
    parser.add_argument("--output", type=Path, help="Plot destination. Defaults to output/plots.")
    args = parser.parse_args()
    print(plotter(args.metrics, args.output))


def run_alpha_pair_plotter() -> None:
    """Run the paired alpha plotter that writes two separate figures."""

    parser = argparse.ArgumentParser(description=plot_alpha_metrics.__doc__)
    parser.add_argument("--metrics", type=Path, help="Path to tick_metrics.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR, help="Directory for alpha_free.png and alpha_aggregate.png.")
    args = parser.parse_args()
    for path in plot_alpha_metrics(args.metrics, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
