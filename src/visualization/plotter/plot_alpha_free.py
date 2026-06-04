from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.visualization.plotter.tick_metrics_plotter import plot_alpha_free, run_fixed_plotter


if __name__ == "__main__":
    run_fixed_plotter(plot_alpha_free)
