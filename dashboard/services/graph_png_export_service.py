from __future__ import annotations
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# TODO potentially deprecated?

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEW_EXPORT_DIR = PROJECT_ROOT / "output" / "graphs" / "views"

NODE_COLOR_MAP = {
    "agent_state": "#4C78A8",
    "environment_field": "#F58518",
    "aggregate": "#B279A2",
    "agent_state_cluster": "#4C78A8",
    "feedback_component": "#E45756",
    "self_feedback": "#F58518",
    "singleton_process": "#72B7B2"
}
DEFAULT_NODE_COLOR = "#9D9D9D"

class GraphPngExportService:
    def __init__(self, output_dir: Path | str = DEFAULT_VIEW_EXPORT_DIR) -> None:
        self.output_dir = Path(output_dir)

    def save_png(self, graph: Any, *, level: str, name_hint: str = "view") -> Path:
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot export an empty graph view.")
        try:
            import networkx as nx
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PNG export requires `networkx`.") from exc
        try:
            import matplotlib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PNG export requires `matplotlib`.") from exc
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output_path = self.output_path(level=level, name_hint=name_hint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = figure_size(graph.number_of_nodes())
        figure, axis = plt.subplots(figsize=(width, height), dpi=180)
        axis.set_axis_off()
        positions = layout_positions(graph)
        node_colors = [node_color(attributes) for _, attributes in graph.nodes(data=True)]
        node_sizes = [node_size(graph, node_id) for node_id in graph.nodes()]
        edge_widths = [edge_width(attributes) for _, _, attributes in graph.edges(data=True)]
        nx.draw_networkx_edges(graph, positions, ax=axis, arrows=graph.is_directed(),
            arrowstyle="-|>", arrowsize=10, edge_color="#7A7A7A", alpha=0.55,
            width=edge_widths, connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_nodes(
            graph, positions, ax=axis, node_color=node_colors,
            node_size=node_sizes, edgecolors="#222222", linewidths=0.45, alpha=0.95)
        if graph.number_of_nodes() <= 160:
            nx.draw_networkx_labels(graph, positions, labels=node_labels(graph), ax=axis, font_size=7, font_color="#111111")
        axis.set_title(f"{level.upper()} graph view", fontsize=12)
        figure.tight_layout(pad=0.6)
        figure.savefig(output_path, format="png", bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output_path

    def output_path(self, *, level: str, name_hint: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{level.lower()}_{clean_filename(name_hint)}_{timestamp}.png"


def layout_positions(graph: Any) -> dict[Any, tuple[float, float]]:
    import networkx as nx
    if graph.number_of_nodes() == 1:
        node_id = next(iter(graph.nodes()))
        return {node_id: (0.0, 0.0)}
    explicit_positions = explicit_visual_positions(graph)
    if explicit_positions is not None:
        return explicit_positions
    layout_graph = weighted_layout_graph(graph)
    components = sorted(nx.connected_components(layout_graph), key=len, reverse=True)
    if len(components) == 1:
        return force_directed_layout(layout_graph)
    return pack_component_layouts(layout_graph, components)


def explicit_visual_positions(graph: Any) -> dict[Any, tuple[float, float]] | None:
    positions: dict[Any, tuple[float, float]] = {}
    for node_id, attributes in graph.nodes(data=True):
        x = numeric(attributes.get("pyvis_x") or attributes.get("vis_x") or attributes.get("layout_x"))
        y = numeric(attributes.get("pyvis_y") or attributes.get("vis_y") or attributes.get("layout_y"))
        if x is None or y is None:
            return None
        positions[node_id] = (x, y)
    return normalize_positions(positions)


def weighted_layout_graph(graph: Any) -> Any:
    import networkx as nx
    layout_graph = nx.Graph()
    for node_id, attributes in graph.nodes(data=True):
        layout_graph.add_node(node_id, **attributes)
    for source, target, attributes in graph.edges(data=True):
        weight = max(0.2, min(8.0, math.sqrt(float_or_default(attributes.get("count"), 1.0))))
        if layout_graph.has_edge(source, target):
            layout_graph.edges[source, target]["weight"] += weight
        else:
            layout_graph.add_edge(source, target, weight=weight)
    return layout_graph


def force_directed_layout(graph: Any) -> dict[Any, tuple[float, float]]:
    import networkx as nx
    if hasattr(nx, "forceatlas2_layout"):
        try:
            return normalize_positions(nx.forceatlas2_layout(graph, seed=42, max_iter=650, gravity=1.0, scaling_ratio=1.6, weight="weight"))
        except TypeError:
            pass
    node_count = max(1, graph.number_of_nodes())
    return normalize_positions(
        nx.spring_layout(graph, seed=42,
            k=1.55 / math.sqrt(node_count), iterations=420 if node_count <= 500 else 180,
            weight="weight", scale=1.0))

def pack_component_layouts(graph: Any, components: list[set[Any]]) -> dict[Any, tuple[float, float]]:
    positions: dict[Any, tuple[float, float]] = {}
    component_layouts = []
    for component in components:
        subgraph = graph.subgraph(component).copy()
        local_positions = force_directed_layout(subgraph)
        radius = component_radius(len(component))
        component_layouts.append((component, scale_positions(local_positions, radius), radius))
    centers = component_centers([radius for _, _, radius in component_layouts])
    for (_, local_positions, _), center in zip(component_layouts, centers):
        cx, cy = center
        for node_id, (x, y) in local_positions.items():
            positions[node_id] = (x + cx, y + cy)
    return normalize_positions(positions)


def component_centers(radii: list[float]) -> list[tuple[float, float]]:
    if len(radii) == 1:
        return [(0.0, 0.0)]
    centers: list[tuple[float, float]] = []
    max_radius = max(radii)
    columns = max(1, math.ceil(math.sqrt(len(radii))))
    cell = max(2.6, max_radius * 2.8)
    for index, _ in enumerate(radii):
        row = index // columns
        column = index % columns
        centers.append((column * cell, -row * cell))
    return centers


def component_radius(node_count: int) -> float:
    return max(0.65, min(2.6, 0.34 * math.sqrt(max(1, node_count))))


def normalize_positions(positions: dict[Any, tuple[float, float]]) -> dict[Any, tuple[float, float]]:
    if not positions:
        return {}
    xs = [float(position[0]) for position in positions.values()]
    ys = [float(position[1]) for position in positions.values()]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    return {node_id: ((float(x) - center_x) / span, (float(y) - center_y) / span) for node_id, (x, y) in positions.items()}


def scale_positions(positions: dict[Any, tuple[float, float]], scale: float) -> dict[Any, tuple[float, float]]:
    normalized = normalize_positions(positions)
    return {node_id: (x * scale, y * scale) for node_id, (x, y) in normalized.items()}

def figure_size(node_count: int) -> tuple[float, float]:
    side = math.sqrt(max(1, node_count))
    width = min(32.0, max(10.0, side * 1.45))
    height = min(26.0, max(7.0, side * 1.05))
    return width, height

def node_color(attributes: dict[str, Any]) -> str:
    key = str(attributes.get("semantic_kind") or attributes.get("cluster_kind") or attributes.get("pattern_kind") or attributes.get("kind") or "")
    return NODE_COLOR_MAP.get(key, DEFAULT_NODE_COLOR)

def node_size(graph: Any, node_id: Any) -> float:
    degree = graph.degree(node_id)
    return min(760.0, 160.0 + 42.0 * math.sqrt(max(0, degree)))

def edge_width(attributes: dict[str, Any]) -> float:
    try:
        count = max(1.0, float(attributes.get("count", 1)))
    except (TypeError, ValueError):
        count = 1.0
    return min(4.5, 0.8 + math.sqrt(count) * 0.35)

def node_labels(graph: Any) -> dict[Any, str]:
    return {
        node_id: short_label(attributes.get("display_label") or attributes.get("biological_label") or attributes.get("label") or node_id)
        for node_id, attributes in graph.nodes(data=True)}

def short_label(value: Any, *, max_length: int = 28) -> str:
    label = str(value)
    return label if len(label) <= max_length else label[: max_length - 1] + "..."

def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def float_or_default(value: Any, default: float) -> float:
    converted = numeric(value)
    return default if converted is None else converted

def clean_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return cleaned or "view"
