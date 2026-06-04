from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import networkx as nx
from multilevelgraphs import MultilevelGraph
from src.analysis.graph.g0_builder import build_g0, write_g0_gexf
from src.analysis.schemes import AgentClusteringScheme, TimeContractionScheme


DEFAULT_LOG_DIR = Path("output/simulation/logs")
DEFAULT_GRAPH_DIR = Path("output/analysis/graphs")
MAX_GEPHI_ATTRIBUTE_CHARS = 4096
GEPHI_DROPPED_PROVENANCE_ATTRS = {"original_node_ids", "source_edge_ids", "source_log_node_ids", "edge_ids", "relations", "mechanisms", "causal_kinds", "actions", "outcomes"}
MULTILEVELGRAPH_NODE_RESERVED_ATTRS = {"key", "level", "dec", "component_sets", "supernode"}
MULTILEVELGRAPH_EDGE_RESERVED_ATTRS = {"tail", "head", "level", "dec"}


@dataclass
class MultilevelBuildResult:
    g0: Any
    g1: Any
    g2: Any
    multilevel_graph: Optional[Any]
    output_paths: dict[str, Path]
    report: str


def build_multilevel_graphs(log_dir: Path | str = DEFAULT_LOG_DIR, *, output_dir: Path | str = DEFAULT_GRAPH_DIR, add_g0_continuity: bool = True, include_field_continuity: bool = True, rupture_grace_ticks: int = 2, time_window_size: Optional[int] = None, build_external_graph: bool = False, write_outputs: bool = True, write_g0: bool = False) -> MultilevelBuildResult:
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    g0 = build_g0(log_dir, add_continuity=add_g0_continuity, include_field_continuity=include_field_continuity, rupture_grace_ticks=rupture_grace_ticks)
    time_scheme = TimeContractionScheme(window_size=time_window_size)
    agent_scheme = AgentClusteringScheme()
    g1 = time_scheme.contract_networkx(g0)
    g2 = agent_scheme.contract_networkx(g1)
    ml_graph = build_external_multilevel_graph(g0, time_scheme, agent_scheme) if build_external_graph else None
    external_status = "built" if ml_graph is not None else "skipped"
    output_paths: dict[str, Path] = {}
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        if write_g0:
            output_paths["g0"] = write_g0_gexf(g0, output_dir / "g0.gexf")
        output_paths["g1"] = write_networkx_gexf(g1, output_dir / "g1.gexf")
        output_paths["g2"] = write_networkx_gexf(g2, output_dir / "g2.gexf")
    if write_outputs:
        output_paths["report"] = write_report(g0, g1, g2, output_paths["g0"], output_paths["g1"], output_paths["g2"])
    return MultilevelBuildResult(g0=g0, g1=g1, g2=g2, multilevel_graph=ml_graph, output_paths=output_paths)


def build_external_multilevel_graph(g0, time_scheme: TimeContractionScheme, agent_scheme: AgentClusteringScheme):
    ml_graph = MultilevelGraph(multilevelgraphs_safe_copy(g0), [time_scheme.clone(), agent_scheme.clone()])
    ml_graph.build_contraction_schemes()
    return ml_graph


def multilevelgraphs_safe_copy(graph):
    safe = nx.DiGraph()
    safe.graph.update(graph.graph)
    for node_id, attrs in graph.nodes(data=True):
        safe.add_node(node_id, **rename_reserved_attrs(attrs, MULTILEVELGRAPH_NODE_RESERVED_ATTRS))
    for source, target, attrs in graph.edges(data=True):
        safe.add_edge(source, target, **rename_reserved_attrs(attrs, MULTILEVELGRAPH_EDGE_RESERVED_ATTRS))
    return safe


def rename_reserved_attrs(attrs: dict[str, Any], reserved: set[str]) -> dict[str, Any]:
    renamed = {}
    for key, value in attrs.items():
        renamed[f"raw_{key}" if key in reserved else key] = value
    return renamed


def write_networkx_gexf(graph, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(gephi_serializable_copy(graph), path)
    return path


def gephi_serializable_copy(graph):
    copy = nx.DiGraph(**{key: gephi_scalar_attr(key, value) for key, value in graph.graph.items()})
    for node_id, attrs in graph.nodes(data=True):
        copy.add_node(node_id, **gephi_attrs(attrs))
    for source, target, attrs in graph.edges(data=True):
        copy.add_edge(source, target, **gephi_attrs(attrs))
    return copy


def gephi_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in attrs.items():
        if key in GEPHI_DROPPED_PROVENANCE_ATTRS:
            safe[f"{key}_count"] = provenance_count(value)
            continue
        safe[key] = gephi_scalar_attr(key, value)
    return safe


def gephi_scalar_attr(key: str, value: Any) -> Any:
    scalar = serializable_copy_scalar(value)
    if isinstance(scalar, str) and len(scalar) > MAX_GEPHI_ATTRIBUTE_CHARS:
        return scalar[:MAX_GEPHI_ATTRIBUTE_CHARS] + "...[truncated]"
    return scalar


def serializable_copy_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value if item is not None)
    return str(value)


def provenance_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        return 0 if not value else len(value.split("|"))
    return 1


def write_report(g0, g1, g2, report: str, g0_path: Path, g1_path: Path, g2_path: Path) -> None:
    print(
        f"G0: {g0.number_of_nodes()}, {g0.number_of_edges()} nodes, {g0.number_of_edges()} edges.\n"
        f"Saved to {g0_path}"
        f"G1 (time contraction): {nx.number_of_nodes(g0)}, {nx.number_of_edges(g0)} nodes, {nx.number_of_edges(g0)} edges.\n"
        f"Saved to {g1_path}"
        f"G2 (agent clustering): {nx.number_of_nodes(g0)}, {nx.number_of_edges(g0)} nodes, {nx.number_of_edges(g0)} edges.\n"
        f"Saved to {g2_path}"
        f"G3 (topology contraction): NOT IMPLEMENTED YET"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G0/G1/G2 multilevel graph exports.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--no-g0-continuity", action="store_true")
    parser.add_argument("--no-field-continuity", action="store_true")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2)
    parser.add_argument("--time-window-size", type=int, default=None)
    parser.add_argument("--external-multilevelgraph", action="store_true", help="Also instantiate the external MultilevelGraph object. This can be slow on large G0 traces.")
    parser.add_argument("--write-g0", action="store_true")
    args = parser.parse_args()

    result = build_multilevel_graphs(
        args.log_dir,
        output_dir=args.output_dir,
        add_g0_continuity=not args.no_g0_continuity,
        include_field_continuity=not args.no_field_continuity,
        rupture_grace_ticks=args.rupture_grace_ticks,
        time_window_size=args.time_window_size,
        build_external_graph=args.external_multilevelgraph,
        write_g0=args.write_g0
    )
    print(result.report)


if __name__ == "__main__":
    main()
