from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import networkx as nx
from multilevelgraphs import MultilevelGraph
from src.analysis.graph.g0_builder import build_g0, write_g0_lite_gexf
from src.analysis.schemes import AgentClusteringScheme, TimeContractionScheme, TopologicalSCCContractionScheme


DEFAULT_LOG_DIR = Path("output/run_logs")
DEFAULT_GRAPH_DIR = Path("output/graphs")
MAX_GEPHI_ATTRIBUTE_CHARS = 4096
GEPHI_DROPPED_PROVENANCE_ATTRS = {"original_node_ids", "component_node_ids", "source_edge_ids", "source_log_node_ids", "node_origins", "edge_ids", "relations", "mechanisms", "causal_kinds", "outcomes"}
MULTILEVELGRAPH_NODE_RESERVED_ATTRS = {"key", "level", "dec", "component_sets", "supernode"}
MULTILEVELGRAPH_EDGE_RESERVED_ATTRS = {"tail", "head", "level", "dec"}


@dataclass
class MultilevelBuildResult:
    g0: Any
    g1: Any
    g2: Any
    g3: Any
    multilevel_graph: Optional[Any]
    output_paths: dict[str, Path]


@dataclass
class GraphLevelBuildResult:
    target_level: str
    graphs_by_level: dict[str, Any]
    output_paths: dict[str, Path]


def build_multilevel_graphs(log_dir: Path | str = DEFAULT_LOG_DIR, *, output_dir: Path | str = DEFAULT_GRAPH_DIR, add_g0_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, time_window_size: Optional[int] = None, build_external_graph: bool = False, write_outputs: bool = True, write_g0: bool = True) -> MultilevelBuildResult:
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    g0 = build_g0(log_dir, add_continuity=add_g0_continuity, include_field_continuity=include_field_continuity, include_snapshot_nodes=include_snapshot_nodes, rupture_grace_ticks=rupture_grace_ticks)
    graphs_by_level, (time_scheme, agent_scheme, topology_scheme) = build_contracted_graphs(g0, through_level="g3", time_window_size=time_window_size)
    g1 = graphs_by_level["g1"]
    g2 = graphs_by_level["g2"]
    g3 = graphs_by_level["g3"]
    ml_graph = build_external_multilevel_graph(g0, time_scheme, agent_scheme, topology_scheme) if build_external_graph else None
    output_paths: dict[str, Path] = {}
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        if write_g0:
            output_paths.update(write_level_exports("g0", g0, output_dir=output_dir))
            output_paths["g0_lite_gephi"] = write_g0_lite_gexf(g0, output_dir / "g0.lite.gext")
        for level in ("g1", "g2", "g3"):
            output_paths.update(write_level_exports(level, graphs_by_level[level], output_dir=output_dir))
    return MultilevelBuildResult(g0=g0, g1=g1, g2=g2, g3=g3, multilevel_graph=ml_graph, output_paths=output_paths)


def build_g1_exports(log_dir: Path | str = DEFAULT_LOG_DIR, *, output_dir: Path | str = DEFAULT_GRAPH_DIR, add_g0_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, time_window_size: Optional[int] = None) -> GraphLevelBuildResult:
    return build_graph_level_exports(
        "g1",
        log_dir,
        output_dir=output_dir,
        add_g0_continuity=add_g0_continuity,
        include_field_continuity=include_field_continuity,
        include_snapshot_nodes=include_snapshot_nodes,
        rupture_grace_ticks=rupture_grace_ticks,
        time_window_size=time_window_size,
    )


def build_g2_exports(log_dir: Path | str = DEFAULT_LOG_DIR, *, output_dir: Path | str = DEFAULT_GRAPH_DIR, add_g0_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, time_window_size: Optional[int] = None) -> GraphLevelBuildResult:
    return build_graph_level_exports(
        "g2",
        log_dir,
        output_dir=output_dir,
        add_g0_continuity=add_g0_continuity,
        include_field_continuity=include_field_continuity,
        include_snapshot_nodes=include_snapshot_nodes,
        rupture_grace_ticks=rupture_grace_ticks,
        time_window_size=time_window_size,
    )


def build_graph_level_exports(target_level: str, log_dir: Path | str = DEFAULT_LOG_DIR, *, output_dir: Path | str = DEFAULT_GRAPH_DIR, add_g0_continuity: bool = True, include_field_continuity: bool = True, include_snapshot_nodes: bool = True, rupture_grace_ticks: int = 2, time_window_size: Optional[int] = None) -> GraphLevelBuildResult:
    target_level = normalize_target_level(target_level)
    output_dir = Path(output_dir)
    g0 = build_g0(log_dir, add_continuity=add_g0_continuity, include_field_continuity=include_field_continuity, include_snapshot_nodes=include_snapshot_nodes, rupture_grace_ticks=rupture_grace_ticks)
    graphs_by_level, _ = build_contracted_graphs(g0, through_level=target_level, time_window_size=time_window_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = write_level_exports(target_level, graphs_by_level[target_level], output_dir=output_dir)
    return GraphLevelBuildResult(target_level=target_level.upper(), graphs_by_level=graphs_by_level, output_paths=output_paths)


def build_contracted_graphs(g0, *, through_level: str, time_window_size: Optional[int]) -> tuple[dict[str, Any], tuple[TimeContractionScheme, AgentClusteringScheme, TopologicalSCCContractionScheme]]:
    through_level = normalize_target_level(through_level, allowed={"g1", "g2", "g3"})
    time_scheme = TimeContractionScheme(window_size=time_window_size)
    agent_scheme = AgentClusteringScheme()
    topology_scheme = TopologicalSCCContractionScheme()
    graphs_by_level: dict[str, Any] = {"g0": g0}
    graphs_by_level["g1"] = time_scheme.contract(g0)
    if through_level in {"g2", "g3"}:
        graphs_by_level["g2"] = agent_scheme.contract(graphs_by_level["g1"])
    if through_level == "g3":
        graphs_by_level["g3"] = topology_scheme.contract(graphs_by_level["g2"])
    return graphs_by_level, (time_scheme, agent_scheme, topology_scheme)


def build_external_multilevel_graph(g0, time_scheme: TimeContractionScheme, agent_scheme: AgentClusteringScheme, topology_scheme: TopologicalSCCContractionScheme):
    ml_graph = MultilevelGraph(multilevelgraphs_safe_copy(g0), [time_scheme.clone(), agent_scheme.clone(), topology_scheme.clone()])
    ml_graph.build_contraction_schemes()
    return ml_graph


def build_g3_from_g2_gexf(g2_path: Path | str = DEFAULT_GRAPH_DIR / "g2.gexf", *, output_dir: Path | str = DEFAULT_GRAPH_DIR) -> dict[str, Path]:
    g2_path = Path(g2_path)
    output_dir = Path(output_dir)
    g2 = nx.read_gexf(g2_path)
    g3 = TopologicalSCCContractionScheme().contract_networkx(g2)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_level_exports("g3", g3, output_dir=output_dir)


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


def normalize_target_level(level: str, *, allowed: set[str] | None = None) -> str:
    allowed_levels = allowed or {"g1", "g2"}
    normalized = str(level).lower()
    if normalized not in allowed_levels:
        expected = ", ".join(sorted(allowed_levels))
        raise ValueError(f"Unsupported target graph level: {level!r}. Expected one of: {expected}")
    return normalized


def write_level_exports(level: str, graph, *, output_dir: Path) -> dict[str, Path]:
    return {f"{level}_gephi": write_networkx_gexf(graph, output_dir / f"{level}.gexf")}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build G0/G1/G2/G3 multilevel graph exports.")
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--skip-g0", action="store_true", help="Do not export G0 files. G1/G2/G3 are still built from G0.")
    parser.add_argument("--no-g0-continuity", action="store_true")
    parser.add_argument("--no-field-continuity", action="store_true")
    parser.add_argument("--no-snapshot-nodes", action="store_true")
    parser.add_argument("--rupture-grace-ticks", type=int, default=2)
    parser.add_argument("--time-window-size", type=int, default=None)
    parser.add_argument("--external-multilevelgraph", action="store_true", help="Also instantiate the external MultilevelGraph object. This can be slow on large G0 traces.")
    args = parser.parse_args()

    result = build_multilevel_graphs(
        args.log_dir,
        output_dir=args.output_dir,
        add_g0_continuity=not args.no_g0_continuity,
        include_field_continuity=not args.no_field_continuity,
        include_snapshot_nodes=not args.no_snapshot_nodes,
        rupture_grace_ticks=args.rupture_grace_ticks,
        time_window_size=args.time_window_size,
        build_external_graph=args.external_multilevelgraph,
        write_g0=not args.skip_g0,
    )
    for path in result.output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
