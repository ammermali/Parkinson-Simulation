from __future__ import annotations
import html
from typing import Any
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

NODE_STYLE_MAP = {
    "agent_state": {"shape": "dot","color": "#4C78A8"},
    "environment_field": {"shape": "box","color": "#F58518"},
    "aggregate": {"shape": "diamond","color": "#B279A2"},
    "agent_state_cluster": {"shape": "dot","color": "#4C78A8"},
    "singleton": {"shape": "ellipse","color": "#9D9D9D"},
    "feedback_component": {"shape": "diamond","color": "#E45756"},
    "self_feedback": {"shape": "diamond","color": "#F58518"},
    "singleton_process": {"shape": "ellipse","color": "#72B7B2"}}
DEFAULT_NODE_STYLE = {"shape": "dot","color": "#9D9D9D"}
EDGE_COLOR_MAP = {
    "continuity": "#A0A0A0",
    "state_transition": "#54A24B",
    "threshold_trigger": "#E45756",
    "field_effect": "#F58518",
    "internal_field_effect": "#FFBF79",
    "aggregation": "#B279A2",
    "degradation": "#72B7B2"}
DEFAULT_EDGE_COLOR = "#777777"

def render_g0_graph(graph: nx.DiGraph,*,central_entity_key: str | None = None,height: int = 760) -> None:
    network = Network(height=f"{height}px",width="100%",directed=True,notebook=False,cdn_resources="in_line")
    for node_id, attributes in graph.nodes(data=True):
        semantic_kind = str(attributes.get("semantic_kind") or attributes.get("kind") or attributes.get("cluster_kind") or attributes.get("pattern_kind") or "unknown")
        style = NODE_STYLE_MAP.get(semantic_kind, DEFAULT_NODE_STYLE)
        is_central = central_entity_key is not None and str(attributes.get("entity_key", "")) == central_entity_key
        network.add_node(
            node_id,label=_node_label(node_id, attributes),title=_node_tooltip(node_id, attributes),
            shape=style["shape"],color=style["color"],size=22 if is_central else 14,
            borderWidth=4 if is_central else 1,borderWidthSelected=5,
            font={"size": 14 if is_central else 11,"face": "Arial"})
    for source, target, attributes in graph.edges(data=True):
        relation = str(attributes.get("relation") or attributes.get("causal_kind") or "unknown")
        edge_width = _edge_width(attributes)
        network.add_edge(source, target, label=relation, title=_edge_tooltip(source, target, attributes), color=EDGE_COLOR_MAP.get(relation, DEFAULT_EDGE_COLOR), width=edge_width, arrows="to")
    network.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -55,
              "centralGravity": 0.01,
              "springLength": 135,
              "springConstant": 0.045,
              "damping": 0.4,
              "avoidOverlap": 0.35
            },
            "stabilization": {
              "enabled": true,
              "iterations": 350,
              "updateInterval": 25,
              "fit": true
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true,
            "hideEdgesOnDrag": true
          },
          "nodes": {
            "shadow": false
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "font": {
              "size": 9,
              "align": "middle",
              "strokeWidth": 3,
              "strokeColor": "#ffffff"
            }
          }
        }
        """)
    generated_html = network.generate_html()
    components.html(generated_html, height=height, scrolling=True)

def render_graph(graph: nx.DiGraph, *, central_entity_key: str | None = None, height: int = 760) -> None:
    render_g0_graph(graph, central_entity_key=central_entity_key, height=height)


def _node_label(node_id: str, attributes: dict[str, Any]) -> str:
    manual_label = attributes.get("display_label") or attributes.get("biological_label")
    if manual_label:
        return _short_label(str(manual_label))
    tick = attributes.get("tick")
    field = attributes.get("field")
    state = attributes.get("state")
    uid = attributes.get("uid")
    agent_type = attributes.get("agent_type") or "Agent"
    label = attributes.get("label")
    if not any((field, state, attributes.get("agent_type"), label)):
        return str(node_id)
    if field:
        return f"{field}@{tick}" if tick not in (None, "") else str(field)
    if state:
        suffix = f"@{tick}" if tick not in (None, "") else _time_span(attributes)
        return f"{agent_type}\n{state}{suffix}"
    if label:
        return _short_label(str(label))
    suffix = f"@{tick}" if tick not in (None, "") else ""
    return f"{agent_type}{suffix}_{uid}"

def _node_tooltip(node_id: str, attributes: dict[str, Any]) -> str:
    return _tooltip({
            "Node": node_id,
            "Biological label": attributes.get("biological_label"),
            "Semantic note": attributes.get("semantic_note"),
            "Entity key": attributes.get("entity_key"),
            "Kind": attributes.get("semantic_kind"),
            "Cluster kind": attributes.get("cluster_kind"),
            "Pattern kind": attributes.get("pattern_kind"),
            "Agent type": attributes.get("agent_type"),
            "UID": attributes.get("uid"),
            "State": attributes.get("state"),
            "Field": attributes.get("field"),
            "Value": attributes.get("value"),
            "Tick": attributes.get("tick"),
            "First seen": attributes.get("first_seen"),
            "Last seen": attributes.get("last_seen"),
            "Members": attributes.get("member_count"),
            "Component size": attributes.get("component_size"),
            "Rank": attributes.get("rank"),
            "Owner UID": attributes.get("owner_uid"),
            "Compartment": attributes.get("compartment")})

def _edge_tooltip(source: str, target: str, attributes: dict[str, Any]) -> str:
    return _tooltip({
            "Source": source,
            "Target": target,
            "Relation": attributes.get("relation"),
            "Causal kind": attributes.get("causal_kind"),
            "Mechanism": attributes.get("mechanism"),
            "Tick": attributes.get("tick"),
            "First tick": attributes.get("first_tick"),
            "Last tick": attributes.get("last_tick"),
            "Count": attributes.get("count"),
            "Total effect": attributes.get("total_effect"),
            "Mean effect": attributes.get("mean_effect"),
            "Probability": attributes.get("probability"),
            "RNG value": attributes.get("rng_value"),
            "Outcome": attributes.get("outcome")})

def _edge_width(attributes: dict[str, Any]) -> float:
    raw_count = attributes.get("count", 1)
    try:
        count = max(1.0, float(raw_count))
    except (TypeError,ValueError):
        count = 1.0
    return min(8.0, 1.0 + count ** 0.5)

def _time_span(attributes: dict[str, Any]) -> str:
    first_seen = attributes.get("first_seen")
    last_seen = attributes.get("last_seen")
    if first_seen in (None, "") and last_seen in (None, ""):
        return ""
    if first_seen == last_seen or last_seen in (None, ""):
        return f"\nt={first_seen}"
    return f"\nt={first_seen}-{last_seen}"

def _short_label(label: str, *, max_length: int = 36) -> str:
    return label if len(label) <= max_length else label[: max_length - 1] + "..."

def _tooltip(values: dict[str, Any]) -> str:
    rows: list[str] = []
    for key, value in values.items():
        if value is None or value == "":
            continue
        rows.append(f"<b>{html.escape(str(key))}:</b> {html.escape(str(value))}")
    return "<br>".join(rows)
