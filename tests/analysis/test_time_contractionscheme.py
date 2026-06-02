import pytest
nx = pytest.importorskip("networkx")
from src.analysis.schemes.time_contractionscheme import TimeContraption


def agent_node(uid, state, tick):
    return {"semantic_kind": "agent_state", "agent_type": "AlphaSynuclein", "agent_uid": uid, "uid": uid, "state": state, "tick": tick, "entity_key": f"AlphaSynuclein:{uid}"}

def field_node(field, tick):
    return {"semantic_kind": "environment_field", "agent_type": "SubstantiaNigra", "uid": "SN", "field": field, "tick": tick, "entity_key": f"SN:{field}"}

class TestTimeContraption:
    def test_contracts_same_agent_state_across_time(self):
        g0 = nx.MultiDiGraph(level="G0")
        g0.add_node("a1@1.2", **agent_node("a:1", "Misfolded", 1))
        g0.add_node("a1@2.2", **agent_node("a:1", "Misfolded", 2))
        g0.add_edge("a1@1.2", "a1@2.2", edge_id="c1", relation="continuity", causal_kind="continuity", tick=2)
        g1 = TimeContraption().contract(g0)
        assert list(g1.nodes) == ["AlphaSynuclein_a:1_Misfolded"]
        node = g1.nodes["AlphaSynuclein_a:1_Misfolded"]
        assert node["observation_count"] == 2
        assert node["absorbed_edge_count"] == 1
        assert g1.number_of_edges() == 0

    def test_keeps_state_transitions_between_distinct_supernodes(self):
        g0 = nx.MultiDiGraph(level="G0")
        g0.add_node("a1@1.0", **agent_node("a:1", "Monomer", 1))
        g0.add_node("a1@1.2", **agent_node("a:1", "Misfolded", 1))
        g0.add_edge("a1@1.0", "a1@1.2", edge_id="t1", relation="state_transition", causal_kind="transition", tick=1)
        g1 = TimeContraption().contract(g0)
        assert g1.has_edge("AlphaSynuclein_a:1_Monomer", "AlphaSynuclein_a:1_Misfolded")
        edge = g1.edges["AlphaSynuclein_a:1_Monomer", "AlphaSynuclein_a:1_Misfolded"]
        assert edge["count"] == 1
        assert edge["sign"] == "state"

    def test_compacts_parallel_edges_with_effect_summary(self):
        g0 = nx.MultiDiGraph(level="G0")
        g0.add_node("sn@1.5", **field_node("oxidative_stress", 1))
        g0.add_node("sn@2.5", **field_node("oxidative_stress", 2))
        g0.add_node("a1@1.2", **agent_node("a:1", "Misfolded", 1))
        g0.add_edge("sn@1.5", "a1@1.2", edge_id="p1", relation="threshold_trigger", causal_kind="perception", tick=1, effect=0.2, sign="positive")
        g0.add_edge("sn@2.5", "a1@1.2", edge_id="p2", relation="threshold_trigger", causal_kind="perception", tick=2, effect=0.4, sign="positive")
        g1 = TimeContraption().contract(g0)
        assert g1.has_edge("SN_oxidative_stress", "AlphaSynuclein_a:1_Misfolded")
        edge = g1.edges["SN_oxidative_stress", "AlphaSynuclein_a:1_Misfolded"]
        assert edge["count"] == 2
        assert edge["total_effect"] == pytest.approx(0.6)
        assert edge["mean_effect"] == pytest.approx(0.3)
        assert edge["first_seen"] == 1
        assert edge["last_seen"] == 2
        assert edge["sign"] == "+"
