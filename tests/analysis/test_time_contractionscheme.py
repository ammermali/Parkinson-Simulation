import pytest

nx = pytest.importorskip("networkx")

from src.analysis.schemes.agent_clustering_scheme import AgentClusteringScheme
from src.analysis.schemes.time_contractionscheme import TimeContractionScheme
from src.analysis.schemes.topological_scc_scheme import TopologicalSCCContractionScheme


def agent_node(uid, state, tick):
    return {
        "semantic_kind": "agent_state",
        "agent_type": "AlphaSynuclein",
        "uid": uid,
        "state": state,
        "tick": tick,
        "entity_key": f"AlphaSynuclein:{uid}",
    }


def neuron_node(uid, state, tick):
    return {
        "semantic_kind": "agent_state",
        "agent_type": "Neuron",
        "uid": uid,
        "state": state,
        "tick": tick,
        "entity_key": f"Neuron:{uid}",
    }


def field_node(field, tick):
    return {
        "semantic_kind": "environment_field",
        "agent_type": "SubstantiaNigra",
        "uid": "SN",
        "field": field,
        "tick": tick,
        "entity_key": f"SN:{field}",
    }


class TestTimeContraction:
    def test_contracts_same_agent_state_across_ticks(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("a1@1", **agent_node("1:3:0", "Misfolded", 1))
        g0.add_node("a1@2", **agent_node("1:3:0", "Misfolded", 2))
        g0.add_node("a2@2", **agent_node("2:3:0", "Misfolded", 2))

        g1 = TimeContractionScheme().contract(g0)

        assert set(g1.nodes) == {"AlphaSynuclein_1_3_0_Misfolded", "AlphaSynuclein_2_3_0_Misfolded"}
        assert g1.nodes["AlphaSynuclein_1_3_0_Misfolded"]["observation_count"] == 2

    def test_preserves_agent_identity_and_state(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("a1_monomer@1", **agent_node("1:3:0", "Monomer", 1))
        g0.add_node("a1_misfolded@2", **agent_node("1:3:0", "Misfolded", 2))
        g0.add_node("a2_misfolded@2", **agent_node("2:3:0", "Misfolded", 2))

        g1 = TimeContractionScheme().contract(g0)

        assert set(g1.nodes) == {
            "AlphaSynuclein_1_3_0_Monomer",
            "AlphaSynuclein_1_3_0_Misfolded",
            "AlphaSynuclein_2_3_0_Misfolded",
        }

    def test_preserves_rank_scoped_neuron_identity(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("n0_rank0@1", **neuron_node("0:0:0", "Healthy", 1))
        g0.add_node("n0_rank0@2", **neuron_node("0:0:0", "Healthy", 2))
        g0.add_node("n0_rank1@1", **neuron_node("0:0:1", "Healthy", 1))

        g1 = TimeContractionScheme().contract(g0)

        assert set(g1.nodes) == {"Neuron_0_0_0_Healthy", "Neuron_0_0_1_Healthy"}
        assert g1.nodes["Neuron_0_0_0_Healthy"]["observation_count"] == 2
        assert g1.nodes["Neuron_0_0_1_Healthy"]["observation_count"] == 1

    def test_contracts_environment_field_across_ticks(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("sn_debris@1", **field_node("extracellular_debris", 1))
        g0.add_node("sn_debris@2", **field_node("extracellular_debris", 2))

        g1 = TimeContractionScheme().contract(g0)

        assert set(g1.nodes) == {"SN_extracellular_debris"}
        assert g1.nodes["SN_extracellular_debris"]["observation_count"] == 2
        assert g1.nodes["SN_extracellular_debris"]["first_seen"] == 1
        assert g1.nodes["SN_extracellular_debris"]["last_seen"] == 2

    def test_absorbs_continuity_inside_time_supernode(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("a1@1", **agent_node("1:3:0", "Misfolded", 1))
        g0.add_node("a1@2", **agent_node("1:3:0", "Misfolded", 2))
        g0.add_edge("a1@1", "a1@2", relation="continuity", causal_kind="continuity", tick=1)

        g1 = TimeContractionScheme().contract(g0)

        assert g1.number_of_edges() == 0
        assert g1.nodes["AlphaSynuclein_1_3_0_Misfolded"]["absorbed_edge_count"] == 1

    def test_compacts_edges_between_time_supernodes(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("sn@1", **field_node("oxidative_stress", 1))
        g0.add_node("a1@1", **agent_node("1:3:0", "Misfolded", 1))
        g0.add_node("a1@2", **agent_node("1:3:0", "Misfolded", 2))
        g0.add_edge("sn@1", "a1@1", relation="threshold_trigger", causal_kind="perception", tick=1, effect_value=0.2)
        g0.add_edge("sn@1", "a1@2", relation="threshold_trigger", causal_kind="perception", tick=2, effect_value=0.4)

        g1 = TimeContractionScheme().contract(g0)

        assert g1.has_edge("SN_oxidative_stress", "AlphaSynuclein_1_3_0_Misfolded")
        edge = g1.edges["SN_oxidative_stress", "AlphaSynuclein_1_3_0_Misfolded"]
        assert edge["count"] == 2
        assert edge["total_effect"] == pytest.approx(0.6)
        assert edge["mean_effect"] == pytest.approx(0.3)
        assert edge["first_seen"] == 1
        assert edge["last_seen"] == 2

    def test_supports_optional_time_windows(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("a1@1", **agent_node("1:3:0", "Monomer", 1))
        g0.add_node("a1@2", **agent_node("1:3:0", "Monomer", 2))
        g0.add_node("a1@3", **agent_node("1:3:0", "Monomer", 3))

        g1 = TimeContractionScheme(window_size=2).contract(g0)

        assert set(g1.nodes) == {"AlphaSynuclein_1_3_0_Monomer_t0_1", "AlphaSynuclein_1_3_0_Monomer_t2_3"}
        assert g1.nodes["AlphaSynuclein_1_3_0_Monomer_t2_3"]["observation_count"] == 2

    def test_raises_on_missing_tick_when_windowing(self):
        g0 = nx.DiGraph(level="G0")
        g0.add_node("a1@missing", semantic_kind="agent_state", agent_type="AlphaSynuclein", uid="1:3:0", state="Monomer")

        with pytest.raises(ValueError, match="tick"):
            TimeContractionScheme(window_size=2).contract(g0)

    def test_raises_on_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            TimeContractionScheme(window_size=0)


class TestAgentClusteringScheme:
    def test_clusters_agents_by_class_and_state(self):
        g1 = nx.DiGraph(level="G1")
        g1.add_node("AlphaSynuclein_1_3_0_Misfolded", semantic_kind="agent_state", agent_type="AlphaSynuclein", uid="1:3:0", state="Misfolded", observation_count=2)
        g1.add_node("AlphaSynuclein_2_3_0_Misfolded", semantic_kind="agent_state", agent_type="AlphaSynuclein", uid="2:3:0", state="Misfolded", observation_count=1)
        g1.add_node("SN_dopamine_output", semantic_kind="environment_field", agent_type="SubstantiaNigra", uid="SN", field="dopamine_output", observation_count=2)
        g1.add_edge("SN_dopamine_output", "AlphaSynuclein_1_3_0_Misfolded", relation="threshold_trigger", count=2, total_effect=0.4, mean_effect=0.2)
        g1.add_edge("SN_dopamine_output", "AlphaSynuclein_2_3_0_Misfolded", relation="threshold_trigger", count=1, total_effect=0.3, mean_effect=0.3)

        g2 = AgentClusteringScheme().contract(g1)

        assert set(g2.nodes) == {"AlphaSynuclein_Misfolded", "SN_dopamine_output"}
        assert g2.nodes["AlphaSynuclein_Misfolded"]["member_count"] == 2
        assert g2.has_edge("SN_dopamine_output", "AlphaSynuclein_Misfolded")
        edge = g2.edges["SN_dopamine_output", "AlphaSynuclein_Misfolded"]
        assert edge["count"] == 3
        assert edge["mean_effect"] == pytest.approx(0.7 / 3)

    def test_clusters_neuron_local_fields_into_one_internal_environment_node(self):
        g1 = nx.DiGraph(level="G1")
        g1.add_node("Neuron_0_0_0_cell_damage", semantic_kind="environment_field", agent_type="Neuron", field="cell_damage", uid="0:0:0", observation_count=2)
        g1.add_node("Neuron_56_0_1_energy_demand", semantic_kind="environment_field", agent_type="Neuron", field="energy_demand", uid="56:0:1", observation_count=1)

        g2 = AgentClusteringScheme().contract(g1)

        assert set(g2.nodes) == {"Neuron_internal_environment"}
        assert g2.nodes["Neuron_internal_environment"]["member_count"] == 2


class TestTopologicalSCCContractionScheme:
    def test_contracts_strongly_connected_feedback_patterns(self):
        g2 = nx.DiGraph(level="G2")
        g2.add_node("SN_inflammation_level", semantic_kind="environment_field", agent_type="SubstantiaNigra", uid="SN", field="inflammation_level")
        g2.add_node("Microglia_Activated", semantic_kind="agent_state", agent_type="Microglia", state="Activated")
        g2.add_node("Neuron_Compromised", semantic_kind="agent_state", agent_type="Neuron", state="Compromised")
        g2.add_edge("SN_inflammation_level", "Microglia_Activated", relation="threshold_trigger", causal_kind="perception", count=4, total_effect=0.8, mean_effect=0.2)
        g2.add_edge("Microglia_Activated", "SN_inflammation_level", relation="field_effect", causal_kind="action", count=3, total_effect=0.6, mean_effect=0.2)
        g2.add_edge("SN_inflammation_level", "Neuron_Compromised", relation="threshold_trigger", causal_kind="perception", count=2, total_effect=0.3, mean_effect=0.15)

        g3 = TopologicalSCCContractionScheme().contract(g2)
        feedback_nodes = [
            attrs
            for _, attrs in g3.nodes(data=True)
            if attrs.get("pattern_kind") == "feedback_component"
        ]

        assert len(feedback_nodes) == 1
        assert feedback_nodes[0]["component_size"] == 2
        assert feedback_nodes[0]["internal_event_count"] == 7
        assert feedback_nodes[0]["is_feedback_pattern"] is True
        assert any(attrs.get("pattern_kind") == "singleton_process" for _, attrs in g3.nodes(data=True))
