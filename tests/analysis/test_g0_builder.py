from src.analysis.graph.g0_builder import agent_node_id, field_node_id


class TestG0BuilderIdentity:
    def test_agent_node_id_preserves_full_runtime_uid(self):
        assert agent_node_id("Neuron", "0:0:0", "Healthy", 1) == "Neuron_0_0_0_Healthy@1"
        assert agent_node_id("Neuron", "0:0:1", "Healthy", 1) == "Neuron_0_0_1_Healthy@1"

    def test_internal_field_node_id_preserves_owner_runtime_uid(self):
        assert field_node_id("oxidative_stress", None, "Neuron", "0:0:0", 1) == "Neuron_0_0_0_oxidative_stress@1"
        assert field_node_id("oxidative_stress", None, "Neuron", "0:0:1", 1) == "Neuron_0_0_1_oxidative_stress@1"
