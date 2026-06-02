from types import SimpleNamespace

from testhelpers import TestRng, import_any


alpha_module = import_any("src.simulation.agents.alphasynuclein")
AlphaSynuclein = alpha_module.AlphaSynuclein
AlphaSynucleinCompartment = alpha_module.AlphaSynucleinCompartment
AlphaSynucleinConfig = alpha_module.AlphaSynucleinConfig
AlphaSynucleinPerception = alpha_module.AlphaSynucleinPerception
AlphaSynucleinState = alpha_module.AlphaSynucleinState


def test_misfolded_alpha_waits_before_oligomerization_intention():
    config = AlphaSynucleinConfig(
        perception_radius=1,
        move_radius=1,
        move_probability=0.0,
        oxidative_stress_high_threshold=0.5,
        oligomerization_probability_scale=1.0,
        min_misfolded_ticks_before_oligomerization=2
    )
    alpha = AlphaSynuclein(
        local_id=1,
        rank=0,
        type_id=9,
        config=config,
        compartment=AlphaSynucleinCompartment.INTRACELLULAR,
        owner_neuron=SimpleNamespace()
    )
    alpha.state = AlphaSynucleinState.MISFOLDED
    alpha.rng = TestRng(random_value=0.0)
    alpha.last_perception = AlphaSynucleinPerception(
        position=None,
        oxidative_stress=0.0,
        local_aggregate_density=0.0,
        neighbors=[SimpleNamespace(ptype=9)]
    )
    alpha.next()
    assert alpha.wants_oligomerization is False
    alpha.next()
    assert alpha.wants_oligomerization is False
    alpha.next()
    assert alpha.wants_oligomerization is True
