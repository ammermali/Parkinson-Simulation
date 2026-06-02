NEURON_METRIC_STATES = ("Healthy", "Compromised", "Apoptotic", "Ruptured", "Unknown")
ASTROCYTE_METRIC_STATES = ("Supportive", "Reactive", "Unknown")
MICROGLIA_METRIC_STATES = ("Resting", "Clearing", "Activated", "Unknown")
ALPHA_METRIC_STATES = ("Monomer", "Misfolded", "Oligomer", "LewyBody", "Cleared", "Unknown")
AGGREGATE_METRIC_STATES = ("Oligomer", "LewyBody", "Unknown")

TICK_METRIC_COUNT_KEYS = ("neurons_healthy", "neurons_compromised", "neurons_apoptotic", "neurons_ruptures", "free_alpha", "alpha_aggregate")
TICK_METRIC_COLUMNS = ("tick", "debris", "inflammation", "dopamine", *TICK_METRIC_COUNT_KEYS)

FINAL_MAX_METRIC_KEYS = ("aggregates.max_size", "aggregates.intracellular.max_size", "aggregates.extracellular.max_size")
FINAL_SUM_METRIC_KEYS = (
    *(f"neurons.{state}" for state in NEURON_METRIC_STATES),
    "neurons.transitions.healthy_to_compromised.count",
    "neurons.transitions.compromised_to_apoptotic.count",
    "neurons.transitions.apoptotic_to_ruptured.count",
    "neurons.state_time.compromised_ticks_total",
    "neurons.state_time.apoptotic_ticks_total",
    "neurons.state_time.compromised_neuron_count", "neurons.state_time.apoptotic_neuron_count",
    "neurons.recoveries.compromised_to_healthy", "neurons.blocks.min_ticks_compromised",
    "neurons.blocks.apoptotic_internal_damage_threshold",
    "neurons.ever_compromised", "neurons.ever_apoptotic", "neurons.ever_recovered",
    *(f"astrocytes.{state}" for state in ASTROCYTE_METRIC_STATES),
    *(f"microglia.{state}" for state in MICROGLIA_METRIC_STATES),
    *(f"alpha.free.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.intracellular.free.total",
    *(f"alpha.intracellular.free.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.extracellular.free.total",
    *(f"alpha.extracellular.free.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.members", "alpha.intracellular.members", "alpha.extracellular.members",
    *(f"alpha.members.{state}" for state in ALPHA_METRIC_STATES),
    "alpha.orphan_lewy", "aggregates.total",
    *(f"aggregates.{state}" for state in AGGREGATE_METRIC_STATES),
    "aggregates.size_total", "aggregates.intracellular.total",
    *(f"aggregates.intracellular.{state}" for state in AGGREGATE_METRIC_STATES),
    "aggregates.intracellular.size_total", "aggregates.extracellular.total",
    *(f"aggregates.extracellular.{state}" for state in AGGREGATE_METRIC_STATES),
    "aggregates.extracellular.size_total", "aggregates.invariant_failures",
    "aggregates.intracellular.invariant_failures", "aggregates.extracellular.invariant_failures"
)
