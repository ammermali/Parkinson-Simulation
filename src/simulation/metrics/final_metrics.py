from __future__ import annotations
from typing import Any, Callable
from src.simulation.agents.aggregate import AlphaAggregate
from src.simulation.agents.alphasynuclein import AlphaSynuclein, AlphaSynucleinCompartment
from src.simulation.agents.astrocyte import Astrocyte
from src.simulation.agents.microglia import Microglia
from src.simulation.agents.neuron import Neuron
from src.simulation.agents.structure.states import AggregateState, AlphaSynucleinState
from src.simulation.metrics.key import (
    AGGREGATE_METRIC_STATES,
    ALPHA_METRIC_STATES,
    ASTROCYTE_METRIC_STATES,
    FINAL_MAX_METRIC_KEYS,
    FINAL_SUM_METRIC_KEYS,
    MICROGLIA_METRIC_STATES,
    NEURON_METRIC_STATES,
)


class FinalMetricsCollector:
    """Collect and reduce final simulation metrics."""
    def __init__(self, *, global_sum: Callable[[float], float], global_max: Callable[[float], float]) -> None:
        self.global_sum = global_sum
        self.global_max = global_max

    def collect(self, *, context: Any, environment: Any) -> dict[str, float]:
        return self.reduce(collect_local_final_metrics(context, environment))

    def reduce(self, local: dict[str, float]) -> dict[str, float]:
        return reduce_final_metrics(local, global_sum=self.global_sum, global_max=self.global_max)

def reduce_final_metrics(local: dict[str, float], *, global_sum: Callable[[float], float], global_max: Callable[[float], float]) -> dict[str, float]:
    global_metrics = {
        key: global_sum(local.get(key, 0.0))
        for key in FINAL_SUM_METRIC_KEYS
    }
    for key in FINAL_MAX_METRIC_KEYS:
        global_metrics[key] = global_max(local.get(key, 0.0))
    aggregate_total = global_metrics["aggregates.total"]
    if aggregate_total > 0:
        global_metrics["aggregates.avg_size"] = global_metrics["aggregates.size_total"] / aggregate_total
    else:
        global_metrics["aggregates.avg_size"] = 0.0
    for compartment in ("intracellular", "extracellular"):
        compartment_total = global_metrics[f"aggregates.{compartment}.total"]
        if compartment_total > 0:
            global_metrics[f"aggregates.{compartment}.avg_size"] = global_metrics[f"aggregates.{compartment}.size_total"] / compartment_total
        else:
            global_metrics[f"aggregates.{compartment}.avg_size"] = 0.0
    compromised_count = global_metrics["neurons.state_time.compromised_neuron_count"]
    apoptotic_count = global_metrics["neurons.state_time.apoptotic_neuron_count"]
    global_metrics["neurons.state_time.compromised_avg_ticks"] = (
        global_metrics["neurons.state_time.compromised_ticks_total"] / compromised_count
        if compromised_count > 0
        else 0.0
    )
    global_metrics["neurons.state_time.apoptotic_avg_ticks"] = (
        global_metrics["neurons.state_time.apoptotic_ticks_total"] / apoptotic_count
        if apoptotic_count > 0
        else 0.0
    )
    return global_metrics


def collect_local_final_metrics(context: Any, environment: Any) -> dict[str, float]:
    metrics = empty_final_metrics()
    aggregate_registry = getattr(environment, "aggregate_registry", None)
    for agent in context.agents():
        if isinstance(agent, Neuron):
            state = metric_state_value(getattr(agent, "state", None), NEURON_METRIC_STATES)
            metrics[f"neurons.{state}"] += 1
            collect_neuron_transition_metrics(agent, metrics)
            collect_neuron_internal_metrics(agent, metrics, aggregate_registry=aggregate_registry)
        elif isinstance(agent, Astrocyte):
            state = metric_state_value(getattr(agent, "state", None), ASTROCYTE_METRIC_STATES)
            metrics[f"astrocytes.{state}"] += 1
        elif isinstance(agent, Microglia):
            state = metric_state_value(getattr(agent, "state", None), MICROGLIA_METRIC_STATES)
            metrics[f"microglia.{state}"] += 1
    collect_extracellular_alpha_metrics(environment, metrics)
    return metrics


def empty_final_metrics() -> dict[str, float]:
    metrics = {key: 0 for key in FINAL_SUM_METRIC_KEYS}
    metrics.update({key: 0 for key in FINAL_MAX_METRIC_KEYS})
    return metrics


def collect_neuron_transition_metrics(neuron: Neuron, metrics: dict[str, float]) -> None:
    if getattr(neuron, "first_compromised_tick", None) is not None:
        metrics["neurons.transitions.healthy_to_compromised.count"] += 1
        metrics["neurons.ever_compromised"] += 1
    if getattr(neuron, "first_apoptotic_tick", None) is not None:
        metrics["neurons.transitions.compromised_to_apoptotic.count"] += 1
        metrics["neurons.ever_apoptotic"] += 1
    if getattr(neuron, "first_ruptured_tick", None) is not None:
        metrics["neurons.transitions.apoptotic_to_ruptured.count"] += 1
    compromised_ticks = getattr(neuron, "compromised_ticks_total", 0)
    apoptotic_ticks = getattr(neuron, "apoptotic_ticks_total", 0)
    metrics["neurons.state_time.compromised_ticks_total"] += compromised_ticks
    metrics["neurons.state_time.apoptotic_ticks_total"] += apoptotic_ticks
    if compromised_ticks > 0:
        metrics["neurons.state_time.compromised_neuron_count"] += 1
    if apoptotic_ticks > 0:
        metrics["neurons.state_time.apoptotic_neuron_count"] += 1
    metrics["neurons.recoveries.compromised_to_healthy"] += getattr(neuron, "compromised_recoveries", 0)
    metrics["neurons.blocks.min_ticks_compromised"] += getattr(neuron, "blocked_by_min_ticks_compromised", 0)
    metrics["neurons.blocks.apoptotic_internal_damage_threshold"] += getattr(neuron, "blocked_by_apoptotic_internal_damage_threshold", 0)
    if getattr(neuron, "compromised_recoveries", 0) > 0:
        metrics["neurons.ever_recovered"] += 1


def collect_neuron_internal_metrics(neuron: Neuron, metrics: dict[str, float], aggregate_registry=None) -> None:
    """Collect intracellular alpha and aggregate metrics for one neuron.

    Runtime neurons read aggregate identity from the environment-level registry.
    Tests may build isolated neurons without binding that environment, so this
    collector accepts the registry from the caller and degrades gracefully when
    only a partial test double is available.
    """

    registry = aggregate_registry or aggregate_registry_from_neuron(neuron)
    validate_invariants = getattr(registry, "validate_invariants", None)
    if callable(validate_invariants):
        try:
            validate_invariants(neuron)
        except RuntimeError:
            metrics["aggregates.invariant_failures"] += 1
            metrics["aggregates.intracellular.invariant_failures"] += 1

    aggregate_agents = intracellular_aggregates(neuron, registry)
    members_for = getattr(registry, "members", None)
    for aggregate in aggregate_agents:
        members = members_for(aggregate.aggregate_id) if callable(members_for) else getattr(aggregate, "member_agents", None)
        collect_aggregate_metrics(
            aggregate,
            metrics,
            compartment="intracellular",
            members=members
        )
    for internal_agent in getattr(getattr(neuron, "grid", None), "agent_registry", []):
        if not isinstance(internal_agent, AlphaSynuclein):
            continue
        state = metric_state_value(internal_agent.state, ALPHA_METRIC_STATES)
        if state == AlphaSynucleinState.LEWY_BODY.value and internal_agent.aggregate_id is None:
            metrics["alpha.orphan_lewy"] += 1
        if internal_agent.aggregate_id is None:
            metrics[f"alpha.free.{state}"] += 1
            metrics["alpha.intracellular.free.total"] += 1
            metrics[f"alpha.intracellular.free.{state}"] += 1


def aggregate_registry_from_neuron(neuron: Neuron):
    """Return a neuron's environment registry without triggering strict errors."""

    return getattr(getattr(neuron, "environment", None), "aggregate_registry", None)


def intracellular_aggregates(neuron: Neuron, registry=None) -> list[AlphaAggregate]:
    """Return intracellular aggregate agents from registry or local grid."""

    aggregates = getattr(registry, "aggregates", None)
    if callable(aggregates):
        return list(aggregates(neuron))
    return [
        agent
        for agent in getattr(getattr(neuron, "grid", None), "agent_registry", [])
        if isinstance(agent, AlphaAggregate)
    ]


def collect_extracellular_alpha_metrics(environment: Any, metrics: dict[str, float]) -> None:
    grid = getattr(environment, "grid", None)
    registry = getattr(environment, "aggregate_registry", None)
    agents = getattr(grid, "agent_registry", [])
    for agent in agents:
        if isinstance(agent, AlphaSynuclein):
            state = metric_state_value(agent.state, ALPHA_METRIC_STATES)
            if agent.aggregate_id is None:
                metrics[f"alpha.free.{state}"] += 1
                metrics["alpha.extracellular.free.total"] += 1
                metrics[f"alpha.extracellular.free.{state}"] += 1
        elif isinstance(agent, AlphaAggregate):
            if not extracellular_aggregate_invariants_hold(agent, agents, registry):
                metrics["aggregates.invariant_failures"] += 1
                metrics["aggregates.extracellular.invariant_failures"] += 1
            collect_aggregate_metrics(
                agent,
                metrics,
                compartment="extracellular"
            )


def collect_aggregate_metrics(aggregate: AlphaAggregate, metrics: dict[str, float], compartment: str, members=None) -> None:
    state = metric_state_value(aggregate.state, AGGREGATE_METRIC_STATES)
    size = aggregate.size
    metrics["aggregates.total"] += 1
    metrics[f"aggregates.{state}"] += 1
    metrics["aggregates.size_total"] += size
    metrics["aggregates.max_size"] = max(metrics["aggregates.max_size"], size)
    metrics[f"aggregates.{compartment}.total"] += 1
    metrics[f"aggregates.{compartment}.{state}"] += 1
    metrics[f"aggregates.{compartment}.size_total"] += size
    metrics[f"aggregates.{compartment}.max_size"] = max(
        metrics[f"aggregates.{compartment}.max_size"],
        size
    )
    member_agents = set(members) if members is not None else set(aggregate.member_agents)
    if member_agents:
        for member in member_agents:
            member_state = state_value(getattr(member, "state", None))
            count_aggregate_member(metrics, compartment, member_state)
        return
    inferred_member_state = alpha_state_for_aggregate(aggregate.state)
    for _ in range(size):
        count_aggregate_member(metrics, compartment, inferred_member_state)


def count_aggregate_member(metrics: dict[str, float], compartment: str, member_state: str) -> None:
    member_state = metric_state_value(member_state, ALPHA_METRIC_STATES)
    metrics["alpha.members"] += 1
    metrics[f"alpha.{compartment}.members"] += 1
    metrics[f"alpha.members.{member_state}"] += 1


def extracellular_aggregate_invariants_hold(aggregate: AlphaAggregate, agents, registry=None) -> bool:
    if aggregate.owner_neuron is not None:
        return False
    if registry is not None and registry.aggregate_for(aggregate.aggregate_id) is not aggregate:
        return False
    if aggregate.size <= 0 or not aggregate.member_ids:
        return False
    member_agents = set(getattr(aggregate, "member_agents", set()))
    if not member_agents:
        return False
    if registry is not None:
        registered_members = registry.members(aggregate.aggregate_id)
        if registered_members and registered_members != member_agents:
            return False
    member_ids = {alpha_member_id(member) for member in member_agents}
    if member_ids != set(aggregate.member_ids):
        return False
    active_agents = set(agents)
    aggregate_state = metric_state_value(aggregate.state, AGGREGATE_METRIC_STATES)
    expected_state = AlphaSynucleinState.LEWY_BODY if aggregate_state == AggregateState.LEWY_BODY.value else AlphaSynucleinState.OLIGOMER
    for member in member_agents:
        if not isinstance(member, AlphaSynuclein):
            return False
        if member in active_agents:
            return False
        if member.aggregate_id != aggregate.aggregate_id:
            return False
        if member.state != expected_state:
            return False
        if member.compartment != AlphaSynucleinCompartment.EXTRACELLULAR:
            return False
        if member.owner_neuron is not None:
            return False
    return True


def state_value(state: Any) -> str:
    return getattr(state, "value", str(state))


def metric_state_value(state: Any, allowed_states: tuple[str, ...]) -> str:
    value = state_value(state)
    if value in allowed_states:
        return value
    return "Unknown"


def alpha_member_id(alpha: AlphaSynuclein):
    return getattr(alpha, "uid", id(alpha))


def alpha_state_for_aggregate(state: Any) -> str:
    if state_value(state) == AggregateState.LEWY_BODY.value:
        return AlphaSynucleinState.LEWY_BODY.value
    return AlphaSynucleinState.OLIGOMER.value
