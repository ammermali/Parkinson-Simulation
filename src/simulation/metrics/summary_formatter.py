from __future__ import annotations
from collections import Counter
from typing import Any, Mapping, Sequence


def format_summary_lines(metrics: dict[str, float], *, transition_details: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    details = list(transition_details or [])
    return [
        (
            "[summary] neurons="
            f"Healthy:{metrics['neurons.Healthy']} "
            f"Compromised:{metrics['neurons.Compromised']} "
            f"Apoptotic:{metrics['neurons.Apoptotic']} "
            f"Ruptured:{metrics['neurons.Ruptured']} | "
            "astrocytes="
            f"Supportive:{metrics['astrocytes.Supportive']} "
            f"Reactive:{metrics['astrocytes.Reactive']} | "
            "microglia="
            f"Resting:{metrics['microglia.Resting']} "
            f"Clearing:{metrics['microglia.Clearing']} "
            f"Activated:{metrics['microglia.Activated']}"
        ),
        (
            "[summary] alpha="
            f"free_monomer:{metrics['alpha.free.Monomer']} "
            f"free_misfolded:{metrics['alpha.free.Misfolded']} "
            f"free_oligomer:{metrics['alpha.free.Oligomer']} "
            f"free_lewy:{metrics['alpha.free.LewyBody']} "
            f"extracellular_free:{metrics['alpha.extracellular.free.total']} "
            f"extracellular_members:{metrics['alpha.extracellular.members']} "
            f"intracellular_free:{metrics['alpha.intracellular.free.total']} "
            f"intracellular_members:{metrics['alpha.intracellular.members']} "
            f"members:{metrics['alpha.members']} "
            f"oligomer_members:{metrics['alpha.members.Oligomer']} "
            f"lewy_members:{metrics['alpha.members.LewyBody']} "
            f"cleared_members:{metrics['alpha.members.Cleared']} "
            f"orphan_lewy:{metrics['alpha.orphan_lewy']} | "
            "aggregates="
            f"total:{metrics['aggregates.total']} "
            f"oligomer:{metrics['aggregates.Oligomer']} "
            f"lewy:{metrics['aggregates.LewyBody']} "
            f"avg_size:{metrics['aggregates.avg_size']:.2f} "
            f"max_size:{metrics['aggregates.max_size']} "
            f"intracellular_total:{metrics['aggregates.intracellular.total']} "
            f"extracellular_total:{metrics['aggregates.extracellular.total']} "
            f"extracellular_lewy:{metrics['aggregates.extracellular.LewyBody']} "
            f"invariant_failures:{metrics['aggregates.invariant_failures']} "
            f"intracellular_invariant_failures:{metrics['aggregates.intracellular.invariant_failures']} "
            f"extracellular_invariant_failures:{metrics['aggregates.extracellular.invariant_failures']}"
        ),
        (
            "[summary] neuron_progression="
            f"h2c:{metrics['neurons.transitions.healthy_to_compromised.count']} "
            f"c2a:{metrics['neurons.transitions.compromised_to_apoptotic.count']} "
            f"a2r:{metrics['neurons.transitions.apoptotic_to_ruptured.count']} "
            f"avg_compromised_ticks:{metrics['neurons.state_time.compromised_avg_ticks']:.2f} "
            f"avg_apoptotic_ticks:{metrics['neurons.state_time.apoptotic_avg_ticks']:.2f} "
            f"compromised_recoveries:{metrics['neurons.recoveries.compromised_to_healthy']} "
            f"number_of_neurons_ever_compromised:{metrics['neurons.ever_compromised']} "
            f"number_of_neurons_ever_apoptotic:{metrics['neurons.ever_apoptotic']} "
            f"number_of_neurons_ever_recovered:{metrics['neurons.ever_recovered']} "
            f"blocked_by_min_ticks_compromised:{metrics['neurons.blocks.min_ticks_compromised']} "
            f"blocked_by_apoptotic_internal_damage_threshold:{metrics['neurons.blocks.apoptotic_internal_damage_threshold']} "
            f"final_by_rank:{format_final_state_by_rank(details)}"
        ),
        (
            "[summary] neuron_transition_ticks="
            f"h2c:{format_transition_ticks(details, 'first_compromised_tick')} "
            f"c2a:{format_transition_ticks(details, 'first_apoptotic_tick')} "
            f"a2r:{format_transition_ticks(details, 'first_ruptured_tick')}"
        ),
    ]

def format_transition_ticks(details: Sequence[Mapping[str, Any]], field: str) -> str:
    values = [
        f"{detail['uid']}:{detail[field]}"
        for detail in sorted(details, key=lambda item: item["uid"])
        if detail.get(field) is not None
    ]
    return ",".join(values) if values else "none"

def format_final_state_by_rank(details: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[int, Counter] = {}
    for detail in details:
        rank = int(detail.get("rank", 0))
        grouped.setdefault(rank, Counter())[detail.get("final_state", "Unknown")] += 1
    if not grouped:
        return "none"
    parts = []
    for rank in sorted(grouped):
        counts = grouped[rank]
        state_counts = "/".join(
            f"{state}:{counts.get(state, 0)}"
            for state in ("Healthy", "Compromised", "Apoptotic", "Ruptured", "Unknown")
            if counts.get(state, 0) > 0
        )
        parts.append(f"rank{rank}={state_counts or 'none'}")
    return ";".join(parts)
