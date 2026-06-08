from math import ceil
from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.aggregate import AlphaAggregate
from src.simulation.utils import InternalHabitatMixin, RNG
from src.simulation.utils.grid import LocalGrid, clamp
from src.simulation.agents.structure import NeuronAction, AdaptiveAgent, NeuronConfig, NeuronInternalConfig, NeuronInternalEffects, NeuronInternalScalars, NeuronPerception, NeuronState, AlphaSynucleinState
from src.simulation.agents.aggregate_registry import AggregateRegistry
from src.simulation.agents.alphasynuclein import AlphaSynuclein
from src.simulation.logger.agent_logging import bind_event_logger, event_logger_from

class Neuron(InternalHabitatMixin, AdaptiveAgent):
    """Neuron macro-agent with an intracellular habitat.
    The neuron owns a local grid where organelles, alpha-synuclein proteins,
    aggregates and lysosomes interact. It also exposes small buffer APIs used
    by intracellular agents to coordinate deferred work, such as lysosomal
    degradation targets. Aggregate identity and membership are delegated to the
    environment-level AggregateRegistry."""
    def __init__(
        self,
        local_id: int,
        rank: int,
        type_id: int,
        config: NeuronConfig,
        alpha_type_id:int,
        internal_config: Optional[NeuronInternalConfig] = None,
        environment=None
    ):
        super().__init__(local_id, type_id, rank)
        # Adaptive agent fields
        self.state: NeuronState = NeuronState.HEALTHY
        self.cfg = config
        self.alpha_type_id = alpha_type_id
        self.last_perception: Optional[NeuronPerception] = None
        self.pending_action: Optional[NeuronAction] = None
        self.rng = RNG
        self.environment = environment

        # Cumulative value for cell damage
        self.cell_damage: float = 0.0
        self._rupture_payload_released = False
        self.ticks_in_state: int = 0
        self.first_compromised_tick: Optional[int] = None
        self.first_apoptotic_tick: Optional[int] = None
        self.first_ruptured_tick: Optional[int] = None
        self.compromised_ticks_total: int = 0
        self.apoptotic_ticks_total: int = 0
        self.compromised_recoveries: int = 0
        self.blocked_by_min_ticks_compromised: int = 0
        self.blocked_by_apoptotic_internal_damage_threshold: int = 0
        self._current_tick: Optional[int] = None

        # Environmental state
        self.internal_cfg = internal_config or NeuronInternalConfig()
        self.internal_scalars = NeuronInternalScalars(
            energy_demand=self.internal_cfg.energy_demand_baseline,
        )
        self.internal_effects = NeuronInternalEffects()

        # Local (Environmental) Grid
        self.grid = LocalGrid(width = self.internal_cfg.width, height = self.internal_cfg.height)

        # Lysosome targeting bridge. Producers register degradable agents here;
        # lysosomes claim one unassigned target at a time.
        self.degradation_targets: list[AdaptiveAgent] = []
        self.degradation_assignment: dict[AdaptiveAgent, AdaptiveAgent] = {}

    @property
    def aggregate_registry(self) -> AggregateRegistry:
        """Return the environment-level aggregate registry visible to this neuron.

        Runtime neurons are bound to SubstantiaNigra, which owns the shared
        registry for the rank. A neuron without such an environment cannot
        process alpha aggregation because aggregate identity is global.
        """

        registry = getattr(getattr(self, "environment", None), "aggregate_registry", None)
        if registry is not None:
            return registry
        raise RuntimeError("Neuron requires an environment with aggregate_registry.")

    def bind_environment(self, environment):
        """Expose the shared extracellular environment to this neuron."""

        self.environment = environment
        return self

    def see(self, model) -> NeuronPerception:
        """Build the neuron perception from external and internal signals."""
        bind_event_logger(self, model)
        env = model.environment
        self.bind_environment(env)
        position = env.position_of(self)
        if position is None:
            nearby_alpha = 0.0
        else:
            nearby_alpha = env.density_of_type(center=position, radius=self.cfg.per_radius, agent_type=self.alpha_type_id,include_center=True)
        perception = NeuronPerception(
            position=position,
            nearby_alpha=nearby_alpha,
            inflammatory_levels=env.scalars.inflammation_level,
            extracellular_debris=env.scalars.extracellular_debris,
            oxidative_stress=self.internal_scalars.oxidative_stress,
            intracellular_debris=self.internal_scalars.intracellular_debris,
            energy_demand=self.internal_scalars.energy_demand,
            internal_damage=self.compute_internal_damage(),
            alpha_load=self.compute_alpha_load(),
            cell_damage=self.cell_damage
        )
        self.last_perception = perception
        return perception

    def next(self) -> NeuronState:
        """Advance the neuronal damage state from the last perception."""
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        if old_state == NeuronState.RUPTURED:
            self.cell_damage = max(self.cell_damage, self.cfg.ruptured_threshold)
            self._record_state_tick(self.state)
            self.ticks_in_state += 1
            return self.state
        p = self.last_perception
        external_stress = self._compute_external_stress(p)
        internal_damage = p.internal_damage

        total_stress = clamp(0.5*external_stress + 0.5*internal_damage) # Mean between external and internal stress

        if total_stress <= self.cfg.low_stress_threshold:
            if self.state in (NeuronState.HEALTHY, NeuronState.COMPROMISED):
                self.cell_damage = clamp(self.cell_damage - self.cfg.damage_recovery_rate)
        else:
            damage_increment = min(
                total_stress * self.cfg.damage_accumulation_rate,
                self.cfg.max_damage_increment_per_tick
            )
            self.cell_damage = clamp(self.cell_damage + damage_increment)

        candidate_state = self._state_from_damage(self.cell_damage)
        final_state, block_reasons = self._apply_transition_gates(candidate_state, p)
        self.state = final_state
        self._record_transition_metrics(old_state, self.state)
        if self.state == old_state:
            self.ticks_in_state += 1
        else:
            self.ticks_in_state = 0
        logger = event_logger_from(self)
        if logger is not None:
            if old_state != self.state:
                logger.state_transition(
                    self,
                    old_state,
                    self.state,
                    "damage_accumulation",
                    rule_id="NEURON_DAMAGE_ACCUMULATION"
                )
        return self.state

    def _state_from_damage(self, cell_damage: float) -> NeuronState:
        """Map cumulative cell damage to the ungated target state."""

        if cell_damage >= self.cfg.ruptured_threshold:
            return NeuronState.RUPTURED
        if cell_damage >= self.cfg.apoptotic_threshold:
            return NeuronState.APOPTOTIC
        if cell_damage >= self.cfg.compromised_threshold:
            return NeuronState.COMPROMISED
        return NeuronState.HEALTHY

    def _apply_transition_gates(self, candidate_state: NeuronState, perception: NeuronPerception) -> tuple[NeuronState, list[str]]:
        """Slow pathological progression by enforcing state dwell and rupture gates."""

        block_reasons: list[str] = []
        if self.state == NeuronState.HEALTHY and candidate_state in (NeuronState.APOPTOTIC, NeuronState.RUPTURED):
            block_reasons.append("blocked_by_stepwise_progression")
            return NeuronState.COMPROMISED, block_reasons

        if self.state == NeuronState.COMPROMISED and candidate_state in (NeuronState.APOPTOTIC, NeuronState.RUPTURED):
            if self.ticks_in_state < self.cfg.min_ticks_compromised_before_apoptotic:
                block_reasons.append("blocked_by_min_ticks_compromised")
                self.blocked_by_min_ticks_compromised += 1
                return NeuronState.COMPROMISED, block_reasons
            if perception.internal_damage < self.cfg.apoptotic_internal_damage_threshold:
                block_reasons.append("blocked_by_apoptotic_internal_damage_threshold")
                self.blocked_by_apoptotic_internal_damage_threshold += 1
                return NeuronState.COMPROMISED, block_reasons
            if candidate_state == NeuronState.RUPTURED:
                block_reasons.append("blocked_by_stepwise_progression")
            return NeuronState.APOPTOTIC, block_reasons

        if self.state == NeuronState.APOPTOTIC and candidate_state == NeuronState.RUPTURED:
            if self.ticks_in_state < self.cfg.min_ticks_apoptotic_before_ruptured:
                block_reasons.append("blocked_by_min_ticks_apoptotic")
            if perception.internal_damage < self.cfg.rupture_internal_damage_threshold:
                block_reasons.append("blocked_by_internal_damage_threshold")
            if perception.intracellular_debris < self.cfg.rupture_intracellular_debris_threshold:
                block_reasons.append("blocked_by_intracellular_debris_threshold")
            if block_reasons:
                return NeuronState.APOPTOTIC, block_reasons

        return candidate_state, block_reasons

    def _record_transition_metrics(self, old_state: NeuronState, new_state: NeuronState):
        """Track first transition ticks, dwell time and compromised recovery."""

        self._record_state_tick(new_state)
        if old_state == new_state:
            return
        tick = self._current_tick
        if old_state == NeuronState.HEALTHY and new_state == NeuronState.COMPROMISED and self.first_compromised_tick is None:
            self.first_compromised_tick = tick
        elif old_state == NeuronState.COMPROMISED and new_state == NeuronState.APOPTOTIC and self.first_apoptotic_tick is None:
            self.first_apoptotic_tick = tick
        elif old_state == NeuronState.APOPTOTIC and new_state == NeuronState.RUPTURED and self.first_ruptured_tick is None:
            self.first_ruptured_tick = tick
        elif old_state == NeuronState.COMPROMISED and new_state == NeuronState.HEALTHY:
            self.compromised_recoveries += 1

    def _record_state_tick(self, state: NeuronState):
        """Accumulate time spent in intermediate pathological states."""

        if state == NeuronState.COMPROMISED:
            self.compromised_ticks_total += 1
        elif state == NeuronState.APOPTOTIC:
            self.apoptotic_ticks_total += 1

    def action(self) -> Optional[NeuronAction]:
        """Choose the neuron-level action for this tick."""
        if self.last_perception is None:
            raise RuntimeError()
        p = self.last_perception
        if self.state == NeuronState.RUPTURED:
            if self._rupture_payload_released:
                self.pending_action = NeuronAction.IDLE
            else:
                self.pending_action = NeuronAction.DUMP_DEBRIS

        elif self.state != NeuronState.APOPTOTIC and p.nearby_alpha >= self.cfg.nearby_alpha_high_threshold:
            self.pending_action = NeuronAction.A_ALPHASYNUCLEIN

        elif self.state == NeuronState.APOPTOTIC or p.alpha_load >= self.cfg.alpha_load_release_threshold:
            self.pending_action = NeuronAction.R_ALPHASYNUCLEIN

        elif self.state == NeuronState.HEALTHY:
            if (
                p.inflammatory_levels >= self.cfg.inflammation_high_threshold
                or p.extracellular_debris >= self.cfg.debris_high_threshold
            ):
                self.pending_action = NeuronAction.STRESS
            else:
                self.pending_action = NeuronAction.R_DOPAMINE
        elif self.state == NeuronState.COMPROMISED:
            self.pending_action = NeuronAction.R_DOPAMINE
        else:
            self.pending_action = NeuronAction.STRESS
        return self.pending_action

    def do(self, model):
        """Apply the selected neuron-level effect to the model."""
        if self.last_perception is None:
            return
        env = model.environment
        if self.pending_action == NeuronAction.R_DOPAMINE:
            dopamine = self._release_dopamine(env)
            logger = event_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "dopamine_output",
                    dopamine,
                    "neuron_dopamine_release"
                )
        elif self.pending_action == NeuronAction.STRESS:
            env.add_inflammation(self.cfg.stress_inflammation_release_rate)
            logger = event_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "inflammation_level",
                    self.cfg.stress_inflammation_release_rate,
                    "neuron_stress_inflammation_release",
                )
        elif self.pending_action == NeuronAction.DUMP_DEBRIS:
            debris = self._dump_debris_payload()
            if debris > 0.0:
                env.add_debris(debris)
            self.internal_scalars.intracellular_debris = 0.0
            released = self.release_alpha(model)
            logger = event_logger_from(self)
            if logger is not None and debris > 0.0:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "extracellular_debris",
                    debris,
                    "neuron_dump_debris"
                )
        elif self.pending_action == NeuronAction.A_ALPHASYNUCLEIN:
            self.absorb_alpha(model)
        elif self.pending_action == NeuronAction.R_ALPHASYNUCLEIN:
            self.release_alpha(model)
            dopamine = self._release_dopamine(
                env,
                self.cfg.alpha_release_dopamine_fraction
            )
            logger = event_logger_from(self)
            if logger is not None and dopamine > 0.0:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "dopamine_output",
                    dopamine,
                    "neuron_residual_dopamine_during_alpha_release"
                )
        elif self.pending_action == NeuronAction.IDLE:
            return

    def step(self, model):
        """Run one intracellular phase pass followed by the neuron macro step."""

        bind_event_logger(self, model)
        self.bind_environment(model.environment)
        self._current_tick = getattr(model, "tick_count", None)
        if self.state == NeuronState.RUPTURED:
            self.begin_tick()
            self.see(model)
            self.action()
            self.do(model)
            return
        self.begin_tick()
        internal_agents = [
            agent
            for agent in list(self.grid.agent_registry)
            if isinstance(agent, AdaptiveAgent)
        ]
        for agent in internal_agents:
            agent.see(model)
        for agent in internal_agents:
            agent.next()
        self.aggregate_registry.process(self)
        active_agents = [
            agent
            for agent in internal_agents
            if self.position_of(agent) is not None
        ]
        for agent in active_agents:
            agent.action()
        for agent in active_agents:
            agent.do(model)
        self.commit_effects()
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def begin_tick(self):
        """Reset buffered intracellular effects for a new tick."""

        self.internal_effects = NeuronInternalEffects()

    def commit_effects(self):
        """Apply buffered intracellular effects at the end of the tick."""
        cfg = self.internal_cfg
        s = self.internal_scalars
        e = self.internal_effects
        s.oxidative_stress = clamp(s.oxidative_stress + e.oxidative_stress_added - cfg.oxidative_stress_decay * s.oxidative_stress)
        s.intracellular_debris = clamp(s.intracellular_debris + e.debris_added - cfg.intracellular_debris_decay * s.intracellular_debris)
        baseline_pull = cfg.energy_demand_recovery_rate * (cfg.energy_demand_baseline - s.energy_demand)
        s.energy_demand = clamp(s.energy_demand + e.energy_demand_added + baseline_pull)

    def _compute_external_stress(self, perception: NeuronPerception) -> float:
        """Combine extracellular inflammatory, debris and alpha stress."""

        return clamp(perception.inflammatory_levels * self.cfg.inflammation_damage_weight + perception.extracellular_debris * self.cfg.debris_damage_weight + perception.nearby_alpha * self.cfg.alpha_damage_weight)

    def _dopamine_release_amount(self) -> float:
        """Return dopamine released by the current neuronal state."""
        if self.state == NeuronState.HEALTHY:
            state_factor = self.cfg.dopamine_factor_healthy
        elif self.state == NeuronState.COMPROMISED:
            state_factor = self.cfg.dopamine_factor_compromised
        elif self.state == NeuronState.APOPTOTIC:
            state_factor = self.cfg.dopamine_factor_apoptotic
        elif self.state == NeuronState.RUPTURED:
            state_factor = self.cfg.dopamine_factor_ruptured
        else:
            state_factor = 0.0
        return self.cfg.dopamine_release_rate * clamp(state_factor)

    def _release_dopamine(self, env, fraction: float = 1.0) -> float:
        """Release state-scaled dopamine and return the emitted amount."""
        dopamine = self._dopamine_release_amount() * clamp(fraction)
        if dopamine > 0.0:
            env.release_dopamine(dopamine)
        return dopamine

    def _dump_debris_payload(self) -> float:
        """Return debris released by a ruptured neuron in this action.

        Intracellular debris is released whenever present. The configured
        rupture payload is released only once, so a ruptured neuron does not
        create an infinite debris source after it has already spilled its
        contents.
        """

        if self._rupture_payload_released:
            return 0.0
        payload = self.internal_scalars.intracellular_debris
        payload += self.cfg.debris_release_rate
        self._rupture_payload_released = True
        return payload

    def compute_alpha_load(self) -> float:
        """Compute total intracellular alpha pathology load over grid capacity."""

        width = max(1, self.internal_cfg.width)
        height = max(1, self.internal_cfg.height)
        capacity = width * height
        aggregate_score = sum(
            self.aggregate_weight(agent)
            for agent in self.grid.agent_registry
        )
        return clamp(aggregate_score / capacity)

    def compute_internal_damage(self) -> float:
        """Compute weighted intracellular damage from scalars and alpha load."""

        cfg = self.internal_cfg
        s = self.internal_scalars
        aggregate_density = self.compute_alpha_load()
        value = (
                cfg.internal_damage_oxidative_weight * s.oxidative_stress +
                cfg.internal_damage_aggregate_weight * aggregate_density +
                cfg.internal_damage_debris_weight * s.intracellular_debris
        )
        return clamp(value)

    def absorb_alpha(self, model):
        """Gradually absorb one extracellular alpha pathology agent.

        Absorption is intentionally gradual: one eligible extracellular
        AlphaSynuclein or AlphaAggregate near this neuron may enter the
        intracellular grid per call, controlled by alpha_absorption_rate.
        """

        env = model.environment
        self.bind_environment(env)
        position = env.position_of(self)
        if position is None:
            return None
        candidates = [
            agent
            for agent in env.agents_in_radius(
                center=position,
                radius=self.cfg.per_radius,
                include_center=True,
            )
            if self._is_absorbable_alpha_pathology(agent)
        ]
        for candidate in candidates:
            draw = self.rng.random()
            if draw <= self.cfg.alpha_absorption_rate:
                self._absorb_alpha_agent(env, candidate)
                return candidate
        return None

    def release_alpha(self, model):
        """Release intracellular alpha pathology into the Substantia Nigra.

        Rupture is modeled as an immediate spill of every visible alpha
        pathology agent. Non-ruptured release, such as apoptotic leakage or
        high alpha load, is gradual and releases a configured fraction.
        """

        env = model.environment
        self.bind_environment(env)
        release_point = env.position_of(self)
        if release_point is None:
            return []
        pathology = self.alpha_pathology_agents()
        if not pathology:
            return []
        release_count = len(pathology) if self.state == NeuronState.RUPTURED else self._gradual_alpha_release_count(len(pathology))
        released = []
        for agent in pathology[:release_count]:
            self._release_alpha_agent(env, agent, release_point)
            released.append(agent)
        return released

    def alpha_pathology_agents(self) -> list[AdaptiveAgent]:
        """Return visible intracellular, uncleared alpha-synuclein agents."""

        return [
            agent
            for agent in list(self.grid.agent_registry)
            if isinstance(agent, AlphaAggregate)
            or (
                isinstance(agent, AlphaSynuclein)
                and agent.state != AlphaSynucleinState.CLEARED
            )
        ]

    def _gradual_alpha_release_count(self, total: int) -> int:
        """Number of alpha pathology agents released by non-rupture leakage."""

        if total <= 0 or self.cfg.alpha_release_amount <= 0.0:
            return 0
        if self.cfg.alpha_release_amount >= 1.0:
            return min(total, int(self.cfg.alpha_release_amount))
        return max(1, min(total, ceil(total * self.cfg.alpha_release_amount)))

    def _release_alpha_agent(self, env, agent: AdaptiveAgent, point: DiscretePoint):
        """Move one alpha pathology agent from neuron grid to environment grid."""

        self._transfer_alpha_out_of_neuron(agent)
        if isinstance(agent, AlphaSynuclein):
            agent.release_to_environment()
        elif isinstance(agent, AlphaAggregate):
            agent.release_to_environment()
        env.add_agent(agent, point)

    def _absorb_alpha_agent(self, env, agent: AdaptiveAgent):
        """Move one extracellular alpha pathology agent into this neuron."""

        env.remove_agent(agent)
        point = self._default_internal_point()
        if isinstance(agent, AlphaSynuclein):
            agent.absorb_into_neuron(self)
            self.add_agent(agent, point)
        elif isinstance(agent, AlphaAggregate):
            agent.absorb_into_neuron(self)
            for member in agent.member_agents:
                member.absorb_into_neuron(self)
            self.add_agent(agent, point)
            self.aggregate_registry.register_existing_aggregate(self, agent)

    def _transfer_alpha_out_of_neuron(self, agent: AdaptiveAgent):
        """Remove alpha pathology from neuron ownership without clearing it."""

        self.unregister_degradation_target(agent)
        self.clear_degradation_assignment(agent)
        self.clear_assignments_for_target(agent)
        if isinstance(agent, AlphaAggregate):
            members = self.aggregate_registry.members(agent.aggregate_id)
            agent.member_agents.update(members)
            for member in members:
                member.release_to_environment()
        self.grid.remove_agent(agent)

    def _default_internal_point(self) -> DiscretePoint:
        """Return a stable default point for absorbed intracellular agents."""

        return DiscretePoint(
            self.internal_cfg.width // 2,
            self.internal_cfg.height // 2,
        )

    def _is_absorbable_alpha_pathology(self, agent: AdaptiveAgent) -> bool:
        """Return whether an extracellular agent can be absorbed by this neuron."""

        if isinstance(agent, AlphaAggregate):
            return agent.owner_neuron is None
        if isinstance(agent, AlphaSynuclein):
            return agent.owner_neuron is None and agent.state != AlphaSynucleinState.CLEARED
        return False

    # Degradation Buffer Functions
    def register_degradation_target(self, agent: AdaptiveAgent):
        """Register an intracellular agent as available for lysosomal cleanup."""

        self._prune_degradation_buffers()
        if agent in self.grid.agent_registry and agent not in self.degradation_targets and not self.is_target_assigned(agent):
            self.degradation_targets.append(agent)

    def available_degradation_targets(self) -> list[AdaptiveAgent]:
        """Return unassigned targets that are still present in the local grid."""

        self._prune_degradation_buffers()
        assigned_targets = set(self.degradation_assignment.values())
        return [target for target in self.degradation_targets if target in self.grid.agent_registry and target not in assigned_targets]

    def assign_degradation_target(self, lysosome: AdaptiveAgent, target: AdaptiveAgent) -> bool:
        """Assign one available target to one lysosome.
        Returns True when the assignment is accepted. A target can only be
        assigned to one lysosome at a time; if the lysosome already had a
        different target, that old target is returned to the available buffer.
        """

        self._prune_degradation_buffers()
        if lysosome not in self.grid.agent_registry or target not in self.grid.agent_registry:
            return False
        if target in self.degradation_assignment.values():
            return self.degradation_assignment.get(lysosome) is target
        old_target = self.degradation_assignment.pop(lysosome, None)
        if old_target is not None and old_target is not target:
            self.register_degradation_target(old_target)
        self.degradation_assignment[lysosome] = target
        if target in self.degradation_targets:
            self.degradation_targets.remove(target)
        return True

    def target_for(self, lysosome: AdaptiveAgent) -> Optional[AdaptiveAgent]:
        """Return the target currently assigned to a lysosome, if any."""

        self._prune_degradation_buffers()
        return self.degradation_assignment.get(lysosome)

    def clear_degradation_assignment(self, lysosome: AdaptiveAgent, requeue_target: bool = False):
        """Clear a lysosome assignment and optionally make the target available."""

        target = self.degradation_assignment.pop(lysosome, None)
        if requeue_target and target is not None:
            self.register_degradation_target(target)

    def clear_assignments_for_target(self, target: AdaptiveAgent):
        """Remove every lysosome assignment pointing at a target."""

        for lysosome, assigned_target in list(self.degradation_assignment.items()):
            if assigned_target is target:
                del self.degradation_assignment[lysosome]

    def is_target_assigned(self, target: AdaptiveAgent) -> bool:
        """Return whether a target is already claimed by any lysosome."""

        self._prune_degradation_buffers()
        return target in self.degradation_assignment.values()

    def unregister_degradation_target(self, target: AdaptiveAgent):
        """Remove a target from all degradation buffers without touching grid state."""

        if target in self.degradation_targets:
            self.degradation_targets.remove(target)
        self.clear_assignments_for_target(target)

    def _prune_degradation_buffers(self):
        """Drop stale targets and assignments whose agents left the local grid."""

        self.degradation_targets = [
            target
            for target in self.degradation_targets
            if target in self.grid.agent_registry
        ]
        for lysosome, target in list(self.degradation_assignment.items()):
            if lysosome not in self.grid.agent_registry or target not in self.grid.agent_registry:
                del self.degradation_assignment[lysosome]


    # Internal Scalar Functions
    def add_oxidative_stress(self, amount: float):
        """Buffer a change in intracellular oxidative stress."""

        self.internal_effects.oxidative_stress_added += amount

    def oxidative_stress_at(self, position: Optional[DiscretePoint] = None) -> float:
        """Return global intracellular oxidative stress."""

        return self.internal_scalars.oxidative_stress

    def add_intracellular_debris(self, amount: float):
        """Buffer a change in intracellular debris."""

        self.internal_effects.debris_added += amount

    def add_energy_demand(self, amount: float):
        """Change unmet cellular energy demand through the effect buffer."""
        self.internal_effects.energy_demand_added += amount

    def energy_demand_at(self, position: Optional[DiscretePoint] = None) -> float:
        """Return global unmet cellular energy demand."""

        return self.internal_scalars.energy_demand

    def local_aggregate_density_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        """Return local alpha aggregate density around a grid position."""

        if position is None:
            return 0.0
        points = list(self.grid.neighbor_points(position, radius, include_center))
        if not points:
            return 0.0
        aggregate_score = 0.0
        for point in points:
            for agent in self.grid.agents_at(point):
                aggregate_score += self.aggregate_weight(agent)
        return clamp(aggregate_score / len(points))

    def remove_agent(self, agent: AdaptiveAgent):
        """Remove an intracellular agent and all neuron bookkeeping for it."""
        self.grid.remove_agent(agent)
        self.aggregate_registry.remove(agent, habitat=self)
        self.clear_degradation_assignment(agent)
        self.clear_assignments_for_target(agent)
        if agent in self.degradation_targets:
            self.degradation_targets.remove(agent)

    def aggregate_weight(self, agent: AdaptiveAgent) -> float:
        """Return the contribution of an agent to alpha pathology load."""
        if isinstance(agent, AlphaSynuclein) or isinstance(agent, AlphaAggregate):
            return agent.aggregate_weight
        else:
            return 0.0

    def local_debris_density_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        """Return local debris-like agent density around a grid position."""
        if position is None:
            return 0.0
        points = list(self.grid.neighbor_points(position, radius, include_center))
        if not points:
            return 0.0
        debris_score = 0.0
        for point in points:
            for agent in self.grid.agents_at(point):
                try:
                    state = agent.state
                except AttributeError:
                    continue
                try:
                    state_value = state.value
                except AttributeError:
                    state_value = state
                if state_value == "Debris":
                    debris_score += 1.0
        return clamp(debris_score / len(points))

    def local_debris_at(self, position: Optional[DiscretePoint] = None, radius: int = 1, include_center: bool = True) -> float:
        """Backward-compatible alias for local_debris_density_at."""
        return self.local_debris_density_at(position, radius, include_center)
