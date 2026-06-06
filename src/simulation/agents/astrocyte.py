from typing import Optional
from src.simulation.agents.structure import AstrocyteState, AstrocytePerception, AstrocyteAction, AstrocyteConfig, AdaptiveAgent
from src.simulation.utils import RNG, clamp
from src.simulation.logger.agent_logging import bind_event_logger, event_logger_from

class Astrocyte(AdaptiveAgent):
    """Extracellular support agent that dampens or amplifies inflammation."""
    def __init__(self, local_id: int, rank: int, type_id: int, config: AstrocyteConfig):
        super().__init__(local_id, type_id, rank)
        self.state: AstrocyteState = AstrocyteState.SUPPORTIVE
        self.cfg = config
        self.last_perception: Optional[AstrocytePerception] = None
        self.pending_action: Optional[AstrocyteAction] = None
        self.stress_memory: float = 0.0
        self.last_transition_sample = {}
        self.rng = RNG

    def see(self, model) -> AstrocytePerception:
        """Read extracellular inflammatory and debris state."""
        bind_event_logger(self, model)
        env = model.environment
        position = env.position_of(self)
        perception = AstrocytePerception(
            position=position,
            inflammation_level=env.scalars.inflammation_level,
            extracellular_debris=env.scalars.extracellular_debris
        )
        self.last_perception = perception
        return perception

    def next(self) -> AstrocyteState:
        """Update astrocyte state from stress memory.
        Astrocytes read global inflammatory scalars, so a purely deterministic
        threshold would synchronize the full population. Each instance instead
        integrates stress over time and samples a transition probability from
        that memory. Defaults preserve the old deterministic behavior for unit
        tests and small toy runs.
        """
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        p = self.last_perception
        pressure = self._stress_pressure(p)
        self.stress_memory = clamp(
            self.cfg.stress_memory_decay * self.stress_memory
            + (1.0 - self.cfg.stress_memory_decay) * pressure
        )
        self.last_transition_sample = {
            "stress_pressure": pressure,
            "stress_memory": self.stress_memory,
            "probability": 0.0,
            "draw": None,
        }
        if self.state == AstrocyteState.SUPPORTIVE:
            probability = clamp(self.cfg.reactive_transition_rate * self.stress_memory)
            draw = self.rng.random()
            self.last_transition_sample.update({
                "probability": probability,
                "draw": draw,
            })
            if draw < probability:
                self.state = AstrocyteState.REACTIVE
        elif self.state == AstrocyteState.REACTIVE:
            low_environment = p.inflammation_level <= self.cfg.inflammation_low_threshold
            probability = clamp(self.cfg.supportive_recovery_rate * (1.0 - self.stress_memory))
            draw = self.rng.random()
            self.last_transition_sample.update({
                "probability": probability,
                "draw": draw,
            })
            if low_environment and draw < probability:
                self.state = AstrocyteState.SUPPORTIVE
        if old_state != self.state:
            self._log_event_state_trigger(old_state, p)
        return self.state

    def action(self) -> AstrocyteAction:
        """Map the current astrocyte state to one environmental action.

        Reactive astrocytes are not automatically inflammatory. They keep
        providing support while stress memory is low, and become inflammatory
        only after stress has persisted enough to cross the configured memory
        threshold.
        """
        if self.state == AstrocyteState.SUPPORTIVE:
            self.pending_action = AstrocyteAction.SUPPORT
        elif self.state == AstrocyteState.REACTIVE:
            if self.stress_memory >= self.cfg.inflammatory_memory_threshold:
                self.pending_action = AstrocyteAction.INFLAMMATION
            else:
                self.pending_action = AstrocyteAction.SUPPORT
        logger = event_logger_from(self)
        if logger is not None:
            logger.action_selection(self, self.pending_action, "astrocyte_state_action_policy")
        return self.pending_action

    def do(self, model):
        """Apply the selected action to the Substantia Nigra environment."""
        if self.pending_action is None:
            return
        env = model.environment
        if self.pending_action == AstrocyteAction.SUPPORT:
            env.remove_inflammation(self.cfg.support_inflammation_reduction_rate)
            logger = event_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "inflammation_level",
                    -self.cfg.support_inflammation_reduction_rate,
                    "astrocyte_support_inflammation_reduction"
                )
        elif self.pending_action == AstrocyteAction.INFLAMMATION:
            amount = self._inflammation_release_amount()
            env.add_inflammation(amount)
            logger = event_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "inflammation_level",
                    amount,
                    "astrocyte_reactive_inflammation_release"
                )

    def step(self, model):
        """Run one extracellular astrocyte phase: see, next, action, do."""
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def _log_event_state_trigger(self, old_state: AstrocyteState, p: AstrocytePerception) -> None:
        """Log only event predicates that produced an astrocyte transition."""

        logger = event_logger_from(self)
        if logger is None:
            return
        if self.state == AstrocyteState.REACTIVE:
            if p.inflammation_level >= self.cfg.inflammation_high_threshold:
                source = logger.env_field_node("SN.inflammation_level", "inflammation_level", "1_perception", p.inflammation_level)
                logger.threshold_trigger(
                    source,
                    self,
                    self.state,
                    "astrocyte_reactive_by_inflammation",
                    "ASTROCYTE_REACTIVE_STRESS_HIGH",
                    "inflammation_level >= inflammation_high_threshold"
                )
            elif p.extracellular_debris >= self.cfg.debris_high_threshold:
                source = logger.env_field_node("SN.extracellular_debris", "extracellular_debris", "1_perception", p.extracellular_debris)
                logger.threshold_trigger(
                    source,
                    self,
                    self.state,
                    "astrocyte_reactive_by_debris",
                    "ASTROCYTE_REACTIVE_STRESS_HIGH",
                    "extracellular_debris >= debris_high_threshold"
                )
        logger.state_transition(
            self,
            old_state,
            self.state,
            "astrocyte_state_update",
            probability=self.last_transition_sample.get("probability"),
            rng_value=self.last_transition_sample.get("draw")
        )

    def _stress_pressure(self, p: AstrocytePerception) -> float:
        """Return normalized extracellular stress above low thresholds."""
        inflammation_pressure = self._normalized_pressure(
            p.inflammation_level,
            self.cfg.inflammation_low_threshold,
            self.cfg.inflammation_high_threshold
        )
        debris_pressure = self._normalized_pressure(
            p.extracellular_debris,
            self.cfg.debris_low_threshold,
            self.cfg.debris_high_threshold
        )
        return max(inflammation_pressure, clamp(self.cfg.debris_stress_weight) * debris_pressure)

    def _normalized_pressure(self, value: float, low: float, high: float) -> float:
        """Normalize a scalar between low and high thresholds."""
        if high <= low:
            return 1.0 if value >= high else 0.0
        return clamp((value - low) / (high - low))

    def _inflammation_release_amount(self) -> float:
        """Scale reactive inflammatory output by stress memory when enabled."""
        weight = clamp(self.cfg.inflammation_memory_weight)
        multiplier = (1.0 - weight) + weight * self.stress_memory
        return self.cfg.inflammation_release_rate * multiplier
