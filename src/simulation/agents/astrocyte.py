from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from src.simulation.logger.causal_trace_logger import bind_causal_logger, causal_logger_from

# Internal State Set
class AstrocyteState(str, AdaptiveAgentState):
    """Functional extracellular state of an astrocyte."""
    SUPPORTIVE = "Supportive"
    REACTIVE = "Reactive"

# Action Set
class AstrocyteAction(str, AdaptiveAgentAction):
    """Actions an astrocyte can apply to the shared environment."""
    SUPPORT = "provide_support"
    INFLAMMATION = "release_inflammation"

@dataclass(frozen=True)
class AstrocytePerception(AdaptiveAgentPerception):
    """Extracellular inflammatory and debris signals sensed by astrocytes."""
    position: Optional[DiscretePoint]
    inflammation_level: float
    extracellular_debris: float

@dataclass
class AstrocyteConfig:
    """Astrocyte sensing thresholds and inflammatory effect rates."""
    inflammation_high_threshold: float
    inflammation_low_threshold: float
    debris_high_threshold: float
    debris_low_threshold: float
    support_inflammation_reduction_rate: float
    inflammation_release_rate: float

class Astrocyte(AdaptiveAgent):
    """Extracellular support agent that dampens or amplifies inflammation."""
    def __init__(self, local_id: int, rank: int, type_id: int, config: AstrocyteConfig):
        super().__init__(local_id, type_id, rank)
        self.state: AstrocyteState = AstrocyteState.SUPPORTIVE
        self.cfg = config
        self.last_perception: Optional[AstrocytePerception] = None
        self.pending_action: Optional[AstrocyteAction] = None

    def see(self, model) -> AstrocytePerception:
        """Read extracellular inflammatory and debris state."""
        bind_causal_logger(self, model)
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
        """Update astrocyte state from the last perception."""
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        p = self.last_perception
        if self.state == AstrocyteState.SUPPORTIVE:
            if p.inflammation_level >= self.cfg.inflammation_high_threshold or p.extracellular_debris >= self.cfg.debris_high_threshold:
                self.state = AstrocyteState.REACTIVE
        elif self.state == AstrocyteState.REACTIVE:
            if p.inflammation_level <= self.cfg.inflammation_low_threshold and p.extracellular_debris <= self.cfg.debris_low_threshold:
                self.state = AstrocyteState.SUPPORTIVE
        if old_state != self.state:
            self._log_causal_state_trigger(old_state, p)
        return self.state

    def action(self) -> AstrocyteAction:
        """Map the current astrocyte state to one environmental action."""
        if self.state == AstrocyteState.SUPPORTIVE:
            self.pending_action = AstrocyteAction.SUPPORT
        elif self.state == AstrocyteState.REACTIVE:
            self.pending_action = AstrocyteAction.INFLAMMATION
        logger = causal_logger_from(self)
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
            logger = causal_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "inflammation_level",
                    -self.cfg.support_inflammation_reduction_rate,
                    "negative",
                    "astrocyte_support_inflammation_reduction"
                )
        elif self.pending_action == AstrocyteAction.INFLAMMATION:
            env.add_inflammation(self.cfg.inflammation_release_rate)
            logger = causal_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    self.pending_action,
                    "inflammation_level",
                    self.cfg.inflammation_release_rate,
                    "positive",
                    "astrocyte_reactive_inflammation_release"
                )

    def step(self, model):
        """Run one extracellular astrocyte phase: see, next, action, do."""
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def _log_causal_state_trigger(self, old_state: AstrocyteState, p: AstrocytePerception) -> None:
        """Log only causal predicates that produced an astrocyte transition."""

        logger = causal_logger_from(self)
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
        logger.state_transition(self, old_state, self.state, "astrocyte_state_update")
