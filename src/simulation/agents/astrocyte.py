from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass

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
        p = self.last_perception
        if self.state == AstrocyteState.SUPPORTIVE:
            if p.inflammation_level >= self.cfg.inflammation_high_threshold or p.extracellular_debris >= self.cfg.debris_high_threshold:
                self.state = AstrocyteState.REACTIVE
        elif self.state == AstrocyteState.REACTIVE:
            if p.inflammation_level <= self.cfg.inflammation_low_threshold and p.extracellular_debris <= self.cfg.debris_low_threshold:
                self.state = AstrocyteState.SUPPORTIVE
        return self.state

    def action(self) -> AstrocyteAction:
        """Map the current astrocyte state to one environmental action."""
        if self.state == AstrocyteState.SUPPORTIVE:
            self.pending_action = AstrocyteAction.SUPPORT
        elif self.state == AstrocyteState.REACTIVE:
            self.pending_action = AstrocyteAction.INFLAMMATION
        return self.pending_action

    def do(self, model):
        """Apply the selected action to the Substantia Nigra environment."""
        if self.pending_action is None:
            return
        env = model.environment
        if self.pending_action == AstrocyteAction.SUPPORT:
            env.remove_inflammation(self.cfg.support_inflammation_reduction_rate)
        elif self.pending_action == AstrocyteAction.INFLAMMATION:
            env.add_inflammation(self.cfg.inflammation_release_rate)

    def step(self, model):
        """Run one extracellular astrocyte phase: see, next, action, do."""
        self.see(model)
        self.next()
        self.action()
        self.do(model)