from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent
from dataclasses import dataclass
from enum import Enum

# Internal State Set
class AstrocyteState(str, Enum):
    SUPPORTIVE = "Supportive",
    REACTIVE = "Reactive"

# Action Set
class AstrocyteAction(str, Enum):
    SUPPORT = "provide_support",
    INFLAMMATION = "release_inflammation"

@dataclass(frozen=True)
class AstrocytePerception:
    position: Optional[DiscretePoint]
    inflammation_level: float
    extracellular_debris: float

@dataclass
class AstrocyteConfig:
    inflammation_high_threshold: float
    inflammation_low_threshold: float
    debris_high_threshold: float
    debris_low_threshold: float
    support_inflammation_reduction_rate: float
    inflammation_release_rate: float

class Astrocyte(AdaptiveAgent):
    def __init__(self, local_id: int, rank: int, type_id: int, config: AstrocyteConfig):
        super().__init__(local_id, type_id, rank)
        self.state = AstrocyteState.SUPPORTIVE
        self.cfg = config
        self.last_perception: Optional[AstrocytePerception] = None
        self.pending_action: Optional[AstrocyteAction] = None

    def see(self, model):
        env = model.environment
        position = env.position_of(self)
        self.last_perception = AstrocytePerception(position=position,inflammation_level=env.inflammation_level,extracellular_debris=env.extracellular_debris)

    def next(self):
        perception = self.last_perception
        if self.state == AstrocyteState.SUPPORTIVE:
            if perception.inflammation_level >= self.cfg.inflammation_high_threshold or perception.extracellular_debris >= self.cfg.debris_high_threshold:
                self.state = AstrocyteState.REACTIVE
        elif self.state == AstrocyteState.REACTIVE:
            if perception.inflammation_level <= self.cfg.inflammation_low_threshold and perception.extracellular_debris <= self.cfg.debris_low_threshold:
                self.state = AstrocyteState.SUPPORTIVE

    def action(self):
        if self.state == AstrocyteState.SUPPORTIVE:
            self.pending_action = AstrocyteAction.SUPPORT
        elif self.state == AstrocyteState.REACTIVE:
            self.pending_action = AstrocyteAction.INFLAMMATION

    def do(self, model):
        env = model.environment
        if self.pending_action == AstrocyteAction.SUPPORT:
            env.remove_inflammation(self.cfg.support_inflammation_reduction_rate)
        elif self.pending_action == AstrocyteAction.INFLAMMATION:
            env.add_inflammation(self.cfg.inflammation_release_rate)

    def step(self, model):
        self.see(model)
        self.next()
        self.action()
        self.do(model)