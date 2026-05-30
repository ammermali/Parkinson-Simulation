from abc import abstractmethod, ABC
from typing import Optional
from repast4py import core
from enum import Enum

class AdaptiveAgentAction(Enum):
    pass
class AdaptiveAgentState(Enum):
    pass
class AdaptiveAgentPerception:
    pass

class AdaptiveAgent(core.Agent, ABC):
    state = None
    pt: (int,int) = None
    last_perception = None
    pending_action = None

    # Common Functions for all Adaptive Agents
    def __init__(self, id, type, rank):
        super().__init__(id, type, rank)
        self.ptype = type

    @abstractmethod
    def see(self, model) -> AdaptiveAgentPerception:
        pass
    @abstractmethod
    def next(self) -> AdaptiveAgentState:
        pass
    @abstractmethod
    def action(self) -> Optional[AdaptiveAgentAction]:
        pass
    @abstractmethod
    def do(self, model):
        pass