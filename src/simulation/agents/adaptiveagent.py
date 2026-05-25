from abc import abstractmethod, ABC

from repast4py import core

class AdaptiveAgent(core.Agent, ABC):
    # Common Fields for all Adaptive Agents
    state = None
    pt: (int,int) = None
    last_perception = None
    pending_action = None

    # Common Functions for all Adaptive Agents
    @abstractmethod
    def see(self, model):
        pass
    @abstractmethod
    def next(self):
        pass
    @abstractmethod
    def action(self):
        pass
    @abstractmethod
    def do(self, model):
        pass