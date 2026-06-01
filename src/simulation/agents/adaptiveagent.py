from abc import abstractmethod, ABC
from typing import Optional
from repast4py import core
from enum import Enum

class AdaptiveAgentAction(Enum):
    """Base enum for action labels emitted by adaptive agents."""
class AdaptiveAgentState(Enum):
    """Base enum for state labels emitted by adaptive agents."""
class AdaptiveAgentPerception:
    """Base class for perception snapshots."""

class AdaptiveAgent(core.Agent, ABC):
    """Repast agent with the project-wide see/next/action/do contract."""
    def __init__(self, id, type, rank):
        super().__init__(id, type, rank)
        self.ptype = type

    @abstractmethod
    def see(self, model) -> AdaptiveAgentPerception:
        """Read the environment and store the current perception."""
    @abstractmethod
    def next(self) -> AdaptiveAgentState:
        """Update internal state from the latest perception."""
    @abstractmethod
    def action(self) -> Optional[AdaptiveAgentAction]:
        """Choose the action to execute during the current tick."""
    @abstractmethod
    def do(self, model):
        """Apply the selected action to the model or local habitat."""