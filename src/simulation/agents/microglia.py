from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from src.simulation.utils import RNG

# Internal State Set
class MicrogliaState(str, AdaptiveAgentState):
    """Functional extracellular state of a microglial agent."""
    RESTING = "Resting"
    CLEARING = "Clearing"
    ACTIVATED = "Activated"


# Action Set
class MicrogliaAction(str, AdaptiveAgentAction):
    """Actions a microglial agent can apply to the shared environment."""
    SCAN = "scan"
    CLEAR_DEBRIS = "clear_debris"
    INFLAMMATION = "release_inflammation"

# Set of possible perceptions
@dataclass(frozen=True)
class MicrogliaPerception(AdaptiveAgentPerception):
    """Extracellular signals sensed by microglia."""
    position: Optional[DiscretePoint]
    extracellular_debris: float
    inflammation_level: float
    nearby_alpha: float

#Params of the specific Microglia
@dataclass
class MicrogliaConfig:
    """Microglia sensing thresholds and environmental effect rates."""
    per_radius: int
    debris_high_threshold: float
    debris_low_threshold: float
    inflammation_high_threshold: float
    inflammation_low_threshold: float
    nearby_alpha_high_threshold: float
    nearby_alpha_low_threshold: float
    debris_clearance_rate: float
    inflammation_release_rate: float
    move_probability: float

class Microglia(AdaptiveAgent):
    """Extracellular immune agent that clears debris or amplifies inflammation."""
    def __init__(self, local_id: int, rank: int, type_id: int, config: MicrogliaConfig, alpha_type_id:int):
        super().__init__(local_id, type_id, rank)
        self.state: MicrogliaState = MicrogliaState.RESTING
        self.cfg = config
        self.alpha_type_id = alpha_type_id
        self.last_perception: Optional[MicrogliaPerception] = None
        self.pending_action: Optional[MicrogliaAction] = None
        self.rng = RNG

    def see(self, model) -> MicrogliaPerception:
        """Read extracellular debris, inflammation and nearby alpha density."""

        env = model.environment
        position = env.position_of(self)
        if position is None:
            nearby_alpha = 0.0
        else:
            nearby_alpha = env.density_of_type(
                center = position,
                radius = self.cfg.per_radius,
                agent_type=self.alpha_type_id,
                include_center=True
            )

        perception = MicrogliaPerception(position=position, extracellular_debris=env.scalars.extracellular_debris,inflammation_level=env.scalars.inflammation_level,nearby_alpha=nearby_alpha)
        self.last_perception = perception
        return perception

    def next(self) -> MicrogliaState:
        """Update state deterministically from the last perception."""
        if self.last_perception is None:
            raise RuntimeError()
        p = self.last_perception
        if self.state == MicrogliaState.RESTING:
            if p.extracellular_debris >= self.cfg.debris_high_threshold:
                self.state = MicrogliaState.CLEARING
            elif p.inflammation_level >= self.cfg.inflammation_high_threshold or p.nearby_alpha >= self.cfg.nearby_alpha_high_threshold:
                self.state = MicrogliaState.ACTIVATED
        elif self.state == MicrogliaState.CLEARING:
            if p.inflammation_level >= self.cfg.inflammation_high_threshold:
                self.state = MicrogliaState.ACTIVATED
            elif p.extracellular_debris <= self.cfg.debris_low_threshold:
                self.state = MicrogliaState.RESTING
        elif self.state == MicrogliaState.ACTIVATED:
            if p.inflammation_level <= self.cfg.inflammation_low_threshold and p.nearby_alpha <= self.cfg.nearby_alpha_low_threshold and p.extracellular_debris <= self.cfg.debris_low_threshold:
                self.state = MicrogliaState.RESTING
        return self.state

    def action(self) -> MicrogliaAction:
        """Map the current microglial state to one extracellular action."""
        if self.state == MicrogliaState.RESTING:
            self.pending_action = MicrogliaAction.SCAN
        elif self.state == MicrogliaState.CLEARING:
            self.pending_action = MicrogliaAction.CLEAR_DEBRIS
        elif self.state == MicrogliaState.ACTIVATED:
            self.pending_action = MicrogliaAction.INFLAMMATION
        return self.pending_action

    def do(self, model):
        """Apply the selected action to the Substantia Nigra environment."""
        if self.pending_action is None:
            return
        env = model.environment
        action = self.pending_action

        if action == MicrogliaAction.SCAN:
            position = env.position_of(self)
            if position is None:
                return
            if self.rng.random() > self.cfg.move_probability:
                return
            candidate_points = list(env.neighbor_points(position, 1, True))
            if not candidate_points:
                return
            newPos = self.rng.choice(candidate_points)
            env.move_to(self, newPos)

        if action == MicrogliaAction.CLEAR_DEBRIS:
            env.remove_debris(self.cfg.debris_clearance_rate)
        elif action == MicrogliaAction.INFLAMMATION:
            env.add_inflammation(self.cfg.inflammation_release_rate)

    def step(self, model):
        """Run one extracellular microglia phase: see, next, action, do."""
        self.see(model)
        self.next()
        self.action()
        self.do(model)
