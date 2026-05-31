from typing import Optional
from repast4py.space import DiscretePoint
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass
from src.simulation.utils import RNG
from src.simulation.logger.causal_trace_logger import bind_causal_logger, causal_logger_from

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

        bind_causal_logger(self, model)
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
        old_state = self.state
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
        if old_state != self.state:
            self._log_causal_state_trigger(old_state, p)
        return self.state

    def action(self) -> MicrogliaAction:
        """Map the current microglial state to one extracellular action."""
        if self.state == MicrogliaState.RESTING:
            self.pending_action = MicrogliaAction.SCAN
        elif self.state == MicrogliaState.CLEARING:
            self.pending_action = MicrogliaAction.CLEAR_DEBRIS
        elif self.state == MicrogliaState.ACTIVATED:
            self.pending_action = MicrogliaAction.INFLAMMATION
        logger = causal_logger_from(self)
        if logger is not None:
            logger.action_selection(self, self.pending_action, "microglia_state_action_policy")
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
            draw = self.rng.random()
            if draw > self.cfg.move_probability:
                return
            candidate_points = list(env.neighbor_points(position, 1, True))
            if not candidate_points:
                return
            newPos = self.rng.choice(candidate_points)
            env.move_to(self, newPos)

        if action == MicrogliaAction.CLEAR_DEBRIS:
            env.remove_debris(self.cfg.debris_clearance_rate)
            logger = causal_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    action,
                    "extracellular_debris",
                    -self.cfg.debris_clearance_rate,
                    "negative",
                    "microglia_debris_clearance"
                )
        elif action == MicrogliaAction.INFLAMMATION:
            env.add_inflammation(self.cfg.inflammation_release_rate)
            logger = causal_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    action,
                    "inflammation_level",
                    self.cfg.inflammation_release_rate,
                    "positive",
                    "microglia_inflammation_release"
                )

    def step(self, model):
        """Run one extracellular microglia phase: see, next, action, do."""
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def _log_causal_state_trigger(self, old_state: MicrogliaState, p: MicrogliaPerception) -> None:
        """Log only causal predicates that produced a microglia transition."""

        logger = causal_logger_from(self)
        if logger is None:
            return
        if old_state == MicrogliaState.RESTING and self.state == MicrogliaState.CLEARING:
            source = logger.env_field_node("SN.extracellular_debris", "extracellular_debris", "1_perception", p.extracellular_debris)
            logger.threshold_trigger(
                source,
                self,
                self.state,
                "microglia_clearing_by_debris",
                "MICROGLIA_CLEARING_DEBRIS_HIGH",
                "extracellular_debris >= debris_high_threshold"
            )
        elif self.state == MicrogliaState.ACTIVATED:
            if p.inflammation_level >= self.cfg.inflammation_high_threshold:
                source = logger.env_field_node("SN.inflammation_level", "inflammation_level", "1_perception", p.inflammation_level)
                logger.threshold_trigger(
                    source,
                    self,
                    self.state,
                    "microglia_activation_by_inflammation",
                    "MICROGLIA_ACTIVATION_INFLAMMATION_HIGH",
                    "inflammation_level >= inflammation_high_threshold"
                )
            elif p.nearby_alpha >= self.cfg.nearby_alpha_high_threshold:
                source = logger.env_field_node("SN.nearby_alpha_density", "nearby_alpha_density", "1_perception", p.nearby_alpha)
                logger.threshold_trigger(
                    source,
                    self,
                    self.state,
                    "microglia_activation_by_nearby_alpha",
                    "MICROGLIA_ACTIVATION_ALPHA_HIGH",
                    "nearby_alpha >= nearby_alpha_high_threshold"
                )
        logger.state_transition(self, old_state, self.state, "microglia_state_update")
