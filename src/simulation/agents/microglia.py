from typing import Optional
from src.simulation.agents.structure import MicrogliaConfig, MicrogliaAction, MicrogliaState, MicrogliaPerception, AdaptiveAgent
from src.simulation.utils import RNG, clamp
from src.simulation.logger.agent_logging import bind_event_logger, event_logger_from

class Microglia(AdaptiveAgent):
    """Extracellular immune agent that clears debris or amplifies inflammation."""
    def __init__(self, local_id: int, rank: int, type_id: int, config: MicrogliaConfig, alpha_type_id:int):
        super().__init__(local_id, type_id, rank)
        self.state: MicrogliaState = MicrogliaState.RESTING
        self.cfg = config
        self.alpha_type_id = alpha_type_id
        self.last_perception: Optional[MicrogliaPerception] = None
        self.pending_action: Optional[MicrogliaAction] = None
        self.last_transition_sample = {}
        self.rng = RNG

    def see(self, model) -> MicrogliaPerception:
        """Read extracellular debris, inflammation and nearby alpha density."""

        bind_event_logger(self, model)
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
        """Update state probabilistically from the last perception.
        The transition probabilities are still threshold-driven, but each
        microglial agent samples its own transition. This avoids a fully
        synchronized population response when all agents read the same global
        debris and inflammation scalars.
        """
        if self.last_perception is None:
            raise RuntimeError()
        old_state = self.state
        p = self.last_perception
        self.last_transition_sample = {}
        debris_pressure = self._debris_pressure(p)
        inflammation_pressure = self._inflammation_pressure(p)
        alpha_pressure = self._alpha_pressure(p)
        activation_pressure = max(inflammation_pressure, alpha_pressure)
        if self.state == MicrogliaState.RESTING:
            if debris_pressure > 0 and self._sample_transition(
                "resting_to_clearing",
                self.cfg.clearing_transition_rate * debris_pressure,
            ):
                self.state = MicrogliaState.CLEARING
            elif activation_pressure > 0 and self._sample_transition(
                "resting_to_activated",
                self.cfg.activation_transition_rate * activation_pressure,
            ):
                self.state = MicrogliaState.ACTIVATED
        elif self.state == MicrogliaState.CLEARING:
            if activation_pressure > 0 and self._sample_transition(
                "clearing_to_activated",
                self.cfg.activation_transition_rate * activation_pressure,
            ):
                self.state = MicrogliaState.ACTIVATED
            elif debris_pressure == 0 and self._sample_transition(
                "clearing_to_resting",
                self.cfg.recovery_transition_rate,
            ):
                self.state = MicrogliaState.RESTING
        elif self.state == MicrogliaState.ACTIVATED:
            if activation_pressure == 0:
                if debris_pressure > 0 and self._sample_transition(
                    "activated_to_clearing",
                    self.cfg.clearing_transition_rate * debris_pressure,
                ):
                    self.state = MicrogliaState.CLEARING
                elif debris_pressure == 0 and self._sample_transition(
                    "activated_to_resting",
                    self.cfg.recovery_transition_rate,
                ):
                    self.state = MicrogliaState.RESTING
        if old_state != self.state:
            self._log_event_state_trigger(old_state, p)
        return self.state

    def action(self) -> MicrogliaAction:
        """Map the current microglial state to one extracellular action."""
        if self.state == MicrogliaState.RESTING:
            self.pending_action = MicrogliaAction.SCAN
        elif self.state == MicrogliaState.CLEARING:
            self.pending_action = MicrogliaAction.CLEAR_DEBRIS
        elif self.state == MicrogliaState.ACTIVATED:
            if self._should_release_inflammation():
                self.pending_action = MicrogliaAction.INFLAMMATION
            else:
                self.pending_action = MicrogliaAction.SCAN
        logger = event_logger_from(self)
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
            logger = event_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    action,
                    "extracellular_debris",
                    -self.cfg.debris_clearance_rate,
                    "microglia_debris_clearance"
                )
        elif action == MicrogliaAction.INFLAMMATION:
            env.add_inflammation(self.cfg.inflammation_release_rate)
            logger = event_logger_from(self)
            if logger is not None:
                logger.field_effect(
                    self,
                    action,
                    "inflammation_level",
                    self.cfg.inflammation_release_rate,
                    "microglia_inflammation_release"
                )

    def step(self, model):
        """Run one extracellular microglia phase: see, next, action, do."""
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def _log_event_state_trigger(self, old_state: MicrogliaState, p: MicrogliaPerception) -> None:
        """Log only event predicates that produced a microglia transition."""

        logger = event_logger_from(self)
        if logger is None:
            return
        if self.state == MicrogliaState.CLEARING:
            source = logger.env_field_node("SN.extracellular_debris", "extracellular_debris", "1_perception", p.extracellular_debris)
            logger.threshold_trigger(
                source,
                self,
                self.state,
                "microglia_clearing_by_debris_pressure",
                "MICROGLIA_CLEARING_DEBRIS_PRESSURE",
                "extracellular_debris pressure > 0"
            )
        elif self.state == MicrogliaState.ACTIVATED:
            if self._inflammation_pressure(p) > 0:
                source = logger.env_field_node("SN.inflammation_level", "inflammation_level", "1_perception", p.inflammation_level)
                logger.threshold_trigger(
                    source,
                    self,
                    self.state,
                    "microglia_activation_by_inflammation",
                    "MICROGLIA_ACTIVATION_INFLAMMATION_HIGH",
                    "inflammation pressure > 0"
                )
            elif self._alpha_pressure(p) > 0:
                source = logger.env_field_node("SN.nearby_alpha_density", "nearby_alpha_density", "1_perception", p.nearby_alpha)
                logger.threshold_trigger(
                    source,
                    self,
                    self.state,
                    "microglia_activation_by_nearby_alpha",
                    "MICROGLIA_ACTIVATION_ALPHA_HIGH",
                    "nearby_alpha pressure > 0"
                )
        logger.state_transition(
            self,
            old_state,
            self.state,
            "microglia_state_update",
            probability=self.last_transition_sample.get("probability"),
            rng_value=self.last_transition_sample.get("draw")
        )

    def _sample_transition(self, check: str, probability: float) -> bool:
        """Sample one microglial transition probability."""
        probability = clamp(probability)
        draw = self.rng.random()
        self.last_transition_sample = {
            "check": check,
            "probability": probability,
            "draw": draw
        }
        return probability >= 1.0 or draw < probability

    def _debris_pressure(self, p: MicrogliaPerception) -> float:
        """Return normalized debris pressure above the low threshold."""
        return self._normalized_pressure(
            p.extracellular_debris,
            self.cfg.debris_low_threshold,
            self.cfg.debris_high_threshold
        )

    def _inflammation_pressure(self, p: MicrogliaPerception) -> float:
        """Return normalized inflammatory pressure above the low threshold."""
        return self._normalized_pressure(
            p.inflammation_level,
            self.cfg.inflammation_low_threshold,
            self.cfg.inflammation_high_threshold
        )

    def _alpha_pressure(self, p: MicrogliaPerception) -> float:
        """Return normalized nearby-alpha pressure above the low threshold."""
        return self._normalized_pressure(
            p.nearby_alpha,
            self.cfg.nearby_alpha_low_threshold,
            self.cfg.nearby_alpha_high_threshold
        )

    def _should_release_inflammation(self) -> bool:
        """Return whether Activated microglia should release inflammation now."""
        if self.last_perception is None:
            return True
        pressure = max(self._alpha_pressure(self.last_perception), self._inflammation_pressure(self.last_perception), 0.5 * self._debris_pressure(self.last_perception))
        return pressure >= self.cfg.inflammatory_action_threshold

    def _normalized_pressure(self, value: float, low: float, high: float) -> float:
        """Normalize a scalar between low and high thresholds."""
        if high <= low:
            return 1.0 if value >= high else 0.0
        return clamp((value - low) / (high - low))
