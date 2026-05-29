from typing import List, Optional, Iterable
from repast4py.space import DiscretePoint
from src.simulation.utils.grid import LocalGrid, clamp
from src.simulation.agents.adaptiveagent import AdaptiveAgent, AdaptiveAgentState, AdaptiveAgentAction, AdaptiveAgentPerception
from dataclasses import dataclass


# Internal State Set
class NeuronState(str, AdaptiveAgentState):
    HEALTHY = "Healthy"
    COMPROMISED = "Compromised"
    APOPTOTIC = "Apoptotic"
    RUPTURED = "Ruptured"


# Action Set
class NeuronAction(str, AdaptiveAgentAction):
    R_DOPAMINE = "release_dopamine"
    R_ALPHASYNUCLEIN = "release_alphasynuclein"
    DUMP_DEBRIS = "dump_debris"
    A_ALPHASYNUCLEIN = "absorb_alphasynuclein"
    STRESS = "signal_stress"

# Perception
@dataclass(frozen=True)
class NeuronPerception(AdaptiveAgentPerception):
    # External Perception
    position: Optional[DiscretePoint]
    nearby_alpha: float
    inflammatory_levels: float
    extracellular_debris: float

    # Internal Perception
    oxidative_stress: float
    intracellular_debris: float
    energy_demand: float

    # Derived values
    internal_damage: float
    alpha_load: float

    # Cumulative damage value
    cell_damage: float

@dataclass
class NeuronConfig:
    per_radius: int
    nearby_alpha_high_threshold: float
    inflammation_high_threshold: float
    debris_high_threshold: float
    alpha_load_release_threshold: float
    damage_accumulation_rate: float
    damage_recovery_rate: float
    low_stress_threshold: float
    inflammation_damage_weight: float
    debris_damage_weight: float
    alpha_damage_weight: float
    compromised_threshold: float
    apoptotic_threshold: float
    ruptured_threshold: float
    dopamine_release_rate: float
    stress_inflammation_release_rate: float
    debris_release_rate: float
    alpha_absorption_rate: float
    alpha_release_amount: float

@dataclass
class NeuronInternalConfig:
    width: int = 10
    height: int = 10
    oxidative_stress_decay: float = 0.01
    intracellular_debris_decay: float = 0.005
    internal_damage_oxidative_weight: float = 0.4
    internal_damage_aggregate_weight: float = 0.4
    internal_damage_debris_weight: float = 0.2

@dataclass
class NeuronInternalScalars:
    oxidative_stress: float = 0.0
    aggregate_density: float = 0.0
    intracellular_debris: float = 0.0
    energy_demand: float = 0.5

@dataclass
class NeuronInternalEffects:
    oxidative_stress_added: float = 0.0
    aggregate_density_added: float = 0.0
    debris_added: float = 0.0

class Neuron(AdaptiveAgent):
    def __init__(
            self,
            local_id: int,
            rank: int,
            type_id: int,
            config: NeuronConfig,
            alpha_type_id:int):

        super().__init__(local_id, type_id, rank)

        # Adaptive agent fields
        self.state = NeuronState.HEALTHY
        self.cfg = config
        self.alpha_type_id = alpha_type_id
        self.last_perception: Optional[NeuronPerception] = None
        self.pending_action: Optional[NeuronAction] = None

        # Cumulative value for cell damage
        self.cell_damage: float = 0.0

        # Environmental state
        self.internal_cfg = NeuronInternalConfig()
        self.internal_scalars = NeuronInternalScalars()
        self.internal_effects = NeuronInternalEffects()

        # Local (Environmental) Grid
        self.grid = LocalGrid(
            width = self.internal_cfg.width,
            height = self.internal_cfg.height
        )

        # Lysosome targetting bridging
        self.degradation_targets: list[AdaptiveAgent] = []
        self.degradation_assignment: dict[AdaptiveAgent, AdaptiveAgent] = {}

    def see(self, model):
        env = model.environment
        position = env.position_of(self)
        if position is None:
            nearby_alpha = 0.0
        else:
            nearby_alpha = env.density_of_type(center=position, radius=self.cfg.per_radius, agent_type=self.alpha_type_id,include_center=True)
        perception = NeuronPerception(
            position=position,
            nearby_alpha=nearby_alpha,
            inflammatory_levels=env.scalars.inflammation_level,
            extracellular_debris=env.scalars.extracellular_debris,
            oxidative_stress=self.internal_scalars.oxidative_stress,
            intracellular_debris=self.internal_scalars.intracellular_debris,
            energy_demand=self.internal_scalars.energy_demand,
            internal_damage=self.compute_internal_damage(),
            alpha_load=self.compute_alpha_load(),
            cell_damage=self.cell_damage
        )
        self.last_perception = perception
        return perception

    def next(self):
        if self.last_perception is None:
            raise RuntimeError()
        p = self.last_perception
        external_stress = self._compute_external_stress(p)
        internal_damage = p.internal_damage

        total_stress = clamp(0.5*external_stress + 0.5*internal_damage) # Mean between external and internal stress

        if total_stress <= self.cfg.low_stress_threshold:
            self.cell_damage = clamp(self.cell_damage - self.cfg.damage_recovery_rate)
        else:
            self.cell_damage = clamp(self.cell_damage + total_stress * self.cfg.damage_accumulation_rate)

        if self.cell_damage >= self.cfg.ruptured_threshold:
            self.state = NeuronState.RUPTURED
        elif self.cell_damage >= self.cfg.apoptotic_threshold:
            self.state = NeuronState.APOPTOTIC
        elif self.cell_damage >= self.cfg.compromised_threshold:
            self.state = NeuronState.COMPROMISED
        else:
            self.state = NeuronState.HEALTHY
        return self.state

    def action(self):
        if self.last_perception is None:
            raise RuntimeError()
        p = self.last_perception
        if self.state == NeuronState.RUPTURED:
            self.pending_action = NeuronAction.DUMP_DEBRIS

        elif self.state != NeuronState.APOPTOTIC and p.nearby_alpha >= self.cfg.nearby_alpha_high_threshold:
            self.pending_action = NeuronAction.A_ALPHASYNUCLEIN

        elif self.state == NeuronState.APOPTOTIC or p.alpha_load >= self.cfg.alpha_load_release_threshold:
            self.pending_action = NeuronAction.R_ALPHASYNUCLEIN

        elif self.state == NeuronState.HEALTHY:
            if p.inflammatory_levels >= self.cfg.inflammation_high_threshold:
                self.pending_action = NeuronAction.STRESS
            else:
                self.pending_action = NeuronAction.R_DOPAMINE
        else:
            self.pending_action = NeuronAction.STRESS

    def do(self, model):
        if self.last_perception is None:
            pass
        env = model.environment
        if self.pending_action == NeuronAction.R_DOPAMINE:
            env.release_dopamine(self.cfg.dopamine_release_rate)
        elif self.pending_action == NeuronAction.STRESS:
            env.add_inflammation(self.cfg.stress_inflammation_release_rate)
        elif self.pending_action == NeuronAction.DUMP_DEBRIS:
            env.add_debris(self.internal_scalars.intracellular_debris)
            self.internal_scalars.intracellular_debris = 0.0
        elif self.pending_action == NeuronAction.A_ALPHASYNUCLEIN:
            self.absorb_alpha(model)
        elif self.pending_action == NeuronAction.R_ALPHASYNUCLEIN:
            self.release_alpha(model)

    def step(self, model):
        self.begin_tick()
        if self.grid is not None and self.grid.agent_registry is not None:
            for agent in list(self.grid.agent_registry):
                if hasattr(agent, "step"):
                    agent.step(model)
        self.commit_effects()
        super().step(model)

    def begin_tick(self):
        self.internal_effects = NeuronInternalEffects(0,0,0)

    def commit_effects(self):
        cfg = self.internal_cfg
        s = self.internal_scalars
        e = self.internal_effects
        s.oxidative_stress = clamp(s.oxidative_stress + e.oxidative_stress_added - cfg.oxidative_stress_decay * s.oxidative_stress)
        s.aggregate_density = clamp(s.aggregate_density + e.aggregate_density_added)
        s.intracellular_debris = clamp(s.intracellular_debris + e.debris_added - cfg.intracellular_debris_decay * s.intracellular_debris)

    def _compute_external_stress(self, perception: NeuronPerception) -> float:
        return clamp(perception.inflammatory_levels * self.cfg.inflammation_damage_weight + perception.extracellular_debris * self.cfg.debris_damage_weight + perception.nearby_alpha * self.cfg.alpha_damage_weight)

    def compute_alpha_load(self) -> float:
        width = max(1, self.internal_cfg.width)
        height = max(1, self.internal_cfg.height)
        capacity = width * height
        aggregate_score = sum(
            self.aggregate_weight(agent)
            for agent in self.grid.agent_registry
        )
        return clamp(aggregate_score / capacity)

    def compute_internal_damage(self) -> float:
        cfg = self.internal_cfg
        s = self.internal_scalars
        aggregate_density = self.compute_alpha_load()
        value = (
            cfg.internal_damage_oxidative_weight * s.oxidative_stress +
            cfg.internal_damage_aggregate_weight * aggregate_density +
            cfg.internal_damage_debris_weight * s.intracellular_debris
        )
        return clamp(value)

    def absorb_alpha(self, model):
        pass # TODO

    def release_alpha(self, model):
        pass # TODO

    # Degradation Buffer Functions
    def register_degradation_target(self, agent: AdaptiveAgent):
        if agent in self.grid.agent_registry and agent not in self.degradation_targets:
            self.degradation_targets.append(agent)

    def available_degradation_targets(self) -> list[AdaptiveAgent]:
        assigned_targets = set(self.degradation_assignment.values())
        return [target for target in self.degradation_targets if target in self.grid.agent_registry and target not in assigned_targets]

    def assign_degradation_target(self, lysosome: AdaptiveAgent, target: AdaptiveAgent):
        if lysosome not in self.grid.agent_registry or target not in self.grid.agent_registry:
            pass
        if target in self.degradation_assignment.values():
            pass
        self.degradation_assignment[lysosome] = target
        if target in self.degradation_targets:
            self.degradation_targets.remove(target)

    def target_for(self, lysosome: AdaptiveAgent) -> Optional[AdaptiveAgent]:
        return self.degradation_assignment.get(lysosome)

    def clear_degradation_assignment(self, lysosome: AdaptiveAgent):
        if lysosome in self.degradation_assignment:
            del self.degradation_assignment[lysosome]

    def is_target_assigned(self, target: AdaptiveAgent) -> bool:
        return target in self.degradation_assignment.values()


    # Internal Scalar Functions
    def add_oxidative_stress(self, amount: float):
        self.internal_effects.oxidative_stress_added += amount

    def oxidative_stress_at(self, position: Optional[DiscretePoint] = None) -> float:
        return self.internal_scalars.oxidative_stress # TODO: potential future cell-by-cell expansion. Not required as of right now

    def local_aggregate_density_at(
            self,
            position: Optional[DiscretePoint] = None,
            radius: int = 1,
            include_center: bool = True
    ) -> float:
        if position is None:
            return 0.0
        points = list(
            self.grid.neighbor_points(position, radius, include_center)
        )

        if not points:
            return 0.0

        aggregate_score = 0.0
        for point in points:
            for agent in self.grid.agents_at(point):
                aggregate_score += self.aggregate_weight(agent)
        return clamp(aggregate_score / len(points))

    # Local Grid Functions
    def add_agent(self, agent: AdaptiveAgent, point: DiscretePoint):
        self.grid.add_agent(agent, point)

    def remove_agent(self, agent: AdaptiveAgent):
        self.grid.remove_agent(agent)
        self.clear_degradation_assignment(agent)
        self.degradation_targets.remove(agent)
        # TODO remove from degradation logic

    def position_of(self, agent: AdaptiveAgent) -> Optional[DiscretePoint]:
        return self.grid.position_of(agent)

    def move_to(self, agent: AdaptiveAgent, point: DiscretePoint) -> Optional[DiscretePoint]:
        return self.grid.move_to(agent, point)

    def agents_at(self, point: DiscretePoint) -> List[AdaptiveAgent]:
        return self.grid.agents_at(point)

    def agents_in_radius(self, center: DiscretePoint, radius: int) -> Iterable:
        return self.grid.agents_in_radius(center, radius)

    def neighbor_points(self, center: DiscretePoint, radius: int) -> Iterable[DiscretePoint]:
        return self.grid.neighbor_points(center, radius)

    def count_agents_in_radius(self, center: DiscretePoint, radius: int, agent_type: Optional[int] = None) -> int:
        return self.grid.count_agents_in_radius(center, radius, agent_type)

    def density_of_type(self, center: DiscretePoint, radius: int, agent_type: Optional[int] = None, include_center: bool = True) -> float:
        return self.grid.density_of_type(center, radius, agent_type, include_center)

    def aggregate_weight(self, agent: AdaptiveAgent) -> float:
        state = getattr(agent, "state", None)
        state_value = getattr(state, "value", 0.0)
        if state_value == "Misfolded":
            return 0.25 # TODO
        elif state_value == "Oligomer":
            return 0.5 # TODO
        if state_value == "LewyBody":
            return 1.0
        return 0.0