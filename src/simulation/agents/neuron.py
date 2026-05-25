from typing import List, Optional
from repast4py.space import DiscretePoint
from torch.distributed.elastic import agent

from src.simulation.agents.adaptiveagent import AdaptiveAgent
from enum import Enum
from dataclasses import dataclass
from src.simulation.substantia_nigra import clamp


# Internal State Set
class NeuronState(str, Enum):
    HEALTHY = "Healthy"
    COMPROMISED = "Compromised"
    APOPTOTIC = "Apoptotic"
    RUPTURED = "Ruptured"


# Action Set
class NeuronAction(str, Enum):
    R_DOPAMINE = "release_dopamine"
    R_ALPHASYNUCLEIN = "release_alphasynuclein"
    DUMP_DEBRIS = "dump_debris"
    A_ALPHASYNUCLEIN = "absorb_alphasynuclein"
    STRESS = "signal_stress"

@dataclass(frozen=True)
class NeuronPerception:
    position: Optional[DiscretePoint]
    nearby_alpha: float
    inflammatory_levels: float
    extracellular_debris: float
    damage: float # TODO remove it
    alpha_burden: float # TODO remove it

@dataclass
class NeuronConfig:
    per_radius: int
    nearby_alpha_high_threshold: float
    inflammation_high_threshold: float
    debris_high_threshold: float
    alpha_burden_release_threshold: float
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
class NeuronInternalEnvironmentConfig:
    width: int
    height: int
    oxidative_stress_decay: float
    aggregate_density_decay: float
    intracellular_debris_decay: float
    internal_damage_oxidative_weight: float
    internal_damage_aggregate_weight: float
    internal_damage_debris_weight: float

@dataclass
class NeuronInternalScalars:
    oxidative_stress: float
    aggregate_density: float
    intracellular_debris: float
    energy_demand: float

@dataclass
class NeuronInternalEffects:
    oxidative_stress_added: float
    aggregate_density_added: float
    debris_added: float

class Neuron(AdaptiveAgent):
    def __init__(self, local_id: int, rank: int, type_id: int, config: NeuronConfig, alpha_type_id:int):
        super().__init__(local_id, type_id, rank)
        self.state = NeuronState.HEALTHY
        self.cfg = config
        self.alpha_type_id = alpha_type_id
        self.last_perception: Optional[NeuronPerception] = None
        self.pending_action: Optional[NeuronAction] = None
        self.damage: float = 0.0
        self.alpha_burden: float = 0.0

        self.agent_registry: List[AdaptiveAgent] = []

    def see(self, model):
        env = model.environment
        position = env.position_of(self)
        nearby_alpha = env.density_of_type(position, self.cfg.per_radius,self.alpha_type_id,include_center=True)
        perception = NeuronPerception(position=position, nearby_alpha=nearby_alpha, inflammatory_levels=env.scalars.inflammation_level, extracellular_debris=env.scalars.extracellular_debris, damage=self.damage, alpha_burden=self.alpha_burden)
        self.last_perception = perception

    def next(self): # TODO redefine according to agent-environemnt specification
        external_stress = self._compute_external_stress(self.last_perception) # TODO ???
        if external_stress <= self.cfg.low_stress_threshold:
            self.damage = clamp(self.damage - self.cfg.damage_recovery_rate)
        else:
            self.damage = clamp(self.damage + external_stress * self.cfg.damage_accumulation_rate)
        if self.damage >= self.cfg.ruptured_threshold:
            self.state = NeuronState.RUPTURED
        elif self.damage >= self.cfg.apoptotic_threshold:
            self.state = NeuronState.APOPTOTIC
        elif self.damage >= self.cfg.compromised_threshold:
            self.state = NeuronState.COMPROMISED
        else:
            self.state = NeuronState.HEALTHY

    def action(self):
        if self.state == NeuronState.RUPTURED:
            self.pending_action = NeuronAction.DUMP_DEBRIS
        elif self.state != NeuronState.APOPTOTIC and self.last_perception.nearby_alpha >= self.cfg.nearby_alpha_high_threshold:
            self.pending_action = NeuronAction.A_ALPHASYNUCLEIN
        elif self.state == NeuronState.APOPTOTIC or self.alpha_burden >= self.cfg.alpha_burden_release_threshold:
            self.pending_action = NeuronAction.R_ALPHASYNUCLEIN
        elif self.state == NeuronState.HEALTHY:
            if self.last_perception.inflammatory_levels >= self.cfg.inflammation_high_threshold:
                self.pending_action = NeuronAction.STRESS
            else:
                self.pending_action = NeuronAction.R_DOPAMINE

    def do(self, model):
        env = model.environment
        if self.pending_action == NeuronAction.R_DOPAMINE:
            env.release_dopamine(self.cfg.dopamine_release_rate)
        elif self.pending_action == NeuronAction.STRESS:
            env.add_inflammation(self.cfg.stress_inflammation_release_rate)
        elif self.pending_action == NeuronAction.DUMP_DEBRIS:
            env.add_debris(self.cfg.debris_release_rate)
        elif self.pending_action == NeuronAction.A_ALPHASYNUCLEIN:
            # TODO: implement the logic for absorbin alpha agents from the external environment
            pass
        elif self.pending_action == NeuronAction.R_ALPHASYNUCLEIN:
            # TODO: implement the logic for releasing of alpha agents on the external environment
            pass
    def step(self, model):
        self.see(model)
        self.next()
        self.action()
        self.do(model)

    def _compute_external_stress(self, perception: NeuronPerception) -> float:
        return clamp(perception.inflammatory_levels * self.cfg.inflammation_damage_weight + perception.extracellular_debris * self.cfg.debris_damage_weight + perception.nearby_alpha * self.cfg.alpha_damage_weight)

class NeuronInternalEnvironment:
    def __init__(self, config: NeuronInternalEnvironmentConfig):
        self.cfg = config
        self.scalars = NeuronInternalScalars
        self.effects = NeuronInternalEffects
        self.agent_registry: List[AdaptiveAgent] = []
        self._locations: dict[AdaptiveAgent, DiscretePoint] = {}
        self._cells: dict[tuple[int,int], list[AdaptiveAgent]] = {}
        self._offset_cache: dict[tuple[int,int], list[tuple[int,int]]] = {}

        def begin_tick(self):
            self.effects = NeuronInternalEffects

        def commit_effects(self):
            cfg = self.config
            old = self.scalars
            eff = self.effects

            self.scalars.oxidative_stress = clamp(old.oxidative_stress + eff.oxidative_stress - cfg.oxidative_stress_decay * old.aggregate_density)
            self.scalarse.intracellular_debris = clamp(old.intracellular_debris + eff.debris_added - cfg.intracellular_debris_decay * old.intracellular_debris)


        # TODO: refactor the following methods to a new EnvUtils.py class
        def add_agent(self, agent: AdaptiveAgent, point: DiscretePoint):
            self.agent_registry.append(agent)
            self._locations[agent] = point
            self._cells.setdefault((point.x, point.y), []).append(agent)

        def remove_agent(self, agent: AdaptiveAgent):
            point = self._locations.get(agent)
            if point is not None:
                key = (point.x, point.y)
                self._cells[key].remove(agent)
                if not self._cells[key]:
                    del self._cells[key]
                del self._locations[agent]

            if agent in self.agent_registry:
                self.agent_registry.remove(agent)

        def position_of(self,agent) -> Optional[DiscretePoint]:
            return self._locations.get(agent, None)

        def move_to(self,agent,point: DiscretePoint) -> Optional[DiscretePoint]:
            if not self._inside_bounds(point.x, point.y):
                return None
            old_point = self._locations.get(agent)
            if old_point is not None:
                old_key = (old_point.x, old_point.y)
                self._cells[old_key].remove(agent)
                if not self._cells[old_key]:
                    del self._cells[old_key]
            self._locations[agent] = point
            self._cells.setdefault(agent.x, agent.y).append(agent)
            return point

        def agents_at(self, point: DiscretePoint) -> List[AdaptiveAgent]:
            return list(self._cells.get(agent.x, agent.y, []))

        def neighbor_points(self, center: DiscretePoint, radius: int = 1, include_center: bool = True):
            for dx, dy in self._get_offsets(radius, include_center):
                x = center.x + dx
                y = center.y + dy
                if self._inside_bounds(x, y):
                    yield DiscretePoint(x, y)

        def count_agents_in_radius(self, center: DiscretePoint, radius: int = 1, agent_type: Optional[int] = None, include_center: bool = True) -> int:
            total = 0
            for point in self.neighbor_points(center, radius, include_center):
                for agent in self.agents_at(point):
                    if agent_type is None or agent.ptype == agent_type:
                        total += 1
            return total