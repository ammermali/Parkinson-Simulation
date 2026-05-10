from dataclasses import dataclass, asdict, field
from enum import IntEnum

class EnergyState(IntEnum):
    LOW = 0
    HIGH = 1
class ToxicityState(IntEnum):
    CLEAR = 0
    TOXIC = 1
class ClearanceState(IntEnum):
    AVAILABLE = 0
    OVERWHELMED = 1
class StressState(IntEnum):
    NORMAL = 0
    HIGH = 1
class ConcentrationState(IntEnum):
    LOW = 0
    HIGH = 1
class WorkloadState(IntEnum):
    NORMAL = 0
    HIGH = 1
class TargetState(IntEnum):
    NON_TOXIC = 0
    TOXIC = 1


@dataclass
class IntraCellEnv:
    """
    Intracellular environment for a single dopaminergic neuron.

    This environment is shared by:
        - AgAlpha
        - AgMito
        - AgDeg

    It represents the biological intracellular substrate.
    """
    energy: float = 0.0
    toxicity: float = 0.0
    clearance: float = 0.0
    stress: float = 0.0
    concentration: float = 0.0
    workload: float = 0.0
    target: float = 0.0

    def reset(self):
        """
        Restores environment to initial state e0.
        """

        self.energy = 0.0
        self.toxicity = 0.0
        self.clearance = 0.0
        self.stress = 0.0
        self.concentration = 0.0
        self.workload = 0.0
        self.target = 0.0 # TODO ???

    def update(self):
        """
        Applies biological consistency rules after all agents act.
        #TODO: this one needs to be added to the specification and eventually modified accordingly here.
        """
        pass

    def state_vector(self):
        """
        Returns environment state as a vector.
        """
        return (
            float(self.energy),
            float(self.toxicity),
            float(self.clearance),
            float(self.stress),
            float(self.concentration),
            float(self.workload),
            float(self.target),
        )

    def snapshot(self):
        """
        Dictionary representation of the current state.
        """
        return {
            "energy": float(self.energy),
            "toxicity": float(self.toxicity),
            "clearance": float(self.clearance),
            "stress": float(self.stress),
            "concentration": float(self.concentration),
            "workload": float(self.workload),
            "target": float(self.target),
        }

    def save(self):
        return self.state_vector()

    def load(self, data):
        """
        Restore from serialized tuple.
        """
        (
            self.energy,
            self.toxicity,
            self.clearance,
            self.stress,
            self.concentration,
            self.workload,
            self.target,
        ) = data

    def describe(self):
        return {
            "energy": self.energy_state().name,
            "toxicity": self.toxicity_state().name,
            "clearance": self.clearance_state().name,
            "stress": self.stress_state().name,
            "concentration": self.concentration_state().name,
            "workload": self.workload_state().name,
            "target": self.target_state().name,
        }


    def set_healthy(self):
        self.reset()

    def set_parkinsonian_seed(self):
        """
        Experimental pathological initialization:
        introduces early toxic burden.
        """
        self.energy = 0.0
        self.toxicity = 1.0
        self.clearance = 0.0
        self.stress = 1.0
        self.concentration = 1.0
        self.workload = 1.0
        self.target = 1.0

    # functions to increase real-value states
    def increase_energy(self, value: float):
        self.energy = max(0.0, min(1.0, 1.0 + value))
    def increase_toxicity(self, value: float):
        self.toxicity = max(0.0, min(1.0, 1.0 + value))
    def increase_clearance(self, value: float):
        self.clearance = max(0.0, min(1.0, 1.0 + value))
    def increase_stress(self, value: float):
        self.stress = max(0.0, min(1.0, 1.0 + value))
    def increase_concentration(self, value: float):
        self.concentration = max(0.0, min(1.0, 1.0 + value))
    def increase_workload(self, value: float):
        self.workload = max(0.0, min(1.0, 1.0 + value))
    def increase_target(self, value: float):
        self.target = max(0.0, min(1.0, 1.0 + value))


    # method to map from real-values to finite states
    def energy_state(self) -> EnergyState:
        return EnergyState.HIGH if self.energy > 0.5 else EnergyState.LOW
    def toxicity_state(self) -> ToxicityState:
        return ToxicityState.TOXIC if self.toxicity >= 0.7 else ToxicityState.CLEAR
    def clearance_state(self) -> ClearanceState:
        return ClearanceState.OVERWHELMED if self.clearance > 0.5 else ClearanceState.AVAILABLE
    def stress_state(self) -> StressState:
        return StressState.HIGH if self.stress > 0.5 else StressState.NORMAL
    def concentration_state(self) -> ConcentrationState:
        return ConcentrationState.HIGH if self.concentration > 0.5 else ConcentrationState.LOW
    def workload_state(self) -> WorkloadState:
        return WorkloadState.HIGH if self.workload > 0.5 else WorkloadState.NORMAL
    def target_state(self) -> TargetState:
        return TargetState.TOXIC if self.target > 0.5 else TargetState.NON_TOXIC