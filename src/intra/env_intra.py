from dataclasses import dataclass, asdict
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
    energy: int = EnergyState.LOW
    toxicity: int = ToxicityState.CLEAR
    clearance: int = ClearanceState.AVAILABLE
    stress: int = StressState.NORMAL
    concentration: int = ConcentrationState.LOW
    workload: int = WorkloadState.NORMAL
    target: int = TargetState.NON_TOXIC

    def reset(self):
        """
        Restores environment to initial state e0.
        """
        self.energy = EnergyState.LOW
        self.toxicity = ToxicityState.CLEAR
        self.clearance = ClearanceState.AVAILABLE
        self.stress = StressState.NORMAL
        self.concentration = ConcentrationState.LOW
        self.workload = WorkloadState.NORMAL
        self.target = TargetState.NON_TOXIC

    def update(self):
        """
        Applies biological consistency rules after all agents act.
        #TODO: this one needs to be added to the specification and eventually modified accordingly here.
        """

        # Toxic intracellular burden implies cellular stress
        if self.toxicity == ToxicityState.TOXIC:
            self.stress = StressState.HIGH
            self.target = TargetState.TOXIC

        # High alpha-syn concentration increases degradation burden
        if self.concentration == ConcentrationState.HIGH:
            self.workload = WorkloadState.HIGH

        # Combined toxic target + high workload overwhelms clearance
        if (
            self.workload == WorkloadState.HIGH
            and self.target == TargetState.TOXIC
        ):
            self.clearance = ClearanceState.OVERWHELMED

        # Recovery fallback
        if self.toxicity == ToxicityState.CLEAR:
            self.stress = StressState.NORMAL
            self.target = TargetState.NON_TOXIC

        if self.concentration == ConcentrationState.LOW:
            self.workload = WorkloadState.NORMAL

        if (
            self.workload == WorkloadState.NORMAL
            and self.target == TargetState.NON_TOXIC
        ):
            self.clearance = ClearanceState.AVAILABLE

    def state_vector(self):
        """
        Returns environment state as a vector.
        """
        return (
            int(self.energy),
            int(self.toxicity),
            int(self.clearance),
            int(self.stress),
            int(self.concentration),
            int(self.workload),
            int(self.target),
        )

    def snapshot(self):
        """
        Dictionary representation of the current state.
        """
        return {
            "energy": int(self.energy),
            "toxicity": int(self.toxicity),
            "clearance": int(self.clearance),
            "stress": int(self.stress),
            "concentration": int(self.concentration),
            "workload": int(self.workload),
            "target": int(self.target),
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
            "energy": EnergyState(self.energy).name,
            "toxicity": ToxicityState(self.toxicity).name,
            "clearance": ClearanceState(self.clearance).name,
            "stress": StressState(self.stress).name,
            "concentration": ConcentrationState(self.concentration).name,
            "workload": WorkloadState(self.workload).name,
            "target": TargetState(self.target).name,
        }


    """
    Methods with only experimental purposes.
    """

    def set_healthy(self):
        self.reset()

    def set_parkinsonian_seed(self):
        """
        Experimental pathological initialization:
        introduces early toxic burden.
        """
        self.energy = EnergyState.LOW
        self.toxicity = ToxicityState.TOXIC
        self.clearance = ClearanceState.AVAILABLE
        self.stress = StressState.HIGH
        self.concentration = ConcentrationState.HIGH
        self.workload = WorkloadState.HIGH
        self.target = TargetState.TOXIC