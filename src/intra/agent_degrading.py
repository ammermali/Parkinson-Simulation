from enum import IntEnum
from repast4py import core


class DegState(IntEnum):
    INACTIVE = 0
    ACTIVE = 1
    STRESSED = 2
    OVERWHELMED = 3

class DegAction(IntEnum):
    SCAN = 0
    DEGRADE = 1
    UPREGULATE = 2
    FAIL = 3


class AgDeg(core.Agent):
    TYPE = 1
    def __init__(self, local_id: int, rank: int):
        super().__init__(local_id, AgDeg.TYPE, rank)
        self.state = DegState.INACTIVE
        self.pending_actions = []

    def see(self, env):
        """
        Perception of the degrading machinery.
        """
        # return discretized values for compatibility with transitions
        return (
            env.workload_state().value,
            env.target_state().value
        )

    def action(self, perception):
        """
        Action mapping.
        """
        pworkload, ptarget = perception
        actions = []

        # If the DegMach is inactive, it scans for possible bodies to degrade.
        if self.state == DegState.INACTIVE:
            actions.append(DegAction.SCAN)

        # If the DegMach is active, it degrades targets.
        if self.state == DegState.ACTIVE:
            actions.append(DegAction.DEGRADE)

        # If the DegMach is stressed, it upregulates and degrades targets.
        if self.state == DegState.STRESSED:
            actions.append(DegAction.UPREGULATE)
            actions.append(DegAction.DEGRADE)

        # If the DegMach is overwhelmed, it fails.
        if self.state == DegState.OVERWHELMED:
            actions.append(DegAction.FAIL)

        self.pending_actions = actions
        return actions

    def do(self, env):
        """
        Applies chosen actions to intra-cellular environment.
        """

    def do(self, env):
        """
        Applies chosen actions to intra-cellular environment.
        """
        for action in self.pending_actions:
            if action == DegAction.SCAN:
                # Scan requires energy.
                env.increase_energy(0.01)
                # TODO: logic for scanning and individuate targets
            elif action == DegAction.DEGRADE:
                # Degradation efficiency depends on the state of the agent.
                if self.state == DegState.ACTIVE:
                    inefficiency = 0.2
                elif self.state == DegState.STRESSED:
                    inefficiency = 0.35
                else:
                    inefficiency = 0.1
                env.increase_toxicity(-inefficiency)
                env.increase_concentration(-0.05*inefficiency)
                env.increase_target(-0.1*inefficiency)
                env.increase_clearance(-0.1)
                env.increase_energy(0.05)

            elif action == DegAction.UPREGULATE:
                # It upregulates itself, consuming more energy and increasing cellular stress.
                env.increase_energy(0.15)
                env.increase_stress(0.05)

            elif action == DegAction.FAIL:
                # If system fails, toxicity and workload accumulates
                env.increase_toxicity(0.2)
                env.increase_workload(0.2)

    def next(self, perception):
        """
        Internal automaton transition.
        """
        pworkload, ptarget = perception


        # Inactive -> Active
        if self.state == DegState.INACTIVE and pworkload == 1:
            self.state = DegState.ACTIVE

        # Active -> Inactive
        if self.state == DegState.ACTIVE and pworkload == 0 and ptarget == 0:
            self.state = DegState.INACTIVE

        # Active -> Stressed
        if self.state == DegState.ACTIVE and pworkload == 1 and ptarget == 0:
            self.state = DegState.STRESSED

        # Active -> Overwhelmed
        if self.state == DegState.ACTIVE and ptarget == 1:
            self.state = DegState.OVERWHELMED

        # Stressed -> Active
        if self.state == DegState.STRESSED and ptarget == 0 and pworkload == 0:
            self.state = DegState.ACTIVE

        # Stressed -> Overwhelmed
        if self.state == DegState.STRESSED and ptarget == 1:
            self.state = DegState.OVERWHELMED


    def step(self, env):
        """
        Full synchronous local step:
            1. perceive
            2. choose action
            3. apply
            4. transition
        """
        perception = self.see(env)
        self.action(perception)
        self.do(env)
        self.next(perception)

    def save(self):
        return (self.uid, int(self.state))

    def update(self, data):
        self.state = DegState(data[1])

    def describe(self):
        return {
            "agent_id": self.uid,
            "state": self.state.name,
            "pending_actions": [a.name for a in self.pending_actions]
        }