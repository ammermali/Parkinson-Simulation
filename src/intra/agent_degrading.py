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
        return (
            env.workload,
            env.target
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

        #TODO: define, specify and implement do mapping

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