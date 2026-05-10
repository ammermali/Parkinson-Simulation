from enum import IntEnum
from repast4py import core


class AlphaState(IntEnum):
    MONOMER = 0
    MISFOLDED = 1
    OLIGOMER = 2
    LEWY_BODIES = 3
    CLEARED = 4


class AlphaAction(IntEnum):
    MISFOLD = 0
    DEGRADE = 1
    AGGREGATE = 2
    INTOXICATE = 3
    TRANSMIT = 4  # future extracellular expansion


class AgAlpha(core.Agent):
    TYPE = 1
    def __init__(self, local_id: int, rank: int):
        super().__init__(local_id, AgAlpha.TYPE, rank)
        self.state = AlphaState.MONOMER
        self.pending_actions = []

    def see(self, env):
        """
        Perception of the alpha-synuclein.
        """
        return (
            env.stress,
            env.clearance,
            env.concentration
        )

    def action(self, perception):
        """
        Action mapping
        """
        pstress, pclearance, pconcentration = perception
        actions = []

        # If the stress is HIGH, the alphasynuclein starts misfolding.
        if self.state == AlphaState.MONOMER:
            if pstress == 1:
                actions.append(AlphaAction.MISFOLD)

        # Actions mapped to the Misfolded state.
        elif self.state == AlphaState.MISFOLDED:

            # If the stress keeps staying high, the protein keeps misfolding.
            if pstress == 1:
                actions.append(AlphaAction.MISFOLD)

            # If clearance mechanism are available, the protein gets degraded.
            if pclearance == 0:
                actions.append(AlphaAction.DEGRADE)

            # If clearance mechanism are overwhelmed, the misfolded proteins aggregate
            elif pclearance == 1:
                actions.append(AlphaAction.AGGREGATE)

        # If the protein has aggregated into oligomers, it keeps aggregating and it raises the toxicity values.
        elif self.state == AlphaState.OLIGOMER:
            actions.append(AlphaAction.INTOXICATE)
            actions.append(AlphaAction.AGGREGATE)

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
        # Monomer -> Misfolded
        if self.state == AlphaState.MONOMER:
            if AlphaAction.MISFOLD in self.pending_actions:
                self.state = AlphaState.MISFOLDED

        # Misfolded -> Cleared or Oligomer
        elif self.state == AlphaState.MISFOLDED:

            # Clearance takes precedence if degradation succeeds
            if AlphaAction.DEGRADE in self.pending_actions:
                self.state = AlphaState.CLEARED

            elif AlphaAction.AGGREGATE in self.pending_actions:
                self.state = AlphaState.OLIGOMER

        # Oligomer -> Lewy Bodies
        elif self.state == AlphaState.OLIGOMER:
            if AlphaAction.AGGREGATE in self.pending_actions:
                self.state = AlphaState.LEWY_BODIES

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
        self.state = AlphaState(data[1])

    def describe(self):
        return {
            "agent_id": self.uid,
            "state": self.state.name,
            "pending_actions": [a.name for a in self.pending_actions]
        }