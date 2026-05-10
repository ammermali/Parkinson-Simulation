from enum import IntEnum
from repast4py import core


class MitoState(IntEnum):
    HEALTHY = 0
    CONSUMED = 1
    DAMAGED = 2
    DEBRIS = 3

class MitoAction(IntEnum):
    FUSION = 0
    FISSION = 1
    TRANSPORT = 2
    PRODUCE_ATP = 3
    RECYCLE = 4
    FAIL_AND_IMPAIR = 5


class AgMito(core.Agent):
    TYPE = 1
    def __init__(self, local_id: int, rank: int):
        super().__init__(local_id, AgMito.TYPE, rank)
        self.state = MitoState.HEALTHY
        self.pending_actions = []

    def see(self, env):
        """
        Perception of the mitochondrion.
        """
        # perceive discretized states for consistency with env
        return (
            env.energy_state().value,
            env.toxicity_state().value,
            env.clearance_state().value,
        )

    def action(self, perception):
        """
        Action mapping.
        """
        penergy, ptoxicity, pclearance = perception
        actions = []

        # If the energy request is HIGH, the mitochondrion starts producing ATP.
        if self.state == MitoState.HEALTHY:
            if penergy == 1:
                actions.append(MitoAction.PRODUCE_ATP)
                actions.append(MitoAction.FISSION)

        # Actions mapped to the Consumed state.
        elif self.state == MitoState.CONSUMED:
            # If the energy request is still high and there's no toxicity, the mitochondria fuse.
            if penergy == 1 and ptoxicity == 0:
                actions.append(MitoAction.FUSION)

        #Actions mapped to the Damaged state.
        elif self.state == MitoState.DAMAGED:

            # If clearance mechanism are available, the mitochondrion gets recycled.
            if pclearance == 0:
                actions.append(MitoAction.RECYCLE)

            # If clearance mechanism are overwhelmed, the mitochondrion fails.
            elif pclearance == 1:
                actions.append(MitoAction.FAIL_AND_IMPAIR)

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
                if action == MitoAction.PRODUCE_ATP:
                    # Produces energy, lowering the requirement.
                    env.increase_energy(-0.3)
                    # ATP production generates some stress.
                    env.increase_stress(0.02)

                elif action == MitoAction.FUSION or action == MitoAction.FISSION:
                    env.increase_stress(-0.05)

                elif action == MitoAction.TRANSPORT:
                    # Transport consumes energy.
                    env.increase_energy(0.05)

                elif action == MitoAction.RECYCLE:
                    # Mitophagia consumes the damaged mitochondrion, decreasing toxicity, but increasing clearance and workload.
                    env.increase_workload(0.15)
                    env.increase_clearance(0.1)
                    env.increase_toxicity(-0.15)

                elif action == MitoAction.FAIL_AND_IMPAIR:
                    # Failure releases pro-apoptosis factors and stress, causing a peak in toxicity.
                    env.increase_toxicity(0.3)
                    env.increase_stress(0.3)
                    # The energy requirement also increases.
                    env.increase_energy(0.4)

    def next(self, perception):
        """
        Internal automaton transition.
        """
        penergy, ptoxicity, pclearance = perception

        # Healthy -> Consumed
        if self.state == MitoState.HEALTHY and penergy == 1:
            self.state = MitoState.CONSUMED

        # Consumed -> Healthy
        if self.state == MitoState.CONSUMED and ptoxicity == 0:
            self.state = MitoState.HEALTHY

        # Consumed -> Damaged
        if self.state == MitoState.CONSUMED and ptoxicity == 1:
            self.state = MitoState.DAMAGED

        # Damaged -> Healthy
        if self.state == MitoState.DAMAGED and pclearance == 0:
            self.state = MitoState.HEALTHY

        # Damaged -> Debris
        if self.state == MitoState.DAMAGED and pclearance == 1:
            self.state = MitoState.DEBRIS

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
        self.state = MitoState(data[1])

    def describe(self):
        return {
            "agent_id": self.uid,
            "state": self.state.name,
            "pending_actions": [a.name for a in self.pending_actions]
        }