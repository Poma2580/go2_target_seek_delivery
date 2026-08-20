"""MADDPG variant with one shared follower actor for all agents.

The centralized critics remain agent-specific, but every agent calls and
updates the same actor/target-actor.  This is useful for symmetric formation
tasks where go2/go3 should learn one common "go to my slot" policy and the
observation/role/slot_rel tells the shared actor which side it is on.
"""

from .maddpg import MADDPG


class MADDPGSharedActor(MADDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_agents <= 1:
            return

        self.shared_actor = self.agents[0].actor
        self.shared_actor_target = self.agents[0].actor_target
        self.shared_actor_optimizer = self.agents[0].actor_optimizer

        for agent in self.agents:
            agent.actor = self.shared_actor
            agent.actor_target = self.shared_actor_target
            agent.actor_optimizer = self.shared_actor_optimizer

    def update_targets(self):
        """Soft-update shared actor once, and each centralized critic once."""
        if self.num_agents > 0:
            self.soft_update(self.shared_actor_target, self.shared_actor)
        for agent in self.agents:
            self.soft_update(agent.critic_target, agent.critic)
