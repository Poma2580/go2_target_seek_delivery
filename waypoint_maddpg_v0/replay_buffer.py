"""Numpy replay buffer for the discrete two-agent task."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, action_dim, device):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.observations = np.zeros(
            (capacity, num_agents, obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (capacity, num_agents, action_dim), dtype=np.float32
        )
        self.action_masks = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.next_action_masks = np.zeros_like(self.actions, dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observations,
        actions,
        action_masks,
        rewards,
        next_observations,
        next_action_masks,
        done,
    ):
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.action_masks[self.position] = action_masks
        self.rewards[self.position] = rewards
        self.next_observations[self.position] = next_observations
        self.next_action_masks[self.position] = next_action_masks
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("not enough replay samples")
        indices = np.random.randint(0, self.size, size=batch_size)

        def tensor(array):
            return torch.as_tensor(array[indices], dtype=torch.float32, device=self.device)

        return (
            tensor(self.observations),
            tensor(self.actions),
            tensor(self.action_masks),
            tensor(self.rewards),
            tensor(self.next_observations),
            tensor(self.next_action_masks),
            tensor(self.dones),
        )

    def __len__(self):
        return self.size
