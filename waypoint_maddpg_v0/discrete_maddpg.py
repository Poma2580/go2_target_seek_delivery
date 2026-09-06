"""Centralized-critic MADDPG with Gumbel-Softmax discrete actors."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(layer, output=False):
    if output:
        nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    else:
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
    nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        _init_layer(self.net[0])
        _init_layer(self.net[2])
        _init_layer(self.net[4], output=True)

    def forward(self, observation):
        return self.net(observation)


class Critic(nn.Module):
    def __init__(self, joint_obs_dim, joint_action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        _init_layer(self.net[0])
        _init_layer(self.net[2])
        _init_layer(self.net[4], output=True)

    def forward(self, joint_observation, joint_action):
        return self.net(torch.cat([joint_observation, joint_action], dim=-1))


class DiscreteMADDPG:
    def __init__(
        self,
        num_agents,
        obs_dim,
        action_dim,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=5e-4,
        gamma=0.99,
        tau=0.005,
        device=None,
        shared_actor=False,
    ):
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.shared_actor = bool(shared_actor)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        joint_obs_dim = self.num_agents * self.obs_dim
        joint_action_dim = self.num_agents * self.action_dim
        if self.shared_actor:
            self.shared_actor_model = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
            self.shared_target_actor = deepcopy(self.shared_actor_model).eval()
            self.actors = [self.shared_actor_model for _ in range(self.num_agents)]
            self.target_actors = [self.shared_target_actor for _ in range(self.num_agents)]
        else:
            self.actors = [
                Actor(obs_dim, action_dim, hidden_dim).to(self.device)
                for _ in range(self.num_agents)
            ]
            self.target_actors = [deepcopy(actor).eval() for actor in self.actors]
        self.critics = [
            Critic(joint_obs_dim, joint_action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_critics = [deepcopy(critic).eval() for critic in self.critics]
        if self.shared_actor:
            self.actor_optimizers = [
                torch.optim.Adam(self.shared_actor_model.parameters(), lr=actor_lr)
            ]
        else:
            self.actor_optimizers = [
                torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors
            ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics
        ]

    @torch.no_grad()
    def act(self, observations, action_masks=None, epsilon=0.0, deterministic=False):
        observations = np.asarray(observations, dtype=np.float32)
        if action_masks is None:
            action_masks = np.ones(
                (self.num_agents, self.action_dim), dtype=bool
            )
        action_masks = np.asarray(action_masks, dtype=bool)
        if action_masks.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                "action_masks must have shape "
                f"({self.num_agents},{self.action_dim})"
            )
        if np.any(~np.any(action_masks, axis=1)):
            raise ValueError("each agent must have at least one valid action")
        action_indices = []
        for index, actor in enumerate(self.actors):
            if not deterministic and np.random.random() < epsilon:
                valid_actions = np.flatnonzero(action_masks[index])
                action_indices.append(int(np.random.choice(valid_actions)))
                continue
            observation = torch.as_tensor(
                observations[index], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            logits = actor(observation)
            mask = torch.as_tensor(
                action_masks[index], dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
            action_indices.append(int(torch.argmax(logits, dim=-1).item()))
        return np.asarray(action_indices, dtype=np.int64)

    def update(self, batch, temperature=1.0):
        (
            observations,
            actions,
            action_masks,
            rewards,
            next_observations,
            next_action_masks,
            dones,
        ) = batch
        batch_size = observations.shape[0]
        joint_observations = observations.reshape(batch_size, -1)
        joint_actions = actions.reshape(batch_size, -1)
        joint_next_observations = next_observations.reshape(batch_size, -1)

        with torch.no_grad():
            target_actions = []
            for index, actor in enumerate(self.target_actors):
                logits = actor(next_observations[:, index])
                logits = logits.masked_fill(
                    ~next_action_masks[:, index].bool(), -1e9
                )
                discrete = F.one_hot(
                    torch.argmax(logits, dim=-1), num_classes=self.action_dim
                ).float()
                target_actions.append(discrete)
            joint_target_actions = torch.cat(target_actions, dim=-1)

        critic_losses = []
        for agent_index in range(self.num_agents):
            with torch.no_grad():
                next_q = self.target_critics[agent_index](
                    joint_next_observations, joint_target_actions
                )
                target_q = rewards[:, agent_index : agent_index + 1] + self.gamma * (
                    1.0 - dones[:, agent_index : agent_index + 1]
                ) * next_q

            predicted_q = self.critics[agent_index](joint_observations, joint_actions)
            critic_loss = F.mse_loss(predicted_q, target_q)
            self.critic_optimizers[agent_index].zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent_index].parameters(), 1.0)
            self.critic_optimizers[agent_index].step()

            critic_losses.append(float(critic_loss.item()))

        # A shared actor receives one averaged update from both centralized
        # critics, preventing upper/lower roles from drifting into unrelated
        # policies.  Separate-actor checkpoints retain the original behavior.
        for critic in self.critics:
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
        actor_losses = []
        for agent_index in range(self.num_agents):
            policy_actions = []
            for other_index in range(self.num_agents):
                if other_index == agent_index:
                    logits = self.actors[other_index](observations[:, other_index])
                    logits = logits.masked_fill(
                        ~action_masks[:, other_index].bool(), -1e9
                    )
                    policy_actions.append(
                        F.gumbel_softmax(logits, tau=max(float(temperature), 0.05), hard=True)
                    )
                else:
                    policy_actions.append(actions[:, other_index].detach())
            joint_policy_actions = torch.cat(policy_actions, dim=-1)
            actor_losses.append(-self.critics[agent_index](
                joint_observations, joint_policy_actions
            ).mean())

        if self.shared_actor:
            combined_actor_loss = torch.stack(actor_losses).mean()
            optimizer = self.actor_optimizers[0]
            optimizer.zero_grad(set_to_none=True)
            combined_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.shared_actor_model.parameters(), 0.5)
            optimizer.step()
        else:
            # Preserve independent optimization when loading/using legacy mode.
            # Recompute each graph because the losses above share no parameters
            # across actors but cannot be stepped safely after an in-place update.
            for agent_index, actor_loss in enumerate(actor_losses):
                optimizer = self.actor_optimizers[agent_index]
                optimizer.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=agent_index < self.num_agents - 1)
                nn.utils.clip_grad_norm_(self.actors[agent_index].parameters(), 0.5)
                optimizer.step()

        for critic in self.critics:
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

        self._soft_update()
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(
                torch.stack([loss.detach() for loss in actor_losses]).mean().item()
            ),
        }

    def _soft_update(self):
        if self.shared_actor:
            self._polyak(self.shared_target_actor, self.shared_actor_model)
        else:
            for target, source in zip(self.target_actors, self.actors):
                self._polyak(target, source)
        for target, source in zip(self.target_critics, self.critics):
            self._polyak(target, source)

    def _polyak(self, target, source):
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(source_parameter, alpha=self.tau)

    def save(self, path, metadata=None):
        payload = {
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "shared_actor": self.shared_actor,
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["obs_dim"] != self.obs_dim or payload["action_dim"] != self.action_dim:
            raise ValueError(
                f"checkpoint interface ({payload['obs_dim']},{payload['action_dim']}) "
                f"does not match ({self.obs_dim},{self.action_dim})"
            )
        if self.shared_actor:
            self.shared_actor_model.load_state_dict(
                self._average_states(payload["actors"])
            )
            self.shared_target_actor.load_state_dict(
                self._average_states(payload["target_actors"])
            )
        else:
            for model, state in zip(self.actors, payload["actors"]):
                model.load_state_dict(state)
            for model, state in zip(self.target_actors, payload["target_actors"]):
                model.load_state_dict(state)
        for model, state in zip(self.critics, payload["critics"]):
            model.load_state_dict(state)
        for model, state in zip(self.target_critics, payload["target_critics"]):
            model.load_state_dict(state)
        return payload.get("metadata", {})

    @staticmethod
    def _average_states(states):
        if len(states) == 1:
            return states[0]
        return {
            key: torch.stack([state[key].float() for state in states]).mean(dim=0)
            for key in states[0]
        }
