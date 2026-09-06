"""Fast deterministic checks for geometry, observations and one learner update."""

import numpy as np
import torch

from .config import EnvConfig
from .discrete_maddpg import DiscreteMADDPG
from .environment import WaypointSelectionEnv
from .replay_buffer import ReplayBuffer


def main():
    config = EnvConfig(lidar_noise_std=0.0, max_episode_steps=8)
    env = WaypointSelectionEnv(config, seed=1, lidar_noise=False)
    observations, info = env.reset(seed=1)
    assert observations.shape == (2, 83)
    assert np.isfinite(observations).all()
    assert [obstacle["shape"] for obstacle in env.obstacles] == ["square", "circle"]
    assert np.sign(env.obstacles[0]["center"][1]) != np.sign(env.obstacles[1]["center"][1])
    assert env.obstacles[1]["radius"] == 1.0
    candidates = env._candidate_points()
    np.testing.assert_allclose(candidates[0, :, 1], [4, 3, 2, 1, 0], atol=1e-6)
    np.testing.assert_allclose(candidates[1, :, 1], [0, -1, -2, -3, -4], atol=1e-6)
    expected_initial_mask = np.asarray([False, False, True, False, False])
    np.testing.assert_array_equal(env.valid_action_masks()[0], expected_initial_mask)
    try:
        env.step(np.asarray([0, 2]))
    except ValueError as error:
        assert "adjacent candidate" in str(error)
    else:
        raise AssertionError("a direct 0 m -> +2 m jump must be rejected")

    # Only the agent whose own default corridor is blocked may leave its slot.
    for metrics in env._candidate_metrics[0]:
        metrics["blocked"] = False
    env._candidate_metrics[0][2]["blocked"] = True
    env._update_default_path_state()
    np.testing.assert_array_equal(
        env.valid_action_masks()[0],
        np.asarray([False, True, False, True, False]),
    )
    np.testing.assert_array_equal(env.valid_action_masks()[1], expected_initial_mask)
    env.previous_actions[0] = 1
    env._candidate_metrics[0][2]["blocked"] = False
    for _ in range(config.default_clear_release_steps):
        env._update_default_path_state()
    np.testing.assert_array_equal(
        env.valid_action_masks()[0],
        np.asarray([False, False, True, False, False]),
    )
    observations, info = env.reset(seed=1)

    # If a one-metre detour is safe, an unnecessary two-metre detour must have
    # a lower formation reward.  If only two metres is safe, it is not charged.
    safe = lambda blocked: {"blocked": blocked}
    env._candidate_metrics = [
        [safe(True), safe(True), safe(True), safe(False), safe(False)],
        [safe(True), safe(True), safe(True), safe(False), safe(False)],
    ]
    penalties = env._formation_offset_penalties(np.asarray([3, 4]))
    np.testing.assert_allclose(penalties, [0.0, -0.6], atol=1e-6)
    env._candidate_metrics[1][3] = safe(True)
    necessary_detour = env._formation_offset_penalties(np.asarray([3, 4]))
    assert necessary_detour[1] == 0.0
    observations, info = env.reset(seed=1)

    device = torch.device("cpu")
    policy = DiscreteMADDPG(
        2,
        env.obs_dim,
        env.num_actions,
        hidden_dim=32,
        device=device,
        shared_actor=True,
    )
    assert policy.actors[0] is policy.actors[1]
    replay = ReplayBuffer(64, 2, env.obs_dim, env.num_actions, device)
    for _ in range(40):
        action_masks = env.valid_action_masks()
        actions = np.asarray(
            [np.random.choice(np.flatnonzero(mask)) for mask in action_masks]
        )
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        next_action_masks = env.valid_action_masks()
        replay.add(
            observations,
            np.eye(env.num_actions, dtype=np.float32)[actions],
            action_masks,
            rewards,
            next_observations,
            next_action_masks,
            terminated or truncated,
        )
        observations = next_observations
        if terminated or truncated:
            observations, info = env.reset()
    losses = policy.update(replay.sample(16), temperature=1.0)
    assert np.isfinite(list(losses.values())).all()

    # In a clear scene both agents are hard-held at their default slots.
    clear_config = EnvConfig(
        lidar_noise_std=0.0,
        obstacle_spawn_x=(100.0, 100.0),
        max_episode_steps=12,
    )
    default_return = _rollout_return(clear_config, [[2, 2]] * 10)
    print(
        "smoke test passed:",
        {
            "observation_shape": observations.shape,
            "losses": losses,
            "default_return": default_return,
            "last_info": info,
        },
    )


def _rollout_return(config, action_sequence):
    env = WaypointSelectionEnv(config, seed=9, lidar_noise=False)
    observations, _ = env.reset(seed=9)
    del observations
    total = 0.0
    for actions in action_sequence:
        _, rewards, terminated, truncated, _ = env.step(np.asarray(actions))
        total += float(rewards[0])
        if terminated or truncated:
            break
    return total


if __name__ == "__main__":
    main()
