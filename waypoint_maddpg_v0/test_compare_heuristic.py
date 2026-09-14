"""Tests for the random obstacle-filtering comparison controller."""

import numpy as np
import pytest

from .compare_heuristic import NoSafeCandidateError, random_safe_actions


def _metrics(clear_indices):
    return [
        {
            "blocked": index not in clear_indices,
            "path_clearance": 1.0 if index in clear_indices else 0.0,
            "endpoint_clearance": 1.0 if index in clear_indices else 0.0,
        }
        for index in range(5)
    ]


def test_random_safe_actions_never_selects_a_blocked_candidate():
    rng = np.random.default_rng(42)
    metrics = [_metrics({0, 2, 4}), _metrics({1, 3})]
    samples = np.asarray([random_safe_actions(metrics, rng) for _ in range(300)])

    assert set(samples[:, 0]) == {0, 2, 4}
    assert set(samples[:, 1]) == {1, 3}


def test_random_safe_actions_uses_the_only_clear_candidate():
    actions = random_safe_actions(
        [_metrics({4}), _metrics({0})], np.random.default_rng(7)
    )
    np.testing.assert_array_equal(actions, [4, 0])


def test_random_safe_actions_stops_if_a_robot_has_no_clear_candidate():
    with pytest.raises(NoSafeCandidateError, match="agent 1"):
        random_safe_actions(
            [_metrics({2}), _metrics(set())], np.random.default_rng(7)
        )
