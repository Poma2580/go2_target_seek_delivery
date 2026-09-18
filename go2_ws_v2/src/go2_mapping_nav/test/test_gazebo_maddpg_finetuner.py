"""Unit checks for the Gazebo clear-to-obstacle curriculum boundaries."""

import math

from go2_mapping_nav.gazebo_maddpg_finetuner import GazeboMaddpgFinetuner


def _curriculum_stub(step):
    node = object.__new__(GazeboMaddpgFinetuner)
    node.global_step = step
    node.clear_stage_steps = 10_000
    node.one_obstacle_stage_steps = 30_000
    node.total_training_steps = 40_000
    node.epsilon_decay_steps = 30_000
    node.initial_epsilon = 0.10
    node.final_epsilon = 0.03
    return node


def test_curriculum_stage_boundaries():
    assert _curriculum_stub(0)._curriculum_stage() == "clear"
    assert _curriculum_stub(9_999)._curriculum_stage() == "clear"
    assert _curriculum_stub(10_000)._curriculum_stage() == "one_obstacle"
    assert _curriculum_stub(39_999)._curriculum_stage() == "one_obstacle"
    assert _curriculum_stub(40_000)._curriculum_stage() == "complete"


def test_epsilon_restarts_for_obstacle_stage():
    assert math.isclose(_curriculum_stub(0)._epsilon(), 0.10)
    assert math.isclose(_curriculum_stub(10_000)._epsilon(), 0.10)
    assert math.isclose(_curriculum_stub(40_000)._epsilon("one_obstacle"), 0.03)


def test_stage_step_and_budget():
    node = _curriculum_stub(12_345)
    assert node._stage_start_step("clear") == 0
    assert node._stage_budget("clear") == 10_000
    assert node._stage_start_step("one_obstacle") == 10_000
    assert node._stage_budget("one_obstacle") == 30_000
    assert node._stage_step("one_obstacle") == 2_345
