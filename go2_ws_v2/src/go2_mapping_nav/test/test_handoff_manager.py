"""Pure tests for the fail-closed handoff state machine."""

from types import SimpleNamespace

from go2_mapping_nav.dynamic_encircle.handoff_manager import HandoffManager, HandoffState
from go2_mapping_nav.dynamic_encircle.nav_goal_manager import NavGoalManager


class FakeFuture:
    def __init__(self, value):
        self.value = value
        self.callback = None

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callback = callback


class FakeGoalHandle:
    accepted = True

    def __init__(self):
        self.result_future = FakeFuture(SimpleNamespace(status=5))
        self.cancel_future = FakeFuture(SimpleNamespace(goals_canceling=[1]))

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        return self.cancel_future


def make_manager(**overrides):
    values = dict(arrival_hold_duration=1.0, stopped_hold_duration=0.5,
                  cancel_timeout=10.0, stop_timeout=10.0,
                  maddpg_ready_timeout=30.0, maddpg_enable_timeout=5.0)
    values.update(overrides)
    events = []
    manager = HandoffManager(
        SimpleNamespace(**values), lambda: events.append("cancel"),
        lambda value: events.append(("enable", value)),
        lambda value: events.append(("mux", value)),
        lambda value: events.append(("state", value)),
    )
    return manager, events


def test_full_handoff_requires_holds_cancel_ready_and_active():
    manager, events = make_manager()
    manager.select_role(0.0)
    manager.update(0.1, arrived=True)
    manager.update(0.8, arrived=True)
    assert "cancel" not in events
    manager.update(1.1, arrived=True)
    manager.update(1.2, cancel_complete=True)
    manager.update(1.3, stopped=True)
    manager.update(1.8, stopped=True)
    assert manager.state == HandoffState.WAITING_FOR_MADDPG_READY
    manager.update(1.9, maddpg_ready=True)
    assert ("mux", True) not in events
    manager.update(2.0, maddpg_active=True)
    assert manager.state == HandoffState.MADDPG_ACTIVE
    assert events[-2:] == [("mux", True), ("state", "MADDPG_ACTIVE")]


def test_arrival_break_resets_hold_and_timeout_fails_closed():
    manager, events = make_manager(cancel_timeout=0.2)
    manager.select_role(0.0)
    manager.update(0.1, arrived=True)
    manager.update(0.5, arrived=False)
    assert manager.state == HandoffState.NAV2_ACTIVE
    manager.update(1.0, arrived=True)
    manager.update(2.0, arrived=True)
    manager.update(2.3)
    assert manager.state == HandoffState.HANDOFF_FAILED
    assert ("enable", False) in events and ("mux", False) in events


def test_active_drop_fails_without_restarting_nav2():
    manager, events = make_manager()
    manager.select_role(0.0)
    manager.state = HandoffState.MADDPG_ACTIVE
    manager.state_since = 1.0
    manager.update(2.0, maddpg_active=False)
    assert manager.state == HandoffState.HANDOFF_FAILED
    assert events[-3:-1] == [("enable", False), ("mux", False)]


def test_late_accepted_goal_is_cancelled_and_waited_to_terminal():
    manager = NavGoalManager.__new__(NavGoalManager)
    manager.navigation_dogs = ("go2_2", "go2_3")
    manager.pending_goal_sends = {"go2_2": {7}, "go2_3": set()}
    manager.active_goal_handles = {"go2_2": {}, "go2_3": {}}
    manager.cancel_inflight = {"go2_2": set(), "go2_3": set()}
    manager._handoff_cancelling = True
    manager._handoff_cancel_failed = False
    manager.node = SimpleNamespace(
        get_logger=lambda: SimpleNamespace(
            error=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )
    )
    handle = FakeGoalHandle()
    manager._goal_response_callback(
        "go2_2", 7, (0.0, 0.0, 0.0), FakeFuture(handle)
    )
    assert not manager.pending_goal_sends["go2_2"]
    assert 7 in manager.active_goal_handles["go2_2"]
    assert 7 in manager.cancel_inflight["go2_2"]
    handle.cancel_future.callback(handle.cancel_future)
    assert not manager.handoff_cancel_complete()
    handle.result_future.callback(handle.result_future)
    assert manager.handoff_cancel_complete()
