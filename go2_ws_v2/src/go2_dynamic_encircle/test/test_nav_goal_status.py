"""Read-only NavGoalManager status interface tests."""

import json
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

import go2_dynamic_encircle.nav_goal_manager as module


class Future:
    def __init__(self):
        self.callbacks = []
        self.value = None

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def complete(self, value):
        self.value = value
        for callback in tuple(self.callbacks):
            callback(self)

    def result(self):
        return self.value


class GoalHandle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result_future = Future()

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        return Future()


class ActionClient:
    instances = {}

    def __init__(self, _node, _action, name):
        self.name = name
        self.sent = []
        self.ready = True
        ActionClient.instances[name] = self

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        future = Future()
        self.sent.append((goal, future))
        return future


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


class Node:
    def __init__(self):
        self.publisher = Publisher()
        self.now_ns = 12_345_000_000
        self.logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )

    def create_publisher(self, *_args, **_kwargs):
        return self.publisher

    def get_clock(self):
        stamp = Time(sec=12, nanosec=345_000_000)
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: stamp))

    def get_logger(self):
        return self.logger


def manager(monkeypatch):
    ActionClient.instances = {}
    monkeypatch.setattr(module, "ActionClient", ActionClient)
    node = Node()
    value = module.NavGoalManager(
        node, ("go2_1", "go2_2", "go2_3"), "merged_map", 1.0
    )
    value.set_navigation_dogs(("go2_2", "go2_3"))
    value.set_plan(SimpleNamespace(
        slots={"go2_2": (1.0, 2.0, 0.3), "go2_3": (4.0, 5.0, -0.2)},
        route_heading=0.0,
    ))
    return node, value


def test_dispatched_is_published_after_both_real_sends(monkeypatch):
    node, value = manager(monkeypatch)
    assert value.dispatch_if_due(0.0)
    assert all(len(client.sent) == 1 for client in ActionClient.instances.values()
               if client.name.endswith(("go2_2/navigate_to_pose", "go2_3/navigate_to_pose")))
    event = node.publisher.messages[0]
    assert event == {
        "schema_version": 1,
        "stamp": {"sec": 12, "nanosec": 345000000},
        "event": "DISPATCHED",
        "generation": 1,
        "frame_id": "merged_map",
        "navigation_dogs": ["go2_2", "go2_3"],
        "goals": {
            "go2_2": {"goal_x": 1.0, "goal_y": 2.0, "goal_yaw": 0.3},
            "go2_3": {"goal_x": 4.0, "goal_y": 5.0, "goal_yaw": -0.2},
        },
        "robot": None,
        "action_status": None,
    }


def test_unready_server_does_not_dispatch_or_publish(monkeypatch):
    node, value = manager(monkeypatch)
    ActionClient.instances["/go2_3/navigate_to_pose"].ready = False
    assert not value.dispatch_if_due(0.0)
    assert value.state.generation == 0
    assert node.publisher.messages == []


def test_callback_events_keep_original_generation_and_terminal_status(monkeypatch):
    node, value = manager(monkeypatch)
    value.dispatch_if_due(0.0)
    first = ActionClient.instances["/go2_2/navigate_to_pose"].sent[0][1]
    handle = GoalHandle()
    first.complete(handle)
    value.dispatch_if_due(1.0)
    handle.result_future.complete(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))

    assert value.state.generation == 2
    assert [(event["event"], event["generation"]) for event in node.publisher.messages] == [
        ("DISPATCHED", 1), ("ACCEPTED", 1),
        ("DISPATCHED", 2), ("ABORTED", 1),
    ]


def test_rejected_and_success_events_are_reported(monkeypatch):
    node, value = manager(monkeypatch)
    value.dispatch_if_due(0.0)
    clients = ActionClient.instances
    rejected_future = clients["/go2_2/navigate_to_pose"].sent[0][1]
    rejected_future.complete(GoalHandle(accepted=False))
    accepted_future = clients["/go2_3/navigate_to_pose"].sent[0][1]
    accepted = GoalHandle()
    accepted_future.complete(accepted)
    accepted.result_future.complete(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED))
    assert [event["event"] for event in node.publisher.messages] == [
        "DISPATCHED", "REJECTED", "ACCEPTED", "SUCCEEDED"
    ]
