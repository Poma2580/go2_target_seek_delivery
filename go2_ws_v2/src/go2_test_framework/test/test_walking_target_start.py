from types import SimpleNamespace

import pytest

from go2_test_framework.runner import ros_wait
from go2_test_framework.runner.ros_wait import (
    WalkingTargetStartError,
    start_walking_target,
)


class UnreadableFuture:
    def __getattribute__(self, name):
        if name in {"done", "result", "exception"}:
            raise AssertionError("service response must not be inspected")
        return super().__getattribute__(name)


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class FakeClient:
    def __init__(self, ready=True):
        self.ready = ready
        self.requests = 0

    def service_is_ready(self):
        return self.ready

    def wait_for_service(self, timeout_sec):
        return self.ready

    def call_async(self, request):
        self.requests += 1
        return UnreadableFuture()


class FakeNode:
    def __init__(self, client):
        self.client = client
        self.callback = None

    def create_client(self, service_type, name):
        return self.client

    def create_subscription(self, message_type, name, callback, qos):
        self.callback = callback
        return object()

    def destroy_subscription(self, subscription):
        pass

    def destroy_client(self, client):
        pass

    def destroy_node(self):
        pass


def odom(x, y=0.0):
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
            )
        )
    )


def install_fake_ros(monkeypatch, positions, *, service_ready=True):
    clock = FakeTime()
    client = FakeClient(ready=service_ready)
    node = FakeNode(client)
    samples = iter(positions)

    def spin_once(spin_node, timeout_sec):
        clock.now += max(timeout_sec, 0.01)
        sample = next(samples, None)
        if sample is not None:
            spin_node.callback(odom(*sample))

    monkeypatch.setattr(ros_wait.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ros_wait, "Node", lambda _name: node)
    monkeypatch.setattr(ros_wait.rclpy, "init", lambda: None)
    monkeypatch.setattr(ros_wait.rclpy, "ok", lambda: True)
    monkeypatch.setattr(ros_wait.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(ros_wait.rclpy, "spin_once", spin_once)
    return client


def test_movement_is_the_only_success_criterion(monkeypatch):
    client = install_fake_ros(
        monkeypatch, [(0.0, 0.0), (0.0, 0.0), (0.02, 0.0), (0.06, 0.0)]
    )

    result = start_walking_target(timeout_sec=2.0)

    assert result.movement_confirmed
    assert result.displacement_m == pytest.approx(0.06)
    assert result.start_requests_sent == 1
    assert client.requests == 1
    assert set(result.to_dict()) == {
        "movement_confirmed", "displacement_m", "start_requests_sent",
    }


def test_stationary_target_retries_then_fails(monkeypatch):
    client = install_fake_ros(monkeypatch, [(0.0, 0.0)] * 100)

    with pytest.raises(WalkingTargetStartError, match="did not move") as caught:
        start_walking_target(
            timeout_sec=7.0, retry_interval_sec=2.0, max_requests=3
        )

    assert client.requests == 3
    assert caught.value.observation.start_requests_sent == 3
    assert not caught.value.observation.movement_confirmed


def test_initial_odom_is_required(monkeypatch):
    client = install_fake_ros(monkeypatch, [])

    with pytest.raises(WalkingTargetStartError, match="initial .*odom") as caught:
        start_walking_target(timeout_sec=1.0)

    assert client.requests == 0
    assert caught.value.observation.start_requests_sent == 0


def test_start_service_is_required(monkeypatch):
    client = install_fake_ros(
        monkeypatch, [(0.0, 0.0)] * 100, service_ready=False
    )

    with pytest.raises(WalkingTargetStartError, match="start service") as caught:
        start_walking_target(timeout_sec=1.0)

    assert client.requests == 0
    assert caught.value.observation.start_requests_sent == 0


def test_world_health_is_checked_while_waiting(monkeypatch):
    install_fake_ros(monkeypatch, [(0.0, 0.0)] * 100)

    with pytest.raises(RuntimeError, match="world disappeared"):
        start_walking_target(
            timeout_sec=2.0,
            health_check=lambda: (_ for _ in ()).throw(
                RuntimeError("world disappeared")
            ),
        )
