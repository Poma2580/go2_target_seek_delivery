"""Import-level smoke test for the actor state bridge."""

from walking_target_controller.actor_state_publisher import ActorStatePublisher, main


def test_actor_state_publisher_exports_node_and_entry_point():
    assert ActorStatePublisher.__name__ == "ActorStatePublisher"
    assert callable(main)
