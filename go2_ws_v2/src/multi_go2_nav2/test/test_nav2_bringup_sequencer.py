from multi_go2_nav2.nav2_bringup_sequencer import (
    lifecycle_manager_sequence,
    run_bringup_sequence,
)


def test_lifecycle_managers_start_map_then_each_robot():
    sequence = lifecycle_manager_sequence(('go2_1', 'go2_2', 'go2_3'))

    assert [stack.label for stack in sequence] == [
        'shared map', 'go2_1', 'go2_2', 'go2_3']
    assert [stack.manage_service for stack in sequence] == [
        '/lifecycle_manager_map/manage_nodes',
        '/go2_1/lifecycle_manager_navigation/manage_nodes',
        '/go2_2/lifecycle_manager_navigation/manage_nodes',
        '/go2_3/lifecycle_manager_navigation/manage_nodes',
    ]


def test_bringup_waits_for_each_stack_before_starting_next():
    sequence = lifecycle_manager_sequence(('go2_1', 'go2_2', 'go2_3'))
    started = []

    assert run_bringup_sequence(
        sequence, lambda stack: started.append(stack.label) or True)
    assert started == ['shared map', 'go2_1', 'go2_2', 'go2_3']


def test_bringup_stops_after_first_failure():
    sequence = lifecycle_manager_sequence(('go2_1', 'go2_2', 'go2_3'))
    started = []

    def start(stack):
        started.append(stack.label)
        return stack.label != 'go2_2'

    assert not run_bringup_sequence(sequence, start)
    assert started == ['shared map', 'go2_1', 'go2_2']
