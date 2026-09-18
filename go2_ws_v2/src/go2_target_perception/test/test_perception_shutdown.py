"""Exercise the real entry point and role callback without loading YOLO."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("selected", ["go2_1", "go2_2", ""])
def test_role_callback_allows_entry_point_to_exit_without_deadlock(selected):
    script = textwrap.dedent("""
        import sys
        from rclpy.node import Node
        from std_msgs.msg import String
        from go2_target_perception import target_perception as module

        selected = sys.argv[1]
        original = module.TargetPerception

        class Probe(Node):
            _role_callback = original._role_callback

            def __init__(self):
                super().__init__('perception_shutdown_probe')
                self.robot_namespace = 'go2_1'
                self._shutdown_requested = False
                self.role_timer = self.create_timer(0.05, self.deliver_role)
                self.create_timer(0.2, self.verify_still_running)

            def deliver_role(self):
                self.role_timer.cancel()
                self._role_callback(String(data=selected))
                assert self._shutdown_requested == (selected == 'go2_2')
                print('callback returned', flush=True)

            def verify_still_running(self):
                assert selected != 'go2_2', 'non-selected process did not exit'
                assert not self._shutdown_requested
                print('selected or unassigned process stayed active', flush=True)
                raise KeyboardInterrupt

        module.TargetPerception = Probe
        module.main()
        print('entry point returned', flush=True)
    """)
    environment = dict(os.environ, ROS_DOMAIN_ID="173", ROS_LOCALHOST_ONLY="1")
    result = subprocess.run(
        [sys.executable, "-c", script, selected],
        env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "callback returned" in result.stdout
    assert "entry point returned" in result.stdout
    if selected != "go2_2":
        assert "stayed active" in result.stdout
