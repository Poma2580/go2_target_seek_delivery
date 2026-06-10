"""Three xterm keyboard windows (ROS official teleop_twist_keyboard), one per Go2.

Each teleop_twist_keyboard runs under namespace go2_i, so it publishes Twist on
'cmd_vel' -> /go2_i/cmd_vel, which the namespaced quadruped_controller picks up
(via its cmd_vel/smooth -> cmd_vel remap).

NOTE: keyboard input only goes to the xterm window that currently has focus.
Click the go2_1 / go2_2 / go2_3 window before driving that dog.
Keys: i forward, , back, j left, l right, k stop, q/z speed up/down.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


ROBOTS = ("go2_1", "go2_2", "go2_3")


def make_teleop(namespace):
    return Node(
        package="teleop_twist_keyboard",
        executable="teleop_twist_keyboard",
        namespace=namespace,
        name="teleop_twist_keyboard",
        prefix=f'xterm -T "{namespace} teleop" -e',
        output="screen",
    )


def generate_launch_description():
    return LaunchDescription([make_teleop(ns) for ns in ROBOTS])
