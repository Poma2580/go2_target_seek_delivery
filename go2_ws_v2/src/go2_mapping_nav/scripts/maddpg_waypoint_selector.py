#!/usr/bin/env python3
"""ROS executable for MADDPG waypoint selection with Nav2 execution."""

import rclpy

from go2_mapping_nav.maddpg_waypoint_selector import MaddpgWaypointSelector


def main(args=None):
    rclpy.init(args=args)
    node = MaddpgWaypointSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
