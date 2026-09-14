#!/usr/bin/env python3
"""ROS executable for Gazebo MADDPG online fine-tuning."""

import rclpy

from go2_mapping_nav.gazebo_maddpg_finetuner import GazeboMaddpgFinetuner


def main(args=None):
    rclpy.init(args=args)
    node = GazeboMaddpgFinetuner()
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
