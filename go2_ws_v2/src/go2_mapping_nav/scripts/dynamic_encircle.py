#!/usr/bin/env python3
"""ROS executable entry point for modular dynamic encirclement."""

import rclpy
from rclpy.signals import SignalHandlerOptions

from go2_mapping_nav.dynamic_encircle.node import DynamicEncircle


def main(args=None):
    """Run the node and preserve ROS context until safe shutdown completes."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = DynamicEncircle()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
