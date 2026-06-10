#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


class GroundTruthOdomRelay(Node):
    def __init__(self):
        super().__init__("ground_truth_odom_relay")

        self.declare_parameter("input_topic", "/odom/ground_truth")
        self.declare_parameter("output_topic", "/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("child_frame", "base_footprint")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("project_to_2d", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.child_frame = self.get_parameter("child_frame").value
        self.publish_tf = self.get_parameter("publish_tf").value
        self.project_to_2d = self.get_parameter("project_to_2d").value

        self.odom_publisher = self.create_publisher(Odometry, self.output_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Odometry, self.input_topic, self.odom_callback, 10)

    def odom_callback(self, msg):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.child_frame

        odom.pose.pose.position.x = msg.pose.pose.position.x
        odom.pose.pose.position.y = msg.pose.pose.position.y
        odom.pose.pose.position.z = msg.pose.pose.position.z
        odom.pose.pose.orientation = msg.pose.pose.orientation

        if self.project_to_2d:
            yaw = yaw_from_quaternion(msg.pose.pose.orientation)
            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            vx_world = msg.twist.twist.linear.x
            vy_world = msg.twist.twist.linear.y
            odom.twist.twist.linear.x = cos_yaw * vx_world + sin_yaw * vy_world
            odom.twist.twist.linear.y = -sin_yaw * vx_world + cos_yaw * vy_world
            odom.twist.twist.linear.z = 0.0
            odom.twist.twist.angular.x = 0.0
            odom.twist.twist.angular.y = 0.0
            odom.twist.twist.angular.z = msg.twist.twist.angular.z
        else:
            odom.twist.twist = msg.twist.twist

        odom.pose.covariance = msg.pose.covariance
        odom.twist.covariance = msg.twist.covariance
        self.odom_publisher.publish(odom)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = odom.child_frame_id
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = odom.pose.pose.position.z
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)


def main():
    rclpy.init()
    node = GroundTruthOdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
