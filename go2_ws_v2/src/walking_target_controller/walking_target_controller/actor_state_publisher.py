#!/usr/bin/env python3
"""行人 actor 状态桥接节点。

把 Gazebo 的 /gazebo/model_states 里 walking_target 的位姿，转换成标准 ROS2 里程计
话题 /walking_target/odom，供动态围捕节点订阅。

注意：actor 由 <script> 动画驱动、不走物理引擎，model_states 里它的 twist 速度通常恒为 0，
因此本节点用相邻两帧位置差分自算世界系速度，并做一阶低通滤波。
"""

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class ActorStatePublisher(Node):
    def __init__(self):
        super().__init__('actor_state_publisher')

        # 参数
        self.declare_parameter('model_name', 'walking_target')
        self.declare_parameter('model_states_topic', '/gazebo/model_states')
        self.declare_parameter('output_topic', '/walking_target/odom')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('vel_lpf_alpha', 0.3)   # 速度低通系数 (0~1，越小越平滑)

        self.model_name = self.get_parameter('model_name').value
        states_topic = self.get_parameter('model_states_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.alpha = float(self.get_parameter('vel_lpf_alpha').value)

        # 差分测速状态
        self.last_x = None
        self.last_y = None
        self.last_stamp = None
        self.vx = 0.0
        self.vy = 0.0

        self._warned = False  # 找不到模型时只节流告警

        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)
        self.sub = self.create_subscription(
            ModelStates, states_topic, self.states_callback, 10)

        self.get_logger().info(
            f'actor_state_publisher 已启动：{states_topic}[{self.model_name}] -> {output_topic}')

    def states_callback(self, msg):
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            if not self._warned:
                self.get_logger().warn(
                    f'/gazebo/model_states 中未找到模型 "{self.model_name}"，'
                    f'当前模型列表：{msg.name}')
                self._warned = True
            return

        pose = msg.pose[idx]
        now = self.get_clock().now()
        x = pose.position.x
        y = pose.position.y

        # 用相邻帧位置差分算世界系速度（actor 的 twist 不可靠），再低通滤波
        if self.last_stamp is not None:
            dt = (now - self.last_stamp).nanoseconds * 1e-9
            if dt > 1e-4:
                raw_vx = (x - self.last_x) / dt
                raw_vy = (y - self.last_y) / dt
                self.vx = self.alpha * raw_vx + (1.0 - self.alpha) * self.vx
                self.vy = self.alpha * raw_vy + (1.0 - self.alpha) * self.vy

        self.last_x = x
        self.last_y = y
        self.last_stamp = now

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.model_name
        odom.pose.pose = pose
        # twist 用自算的世界系速度（注意：这是世界系，下游需要时自行旋转到机体系）
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.linear.z = 0.0
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = ActorStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
