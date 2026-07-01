#!/usr/bin/env python3
"""感知精度评估：对比在线估计 odom 与真值 odom，持续打印位置误差与统计量。

- 估计：/go2_1/target_estimated/odom（target_perception 发布，世界系）
- 真值：/walking_target/odom（actor_state_publisher 发布，世界系 baseline）

估计帧到来时，取时间最近的真值帧配对（时间差超过 match_timeout 则跳过），计算：
  ex, ey, 欧氏距离 e；以及二者速度差（可选参考）。
维护累计统计（样本数 n、均值、最大值、位置 RMSE），按 log_period 节流打印。
可选把当前距离误差发布到 /go2_1/perception_error（Float32），便于 rqt_plot 看曲线。

运行（先 conda deactivate，确认 which python3 为 /usr/bin/python3）：
  ros2 run multi_go2_waypoint perception_eval --ros-args -p use_sim_time:=true
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class PerceptionEval(Node):
    def __init__(self):
        super().__init__('perception_eval')

        self.declare_parameter('estimate_topic', '/go2_1/target_estimated/odom')
        self.declare_parameter('truth_topic', '/walking_target/odom')
        self.declare_parameter('error_topic', '/go2_1/perception_error')
        self.declare_parameter('match_timeout', 0.2)   # 配对最大时间差(s)
        self.declare_parameter('log_period', 1.0)      # 打印周期(s)
        self.declare_parameter('truth_buffer', 200)    # 真值缓存帧数

        est_topic = self.get_parameter('estimate_topic').value
        truth_topic = self.get_parameter('truth_topic').value
        err_topic = self.get_parameter('error_topic').value
        self.match_timeout = float(self.get_parameter('match_timeout').value)
        self.log_period = float(self.get_parameter('log_period').value)
        buf_len = int(self.get_parameter('truth_buffer').value)

        # 真值缓存：(t, x, y, vx, vy)
        self.truth_buf = deque(maxlen=buf_len)

        # 统计量
        self.n = 0
        self.sum_e = 0.0
        self.sum_e2 = 0.0
        self.max_e = 0.0
        self.last_log = self.get_clock().now()
        self.last_line = None

        self.err_pub = self.create_publisher(Float32, err_topic, 10)
        self.create_subscription(Odometry, truth_topic, self._truth_cb, 50)
        self.create_subscription(Odometry, est_topic, self._est_cb, 10)

        self.get_logger().info(
            f'perception_eval 已启动：est={est_topic} vs truth={truth_topic}')

    def _truth_cb(self, msg):
        self.truth_buf.append((
            stamp_to_sec(msg.header.stamp),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        ))

    def _match_truth(self, t):
        """取时间最近的真值帧。"""
        if not self.truth_buf:
            return None
        best = min(self.truth_buf, key=lambda r: abs(r[0] - t))
        if abs(best[0] - t) > self.match_timeout:
            return None
        return best

    def _est_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        truth = self._match_truth(t)
        if truth is None:
            self.get_logger().warn(
                '无可配对真值（时间差过大或真值未发布）。', throttle_duration_sec=2.0)
            return

        ex = msg.pose.pose.position.x - truth[1]
        ey = msg.pose.pose.position.y - truth[2]
        e = math.hypot(ex, ey)

        evx = msg.twist.twist.linear.x - truth[3]
        evy = msg.twist.twist.linear.y - truth[4]
        ev = math.hypot(evx, evy)

        self.n += 1
        self.sum_e += e
        self.sum_e2 += e * e
        self.max_e = max(self.max_e, e)

        self.err_pub.publish(Float32(data=float(e)))

        mean = self.sum_e / self.n
        rmse = math.sqrt(self.sum_e2 / self.n)
        self.last_line = (
            f'误差 e={e:.3f}m (ex={ex:+.3f}, ey={ey:+.3f})  '
            f'速度误差={ev:.3f}m/s  |  mean={mean:.3f}  RMSE={rmse:.3f}  '
            f'max={self.max_e:.3f}  n={self.n}')

        now = self.get_clock().now()
        if (now - self.last_log).nanoseconds * 1e-9 >= self.log_period:
            self.get_logger().info(self.last_line)
            self.last_log = now


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionEval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.last_line is not None:
            node.get_logger().info('最终统计：' + node.last_line)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
