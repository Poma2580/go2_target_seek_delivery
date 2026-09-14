#!/usr/bin/env python3
"""Live terminal dashboard for waypoint-policy obstacle decisions."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _number(value, width=7, precision=2, suffix=""):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{precision}f}{suffix}"


def format_dashboard(payload):
    endpoint_threshold = payload["endpoint_blocked_threshold"]
    path_threshold = payload["path_blocked_threshold"]
    default_action = int(payload["default_action"])
    lines = [
        "MADDPG 选点障碍监测（数据与 Actor 本次决策输入一致）",
        (
            f"leader={payload['leader']}  "
            f"heuristic_mask={payload['heuristic_action_mask']}  "
            f"blocked阈值: endpoint<{endpoint_threshold:.2f}m "
            f"或 path<{path_threshold:.2f}m"
        ),
        "标记: *=最终动作  A=动作掩码允许  B=候选点被识别为阻塞",
    ]
    for agent in payload["agents"]:
        changed = "CHANGED" if agent["action_changed"] else "HOLD"
        lines.extend(
            [
                "",
                (
                    f"[{agent['name']}] action {agent['previous_action']}"
                    f" -> {agent['action']} ({changed})  "
                    f"scan接收年龄={_number(agent['scan_receive_age_ms'], 6, 1, 'ms')}  "
                    f"消息时间戳年龄={_number(agent['scan_header_age_ms'], 6, 1, 'ms')}  "
                    f"扫描周期={_number(agent['scan_period_ms'], 6, 1, 'ms')}"
                ),
                (
                    f"雷达 rays={agent['raw_ray_count']} "
                    f"有效近距回波={agent['raw_valid_hit_count']}  "
                    f"最近策略回波={agent['nearest_policy_range']:.2f}m  "
                    f"中间通道锁存阻塞={agent['default_blocked_latched']}"
                ),
                " action offset(m) endpoint(m) path(m) flags",
            ]
        )
        for candidate in agent["candidates"]:
            flags = "".join(
                [
                    "*" if candidate["selected"] else "-",
                    "A" if candidate["allowed"] else "-",
                    "B" if candidate["blocked"] else "-",
                ]
            )
            lines.append(
                f"   {candidate['action']:>2}   {candidate['offset']:>+7.2f}"
                f"     {candidate['endpoint_clearance']:>7.2f}"
                f"  {candidate['path_clearance']:>7.2f}   {flags}"
            )

        default = agent["candidates"][default_action]
        if agent["action"] != default_action and not default["blocked"]:
            lines.append(
                "  判断：中间候选点为 CLEAR，但策略选择了侧点；"
                "不是 blocked 硬判断直接造成，重点检查策略偏置/奖励。"
            )
        elif agent["action"] != default_action:
            lines.append(
                "  判断：中间候选点被雷达特征判为 BLOCKED；"
                "侧移可能由障碍输入触发，请结合扫描年龄和净空判断误检。"
            )
        else:
            lines.append("  判断：策略当前保持中间候选点。")
    return "\n".join(lines)


class WaypointMonitor(Node):
    def __init__(self):
        super().__init__("maddpg_waypoint_monitor")
        self.declare_parameter(
            "diagnostic_topic", "/maddpg_waypoint/decision_diagnostics"
        )
        topic = str(self.get_parameter("diagnostic_topic").value)
        self.create_subscription(String, topic, self._callback, 10)
        print(f"等待 {topic}；该话题在 MADDPG 接管并开始选点后每秒更新。")

    def _callback(self, message):
        try:
            dashboard = format_dashboard(json.loads(message.data))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(f"Invalid decision diagnostic: {error}")
            return
        print("\033[2J\033[H" + dashboard, flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
