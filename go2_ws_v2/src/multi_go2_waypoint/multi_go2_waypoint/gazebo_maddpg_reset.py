#!/usr/bin/env python3
"""Gazebo reset helper for MADDPG follower fine-tuning.

Default reset mode is ``spawn``:
only call /reset_simulation and let Gazebo restore the model poses created by
the launch/spawn scripts.  This is safer for quadrupeds than teleporting the
base link with /gazebo/set_entity_state, because teleporting can leave leg
contacts and joint states in an unstable configuration.

Reset mode ``none`` does not call /reset_simulation or /gazebo/set_entity_state;
it only publishes zero cmd_vel and waits.  This is useful when Gazebo reset
makes the Go2 fall over and we want to start from an already stable scene.

Reset mode ``teleport`` does not call /reset_simulation.  It only uses
/gazebo/set_entity_state to move the target and robots to the fixed episode
poses, clears model twist to zero, keeps publishing zero cmd_vel, and then waits
for the robots to settle.  This isolates whether /reset_simulation is the cause
of quadruped falls.

Optional reset mode ``fixed`` keeps the earlier hand-authored fixed pose:

- walking target moves straight along +x,
- go1 starts behind the target and faces it,
- go2/go3 start near the final left/right slots but not exactly on them.

The class is also imported by the Gazebo training node so that manual reset and
training reset use the same geometry.
"""

import math
import time
from dataclasses import dataclass

import rclpy
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import Empty


def yaw_to_quaternion(yaw):
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    return 0.0, 0.0, qz, qw


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


class GazeboFixedResetter:
    def __init__(self, node: Node):
        self.node = node
        self._declare_if_needed("reset_service", "/reset_simulation")
        self._declare_if_needed("set_entity_service", "/gazebo/set_entity_state")
        self._declare_if_needed("delete_entity_service", "/delete_entity")
        self._declare_if_needed("spawn_entity_service", "/spawn_entity")
        self._declare_if_needed("target_controller_reset_service", "/walking_target/reset")
        self._declare_if_needed("reset_mode", "spawn")
        self._declare_if_needed("reset_pose_source", "script")
        self._declare_if_needed("target_reset_mode", "set_entity")
        self._declare_if_needed("target_model", "walking_target")
        self._declare_if_needed("go1_model", "go2_1")
        self._declare_if_needed("go2_model", "go2_2")
        self._declare_if_needed("go3_model", "go2_3")
        # Defaults match the Gazebo stage-1 training setup:
        # - go2 spawn poses come from spawn_go2_velodyne_*.launch.py
        # - the resettable training target starts at (-8, 4), moving along +x
        #   with the same speed as QY_MODEL/target_seek: 94 / 464 m/s.
        self._declare_if_needed("target_x", -8.0)
        self._declare_if_needed("target_y", 4.00)
        self._declare_if_needed("target_yaw", 6.2832)
        self._declare_if_needed("target_speed", 94.0 / 464.0)
        self._declare_if_needed("go1_x", -10.0)
        self._declare_if_needed("go1_y", 4.0)
        self._declare_if_needed("go1_yaw", 0.0)
        self._declare_if_needed("go1_z", 0.50)
        self._declare_if_needed("go2_x", -9.0)
        self._declare_if_needed("go2_y", 5.5)
        self._declare_if_needed("go2_yaw", 0.0)
        self._declare_if_needed("go2_z", 0.50)
        self._declare_if_needed("go3_x", -9.0)
        self._declare_if_needed("go3_y", 2.5)
        self._declare_if_needed("go3_yaw", 0.0)
        self._declare_if_needed("go3_z", 0.50)
        self._declare_if_needed("leader_follow_dist", 1.80)
        self._declare_if_needed("side_dist", 1.20)
        self._declare_if_needed("go2_slot_offset_x", -0.60)
        self._declare_if_needed("go2_slot_offset_y", 0.35)
        self._declare_if_needed("go3_slot_offset_x", -0.60)
        self._declare_if_needed("go3_slot_offset_y", -0.35)
        self._declare_if_needed("robot_z", 0.50)
        self._declare_if_needed("target_z", 0.0)
        self._declare_if_needed("settle_time", 1.0)

        self.reset_service = str(self.node.get_parameter("reset_service").value)
        self.set_entity_service = str(self.node.get_parameter("set_entity_service").value)
        self.delete_entity_service = str(self.node.get_parameter("delete_entity_service").value)
        self.spawn_entity_service = str(self.node.get_parameter("spawn_entity_service").value)
        self.target_controller_reset_service = str(
            self.node.get_parameter("target_controller_reset_service").value
        )
        self.reset_mode = str(self.node.get_parameter("reset_mode").value)
        if self.reset_mode not in ("none", "spawn", "teleport", "fixed"):
            raise ValueError("reset_mode must be 'none', 'spawn', 'teleport', or 'fixed'")
        self.reset_pose_source = str(self.node.get_parameter("reset_pose_source").value)
        if self.reset_pose_source not in ("script", "formation"):
            raise ValueError("reset_pose_source must be 'script' or 'formation'")
        self.target_reset_mode = str(self.node.get_parameter("target_reset_mode").value)
        if self.target_reset_mode not in ("none", "set_entity", "respawn", "controller"):
            raise ValueError("target_reset_mode must be 'none', 'set_entity', 'respawn', or 'controller'")
        self.target_model = str(self.node.get_parameter("target_model").value)
        self.go1_model = str(self.node.get_parameter("go1_model").value)
        self.go2_model = str(self.node.get_parameter("go2_model").value)
        self.go3_model = str(self.node.get_parameter("go3_model").value)
        self.target_speed = float(self.node.get_parameter("target_speed").value)
        self.robot_z = float(self.node.get_parameter("robot_z").value)
        self.go1_z = float(self.node.get_parameter("go1_z").value)
        self.go2_z = float(self.node.get_parameter("go2_z").value)
        self.go3_z = float(self.node.get_parameter("go3_z").value)
        self.target_z = float(self.node.get_parameter("target_z").value)
        self.settle_time = float(self.node.get_parameter("settle_time").value)

        self.reset_client = self.node.create_client(Empty, self.reset_service)
        self.set_entity_client = self.node.create_client(SetEntityState, self.set_entity_service)
        self.delete_entity_client = self.node.create_client(DeleteEntity, self.delete_entity_service)
        self.spawn_entity_client = self.node.create_client(SpawnEntity, self.spawn_entity_service)
        self.target_controller_reset_client = self.node.create_client(
            Empty, self.target_controller_reset_service
        )
        self.cmd_pubs = {
            self.go1_model: self.node.create_publisher(Twist, f"/{self.go1_model}/cmd_vel", 10),
            self.go2_model: self.node.create_publisher(Twist, f"/{self.go2_model}/cmd_vel", 10),
            self.go3_model: self.node.create_publisher(Twist, f"/{self.go3_model}/cmd_vel", 10),
            self.target_model: self.node.create_publisher(Twist, f"/{self.target_model}/cmd_vel", 10),
        }

    def _declare_if_needed(self, name, value):
        if not self.node.has_parameter(name):
            self.node.declare_parameter(name, value)

    def fixed_poses(self):
        tx = float(self.node.get_parameter("target_x").value)
        ty = float(self.node.get_parameter("target_y").value)
        tyaw = float(self.node.get_parameter("target_yaw").value)
        if self.reset_pose_source == "script":
            return {
                self.target_model: Pose2D(tx, ty, tyaw),
                self.go1_model: Pose2D(
                    float(self.node.get_parameter("go1_x").value),
                    float(self.node.get_parameter("go1_y").value),
                    float(self.node.get_parameter("go1_yaw").value),
                ),
                self.go2_model: Pose2D(
                    float(self.node.get_parameter("go2_x").value),
                    float(self.node.get_parameter("go2_y").value),
                    float(self.node.get_parameter("go2_yaw").value),
                ),
                self.go3_model: Pose2D(
                    float(self.node.get_parameter("go3_x").value),
                    float(self.node.get_parameter("go3_y").value),
                    float(self.node.get_parameter("go3_yaw").value),
                ),
            }

        leader_follow_dist = float(self.node.get_parameter("leader_follow_dist").value)
        side_dist = float(self.node.get_parameter("side_dist").value)
        go2_dx = float(self.node.get_parameter("go2_slot_offset_x").value)
        go2_dy = float(self.node.get_parameter("go2_slot_offset_y").value)
        go3_dx = float(self.node.get_parameter("go3_slot_offset_x").value)
        go3_dy = float(self.node.get_parameter("go3_slot_offset_y").value)

        forward = (math.cos(tyaw), math.sin(tyaw))
        left = (-forward[1], forward[0])

        leader = Pose2D(tx - leader_follow_dist * forward[0], ty - leader_follow_dist * forward[1], tyaw)
        left_slot = Pose2D(tx + side_dist * left[0], ty + side_dist * left[1], tyaw)
        right_slot = Pose2D(tx - side_dist * left[0], ty - side_dist * left[1], tyaw)

        go2 = Pose2D(
            left_slot.x + go2_dx * forward[0] + go2_dy * left[0],
            left_slot.y + go2_dx * forward[1] + go2_dy * left[1],
            tyaw,
        )
        go3 = Pose2D(
            right_slot.x + go3_dx * forward[0] + go3_dy * left[0],
            right_slot.y + go3_dx * forward[1] + go3_dy * left[1],
            tyaw,
        )
        return {
            self.target_model: Pose2D(tx, ty, tyaw),
            self.go1_model: leader,
            self.go2_model: go2,
            self.go3_model: go3,
        }

    def wait_for_services(self, timeout_sec=10.0):
        deadline = time.time() + timeout_sec
        required = [(self.set_entity_client, self.set_entity_service)]
        if self.reset_mode in ("spawn", "fixed"):
            required.append((self.reset_client, self.reset_service))
        if self.target_reset_mode == "respawn":
            required.extend(
                [
                    (self.delete_entity_client, self.delete_entity_service),
                    (self.spawn_entity_client, self.spawn_entity_service),
                ]
            )
        if self.target_reset_mode == "controller":
            required.append((self.target_controller_reset_client, self.target_controller_reset_service))
        for client, name in required:
            while not client.wait_for_service(timeout_sec=0.2):
                if time.time() > deadline:
                    raise RuntimeError(f"service not available: {name}")

    def publish_zero_cmds(self, repeats=5):
        zero = Twist()
        for _ in range(repeats):
            for pub in self.cmd_pubs.values():
                pub.publish(zero)
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def _call_empty(self, client):
        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        if future.result() is None:
            raise RuntimeError(f"failed to call {client.srv_name}")

    def _set_entity(self, name, pose: Pose2D, z, vx=0.0, vy=0.0, wz=0.0):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.reference_frame = "world"
        req.state.pose.position.x = float(pose.x)
        req.state.pose.position.y = float(pose.y)
        req.state.pose.position.z = float(z)
        qx, qy, qz, qw = yaw_to_quaternion(float(pose.yaw))
        req.state.pose.orientation.x = qx
        req.state.pose.orientation.y = qy
        req.state.pose.orientation.z = qz
        req.state.pose.orientation.w = qw
        req.state.twist.linear.x = float(vx)
        req.state.twist.linear.y = float(vy)
        req.state.twist.angular.z = float(wz)
        future = self.set_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError(f"failed to set entity state: {name}")
        if hasattr(result, "success") and not result.success:
            self.node.get_logger().warn(f"set_entity_state({name}) returned false: {result.status_message}")

    def _delete_entity(self, name):
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError(f"failed to delete entity: {name}")
        if hasattr(result, "success") and not result.success:
            self.node.get_logger().warn(f"delete_entity({name}) returned false: {result.status_message}")

    def _target_model_sdf_xml(self, pose: Pose2D):
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{self.target_model}">
    <pose>{pose.x:.6f} {pose.y:.6f} {self.target_z:.6f} 0 0 {pose.yaw:.6f}</pose>
    <static>false</static>
    <link name="base_link">
      <inertial>
        <mass>60.0</mass>
        <inertia>
          <ixx>4.0</ixx>
          <iyy>4.0</iyy>
          <izz>1.0</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <pose>0 0 0.9 0 0 0</pose>
        <geometry>
          <box>
            <size>0.45 0.35 1.8</size>
          </box>
        </geometry>
      </collision>
      <visual name="person_visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://actor/meshes/SKIN_man_blue_shirt.dae</uri>
            <scale>1.0 1.0 1.0</scale>
          </mesh>
        </geometry>
      </visual>
    </link>
    <plugin name="planar_move" filename="libgazebo_ros_planar_move.so">
      <ros>
        <namespace>/{self.target_model}</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
      </ros>
      <update_rate>100</update_rate>
      <publish_rate>10</publish_rate>
      <publish_odom>false</publish_odom>
      <publish_odom_tf>false</publish_odom_tf>
      <odometry_frame>world</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </model>
</sdf>"""

    def _spawn_target_model(self, pose: Pose2D):
        req = SpawnEntity.Request()
        req.name = self.target_model
        req.xml = self._target_model_sdf_xml(pose)
        req.robot_namespace = ""
        req.reference_frame = "world"
        req.initial_pose.position.x = float(pose.x)
        req.initial_pose.position.y = float(pose.y)
        req.initial_pose.position.z = float(self.target_z)
        qx, qy, qz, qw = yaw_to_quaternion(float(pose.yaw))
        req.initial_pose.orientation.x = qx
        req.initial_pose.orientation.y = qy
        req.initial_pose.orientation.z = qz
        req.initial_pose.orientation.w = qw
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            raise RuntimeError(f"failed to spawn target model: {self.target_model}")
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(f"spawn_entity({self.target_model}) failed: {result.status_message}")

    def reset_fixed_episode(self):
        self.wait_for_services()
        self.publish_zero_cmds()

        if self.reset_mode == "none":
            start = time.time()
            while time.time() - start < self.settle_time:
                self.publish_zero_cmds(repeats=1)
                rclpy.spin_once(self.node, timeout_sec=0.02)
            self.node.get_logger().info(
                "Gazebo episode reset disabled: reset_mode=none, "
                "did not call /reset_simulation or /gazebo/set_entity_state. "
                f"settle_time={self.settle_time:.1f}s"
            )
            return

        if self.reset_mode in ("spawn", "fixed"):
            self._call_empty(self.reset_client)

        if self.reset_mode == "spawn":
            self.publish_zero_cmds()
            start = time.time()
            while time.time() - start < self.settle_time:
                rclpy.spin_once(self.node, timeout_sec=0.02)
            self.node.get_logger().info(
                "Gazebo episode reset with spawn initial state: "
                "called /reset_simulation, did not teleport go2_1/go2_2/go2_3. "
                f"settle_time={self.settle_time:.1f}s"
            )
            return

        poses = self.fixed_poses()
        tyaw = poses[self.target_model].yaw
        target_vx = self.target_speed * math.cos(tyaw)
        target_vy = self.target_speed * math.sin(tyaw)

        if self.target_reset_mode == "none":
            pass
        elif self.target_reset_mode == "controller":
            self._call_empty(self.target_controller_reset_client)
            self._set_entity(self.target_model, poses[self.target_model], self.target_z, 0.0, 0.0, 0.0)
        elif self.target_reset_mode == "respawn":
            self._delete_entity(self.target_model)
            rclpy.spin_once(self.node, timeout_sec=0.05)
            self._spawn_target_model(poses[self.target_model])
        else:
            self._set_entity(self.target_model, poses[self.target_model], self.target_z, target_vx, target_vy, 0.0)
        self._set_entity(self.go1_model, poses[self.go1_model], self.go1_z)
        self._set_entity(self.go2_model, poses[self.go2_model], self.go2_z)
        self._set_entity(self.go3_model, poses[self.go3_model], self.go3_z)
        self.publish_zero_cmds()

        start = time.time()
        while time.time() - start < self.settle_time:
            if self.target_reset_mode == "controller":
                self.publish_zero_cmds(repeats=1)
            rclpy.spin_once(self.node, timeout_sec=0.02)

        if self.target_reset_mode not in ("none", "controller"):
            # If the walking target has a cmd_vel interface this keeps stage1 straight.
            # If not, Gazebo simply ignores the topic and actor_state_publisher still
            # reports the actor script motion.
            target_cmd = Twist()
            target_cmd.linear.x = self.target_speed
            self.cmd_pubs[self.target_model].publish(target_cmd)

        p = poses
        mode_text = (
            "teleport without /reset_simulation"
            if self.reset_mode == "teleport"
            else "fixed reset after /reset_simulation"
        )
        self.node.get_logger().info(
            f"Gazebo episode reset ({mode_text}): "
            f"target=({p[self.target_model].x:.2f},{p[self.target_model].y:.2f},{p[self.target_model].yaw:.2f}), "
            f"go1=({p[self.go1_model].x:.2f},{p[self.go1_model].y:.2f}), "
            f"go2=({p[self.go2_model].x:.2f},{p[self.go2_model].y:.2f}), "
            f"go3=({p[self.go3_model].x:.2f},{p[self.go3_model].y:.2f}), "
            f"pose_source={self.reset_pose_source}, "
            f"target_reset={self.target_reset_mode}, "
            f"z=({self.go1_z:.2f},{self.go2_z:.2f},{self.go3_z:.2f}), "
            f"target_speed={self.target_speed:.3f}, "
            f"settle_time={self.settle_time:.1f}s"
        )


class GazeboMaddpgResetNode(Node):
    def __init__(self):
        super().__init__("gazebo_maddpg_reset")
        self.resetter = GazeboFixedResetter(self)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboMaddpgResetNode()
    try:
        node.resetter.reset_fixed_episode()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
