#!/bin/bash
# Start a clean Gazebo environment for leader-slot MADDPG training/testing:
#   - current target_seek urban world
#   - three Go2 robots only
#   - no YOLO / target_perception
#   - no walking target controller / actor state publisher
#   - no dynamic_encircle or MADDPG controller
#   - remove the built-in walking_target actor after the world starts
#
# Usage:
#   ./Scripts/start_three_go2_maddpg_env.sh
#   ./Scripts/start_three_go2_maddpg_env.sh --headless

set -e

DELIVERY_ROOT=/home/wangantong/KD_all/go2_target_seek_delivery
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
GUI=true

if [ "${1:-}" = "--headless" ]; then
    GUI=false
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "ERROR: gnome-terminal not found."
    exit 1
fi

echo "==== Cleaning old ROS/Gazebo nodes from previous runs ===="
pkill -f "walking_target_actor_controller" 2>/dev/null || true
pkill -f "actor_state_publisher" 2>/dev/null || true
pkill -f "dynamic_encircle" 2>/dev/null || true
pkill -f "gazebo_maddpg_train_stage1" 2>/dev/null || true
pkill -f "gazebo_leader_slot_train_stage1" 2>/dev/null || true
pkill -f "gazebo_leader_slot_controller" 2>/dev/null || true
pkill -f "maddpg_follower_slot_controller" 2>/dev/null || true
pkill -f "target_perception" 2>/dev/null || true
pkill -f "perception_eval" 2>/dev/null || true
pkill -f "rqt_image_view" 2>/dev/null || true
pkill -f "gzserver" 2>/dev/null || true
pkill -f "gzclient" 2>/dev/null || true
sleep 2

COMMON_ENV="
export DELIVERY_ROOT=$DELIVERY_ROOT
cd $WS
conda deactivate 2>/dev/null || true
export PATH=/usr/bin:/bin:\$PATH
echo \"Using python3: \$(which python3)\"
source /opt/ros/humble/setup.bash
source install/setup.bash
export QY_MODEL_ROOT=$QY_MODEL_ROOT
export GAZEBO_MODEL_PATH=\$QY_MODEL_ROOT/models:\$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_DATABASE_URI=\"\"
"

launch_terminal() {
    local title=$1
    local command=$2

    gnome-terminal --title="$title" -- bash -c "
$COMMON_ENV
$command
exec bash
"
}

wait_for_ros_service() {
    local service_name=$1
    echo "等待 ROS service ${service_name} 出现..."
    bash -c "$COMMON_ENV
until ros2 service list | grep -qx '${service_name}'; do
    echo 'waiting for ${service_name} ...'
    sleep 2
done
echo '${service_name} is ready.'
"
}

wait_for_topic() {
    local topic_name=$1
    echo "等待 ROS topic ${topic_name} 出现..."
    bash -c "$COMMON_ENV
until ros2 topic list | grep -qx '${topic_name}'; do
    echo 'waiting for ${topic_name} ...'
    sleep 2
done
echo '${topic_name} is ready.'
"
}

wait_for_controllers_active() {
    local robot_name=$1
    local list_controllers_service="/${robot_name}/controller_manager/list_controllers"
    echo "等待 ${robot_name} 的两个控制器 active..."
    bash -c "$COMMON_ENV
until response=\$(timeout 8 ros2 service call '${list_controllers_service}' controller_manager_msgs/srv/ListControllers '{}' 2>/dev/null) && \
      printf '%s\n' \"\$response\" | grep -q \"name='joint_group_effort_controller', state='active'\" && \
      printf '%s\n' \"\$response\" | grep -q \"name='joint_states_controller', state='active'\"; do
    echo 'waiting for ${robot_name} controllers ...'
    sleep 2
done
echo '${robot_name} controllers are active.'
"
}

delete_walking_target_if_present() {
    echo "==== Removing walking_target actor from Gazebo world if present ===="
    bash -c "$COMMON_ENV
if ros2 service list | grep -qx '/delete_entity'; then
    ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \"{name: 'walking_target'}\" || true
else
    echo 'WARNING: /delete_entity service not found; skip walking_target deletion.'
fi
sleep 1
if timeout 5 ros2 topic echo /gazebo/model_states --once | grep -q \"walking_target\"; then
    echo 'WARNING: walking_target still appears in /gazebo/model_states.'
else
    echo 'walking_target removed or not present.'
fi
"
}

echo "==== Starting target_seek urban world for MADDPG, GUI=${GUI} ===="
launch_terminal "go2_world_maddpg_env" "
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=$GUI world:=$QY_MODEL_ROOT/target_seek
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/gazebo/model_states"
wait_for_topic "/clock"
sleep 3
wait_for_ros_service "/delete_entity"
delete_walking_target_if_present

echo "==== Spawning go2_1 without lidar/camera ===="
launch_terminal "spawn_go2_1" "
ros2 launch go2_config spawn_go2_velodyne_1.launch.py enable_lidar:=false enable_camera:=false use_sim_time:=true
"
wait_for_controllers_active "go2_1"
sleep 2

echo "==== Spawning go2_2 without lidar/camera ===="
launch_terminal "spawn_go2_2" "
ros2 launch go2_config spawn_go2_velodyne_2.launch.py enable_lidar:=false enable_camera:=false use_sim_time:=true
"
wait_for_controllers_active "go2_2"
sleep 2

echo "==== Spawning go2_3 without lidar/camera ===="
launch_terminal "spawn_go2_3" "
ros2 launch go2_config spawn_go2_velodyne_3.launch.py enable_lidar:=false enable_camera:=false use_sim_time:=true
"
wait_for_controllers_active "go2_3"

echo
echo "MADDPG Gazebo base environment is ready."
echo "Started: target_seek world + go2_1/go2_2/go2_3"
echo "Not started: YOLO, target_perception, actor_state_publisher, walking_target_actor_controller, dynamic_encircle"
echo "Deleted from world if present: walking_target"
echo
echo "Check:"
echo "  ros2 topic list | grep -E 'go2_[123]/odom|walking_target|target_perception'"
echo "  ros2 topic info /go2_1/cmd_vel -v"
echo
echo "Next: manually start either gazebo_leader_slot_train_stage1 or gazebo_leader_slot_controller."
