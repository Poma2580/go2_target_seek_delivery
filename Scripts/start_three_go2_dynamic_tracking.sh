#!/bin/bash
# Usage:
# cd /home/bit/go2_target_seek_delivery/Scripts
# ./start_three_go2_dynamic_tracking.sh

DELIVERY_ROOT=/home/bit/go2_target_seek_delivery
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
YOLO_MODEL=$DELIVERY_ROOT/yolov8m.pt

if [ ! -f "$YOLO_MODEL" ]; then
    echo "ERROR: YOLO model not found: $YOLO_MODEL"
    exit 1
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "ERROR: gnome-terminal not found."
    exit 1
fi

COMMON_ENV="
export DELIVERY_ROOT=$DELIVERY_ROOT
cd $WS
conda deactivate 2>/dev/null || true
if [ \"\$(which python3)\" != \"/usr/bin/python3\" ]; then
    echo \"ERROR: python3 is \$(which python3), expected /usr/bin/python3\"
    exit 1
fi
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

# 终端 1：启动 target_seek 世界
launch_terminal "go2_world" "
echo '==== Starting target_seek world ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/gazebo/model_states"
wait_for_topic "/clock"

# 终端 2：启动 go2_1，关闭 3D 雷达，开启 RGB-D 相机（负责目标感知）
launch_terminal "spawn_go2_1" "
echo '==== Spawning go2_1 without lidar with RGB-D camera ===='
ros2 launch go2_config spawn_go2_velodyne_1.launch.py enable_lidar:=false enable_camera:=true use_sim_time:=true
"

wait_for_controllers_active "go2_1"
wait_for_topic "/go2_1/camera/image_raw"
wait_for_topic "/go2_1/camera/depth/image_raw"
wait_for_topic "/go2_1/camera/depth/camera_info"

# # 终端 3：启动 go2_2，关闭 3D 雷达和相机（只参与运动控制）
# launch_terminal "spawn_go2_2" "
# echo '==== Spawning go2_2 without lidar without camera ===='
# ros2 launch go2_config spawn_go2_velodyne_2.launch.py enable_lidar:=false enable_camera:=false use_sim_time:=true
# "

# wait_for_controllers_active "go2_2"

# # 终端 4：启动 go2_3，关闭 3D 雷达和相机（只参与运动控制）
# launch_terminal "spawn_go2_3" "
# echo '==== Spawning go2_3 without lidar without camera ===='
# ros2 launch go2_config spawn_go2_velodyne_3.launch.py enable_lidar:=false enable_camera:=false use_sim_time:=true
# "

# wait_for_controllers_active "go2_3"

# 终端 5：启动行人真值状态广播
launch_terminal "actor_state" "
echo '==== Starting actor_state_publisher ===='
ros2 run multi_go2_waypoint actor_state_publisher --ros-args -p use_sim_time:=true
"

wait_for_topic "/walking_target/odom"

# 终端 6：启动目标感知（go2_1 相机 -> YOLO + 深度估计 -> 目标 odom）
launch_terminal "target_perception" "
echo '==== Starting target_perception ===='
ros2 run multi_go2_waypoint target_perception --ros-args -p use_sim_time:=true -p robot_namespace:=go2_1 -p model_path:=$YOLO_MODEL
"

wait_for_topic "/go2_1/target_perception/debug_image"

# 终端 7：启动三 go2 动态围捕，目标输入使用 go2_1 感知估计的 odom
launch_terminal "dynamic_encircle" "
echo '==== Starting dynamic_encircle ===='
ros2 run multi_go2_waypoint dynamic_encircle --ros-args -p use_sim_time:=true -p target_odom_topic:=/go2_1/target_estimated/odom
"

# 终端 8：打开 RQT 查看检测调试图
launch_terminal "rqt_debug_image" "
echo '==== Starting rqt_image_view debug image ===='
ros2 run rqt_image_view rqt_image_view /go2_1/target_perception/debug_image
"

# 终端 9：启动感知误差评估
launch_terminal "perception_eval" "
echo '==== Starting perception_eval ===='
ros2 run multi_go2_waypoint perception_eval --ros-args -p use_sim_time:=true
"

echo "三 go2 动态追踪启动命令已经全部发出。"
echo "可用以下命令检查："
echo "ros2 topic list | grep -E 'go2_1/(camera|target|perception_error)|walking_target'"
echo "ros2 topic echo /walking_target/odom --once"
echo "ros2 topic echo /go2_1/target_estimated/odom --once"
echo "ros2 topic echo /go2_2/cmd_vel"
