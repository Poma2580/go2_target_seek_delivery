#!/bin/bash
# Usage:
# cd /home/bit/go2_target_seek_delivery/Scripts
# ./start_three_go2_velodyne.sh

DELIVERY_ROOT=/home/bit/go2_target_seek_delivery
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL

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
gnome-terminal --title="go2_world" -- bash -c "
$COMMON_ENV
echo '==== Starting target_seek world ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true
exec bash
"

wait_for_ros_service "/spawn_entity"

# 终端 2：启动 go2_1
gnome-terminal --title="spawn_go2_1" -- bash -c "
$COMMON_ENV
echo '==== Spawning go2_1 without lidar/camera ===='
ros2 launch go2_config spawn_go2_velodyne_1.launch.py enable_lidar:=false enable_camera:=false
exec bash
"

wait_for_controllers_active "go2_1"

# 终端 3：启动 go2_2
gnome-terminal --title="spawn_go2_2" -- bash -c "
$COMMON_ENV
echo '==== Spawning go2_2 without lidar/camera ===='
ros2 launch go2_config spawn_go2_velodyne_2.launch.py enable_lidar:=false enable_camera:=false
exec bash
"

wait_for_controllers_active "go2_2"

# 终端 4：启动 go2_3
gnome-terminal --title="spawn_go2_3" -- bash -c "
$COMMON_ENV
echo '==== Spawning go2_3 without lidar/camera ===='
ros2 launch go2_config spawn_go2_velodyne_3.launch.py enable_lidar:=false enable_camera:=false
exec bash
"

wait_for_controllers_active "go2_3"

echo "全部启动命令已经发出。"
echo "请检查："
echo "ros2 control list_controllers -c /go2_1/controller_manager"
echo "ros2 control list_controllers -c /go2_2/controller_manager"
echo "ros2 control list_controllers -c /go2_3/controller_manager"
