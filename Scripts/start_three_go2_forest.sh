#!/bin/bash
# 启动森林场景、顺序导入三只 Go2，并启动 waypoint_forest 围捕控制。
#
# 用法：
#   bash Scripts/start_three_go2_forest.sh
#   bash Scripts/start_three_go2_forest.sh --lidar
#   bash Scripts/start_three_go2_forest.sh --camera
#   bash Scripts/start_three_go2_forest.sh --all-sensors

set -u

usage() {
    cat <<'EOF'
用法：bash Scripts/start_three_go2_forest.sh [传感器选项]

功能：
  1. 启动森林场景 KD_MODEL/world/forestV3.world
  2. 顺序导入 go2_1 / go2_2 / go2_3
  3. 等待每只 Go2 的两个控制器 active
  4. 启动 multi_go2_waypoint 的 waypoint_forest 围捕节点

传感器选项（默认均关闭）：
  --lidar                  三只 Go2 开启 3D Velodyne
  --camera                 三只 Go2 开启 RGB-D 相机
  --all-sensors            同时开启 3D Velodyne 和 RGB-D 相机
  -h, --help               显示本帮助

森林围捕出生点：
  go2_1: x=20, y=18, z=0.8, yaw=2.19
  go2_2: x=-8, y=42, z=0.8, yaw=0.00
  go2_3: x=36, y=40, z=0.8, yaw=-2.92
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
KD_MODEL_ROOT=$DELIVERY_ROOT/KD_MODEL
WORLD_PATH=$KD_MODEL_ROOT/world/forestV3.world

ENABLE_LIDAR=false
ENABLE_CAMERA=false

for arg in "$@"; do
    case "$arg" in
        --lidar)
            ENABLE_LIDAR=true
            ;;
        --camera)
            ENABLE_CAMERA=true
            ;;
        --all-sensors)
            ENABLE_LIDAR=true
            ENABLE_CAMERA=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: 未知参数：$arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SPAWN_X=("20" "-8" "36")
SPAWN_Y=("18" "42" "40")
SPAWN_Z=("0.8" "0.8" "0.8")
SPAWN_YAW=("2.19" "0.00" "-2.92")

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "ERROR: 未找到 gnome-terminal。请安装后再运行。" >&2
    exit 1
fi

if [ ! -d "$WS/install" ]; then
    echo "ERROR: 未找到 $WS/install。请先编译工作空间。" >&2
    exit 1
fi

if [ ! -f "$WORLD_PATH" ]; then
    echo "ERROR: 森林场景文件不存在：$WORLD_PATH" >&2
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
export KD_MODEL_ROOT=$KD_MODEL_ROOT
export GAZEBO_MODEL_PATH=\$QY_MODEL_ROOT/models:\$KD_MODEL_ROOT/models:\$GAZEBO_MODEL_PATH
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
until response=\$(timeout 8 ros2 service call '${list_controllers_service}' controller_manager_msgs/srv/ListControllers '{}' 2>/dev/null) && \\
  printf '%s\\n' \"\$response\" | grep -q \"name='joint_group_effort_controller', state='active'\" && \\
  printf '%s\\n' \"\$response\" | grep -q \"name='joint_states_controller', state='active'\"; do
    echo 'waiting for ${robot_name} controllers ...'
    sleep 2
done
echo '${robot_name} controllers are active.'
"
}

echo "场景：forest"
echo "world：$WORLD_PATH"
echo "Go2 传感器：lidar=${ENABLE_LIDAR}, camera=${ENABLE_CAMERA}"
echo "Go2 出生点："
echo "  go2_1=(${SPAWN_X[0]}, ${SPAWN_Y[0]}, ${SPAWN_Z[0]}, yaw=${SPAWN_YAW[0]})"
echo "  go2_2=(${SPAWN_X[1]}, ${SPAWN_Y[1]}, ${SPAWN_Z[1]}, yaw=${SPAWN_YAW[1]})"
echo "  go2_3=(${SPAWN_X[2]}, ${SPAWN_Y[2]}, ${SPAWN_Z[2]}, yaw=${SPAWN_YAW[2]})"

launch_terminal "forest_world" "
echo '==== Starting forest world ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true world:=$WORLD_PATH
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/clock"

echo "森林世界已就绪，等待 5 秒以完成 Gazebo 稳定加载..."
sleep 5

for robot_index in 1 2 3; do
    array_index=$((robot_index - 1))
    robot_name="go2_${robot_index}"
    spawn_x=${SPAWN_X[$array_index]}
    spawn_y=${SPAWN_Y[$array_index]}
    spawn_z=${SPAWN_Z[$array_index]}
    spawn_yaw=${SPAWN_YAW[$array_index]}

    launch_terminal "spawn_${robot_name}_forest" "
echo '==== Spawning ${robot_name}: (${spawn_x}, ${spawn_y}, ${spawn_z}, yaw=${spawn_yaw}), lidar=${ENABLE_LIDAR}, camera=${ENABLE_CAMERA} ===='
ros2 launch go2_config spawn_go2_velodyne_${robot_index}.launch.py use_sim_time:=true enable_lidar:=${ENABLE_LIDAR} enable_camera:=${ENABLE_CAMERA} spawn_x:=${spawn_x} spawn_y:=${spawn_y} spawn_z:=${spawn_z} spawn_yaw:=${spawn_yaw}
"
    wait_for_controllers_active "$robot_name"

    if [ "$robot_index" -lt 3 ]; then
        echo "${robot_name} 已就绪，等待 3 秒后启动下一只 Go2..."
        sleep 3
    fi
done

echo "三只 Go2 已就绪，等待 3 秒后启动森林 waypoint 围捕..."
sleep 3

launch_terminal "waypoint_forest" "
echo '==== Starting waypoint_forest encircle controller ===='
ros2 run multi_go2_waypoint waypoint_forest
"

echo "森林围捕启动完成。"
echo "控制器检查："
echo "  ros2 control list_controllers -c /go2_1/controller_manager"
echo "  ros2 control list_controllers -c /go2_2/controller_manager"
echo "  ros2 control list_controllers -c /go2_3/controller_manager"
echo "围捕节点检查："
echo "  ros2 node list | grep waypoint"
