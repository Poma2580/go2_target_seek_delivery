#!/bin/bash
# 启动一个场景、预置 uav1，并顺序导入三只 Go2。
#
# 用法：
#   ./start_three_go2_velodyne.sh [city|forest|airport] [--lidar] [--camera]
#   ./start_three_go2_velodyne.sh forest --all-sensors

set -u

usage() {
    cat <<'EOF'
用法：./start_three_go2_velodyne.sh [场景] [传感器选项]

场景（默认 city）：
  city | qy | target_seek  target_seek 城市场景
  forest                   森林场景
  airport                  机场场景

传感器选项（默认均关闭）：
  --lidar                  三只 Go2 开启 3D Velodyne
  --camera                 三只 Go2 开启 RGB-D 相机
  --all-sensors            同时开启 3D Velodyne 和 RGB-D 相机
  -h, --help               显示本帮助

示例：
  ./start_three_go2_velodyne.sh
  ./start_three_go2_velodyne.sh forest
  ./start_three_go2_velodyne.sh airport --all-sensors
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
KD_MODEL_ROOT=$DELIVERY_ROOT/KD_MODEL

SCENE=city
ENABLE_LIDAR=false
ENABLE_CAMERA=false
SCENE_SET=false

for arg in "$@"; do
    case "$arg" in
        city|qy|target_seek|forest|airport)
            if [ "$SCENE_SET" = true ]; then
                echo "ERROR: 只能指定一个场景。" >&2
                usage >&2
                exit 2
            fi
            SCENE=$arg
            SCENE_SET=true
            ;;
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

case "$SCENE" in
    city|qy|target_seek)
        SCENE=city
        WORLD_PATH=$QY_MODEL_ROOT/target_seek
        ;;
    forest)
        WORLD_PATH=$KD_MODEL_ROOT/world/forestV3.world
        ;;
    airport)
        WORLD_PATH=$KD_MODEL_ROOT/world/airport
        ;;
esac

if [ "$SCENE" = "forest" ] || [ "$SCENE" = "airport" ]; then
    # 原点附近的紧凑三角形：彼此相距 2 m，避免出生时模型重叠。
    SCENE_SPAWN_X=("0" "2" "0")
    SCENE_SPAWN_Y=("-4" "-4" "-6")
    SCENE_SPAWN_Z=("0.50" "0.50" "0.50")
    SCENE_SPAWN_YAW=("0" "0" "0")
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "ERROR: 未找到 gnome-terminal。请安装后再运行。" >&2
    exit 1
fi

if [ ! -d "$WS/install" ] || [ ! -f "$WORLD_PATH" ]; then
    echo "ERROR: 工作空间未编译或场景文件不存在。请先确认路径并运行 colcon build。" >&2
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

echo "场景：${SCENE}"
echo "Go2 传感器：lidar=${ENABLE_LIDAR}, camera=${ENABLE_CAMERA}"
if [ "$SCENE" = "city" ]; then
    echo "Go2 出生点：使用各 launch 的默认位置"
else
    echo "Go2 出生点：go2_1=(0,-4,0.50,0), go2_2=(2,-4,0.50,0), go2_3=(0,-6,0.50,0)"
fi

launch_terminal "go2_world_${SCENE}" "
echo '==== Starting ${SCENE} world with uav1 ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true world:=$WORLD_PATH
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/clock"
wait_for_topic "/uav1/camera/image_raw"

echo "世界与 uav1 已就绪，等待 5 秒以完成 Gazebo 稳定加载..."
sleep 5

for robot_index in 1 2 3; do
    robot_name="go2_${robot_index}"
    spawn_arguments=""
    spawn_description="launch 默认出生点"

    if [ "$SCENE" != "city" ]; then
        position_index=$((robot_index - 1))
        spawn_x=${SCENE_SPAWN_X[$position_index]}
        spawn_y=${SCENE_SPAWN_Y[$position_index]}
        spawn_z=${SCENE_SPAWN_Z[$position_index]}
        spawn_yaw=${SCENE_SPAWN_YAW[$position_index]}
        spawn_arguments="spawn_x:=${spawn_x} spawn_y:=${spawn_y} spawn_z:=${spawn_z} spawn_yaw:=${spawn_yaw}"
        spawn_description="(${spawn_x}, ${spawn_y}, ${spawn_z}, yaw=${spawn_yaw})"
    fi

    launch_terminal "spawn_${robot_name}" "
echo '==== Spawning ${robot_name}: ${spawn_description}, lidar=${ENABLE_LIDAR}, camera=${ENABLE_CAMERA} ===='
ros2 launch go2_config spawn_go2_velodyne_${robot_index}.launch.py use_sim_time:=true enable_lidar:=${ENABLE_LIDAR} enable_camera:=${ENABLE_CAMERA} ${spawn_arguments}
"
    wait_for_controllers_active "$robot_name"

    if [ "$robot_index" -lt 3 ]; then
        echo "${robot_name} 已就绪，等待 3 秒后启动下一只 Go2..."
        sleep 3
    fi
done

echo "全部启动命令已经发出。"
echo "无人机检查："
echo "  ros2 topic list | grep /uav1"
echo "  ros2 topic hz /uav1/camera/image_raw"
echo "  ros2 topic hz /uav1/camera/depth/image_raw"
echo "  ros2 topic hz /uav1/camera/points"
echo "  ros2 topic echo /uav1/custom_cmd_vel"
echo "机器狗控制器检查："
echo "  ros2 control list_controllers -c /go2_1/controller_manager"
echo "  ros2 control list_controllers -c /go2_2/controller_manager"
echo "  ros2 control list_controllers -c /go2_3/controller_manager"
