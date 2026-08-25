#!/bin/bash
# 启动一个场景、预置 uav1，并顺序导入三只 Go2。
#
# 用法：
#   ./start_three_go2_velodyne.sh [city|forest|airport] [--lidar] [--camera] [--mapping-nav]
#   ./start_three_go2_velodyne.sh forest --all-sensors

set -u

usage() {
    cat <<'EOF'
用法：./start_three_go2_velodyne.sh [场景] [传感器选项]

场景（默认 city）：
  city | qy | target_seek  target_seek 城市场景
  forest                   森林场景
  airport                  机场场景

传感器选项（默认读取场景 YAML，当前三个场景均关闭）：
  --lidar                  三只 Go2 开启 3D Velodyne
  --camera                 三只 Go2 开启 RGB-D 相机
  --all-sensors            同时开启 3D Velodyne 和 RGB-D 相机
  --mapping-nav            启动三套建图/导航、已知位姿地图融合和统一 RViz
  -h, --help               显示本帮助

可选环境变量：
  USE_GAZEBO_GUI=false     关闭 Gazebo GUI，减轻三套导航启动时的 CPU 压力
  USE_RVIZ=false           使用 --mapping-nav 时关闭统一融合 RViz
  MERGED_MAP_TIMEOUT=120   等待 /merged_map 首条消息的秒数

示例：
  ./start_three_go2_velodyne.sh
  ./start_three_go2_velodyne.sh forest
  ./start_three_go2_velodyne.sh airport --all-sensors
  ./start_three_go2_velodyne.sh city --all-sensors --mapping-nav
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
KD_MODEL_ROOT=$DELIVERY_ROOT/KD_MODEL

SCENE=city
ENABLE_LIDAR=auto
ENABLE_CAMERA=auto
MAPPING_NAV=false
SCENE_SET=false
USE_GAZEBO_GUI=${USE_GAZEBO_GUI:-true}
MERGED_MAP_TIMEOUT=${MERGED_MAP_TIMEOUT:-120}

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
        --mapping-nav)
            MAPPING_NAV=true
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

if [ "$MAPPING_NAV" = true ]; then
    ENABLE_LIDAR=true
fi

if [ "$USE_GAZEBO_GUI" != true ] && [ "$USE_GAZEBO_GUI" != false ]; then
    echo "ERROR: USE_GAZEBO_GUI 必须是 true 或 false。" >&2
    exit 2
fi
if ! [[ "$MERGED_MAP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MERGED_MAP_TIMEOUT 必须为正整数秒数。" >&2
    exit 2
fi

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

    # When launched from the VS Code Snap, GTK/LD_LIBRARY_PATH points at the
    # Snap runtime and the host gnome-terminal can fail with a GLIBC_PRIVATE
    # libpthread error.  Strip only terminal-loader variables; COMMON_ENV
    # below restores the ROS/Gazebo library paths inside the new shell.
    env \
        -u GTK_PATH \
        -u LD_LIBRARY_PATH \
        -u SNAP \
        -u SNAP_NAME \
        -u SNAP_DATA \
        -u SNAP_USER_DATA \
        -u SNAP_REAL_HOME \
        -u SNAP_LIBRARY_PATH \
        -u SNAP_COMMON \
        -u SNAP_USER_COMMON \
        -u GDK_PIXBUF_MODULE_FILE \
        -u GDK_PIXBUF_MODULEDIR \
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
until ros2 service list 2>/dev/null | awk -v target='${service_name}' '\$0 == target { found=1 } END { exit !found }'; do
    echo 'waiting for ${service_name} ...'
    sleep 2
done
echo '${service_name} is ready.'
"
}

wait_for_ros_action() {
    local action_name=$1
    echo "等待 ROS action ${action_name} 出现..."
    bash -c "$COMMON_ENV
until ros2 action list 2>/dev/null | awk -v target='${action_name}' '\$0 == target { found=1 } END { exit !found }'; do
    echo 'waiting for ${action_name} ...'
    sleep 2
done
echo '${action_name} is ready.'
"
}

wait_for_log_message() {
    local log_path=$1
    local message=$2
    local description=$3
    echo "等待 ${description}..."
    until [ -f "$log_path" ] && grep -Fq "$message" "$log_path"; do
        echo "waiting for ${description} ..."
        sleep 2
    done
    echo "${description} is ready."
}

wait_for_topic() {
    local topic_name=$1
    echo "等待 ROS topic ${topic_name} 出现..."
    bash -c "$COMMON_ENV
until ros2 topic list 2>/dev/null | awk -v target='${topic_name}' '\$0 == target { found=1 } END { exit !found }'; do
    echo 'waiting for ${topic_name} ...'
    sleep 2
done
echo '${topic_name} is ready.'
"
}

wait_for_topic_message() {
    local topic_name=$1
    local timeout_seconds=$2
    echo "等待 ROS topic ${topic_name} 的首条消息（${timeout_seconds}s 超时）..."
    if ! timeout "${timeout_seconds}" bash -c "$COMMON_ENV
ros2 topic echo '${topic_name}' nav_msgs/msg/OccupancyGrid --once >/dev/null 2>&1
"; then
        echo "ERROR: 等待 ${topic_name} 首条消息超时。" >&2
        echo "请检查 /go2_1/map、/go2_2/map、/go2_3/map 和 map_merger.log。" >&2
        return 1
    fi
    echo "${topic_name} has published its first message."
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
echo "world：${WORLD_PATH}"
echo "Go2 传感器覆盖：lidar=${ENABLE_LIDAR}, camera=${ENABLE_CAMERA}"
echo "独立建图导航：${MAPPING_NAV}"
echo "Gazebo GUI：${USE_GAZEBO_GUI}"
echo "Go2 出生点及 auto 传感器值由 go2_scenario_config 的 ${SCENE}.yaml 提供。"

launch_terminal "go2_world_${SCENE}" "
echo '==== Starting ${SCENE} world with uav1 ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=${USE_GAZEBO_GUI} world:=$WORLD_PATH
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/clock"
wait_for_topic "/uav1/camera/image_raw"

echo "世界与 uav1 已就绪，等待 5 秒以完成 Gazebo 稳定加载..."
sleep 5

for robot_index in 1 2 3; do
    robot_name="go2_${robot_index}"

    launch_terminal "spawn_${robot_name}" "
echo '==== Spawning ${robot_name}: scene=${SCENE}, lidar=${ENABLE_LIDAR}, camera=${ENABLE_CAMERA} ===='
ros2 launch go2_config spawn_go2_velodyne_${robot_index}.launch.py scene:=${SCENE} use_sim_time:=true enable_lidar:=${ENABLE_LIDAR} enable_camera:=${ENABLE_CAMERA}
"
    wait_for_controllers_active "$robot_name"

    if [ "$robot_index" -lt 3 ]; then
        echo "${robot_name} 已就绪，等待 3 秒后启动下一只 Go2..."
        sleep 3
    fi
done

if [ "$MAPPING_NAV" = true ]; then
    USE_RVIZ=${USE_RVIZ:-true}
    if [ "$USE_RVIZ" != true ] && [ "$USE_RVIZ" != false ]; then
        echo "ERROR: USE_RVIZ 必须是 true 或 false。" >&2
        exit 2
    fi

    echo "三只 Go2 传感器已启动，等待建图导航输入 topic..."
    for robot_index in 1 2 3; do
        robot_name="go2_${robot_index}"
        wait_for_topic "/${robot_name}/velodyne_points"
        wait_for_topic "/${robot_name}/odom"
    done

    mkdir -p "$WS/runtime/logs"
    map_merger_log="$WS/runtime/logs/map_merger.log"
    : > "$map_merger_log"
    launch_terminal "three_go2_map_merge" "
echo '==== Starting known-pose map merger: scene=${SCENE} ===='
ros2 launch go2_mapping_nav three_go2_map_merge.launch.py use_sim_time:=true use_rviz:=false >${map_merger_log} 2>&1
"

    merged_map_ready=false
    for robot_index in 1 2 3; do
        robot_name="go2_${robot_index}"
        mapping_log="$WS/runtime/logs/${robot_name}_mapping_nav.log"
        : > "$mapping_log"
        launch_terminal "mapping_nav_${robot_name}" "
echo '==== Starting ${robot_name} mapping and navigation ===='
ros2 launch go2_mapping_nav ${robot_name}_mapping_nav.launch.py use_sim_time:=true use_merged_map:=true use_rviz:=false delete_db_on_start:=true >${mapping_log} 2>&1
"
        if [ "$merged_map_ready" = false ]; then
            if ! wait_for_topic_message "/merged_map" "$MERGED_MAP_TIMEOUT"; then
                echo "融合地图启动失败，停止后续 mapping-nav 启动。" >&2
                exit 1
            fi
            merged_map_ready=true
        fi
        # The action name is advertised before Nav2 has finished activating.
        # Watch the launch log instead of polling ROS lifecycle services, then
        # leave a settling interval before starting the next cold stack.
        wait_for_log_message \
            "$mapping_log" \
            "Managed nodes are active" \
            "${robot_name} Nav2 lifecycle"
        wait_for_ros_action "/${robot_name}/navigate_to_pose"
        if [ "$robot_index" -lt 3 ]; then
            echo "${robot_name} 导航已激活，等待 1 秒后启动下一套导航..."
            sleep 1
        fi
    done

    if [ "$USE_RVIZ" = true ]; then
        launch_terminal "three_go2_mapping_rviz" "
echo '==== Starting unified three-Go2 mapping RViz ===='
rviz2 -d \$(ros2 pkg prefix go2_mapping_nav)/share/go2_mapping_nav/rviz/three_go2_mapping_nav.rviz
"
    fi
fi

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
echo "静态目标请通过 Nav2 NavigateToPose 或 Scripts/send_go2_1_static_goal.sh 下发。"
