#!/bin/bash
# Usage: ./start_three_go2_dynamic_tracking.sh

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
YOLO_MODEL=$DELIVERY_ROOT/yolov8s.pt
MERGED_MAP_TIMEOUT=${MERGED_MAP_TIMEOUT:-120}

if [ ! -f "$YOLO_MODEL" ]; then
    echo "ERROR: YOLO model not found: $YOLO_MODEL"
    exit 1
fi

if [ ! -d "$WS/install" ]; then
    echo "ERROR: workspace is not built: $WS/install"
    exit 1
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "ERROR: gnome-terminal not found."
    exit 1
fi

if ! [[ "$MERGED_MAP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MERGED_MAP_TIMEOUT must be a positive integer number of seconds."
    exit 2
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

    # Avoid loading VS Code Snap GTK/runtime libraries in host ROS processes.
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

# 终端 1：启动固定的 target_seek/city 世界。
launch_terminal "go2_world" "
echo '==== Starting target_seek/city world ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/gazebo/model_states"
wait_for_topic "/clock"

echo "Gazebo 世界已就绪，等待 3 秒以完成稳定加载..."
sleep 3

# 终端 2-4：依次生成三只 Go2；全部开启 Velodyne，仅 go2_1 开启 RGB-D。
for robot_index in 1 2 3; do
    robot_name="go2_${robot_index}"
    if [ "$robot_index" -eq 1 ]; then
        enable_camera=true
    else
        enable_camera=false
    fi

    launch_terminal "spawn_${robot_name}" "
echo '==== Spawning ${robot_name}: lidar=true, camera=${enable_camera} ===='
ros2 launch go2_config spawn_go2_velodyne_${robot_index}.launch.py scene:=city use_sim_time:=true enable_lidar:=true enable_camera:=${enable_camera}
"

    wait_for_controllers_active "$robot_name"
    wait_for_topic "/${robot_name}/velodyne_points"
    wait_for_topic "/${robot_name}/odom"

    if [ "$robot_index" -eq 1 ]; then
        wait_for_topic "/go2_1/camera/image_raw"
        wait_for_topic "/go2_1/camera/depth/image_raw"
        wait_for_topic "/go2_1/camera/depth/camera_info"
    fi

    if [ "$robot_index" -lt 3 ]; then
        echo "${robot_name} 已就绪，等待 3 秒后启动下一只 Go2..."
        sleep 3
    fi
done

mkdir -p "$WS/src/go2_mapping_nav/runtime/logs"

# 终端 5：启动已知位姿地图融合。
map_merger_log="$WS/src/go2_mapping_nav/runtime/logs/map_merger.log"
: > "$map_merger_log"
launch_terminal "three_go2_map_merge" "
echo '==== Starting known-pose map merger: scene=city ===='
ros2 launch go2_mapping_nav three_go2_map_merge.launch.py scene:=city use_sim_time:=true use_rviz:=false >${map_merger_log} 2>&1
"

# 速度所有权选择器：Nav2 与 MADDPG 只能通过各自私有输入控制跟随犬。
launch_terminal "follower_cmd_vel_mux" "
echo '==== Starting follower command velocity mux (initial owner: Nav2) ===='
ros2 run go2_mapping_nav follower_cmd_vel_mux.py --ros-args -p use_sim_time:=true
"

# 终端 6-8：依次启动三套 RTAB-Map + Nav2，并统一使用融合地图。
merged_map_ready=false
for robot_index in 1 2 3; do
    robot_name="go2_${robot_index}"
    nav_cmd_vel_arg=""
    if [ "$robot_index" -ge 2 ]; then
        nav_cmd_vel_arg="cmd_vel_topic:=/${robot_name}/nav_cmd_vel"
    fi
    mapping_log="$WS/src/go2_mapping_nav/runtime/logs/${robot_name}_mapping_nav.log"
    : > "$mapping_log"
    launch_terminal "mapping_nav_${robot_name}" "
echo '==== Starting ${robot_name} mapping and navigation ===='
ros2 launch go2_mapping_nav ${robot_name}_mapping_nav.launch.py use_sim_time:=true use_merged_map:=true use_rviz:=false delete_db_on_start:=true ${nav_cmd_vel_arg} >${mapping_log} 2>&1
"

    if [ "$merged_map_ready" = false ]; then
        if ! wait_for_topic_message "/merged_map" "$MERGED_MAP_TIMEOUT"; then
            echo "融合地图启动失败，停止后续 mapping-nav 和感知评估启动。" >&2
            exit 1
        fi
        merged_map_ready=true
    fi

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

# 终端 9：启动统一三机建图导航 RViz。
launch_terminal "three_go2_mapping_rviz" "
echo '==== Starting unified three-Go2 mapping RViz ===='
rviz2 -d \$(ros2 pkg prefix go2_mapping_nav)/share/go2_mapping_nav/rviz/three_go2_mapping_nav.rviz
"

# 终端 10：启动行人真值状态广播。
launch_terminal "actor_state" "
echo '==== Starting actor_state_publisher ===='
ros2 run multi_go2_waypoint actor_state_publisher --ros-args -p use_sim_time:=true
"

wait_for_topic "/walking_target/odom"

# 终端 11：启动目标感知（go2_1 RGB-D -> YOLO + 深度估计 -> 目标 odom）。
launch_terminal "target_perception" "
echo '==== Starting target_perception ===='
ros2 run multi_go2_waypoint target_perception --ros-args -p use_sim_time:=true -p robot_namespace:=go2_1 -p model_path:=$YOLO_MODEL -p imgsz:=640 -p inference_rate:=8.0 -p max_image_age:=0.30
"

wait_for_topic "/go2_1/target_perception/debug_image"

# MADDPG 提前加载模型并等待接管信号；输出进入私有 mux 输入话题。
launch_terminal "maddpg_follower_controller" "
echo '==== Preloading MADDPG follower controller (disabled) ===='
ros2 run multi_go2_waypoint gazebo_leader_slot_controller --ros-args -p use_sim_time:=true -p wait_for_enable:=true -p command_topic_suffix:=maddpg_cmd_vel
"

# 终端 12：启动基于 Nav2 的三 Go2 动态围捕。
launch_terminal "nav2_dynamic_encircle" "
echo '==== Starting Nav2 dynamic encircle ===='
ros2 run go2_mapping_nav dynamic_encircle.py --ros-args -p use_sim_time:=true -p target_odom_topic:=/go2_1/target_estimated/odom
"

# 终端 13：打开 RQT 查看检测调试图。
launch_terminal "rqt_debug_image" "
echo '==== Starting rqt_image_view debug image ===='
ros2 run rqt_image_view rqt_image_view /go2_1/target_perception/debug_image
"

# 终端 14：启动感知误差评估。
launch_terminal "perception_eval" "
echo '==== Starting perception_eval ===='
ros2 run multi_go2_waypoint perception_eval --ros-args -p use_sim_time:=true
"

wait_for_ros_service "/walking_target/start"

# 终端 15：最后启动行人运动。
launch_terminal "start_walking_target" "
echo '==== Starting walking target movement ===='
ros2 service call /walking_target/start std_srvs/srv/Trigger \"{}\"
"

echo "三 Go2 建图导航、动态围捕、目标感知评估与行人启动命令已经全部发出。"
echo "可用以下命令检查："
echo "  ros2 topic list | grep -E 'go2_[123]/(velodyne_points|map)|merged_map'"
echo "  ros2 topic echo /merged_map nav_msgs/msg/OccupancyGrid --once"
echo "  ros2 action list | grep navigate_to_pose"
echo "  ros2 topic echo /walking_target/odom --once"
echo "  ros2 topic echo /go2_1/target_estimated/odom --once"
echo "  ros2 topic list | grep -E 'go2_1/(target_perception|perception_error)'"
echo "  ros2 node list | grep /nav2_dynamic_encircle"
echo "  ros2 action info /go2_2/navigate_to_pose"
echo "  ros2 action info /go2_3/navigate_to_pose"
