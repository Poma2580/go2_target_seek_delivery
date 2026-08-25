#!/bin/bash
# Usage: ./start_three_go2_dynamic_tracking.sh [city|forest|airport]

set -u

usage() {
    cat <<'EOF'
用法：./start_three_go2_dynamic_tracking.sh [场景]

场景（默认 city）：
  city | qy | target_seek  target_seek 城市场景
  forest                   森林动态行人场景
  airport                  机场动态行人场景

可选环境变量：
  MERGED_MAP_TIMEOUT=120       等待 /merged_map 首条消息的秒数
  ACTOR_SERVICE_TIMEOUT=30     等待 /walking_target/start 的秒数
  MAX_GO2_RESTARTS=3           机器狗翻倒时的最大自动重启次数
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WS=$DELIVERY_ROOT/go2_ws_v2
QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
KD_MODEL_ROOT=$DELIVERY_ROOT/KD_MODEL
YOLO_MODEL=$DELIVERY_ROOT/yolov8s.pt
MERGED_MAP_TIMEOUT=${MERGED_MAP_TIMEOUT:-120}
ACTOR_SERVICE_TIMEOUT=${ACTOR_SERVICE_TIMEOUT:-30}
GO2_RESTART_COUNT=${GO2_RESTART_COUNT:-0}
MAX_GO2_RESTARTS=${MAX_GO2_RESTARTS:-3}
SCENE=city
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
        WORLD_PATH=$KD_MODEL_ROOT/world/forestV3_dynamic.world
        ;;
    airport)
        WORLD_PATH=$KD_MODEL_ROOT/world/airport_dynamic.world
        ;;
esac

RUN_ID="go2_tracking_$$_${GO2_RESTART_COUNT}_$(date +%s%N)"
RUNTIME_DIR="$WS/runtime"
PID_DIR="$RUNTIME_DIR/pids/$RUN_ID"

if [ ! -f "$YOLO_MODEL" ]; then
    echo "ERROR: YOLO model not found: $YOLO_MODEL"
    exit 1
fi

if [ ! -d "$WS/install" ] || [ ! -f "$WORLD_PATH" ]; then
    echo "ERROR: workspace is not built or world does not exist: $WORLD_PATH"
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

if ! [[ "$ACTOR_SERVICE_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ACTOR_SERVICE_TIMEOUT must be a positive integer number of seconds."
    exit 2
fi

if ! [[ "$GO2_RESTART_COUNT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GO2_RESTART_COUNT must be a non-negative integer."
    exit 2
fi

if ! [[ "$MAX_GO2_RESTARTS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_GO2_RESTARTS must be a non-negative integer."
    exit 2
fi

mkdir -p "$PID_DIR"

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

collect_process_tree() {
    local parent_pid=$1
    local child_pid

    while IFS= read -r child_pid; do
        [ -n "$child_pid" ] || continue
        collect_process_tree "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    PROCESS_TREE+=("$parent_pid")
}

pid_belongs_to_current_run() {
    local pid=$1
    [ -r "/proc/$pid/environ" ] || return 1
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
        | grep -Fxq "GO2_TRACKING_RUN_ID=$RUN_ID"
}

cleanup_current_run() {
    local pid_file
    local root_pid
    local pid
    local deadline
    local still_running=false
    local -a PROCESS_TREE=()

    [ -d "$PID_DIR" ] || return 0
    echo "正在清理本轮启动的 ROS/Gazebo 进程（run_id=$RUN_ID）..."

    for pid_file in "$PID_DIR"/*.pid; do
        [ -e "$pid_file" ] || continue
        read -r root_pid < "$pid_file" || continue
        if ! [[ "$root_pid" =~ ^[1-9][0-9]*$ ]]; then
            echo "WARNING: 忽略无效 PID 文件：$pid_file" >&2
            continue
        fi
        if ! kill -0 "$root_pid" 2>/dev/null; then
            continue
        fi
        if ! pid_belongs_to_current_run "$root_pid"; then
            echo "WARNING: PID $root_pid 不属于本轮运行，跳过。" >&2
            continue
        fi
        collect_process_tree "$root_pid"
    done

    if [ "${#PROCESS_TREE[@]}" -gt 0 ]; then
        # collect_process_tree 按子进程优先、父进程最后的顺序填充数组。
        kill -TERM "${PROCESS_TREE[@]}" 2>/dev/null || true
        deadline=$((SECONDS + 5))
        while [ "$SECONDS" -lt "$deadline" ]; do
            still_running=false
            for pid in "${PROCESS_TREE[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    still_running=true
                    break
                fi
            done
            [ "$still_running" = false ] && break
            sleep 0.2
        done

        still_running=false
        for pid in "${PROCESS_TREE[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running=true
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        if [ "$still_running" = true ]; then
            echo "部分进程未在 5 秒内退出，已强制终止。"
        fi
    fi

    rm -f "$PID_DIR"/*.pid 2>/dev/null || true
    rmdir "$PID_DIR" 2>/dev/null || true
    echo "本轮进程清理完成。"
}

launch_terminal() {
    local title=$1
    local command=$2
    local pid_file="$PID_DIR/${title}.pid"
    local terminal_pid=""

    if ! [[ "$title" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "ERROR: invalid terminal title for PID tracking: $title" >&2
        cleanup_current_run
        exit 1
    fi
    rm -f "$pid_file"

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
        GO2_TRACKING_RUN_ID="$RUN_ID" \
        gnome-terminal --title="$title" -- bash -c "
export GO2_TRACKING_RUN_ID=$RUN_ID
echo \"\$BASHPID\" > '$pid_file'
$COMMON_ENV
$command
exec bash
"

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if [ -s "$pid_file" ]; then
            read -r terminal_pid < "$pid_file" || true
            if [[ "$terminal_pid" =~ ^[1-9][0-9]*$ ]] \
                && kill -0 "$terminal_pid" 2>/dev/null \
                && pid_belongs_to_current_run "$terminal_pid"; then
                return 0
            fi
        fi
        sleep 0.2
    done

    echo "ERROR: 无法记录或验证终端 $title 的 PID。" >&2
    cleanup_current_run
    exit 1
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

wait_for_ros_service_timeout() {
    local service_name=$1
    local timeout_seconds=$2
    echo "等待 ROS service ${service_name}（${timeout_seconds}s 超时）..."
    timeout "${timeout_seconds}" bash -c "$COMMON_ENV
until ros2 service list 2>/dev/null | awk -v target='${service_name}' '\$0 == target { found=1 } END { exit !found }'; do
    sleep 1
done
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

# 终端 1：启动所选动态行人世界。
echo "动态追踪场景：${SCENE}"
echo "world：${WORLD_PATH}"
launch_terminal "go2_world_${SCENE}" "
echo '==== Starting ${SCENE} dynamic pedestrian world ===='
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true world:=$WORLD_PATH
"

wait_for_ros_service "/spawn_entity"
wait_for_topic "/gazebo/model_states"
wait_for_topic "/clock"

if ! wait_for_ros_service_timeout "/walking_target/start" "$ACTOR_SERVICE_TIMEOUT"; then
    echo "ERROR: ${SCENE} world 未在 ${ACTOR_SERVICE_TIMEOUT}s 内提供 /walking_target/start。" >&2
    echo "请检查 walking_target actor 和 libwalking_target_controller.so。" >&2
    cleanup_current_run
    exit 1
fi

echo "Gazebo 世界已就绪，等待 3 秒以完成稳定加载..."
sleep 3

# 终端 2-4：依次生成三只 Go2；全部开启 Velodyne 和 RGB-D。
for robot_index in 1 2 3; do
    robot_name="go2_${robot_index}"
    enable_camera=true

    launch_terminal "spawn_${robot_name}" "
echo '==== Spawning ${robot_name}: lidar=true, camera=${enable_camera} ===='
ros2 launch go2_config spawn_go2_velodyne_${robot_index}.launch.py scene:=${SCENE} use_sim_time:=true enable_lidar:=true enable_camera:=${enable_camera}
"

    wait_for_controllers_active "$robot_name"
    wait_for_topic "/${robot_name}/velodyne_points"
    wait_for_topic "/${robot_name}/odom"

    wait_for_topic "/${robot_name}/camera/image_raw"
    wait_for_topic "/${robot_name}/camera/depth/image_raw"
    wait_for_topic "/${robot_name}/camera/depth/camera_info"

    if [ "$robot_index" -lt 3 ]; then
        echo "${robot_name} 已就绪，等待 3 秒后启动下一只 Go2..."
        sleep 3
    fi
done

echo "三只 Go2 已完成导入，等待 2 秒后检查姿态..."
sleep 2

bash -c "$COMMON_ENV
ros2 run go2_scenario_config check_three_go2_attitude --ros-args \
    -p model_states_topic:=/gazebo/model_states \
    -p robot_names:='[go2_1,go2_2,go2_3]' \
    -p roll_limit_deg:=90.0 \
    -p required_frames:=3 \
    -p timeout_seconds:=10.0
"
attitude_check_status=$?

case "$attitude_check_status" in
    0)
        echo "三只 Go2 姿态检查通过，继续启动终端 5。"
        ;;
    10)
        echo "检测到机器狗连续 3 帧 abs(roll) > 90 度。" >&2
        cleanup_current_run
        if [ "$GO2_RESTART_COUNT" -ge "$MAX_GO2_RESTARTS" ]; then
            echo "ERROR: 已达到最大自动重启次数 $MAX_GO2_RESTARTS，停止启动。" >&2
            exit 10
        fi
        next_restart_count=$((GO2_RESTART_COUNT + 1))
        echo "1 秒后进行第 ${next_restart_count}/${MAX_GO2_RESTARTS} 次自动重启..."
        sleep 1
        exec env \
            GO2_RESTART_COUNT="$next_restart_count" \
            MAX_GO2_RESTARTS="$MAX_GO2_RESTARTS" \
            "$SCRIPT_PATH" "$@"
        ;;
    *)
        echo "ERROR: 姿态检查失败（退出码 $attitude_check_status），不自动重启。" >&2
        cleanup_current_run
        exit "$attitude_check_status"
        ;;
esac

mkdir -p "$RUNTIME_DIR/logs"

# 终端 5：启动已知位姿地图融合。
map_merger_log="$RUNTIME_DIR/logs/map_merger.log"
: > "$map_merger_log"
launch_terminal "three_go2_map_merge" "
echo '==== Starting known-pose map merger: scene=${SCENE} ===='
ros2 launch go2_mapping_nav three_go2_map_merge.launch.py use_sim_time:=true use_rviz:=false >${map_merger_log} 2>&1
"

# 速度所有权选择器：Nav2 与 MADDPG 只能通过各自私有输入控制跟随犬。
launch_terminal "follower_cmd_vel_mux" "
echo '==== Starting follower command velocity mux (initial owner: Nav2) ===='
ros2 run go2_dynamic_encircle follower_cmd_vel_mux --ros-args -p use_sim_time:=true
"

# 终端 6-8：依次启动三套 RTAB-Map + Nav2，并统一使用融合地图。
merged_map_ready=false
for robot_index in 1 2 3; do
    robot_name="go2_${robot_index}"
    nav_cmd_vel_arg="cmd_vel_topic:=/${robot_name}/nav_cmd_vel"
    mapping_log="$RUNTIME_DIR/logs/${robot_name}_mapping_nav.log"
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

# # 终端 10：启动行人真值状态广播。
# launch_terminal "actor_state" "
# echo '==== Starting actor_state_publisher ===='
# ros2 run walking_target_controller actor_state_publisher --ros-args -p use_sim_time:=true
# "

# wait_for_topic "/walking_target/odom"

# 终端 11：启动三套目标感知和首次稳定发现角色选举。
launch_terminal "three_go2_target_tracking" "
echo '==== Starting three-Go2 target perception and role selector ===='
ros2 launch go2_target_perception three_go2_target_tracking.launch.py use_sim_time:=true model_path:=$YOLO_MODEL
"

wait_for_topic "/target_role/perception_robot"

# MADDPG 提前加载模型并等待接管信号；输出进入私有 mux 输入话题。
launch_terminal "maddpg_follower_controller" "
echo '==== Preloading MADDPG follower controller (disabled) ===='
ros2 run go2_dynamic_encircle gazebo_leader_slot_controller --ros-args -p use_sim_time:=true -p wait_for_enable:=true -p command_topic_suffix:=maddpg_cmd_vel
"

# 终端 12：启动基于 Nav2 的三 Go2 动态围捕。
launch_terminal "nav2_dynamic_encircle" "
echo '==== Starting Nav2 dynamic encircle ===='
ros2 run go2_dynamic_encircle dynamic_encircle --ros-args -p use_sim_time:=true -p scene:=${SCENE} -p perception_robot_topic:=/target_role/perception_robot -p robot_names:="[go2_1,go2_2,go2_3]"
"

# 终端 13-15：分别打开三只狗的压缩相机。
# for robot_index in 1 2 3; do
#     robot_name="go2_${robot_index}"
#     launch_terminal "rqt_image_view_${robot_name}" "
# echo '==== Starting ${robot_name} rqt_image_view ===='
# ros2 run rqt_image_view rqt_image_view /${robot_name}/camera/image_raw/compressed
# "
# done

# # 启动当前感知狗的误差评估。
# launch_terminal "perception_eval" "
# echo '==== Starting perception_eval ===='
# ros2 run go2_target_perception perception_eval --ros-args -p use_sim_time:=true -p perception_robot_topic:=/target_role/perception_robot -p robot_names:="[go2_1,go2_2,go2_3]"
# "

# wait_for_ros_service "/walking_target/start"

# 启动监控阶段切换。
launch_terminal "watch_STAGE_Change" "
echo '==== Starting stage change monitor ===='
ros2 topic echo \
    /dynamic_encircle/handoff_state \
    std_msgs/msg/String \
    --qos-durability transient_local
"

# 最后启动行人运动。
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
echo "  ros2 topic echo /target_role/perception_robot --once"
echo "  ros2 topic list | grep -E 'go2_[123]/(target_estimated|target_perception|perception_error)'"
echo "  ros2 node list | grep /nav2_dynamic_encircle"
echo "  ros2 action info /go2_1/navigate_to_pose"
echo "  ros2 action info /go2_2/navigate_to_pose"
echo "  ros2 action info /go2_3/navigate_to_pose"

# 正常启动完成后不再需要本轮 PID 记录；已启动的终端继续运行。
rm -f "$PID_DIR"/*.pid 2>/dev/null || true
rmdir "$PID_DIR" 2>/dev/null || true
