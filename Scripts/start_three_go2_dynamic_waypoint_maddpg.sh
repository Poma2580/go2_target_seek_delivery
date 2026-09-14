#!/bin/bash
# Full dynamic-pedestrian scenario with discrete MADDPG waypoint selection.
# The legacy continuous-control launcher remains unchanged as a fallback.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LEGACY_SCRIPT="$SCRIPT_DIR/start_three_go2_dynamic_tracking.sh"
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
#MODEL="$DELIVERY_ROOT/waypoint_maddpg_v0/runs/two_obstacles_108rays_final_gpu_20260826/best_model.pt"
MODEL="$DELIVERY_ROOT/waypoint_maddpg_v0/runs/three_seed_obstacle_curriculum_20260909_091240/seed_27/stage2_two_obstacles/best_model.pt"
GAZEBO_GUI=${GAZEBO_GUI:-true}
MADDPG_SELECTOR_EXECUTABLE=${MADDPG_SELECTOR_EXECUTABLE:-maddpg_waypoint_selector.py}
MADDPG_SELECTOR_EXTRA_ARGS=${MADDPG_SELECTOR_EXTRA_ARGS:-}
FOLLOWER_MUX_EXTRA_ARGS=${FOLLOWER_MUX_EXTRA_ARGS:-}
TARGET_FORCED_ROBOT=${TARGET_FORCED_ROBOT:-}
FIXED_LEADER_FINETUNE=${FIXED_LEADER_FINETUNE:-false}
export MADDPG_SELECTOR_EXECUTABLE MADDPG_SELECTOR_EXTRA_ARGS
export FOLLOWER_MUX_EXTRA_ARGS TARGET_FORCED_ROBOT FIXED_LEADER_FINETUNE
CHECK_ONLY=false

case "${1:-}" in
    "") ;;
    --gui) GAZEBO_GUI=true; shift ;;
    --headless) GAZEBO_GUI=false; shift ;;
    --check) CHECK_ONLY=true; shift ;;
    *)
        echo "用法: $0 [--gui|--headless|--check]" >&2
        exit 2
        ;;
esac
if [ "$#" -ne 0 ]; then
    echo "用法: $0 [--gui|--headless|--check]" >&2
    exit 2
fi
if [ "$GAZEBO_GUI" != true ] && [ "$GAZEBO_GUI" != false ]; then
    echo "ERROR: GAZEBO_GUI 只能设置为 true 或 false。" >&2
    exit 2
fi

if [ ! -f "$LEGACY_SCRIPT" ]; then
    echo "ERROR: 基础启动脚本不存在: $LEGACY_SCRIPT" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: MADDPG 选点模型不存在: $MODEL" >&2
    exit 1
fi

# Reuse the proven Gazebo/spawn/lidar/camera/map/Nav2/perception prefix without
# modifying the legacy file.  Only the controller tail is replaced below.
GENERATED_SCRIPT=$(mktemp "$SCRIPT_DIR/.dynamic_waypoint_maddpg.XXXXXX.sh")
cleanup_generated_script() {
    rm -f "$GENERATED_SCRIPT"
}
trap cleanup_generated_script EXIT INT TERM

awk -v gazebo_gui="$GAZEBO_GUI" \
    -v mux_extra_args="$FOLLOWER_MUX_EXTRA_ARGS" \
    -v forced_robot="$TARGET_FORCED_ROBOT" \
    -v fixed_leader="$FIXED_LEADER_FINETUNE" '
    /^# MADDPG 提前加载模型并等待接管信号/ { exit }
    {
        if (fixed_leader == "true" && /^# 终端 11：启动三套目标感知/) {
            skip_target_perception = 1
        }
        if (skip_target_perception) {
            if (/wait_for_topic "\/target_role\/perception_robot"/) {
                skip_target_perception = 0
            }
            next
        }
        sub("gazebo_target_seek_world.launch.py gui:=false",
            "gazebo_target_seek_world.launch.py gui:=" gazebo_gui)
        if (mux_extra_args != "" && /follower_cmd_vel_mux.py --ros-args/) {
            $0 = $0 " " mux_extra_args
        }
        if (forced_robot != "" && /three_go2_target_tracking.launch.py/) {
            $0 = $0 " forced_robot:=" forced_robot
        }
        print
    }
' "$LEGACY_SCRIPT" > "$GENERATED_SCRIPT"

cat >> "$GENERATED_SCRIPT" <<'WAYPOINT_TAIL'

#MODEL="$DELIVERY_ROOT/waypoint_maddpg_v0/runs/two_obstacles_108rays_final_gpu_20260826/best_model.pt"
MODEL="$DELIVERY_ROOT/waypoint_maddpg_v0/runs/three_seed_obstacle_curriculum_20260909_091240/seed_27/stage2_two_obstacles/best_model.pt"

# Preload the discrete policy.  It follows the elected perception role, reads
# both navigation dogs' /scan topics, and stays disabled until Nav2 approach is
# complete.  It publishes NavigateToPose goals only and never publishes cmd_vel.
if [ "$FIXED_LEADER_FINETUNE" = true ]; then
    # perception_robot_topic defaults to an empty string. Omit it instead of
    # passing "name:=", which is not valid ROS 2 parameter YAML.
    MADDPG_ROLE_ARGS="-p wait_for_enable:=false -p enabled:=true"
    selector_description="fixed-leader Gazebo fine-tuner"
else
    MADDPG_ROLE_ARGS="-p perception_robot_topic:=/target_role/perception_robot -p wait_for_enable:=true -p enabled:=false"
    selector_description="role-aware MADDPG waypoint selector (disabled)"
fi
launch_terminal "maddpg_waypoint_selector" "
echo '==== Preloading ${selector_description} ===='
ros2 run go2_mapping_nav ${MADDPG_SELECTOR_EXECUTABLE} --ros-args \
    -p use_sim_time:=true \
    -p model_path:=${MODEL} \
    -p global_frame:=merged_map \
    -p robot_names:='[go2_1,go2_2,go2_3]' \
    ${MADDPG_ROLE_ARGS} \
    -p enable_topic:=/dynamic_encircle/maddpg_enable \
    -p controller_ready_topic:=/maddpg_waypoint/controller_ready \
    -p controller_active_topic:=/maddpg_waypoint/controller_active \
    -p decision_period:=1.0 \
    -p nav_goal_update_period:=8.0 \
    -p require_initial_formation:=false \
    -p leader_speed_tolerance:=0.15 \
    -p speed_tolerance:=0.20 \
    -p dry_run:=false ${MADDPG_SELECTOR_EXTRA_ARGS}
"

wait_for_topic "/maddpg_waypoint/controller_ready"

# Display the exact lidar/candidate evidence used by every policy decision.
# This distinguishes a false BLOCKED observation from a learned side preference.
launch_terminal "maddpg_obstacle_monitor" "
echo '==== Starting MADDPG obstacle/action monitor ===='
ros2 run go2_mapping_nav maddpg_waypoint_monitor.py --ros-args \
    -p use_sim_time:=true
"

if [ "$FIXED_LEADER_FINETUNE" != true ]; then
    # Keep the legacy approach state machine, but use the waypoint selector's
    # handshake topics. Nav2 remains the sole cmd_vel owner after handoff.
    launch_terminal "nav2_dynamic_encircle_waypoint" "
echo '==== Starting Nav2 approach -> MADDPG waypoint handoff ===='
ros2 run go2_mapping_nav dynamic_encircle.py --ros-args \
    -p use_sim_time:=true \
    -p perception_robot_topic:=/target_role/perception_robot \
    -p robot_names:='[go2_1,go2_2,go2_3]' \
    -p maddpg_ready_topic:=/maddpg_waypoint/controller_ready \
    -p maddpg_active_topic:=/maddpg_waypoint/controller_active \
    -p maddpg_enable_topic:=/dynamic_encircle/maddpg_enable \
    -p switch_mux_to_maddpg:=false
"

    launch_terminal "watch_STAGE_Change" "
echo '==== Starting stage change monitor ===='
ros2 topic echo /dynamic_encircle/handoff_state std_msgs/msg/String --qos-durability transient_local
"

    wait_for_ros_service "/walking_target/start"
    launch_terminal "start_walking_target" "
echo '==== Starting walking target movement ===='
ros2 service call /walking_target/start std_srvs/srv/Trigger '{}'
"
    echo "完整流程已启动：三狗靠近行人 -> 安全交接 -> MADDPG 选点。"
else
    echo "固定领航微调已启动：go2_1 以 0.15 m/s 直行，go2_2/go2_3 由 MADDPG+Nav2 选点跟随。"
    echo "行人感知、动态围捕交接和行人运动均未启动。"
fi
echo "旧脚本 start_three_go2_dynamic_tracking.sh 未修改，可随时回退。"
echo "关键检查命令："
echo "  ros2 topic echo /dynamic_encircle/handoff_state --qos-durability transient_local"
echo "  ros2 topic echo /maddpg_waypoint/actions"
echo "  ros2 topic echo /maddpg_waypoint/goals"
echo "  ros2 topic echo /maddpg_waypoint/errors"
echo "  ros2 topic echo /maddpg_waypoint/decision_diagnostics"
echo "  ros2 topic echo /dynamic_encircle/use_maddpg --once --qos-durability transient_local"
echo "正常情况下 use_maddpg 始终为 false，因为 Nav2 始终拥有 cmd_vel。"

rm -f "$PID_DIR"/*.pid 2>/dev/null || true
rmdir "$PID_DIR" 2>/dev/null || true
WAYPOINT_TAIL

chmod +x "$GENERATED_SCRIPT"
if [ "$CHECK_ONLY" = true ]; then
    bash -n "$GENERATED_SCRIPT"
    echo "检查通过：完整派生脚本语法正确，模型文件存在，未启动 Gazebo（gui=$GAZEBO_GUI）。"
    exit 0
fi
echo "Gazebo GUI: $GAZEBO_GUI"
bash "$GENERATED_SCRIPT"
