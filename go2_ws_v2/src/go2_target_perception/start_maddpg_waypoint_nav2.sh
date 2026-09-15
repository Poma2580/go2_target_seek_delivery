#!/bin/bash
# Start only the MADDPG waypoint selector. Gazebo, merged_map and Nav2 must run.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DELIVERY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WS="$DELIVERY_ROOT/go2_ws_v2"
MODEL="$DELIVERY_ROOT/waypoint_maddpg_v0/runs/two_obstacles_108rays_final_gpu_20260826/best_model.pt"
DRY_RUN=true

case "${1:-}" in
    ""|--dry-run) ;;
    --execute) DRY_RUN=false ;;
    *) echo "用法: $0 [--dry-run|--execute]" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL" ]; then
    echo "ERROR: 模型不存在: $MODEL" >&2
    exit 1
fi
if [ ! -f "$WS/install/setup.bash" ]; then
    echo "ERROR: 请先编译 go2_ws_v2。" >&2
    exit 1
fi

export DELIVERY_ROOT
# ROS 2 setup files probe optional environment variables which may be unset.
# Temporarily disable nounset while sourcing them, then restore strict mode.
set +u
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u

echo "GO1 跟踪动态行人；MADDPG 每 1 秒选点，Nav2 同动作目标每 3 秒刷新。"
echo "速度层级：行人约 0.12 m/s，GO1 上限 0.20 m/s，GO2/GO3 上限 0.30 m/s。"
echo "dry_run=$DRY_RUN model=$MODEL"

safe_param_set() {
    local node_name=$1
    local parameter_name=$2
    local parameter_value=$3

    if timeout 5s ros2 param set "$node_name" "$parameter_name" "$parameter_value"; then
        return 0
    fi

    echo "WARN: $node_name 的 $parameter_name 设置超时或失败，继续启动 MADDPG。" >&2
    return 0
}

set_nav2_linear_limit() {
    local robot_name=$1
    local speed=$2
    safe_param_set "/${robot_name}/controller_server" FollowPath.max_vel_x "$speed"
    safe_param_set "/${robot_name}/controller_server" FollowPath.max_speed_xy "$speed"
    safe_param_set "/${robot_name}/velocity_smoother" max_velocity "[$speed, 0.0, 1.0]"
    safe_param_set "/${robot_name}/velocity_smoother" min_velocity "[-$speed, 0.0, -1.0]"
}

if [ "$DRY_RUN" = false ]; then
    echo "将 GO2/GO3 Nav2 线速度限制为 0.30 m/s；GO1 上限为 0.20 m/s。"
    set_nav2_linear_limit go2_2 0.30
    set_nav2_linear_limit go2_3 0.30
fi

echo "正在启动 maddpg_waypoint_nav2.launch.py（应立即出现三个节点的启动日志）..."
exec ros2 launch go2_mapping_nav maddpg_waypoint_nav2.launch.py \
    use_sim_time:=true \
    dry_run:="$DRY_RUN" \
    model_path:="$MODEL" \
    leader_name:=go2_1 \
    follower_1:=go2_2 \
    follower_2:=go2_3 \
    global_frame:=merged_map \
    decision_period:=1.0 \
    nav_goal_update_period:=3.0 \
    initial_formation_tolerance:=0.5 \
    track_pedestrian:=true \
    leader_speed_tolerance:=0.15 \
    follower_speed_tolerance:=0.20
