#!/bin/bash
# Start the normal three-Go2/Nav2 stack, but replace inference with a two-stage
# online curriculum: 10k clear-world steps, then 30k one-box steps.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/start_three_go2_dynamic_waypoint_maddpg.sh"

export MADDPG_SELECTOR_EXECUTABLE=gazebo_maddpg_finetuner.py
export FIXED_LEADER_FINETUNE=true
export FOLLOWER_MUX_EXTRA_ARGS="-p max_navigation_linear_speed:=0.20 -p fixed_leader:=go2_1"
export MADDPG_SELECTOR_EXTRA_ARGS="-p finetune_seed:=2709 \
-p leader_name:=go2_1 \
-p follower_1:=go2_2 \
-p follower_2:=go2_3 \
-p clear_stage_steps:=10000 \
-p one_obstacle_stage_steps:=30000 \
-p clear_success_distance:=10.0 \
-p leader_forward_speed:=0.15 \
-p leader_heading_kp:=1.5 \
-p leader_max_yaw_rate:=0.60 \
-p reset_robots_each_episode:=true \
-p reset_x:='[-15.0, -14.0, -14.0]' \
-p enable_tensorboard:=true \
-p obstacle_min_size:=0.8 \
-p obstacle_max_size:=1.2 \
-p obstacle_height:=1.0 \
-p follower_max_speed:=0.20 \
-p nav_goal_update_period:=1.0 \
-p spawn_min_distance:=6.0 \
-p spawn_max_distance:=9.0 \
-p lane_jitter:=0.4 \
-p replay_size:=50000 \
-p batch_size:=128 \
-p warmup_steps:=256 \
-p actor_lr:=0.00001 \
-p critic_lr:=0.00005 \
-p initial_epsilon:=0.10 \
-p final_epsilon:=0.03 \
-p epsilon_decay_steps:=30000 \
-p checkpoint_interval:=5000"

if [ "$#" -eq 0 ]; then
    set -- --headless
fi
exec "$BASE_SCRIPT" "$@"
