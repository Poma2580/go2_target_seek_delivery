#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="/home/wangantong/KD_all/go2_target_seek_delivery"
WS_DIR="${REPO_ROOT}/go2_ws_v2"
MADDPG_ROOT="${REPO_ROOT}/三角形MADDPG"
GAZEBO_RUN_ROOT="${MADDPG_ROOT}/runs/leader_slot_tracking_v0/GazeboMADDPG"

PRETRAINED_MODEL="${1:-${GAZEBO_RUN_ROOT}/gazebo_leader_stage1_shared_actor_b256_usteps20_alr2e-05_clr5e-05_20260809_152400/best_model.pt}"

cd "${WS_DIR}"
source /home/wangantong/miniconda3/etc/profile.d/conda.sh
conda activate maddpg_gpu
# ROS setup files may read optional variables such as AMENT_TRACE_SETUP_FILES.
# Do not use `set -u` while sourcing them.
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "========================================================================"
echo "Gazebo leader-slot curriculum fine-tuning"
echo "pretrained: ${PRETRAINED_MODEL}"
echo "run root:   ${GAZEBO_RUN_ROOT}"
echo "========================================================================"

ros2 run multi_go2_waypoint gazebo_leader_slot_train_stage1 --ros-args \
  -p use_sim_time:=true \
  -p reset_mode:=teleport \
  -p reset_pose_source:=script \
  -p target_reset_mode:=none \
  -p curriculum_stage:=1 \
  -p shared_actor:=true \
  -p pretrained_model_path:="${PRETRAINED_MODEL}" \
  -p total_timesteps:=20000 \
  -p max_steps:=300 \
  -p batch_size:=256 \
  -p warmup_steps:=1000 \
  -p update_every:=20 \
  -p actor_lr:=0.00001 \
  -p critic_lr:=0.00002 \
  -p noise_scale:=0.01 \
  -p min_noise:=0.002 \
  -p control_rate:=10.0 \
  -p leader_route_speed:=0.18 \
  -p leader_route_yaw:=0.0 \
  -p side_dist:=1.80 \
  -p leader_follow_dist:=2.70 \
  -p follower_max_linear:=0.45 \
  -p follower_max_angular:=0.45 \
  -p follower_accel_lin:=0.30 \
  -p follower_accel_ang:=0.25 \
  -p success_mean_slot_threshold:=2.00 \
  -p success_max_slot_threshold:=2.00 \
  -p success_yaw_threshold:=0.80 \
  -p success_hold_steps:=10 \
  -p early_stop_enable:=true \
  -p early_stop_min_steps:=3000 \
  -p early_stop_success_episodes:=2 \
  -p settle_time:=5.0

STAGE1_MODEL="$(find "${GAZEBO_RUN_ROOT}" -maxdepth 2 -type f -path '*gazebo_leader_stage1_shared_actor*/best_model.pt' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
if [[ -z "${STAGE1_MODEL}" ]]; then
  echo "ERROR: Could not find Stage1 best_model.pt under ${GAZEBO_RUN_ROOT}" >&2
  exit 1
fi

echo "========================================================================"
echo "Stage1 best model: ${STAGE1_MODEL}"
echo "Starting Gazebo curriculum Stage2..."
echo "========================================================================"

ros2 run multi_go2_waypoint gazebo_leader_slot_train_stage1 --ros-args \
  -p use_sim_time:=true \
  -p reset_mode:=teleport \
  -p reset_pose_source:=script \
  -p target_reset_mode:=none \
  -p curriculum_stage:=2 \
  -p shared_actor:=true \
  -p pretrained_model_path:="${STAGE1_MODEL}" \
  -p total_timesteps:=20000 \
  -p max_steps:=300 \
  -p batch_size:=256 \
  -p warmup_steps:=1000 \
  -p update_every:=20 \
  -p actor_lr:=0.00001 \
  -p critic_lr:=0.00002 \
  -p noise_scale:=0.008 \
  -p min_noise:=0.002 \
  -p control_rate:=10.0 \
  -p leader_route_speed:=0.18 \
  -p leader_route_yaw:=0.0 \
  -p side_dist:=1.80 \
  -p leader_follow_dist:=2.70 \
  -p follower_max_linear:=0.45 \
  -p follower_max_angular:=0.45 \
  -p follower_accel_lin:=0.30 \
  -p follower_accel_ang:=0.25 \
  -p success_mean_slot_threshold:=2.00 \
  -p success_max_slot_threshold:=2.00 \
  -p success_yaw_threshold:=0.80 \
  -p success_hold_steps:=20 \
  -p early_stop_enable:=false \
  -p settle_time:=5.0

echo "========================================================================"
echo "Gazebo leader-slot curriculum finished."
echo "Latest Stage2 best model:"
find "${GAZEBO_RUN_ROOT}" -maxdepth 2 -type f -path '*gazebo_leader_stage2_shared_actor*/best_model.pt' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
echo "========================================================================"
