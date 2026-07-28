#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${DELIVERY_ROOT}/go2_ws_v2"
RUNTIME_DIR="${WORKSPACE}/src/go2_mapping_nav/runtime"
LOG_DIR="${RUNTIME_DIR}/logs"
USE_RVIZ="${USE_RVIZ:-true}"

declare -a CHILD_PIDS=()

cleanup() {
    local pid
    trap - EXIT INT TERM
    for pid in "${CHILD_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -INT "${pid}" 2>/dev/null || true
        fi
    done
}

wait_for_topic() {
    local topic_name=$1
    local timeout_seconds=$2
    local deadline=$((SECONDS + timeout_seconds))

    echo "Waiting up to ${timeout_seconds}s for ${topic_name}..."
    while (( SECONDS < deadline )); do
        # Do not require ripgrep at runtime: grep is available on standard Ubuntu/ROS installs.
        if ros2 topic list 2>/dev/null | grep -Fqx -- "${topic_name}"; then
            echo "${topic_name} is ready."
            return 0
        fi
        sleep 1
    done

    echo "ERROR: Timed out waiting for ${topic_name}." >&2
    return 1
}

start_background_launch() {
    local log_file=$1
    shift
    "$@" >"${log_file}" 2>&1 &
    CHILD_PIDS+=("$!")
}

trap cleanup EXIT INT TERM

conda deactivate 2>/dev/null || true
if [[ "$(which python3)" != "/usr/bin/python3" ]]; then
    echo "ERROR: python3 is $(which python3), expected /usr/bin/python3." >&2
    exit 1
fi

if [[ ! -f "${WORKSPACE}/install/setup.bash" ]]; then
    echo "ERROR: Workspace is not built: ${WORKSPACE}/install/setup.bash is missing." >&2
    exit 1
fi
if [[ "${USE_RVIZ}" != "true" && "${USE_RVIZ}" != "false" ]]; then
    echo "ERROR: USE_RVIZ must be true or false." >&2
    exit 2
fi

mkdir -p "${RUNTIME_DIR}/maps" "${LOG_DIR}"
export DELIVERY_ROOT
export QY_MODEL_ROOT="${DELIVERY_ROOT}/QY_MODEL"
export KD_MODEL_ROOT="${DELIVERY_ROOT}/KD_MODEL"
export GAZEBO_MODEL_PATH="${QY_MODEL_ROOT}/models:${KD_MODEL_ROOT}/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=""

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE}/install/setup.bash"
set -u

start_background_launch "${LOG_DIR}/go2_1_mapping_world.log" \
    ros2 launch go2_config gazebo_target_seek_world.launch.py \
    gui:=true "world:=${QY_MODEL_ROOT}/target_seek"
wait_for_topic "/clock" 30
echo "Waiting 5s for the Gazebo world to settle before spawning go2_1..."
sleep 5

start_background_launch "${LOG_DIR}/go2_1_mapping_spawn.log" \
    ros2 launch go2_config spawn_go2_velodyne_1.launch.py \
    scene:=city use_sim_time:=true use_ground_truth_odom:=true \
    enable_lidar:=true enable_camera:=false
wait_for_topic "/go2_1/velodyne_points" 60
wait_for_topic "/go2_1/odom" 60

echo "Starting go2_1 online mapping and Nav2 in the foreground..."
echo "World log: ${LOG_DIR}/go2_1_mapping_world.log"
echo "Spawn log: ${LOG_DIR}/go2_1_mapping_spawn.log"
ros2 launch go2_mapping_nav go2_1_mapping_nav.launch.py \
    use_sim_time:=true "use_rviz:=${USE_RVIZ}" delete_db_on_start:=true
