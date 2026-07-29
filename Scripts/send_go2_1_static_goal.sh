#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--robot go2_1|go2_2|go2_3] <x> <y> [yaw] [segment_length]" >&2
}

is_number() {
    [[ $1 =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

ROBOT_NAME=go2_1
if (( $# >= 1 )) && [[ $1 == "--robot" ]]; then
    if (( $# < 2 )); then
        usage
        exit 2
    fi
    ROBOT_NAME=$2
    shift 2
fi

if [[ ! $ROBOT_NAME =~ ^go2_[123]$ ]]; then
    echo "ERROR: robot must be go2_1, go2_2, or go2_3." >&2
    exit 2
fi

if (( $# < 2 || $# > 4 )); then
    usage
    exit 2
fi

GOAL_X=$1
GOAL_Y=$2
GOAL_YAW=${3:-0}
SEGMENT_LENGTH=${4:-5.0}
for value in "${GOAL_X}" "${GOAL_Y}" "${GOAL_YAW}" "${SEGMENT_LENGTH}"; do
    if ! is_number "${value}"; then
        echo "ERROR: goal coordinates and yaw must be finite numeric values." >&2
        exit 2
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${DELIVERY_ROOT}/go2_ws_v2"

conda deactivate 2>/dev/null || true
if [[ "$(which python3)" != "/usr/bin/python3" ]]; then
    echo "ERROR: python3 is $(which python3), expected /usr/bin/python3." >&2
    exit 1
fi

if [[ ! -f "${WORKSPACE}/install/setup.bash" ]]; then
    echo "ERROR: Workspace is not built: ${WORKSPACE}/install/setup.bash is missing." >&2
    exit 1
fi

export DELIVERY_ROOT
set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE}/install/setup.bash"
set -u

exec ros2 run go2_mapping_nav static_goal_nav.py --ros-args \
    -p action_name:=/${ROBOT_NAME}/navigate_to_pose \
    -p map_topic:=/${ROBOT_NAME}/map \
    -p global_frame:=${ROBOT_NAME}/map \
    -p robot_frame:=${ROBOT_NAME}/base_link \
    -p goal_x:="${GOAL_X}" \
    -p goal_y:="${GOAL_Y}" \
    -p goal_yaw:="${GOAL_YAW}" \
    -p segment_length:="${SEGMENT_LENGTH}"
