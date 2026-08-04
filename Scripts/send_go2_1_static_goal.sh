#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--robot go2_1|go2_2|go2_3] [--map-mode auto|local|merged] <x> <y> [yaw] [segment_length]" >&2
}

is_number() {
    [[ $1 =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

ROBOT_NAME=go2_1
MAP_MODE=auto
while (( $# > 0 )); do
    case "$1" in
        --robot)
            if (( $# < 2 )); then
                usage
                exit 2
            fi
            ROBOT_NAME=$2
            shift 2
            ;;
        --map-mode)
            if (( $# < 2 )); then
                usage
                exit 2
            fi
            MAP_MODE=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ ! $ROBOT_NAME =~ ^go2_[123]$ ]]; then
    echo "ERROR: robot must be go2_1, go2_2, or go2_3." >&2
    exit 2
fi
if [[ ! $MAP_MODE =~ ^(auto|local|merged)$ ]]; then
    echo "ERROR: map mode must be auto, local, or merged." >&2
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

ROS2_BIN=${ROS2_BIN:-ros2}

case "${MAP_MODE}" in
    local)
        GLOBAL_FRAME="${ROBOT_NAME}/map"
        MAP_TOPIC="/${ROBOT_NAME}/map"
        ;;
    merged)
        GLOBAL_FRAME="merged_map"
        MAP_TOPIC="/merged_map"
        ;;
    auto)
        BT_NAVIGATOR="/${ROBOT_NAME}/bt_navigator"
        if ! PARAM_OUTPUT="$(timeout 10 "${ROS2_BIN}" param get "${BT_NAVIGATOR}" global_frame 2>&1)"; then
            echo "ERROR: cannot read global_frame from ${BT_NAVIGATOR}." >&2
            echo "Make sure this robot's Nav2 is running, or pass --map-mode local|merged." >&2
            echo "${PARAM_OUTPUT}" >&2
            exit 1
        fi
        GLOBAL_FRAME="$(
            printf '%s\n' "${PARAM_OUTPUT}" \
                | sed -n 's/^String value is: //p' \
                | tail -n 1
        )"
        case "${GLOBAL_FRAME}" in
            "${ROBOT_NAME}/map")
                MAP_TOPIC="/${ROBOT_NAME}/map"
                ;;
            merged_map)
                MAP_TOPIC="/merged_map"
                ;;
            *)
                echo "ERROR: ${BT_NAVIGATOR} returned unsupported global_frame '${GLOBAL_FRAME}'." >&2
                echo "Expected '${ROBOT_NAME}/map' or 'merged_map'." >&2
                exit 1
                ;;
        esac
        ;;
esac

echo "Using global_frame=${GLOBAL_FRAME}, map_topic=${MAP_TOPIC} (map mode: ${MAP_MODE})."

exec "${ROS2_BIN}" run go2_mapping_nav static_goal_nav.py --ros-args \
    -p action_name:=/${ROBOT_NAME}/navigate_to_pose \
    -p map_topic:="${MAP_TOPIC}" \
    -p global_frame:="${GLOBAL_FRAME}" \
    -p robot_frame:=${ROBOT_NAME}/base_link \
    -p goal_x:="${GOAL_X}" \
    -p goal_y:="${GOAL_Y}" \
    -p goal_yaw:="${GOAL_YAW}" \
    -p segment_length:="${SEGMENT_LENGTH}"
