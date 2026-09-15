#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DELIVERY_ROOT="$(cd -- "${TOOL_ROOT}/../.." && pwd)"
MAP_ROOT="${TOOL_ROOT}/artifacts/maps"
REQUESTED_SCENE="${1:-all}"

set +u
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
if [[ ! -f "${TOOL_ROOT}/install/setup.bash" ]]; then
  echo "ERROR: build tools/gazebo_map_creator before generating maps" >&2
  exit 2
fi
source "${TOOL_ROOT}/install/setup.bash"
set -u

if [[ "$(command -v python3)" != "/usr/bin/python3" ]]; then
  echo "ERROR: ROS tools must use /usr/bin/python3, got $(command -v python3)" >&2
  exit 2
fi
if pgrep -x gzserver >/dev/null; then
  echo "ERROR: another gzserver is running; stop it before map generation" >&2
  exit 2
fi

case "${REQUESTED_SCENE}" in
  all) SCENES=(city forest airport) ;;
  city|forest|airport) SCENES=("${REQUESTED_SCENE}") ;;
  *) echo "Usage: $0 [city|forest|airport|all]" >&2; exit 2 ;;
esac

active_pid=""
cleanup() {
  if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
    kill "${active_pid}" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 "${active_pid}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${active_pid}" 2>/dev/null; then
      kill -KILL "${active_pid}" 2>/dev/null || true
    fi
    wait "${active_pid}" 2>/dev/null || true
  fi
  active_pid=""
}
trap cleanup EXIT INT TERM

for scene in "${SCENES[@]}"; do
  case "${scene}" in
    city)
      world="${DELIVERY_ROOT}/QY_MODEL/target_seek"
      bounds="(-41.3,-40.8,0.05)(116.3,40.8,1.80)"
      width=1576; height=816; origin_x=-41.3; origin_y=-40.8
      scene_models="${DELIVERY_ROOT}/QY_MODEL/models"
      ;;
    forest)
      world="${DELIVERY_ROOT}/KD_MODEL/world/forestV3_dynamic.world"
      bounds="(-68.73,-77.37,0.05)(81.27,72.63,1.80)"
      width=1500; height=1500; origin_x=-68.73; origin_y=-77.37
      scene_models="${DELIVERY_ROOT}/KD_MODEL/models:${DELIVERY_ROOT}/QY_MODEL/models"
      ;;
    airport)
      world="${DELIVERY_ROOT}/KD_MODEL/world/airport_dynamic.world"
      bounds="(-180.0,-75.0,0.05)(180.0,75.0,1.80)"
      width=3600; height=1500; origin_x=-180.0; origin_y=-75.0
      scene_models="${DELIVERY_ROOT}/KD_MODEL/models:${DELIVERY_ROOT}/QY_MODEL/models"
      ;;
  esac

  output_dir="${MAP_ROOT}/${scene}"
  prefix="${output_dir}/${scene}"
  mkdir -p "${output_dir}"
  log_file="$(mktemp "/tmp/gazebo_map_creator_${scene}.XXXXXX.log")"
  export GAZEBO_MODEL_PATH="${scene_models}:${HOME}/.gazebo/models:${GAZEBO_MODEL_PATH:-}"

  echo "Starting ${scene}: ${world}"
  gzserver -s libgazebo_ros_init.so -s libgazebo_map_creator.so "${world}" \
    >"${log_file}" 2>&1 &
  active_pid=$!

  service_ready=false
  for _ in $(seq 1 120); do
    if ! kill -0 "${active_pid}" 2>/dev/null; then
      echo "ERROR: gzserver exited while loading ${scene}; log=${log_file}" >&2
      wait "${active_pid}" || true
      active_pid=""
      exit 2
    fi
    if ros2 service list 2>/dev/null | grep -Fxq '/world/save_map'; then
      service_ready=true
      break
    fi
    sleep 1
  done
  if [[ "${service_ready}" != true ]]; then
    echo "ERROR: /world/save_map did not appear for ${scene}; log=${log_file}" >&2
    exit 2
  fi

  timeout 7200 ros2 run gazebo_map_creator request_map.py \
    --corners "${bounds}" --resolution 0.10 --skip-vertical-scan --filename "${prefix}"

  cleanup
  for suffix in pgm png yaml; do
    if [[ ! -s "${prefix}.${suffix}" ]]; then
      echo "ERROR: missing output ${prefix}.${suffix}; log=${log_file}" >&2
      exit 2
    fi
  done

  /usr/bin/python3 "${SCRIPT_DIR}/inspect_map.py" \
    --yaml "${prefix}.yaml" --png "${prefix}.png" --image-name "${scene}.pgm" \
    --expected-size "${width}" "${height}" \
    --expected-origin "${origin_x}" "${origin_y}" --expected-resolution 0.10
  rm -f -- "${prefix}.pcd" "${prefix}.bt"
  echo "Completed ${scene}; gzserver log=${log_file}"
done
