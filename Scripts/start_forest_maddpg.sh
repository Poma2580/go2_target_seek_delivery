#!/bin/bash
# Start the complete visible forest MADDPG test: world, three Go2 robots,
# lidar/cameras, SLAM/Nav2/map merge, then the waypoint policy.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "森林 MADDPG 测试："
echo "  行人 0.12 m/s；GO1 上限 0.20 m/s；GO2/GO3 上限 0.30 m/s"
echo "  MADDPG 只为 GO2/GO3 选点，Nav2 负责到达选定点"
echo "  三只 Go2 均启用 Velodyne 和 RGB-D 相机"

"$SCRIPT_DIR/start_three_go2_velodyne.sh" \
    forest \
    --all-sensors \
    --mapping-nav \
    --gui

echo "森林、传感器、融合地图和三套 Nav2 已就绪，启动 MADDPG。"

# Match the proven dynamic-tracking launcher: run the policy stack in a clean
# host terminal instead of inheriting VS Code/Snap GTK and loader variables.
# Keep the terminal open on failure so the actual launch error remains visible.
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
    gnome-terminal --title="forest_maddpg_waypoint_nav2" -- bash -c "
cd '$SCRIPT_DIR/..'
'$SCRIPT_DIR/start_maddpg_waypoint_nav2.sh' --execute
status=\$?
echo
echo \"MADDPG launch exited with status \$status\"
exec bash
"

echo "MADDPG 已在独立终端 forest_maddpg_waypoint_nav2 中启动。"
