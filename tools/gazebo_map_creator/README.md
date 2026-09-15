# Gazebo 二维占用地图生成器

本工具从 Gazebo collision 几何生成 city、forest、airport 场景的二维占用地图。
ROS 包、命令入口与基准产物分别存放，且不依赖 `go2_ws_v2/src` 中的源码连接。

## 目录结构

```text
gazebo_map_creator/
├── vendor/            # 上游 ROS 包及许可证
├── scripts/           # 地图生成与结果检查入口
├── artifacts/maps/    # PGM、PNG、Nav2 YAML 基准地图
├── build/             # 独立 colcon 构建目录（Git 忽略）
├── install/           # 独立 colcon 安装目录（Git 忽略）
└── log/               # 独立 colcon 日志目录（Git 忽略）
```

上游版本和本地补丁记录在 [UPSTREAM.md](UPSTREAM.md)。

## 独立构建

必须退出 conda 并使用 ROS 系统 Python：

```bash
conda deactivate 2>/dev/null || true
which python3  # 必须输出 /usr/bin/python3
cd tools/gazebo_map_creator
source /opt/ros/humble/setup.bash
rosdep check --from-paths vendor --ignore-src
colcon build --symlink-install --base-paths vendor
source install/setup.bash
ros2 pkg executables gazebo_map_creator
cd ../..
```

该构建只安装 `gazebo_map_creator` 和 `gazebo_map_creator_interface` 到本工具的
`install` 目录。普通的 `cd go2_ws_v2 && colcon build` 不会发现或构建它们。

## 生成地图

先确认没有其他 `gzserver` 正在运行，再从仓库根目录执行：

```bash
tools/gazebo_map_creator/scripts/generate_maps.sh all
tools/gazebo_map_creator/scripts/generate_maps.sh city
```

输出位于 `tools/gazebo_map_creator/artifacts/maps/<scene>/`。脚本会自行加载
本工具的 `install/setup.bash`，检查地图尺寸、resolution、origin 和图像内容，
并且只停止由自身启动的 Gazebo 进程。

生成的三套地图是
[`pedestrian_map`](../pedestrian_map/README.md) 进行路线验证和机器人位姿生成时的
默认输入。
