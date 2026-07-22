# Go2 多场景使用手册

**Latest Updated 2026.07.21**

本文档说明如何在 Gazebo Classic 中启动 `target_seek`、森林、机场等场景：

- 单狗模式：一只带 2D 激光雷达与深度相机的 Unitree Go2，可在 `target_seek`、`forest`、`airport` 三个场景间切换，支持键盘控制和 SLAM。
- 2D 多狗模式：同一场景里导入 3 只带 2D 激光雷达的 Go2，各自独立键盘控制。
- 3D 多狗模式：同一场景里导入 3 只 Go2，可按需开启 3D Velodyne 和 RGB-D，相互独立控制并可用 RViz 查看数据。
- 多狗围捕模式：三只 Go2 使用命名空间 Nav2 在共享地图上连续导航，并对静态目标完成等边三角形围捕。
- 动态追踪模式：go2_1 用 RGB-D 相机在线视觉感知（YOLO+深度）估计行人位置并稳定跟随（单狗已实现；三狗三角编队尚未调通）。

## 版本要求

```text
Ubuntu 22.04
ROS 2 Humble
Gazebo Classic 11
```


## 第 0 步：拉取项目并新建开发分支

进入你希望存放项目的目录，然后从 GitHub 拉取仓库：

```bash
cd /你希望存放项目的目录
git clone https://github.com/Poma2580/go2_target_seek_delivery.git
cd go2_target_seek_delivery
```

切换到最新 `main`，再新建自己的本地开发分支：

```bash
git checkout main
git pull origin main
git checkout -b feature/姓名-修改内容
```

主要目录应为：

```text
go2_target_seek_delivery/
  go2_ws_v2/
  QY_MODEL/
    models/
  KD_MODEL/
    models/
    world/
  README.md
```

## 第 1 步：安装依赖

```bash
sudo apt update
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-robot-state-publisher \
  ros-humble-robot-localization \
  ros-humble-xacro \
  ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  ros-humble-teleop-twist-keyboard \
  ros-humble-velodyne-description \
  ros-humble-velodyne-gazebo-plugins \
  libassimp-dev \
  libignition-math6-dev \
  libtinyxml2-dev \
  python3-colcon-common-extensions \
  xterm
```

## 第 2 步：下载 OSRF/Gazebo 模型

如果目标机器还没有 Gazebo 官方模型缓存：

```bash
mkdir -p ~/.gazebo
git clone https://github.com/osrf/gazebo_models ~/.gazebo/models
```

如果 `~/.gazebo/models` 已经存在，不想覆盖原目录：

```bash
git clone https://github.com/osrf/gazebo_models ~/gazebo_models_osrf
export GAZEBO_MODEL_PATH=~/gazebo_models_osrf:$GAZEBO_MODEL_PATH
```

## 第 3 步：写入项目环境变量

在项目根目录执行一次，把项目路径写入 `~/.bashrc`：

```bash
cd /实际/项目路径/go2_target_seek_delivery

cat <<EOF >> ~/.bashrc

# go2_target_seek_delivery
export DELIVERY_ROOT="$(pwd)"
export QY_MODEL_ROOT="\$DELIVERY_ROOT/QY_MODEL"
export KD_MODEL_ROOT="\$DELIVERY_ROOT/KD_MODEL"
export GAZEBO_MODEL_PATH="\$QY_MODEL_ROOT/models:\$KD_MODEL_ROOT/models:\$GAZEBO_MODEL_PATH"
export GAZEBO_MODEL_DATABASE_URI=""
EOF

source ~/.bashrc
echo $DELIVERY_ROOT
```

后续新开的终端会自动获得这些变量。运行任何 ROS 命令前先执行 `conda deactivate`，并确认 `which python3` 输出 `/usr/bin/python3`。

## 第 4 步：编译


```bash
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 单狗模式：Gazebo + Go2 + 键盘控制 + SLAM

每个终端先执行：

```
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
```

终端 1：从 3 个场景中选择一个，启动世界和单只 Go2：

```bash
# target_seek 场景
ros2 launch go2_config gazebo_world_2d_lidar.launch.py scene:=qy gui:=true rviz:=false

# 森林场景
ros2 launch go2_config gazebo_world_2d_lidar.launch.py scene:=forest gui:=true rviz:=false

# 机场场景
ros2 launch go2_config gazebo_world_2d_lidar.launch.py scene:=airport gui:=true rviz:=false
```

终端 2：启动键盘控制，控制时鼠标焦点需要停留在该终端窗口内：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

终端 3：按需启动 SLAM：

```bash
ros2 launch go2_config slam.launch.py sim:=true rviz:=true
```


## 多狗模式：三只 Go2 + 2D 雷达

在同一个 `target_seek` 场景里同时导入 3 只带 2D 激光雷达的 Go2，
分别用 `/go2_1`、`/go2_2`、`/go2_3` 命名空间，每只用一个键盘窗口独立控制。

每个终端先执行：

```bash
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
```

终端 1：只启动 `target_seek` 世界：

```bash
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true
```

终端 2/3/4：逐只顺序导入，等上一只稳定后再启动下一只：

```bash
ros2 launch go2_config spawn_go2_1.launch.py
ros2 launch go2_config spawn_go2_2.launch.py
ros2 launch go2_config spawn_go2_3.launch.py
```

终端 5：弹出 3 个 xterm 键盘窗口：

```bash
ros2 launch go2_config teleop_three_go2.launch.py
```

键盘窗口与单狗相同（i 前进、, 后退、j 左转、l 右转、k 停）。
键盘只对当前鼠标焦点所在的 xterm 窗口生效，要控制哪只狗，先点一下对应窗口再按键。

## 多场景三狗：可选 3D Velodyne / RGB-D

脚本支持 `city`（默认）、`forest`、`airport` 三个场景；每个场景均预置 `uav1`。它会等待
世界、`/uav1/camera/image_raw` 就绪并缓冲后，再依次导入三只 Go2，等待每只控制器 active。

```bash
conda deactivate
which python3  # 应为 /usr/bin/python3
cd $DELIVERY_ROOT/Scripts

./start_three_go2_velodyne.sh                 # city
./start_three_go2_velodyne.sh forest           # 森林
./start_three_go2_velodyne.sh airport          # 机场
./start_three_go2_velodyne.sh forest --lidar
./start_three_go2_velodyne.sh airport --all-sensors
```

默认关闭三只 Go2 的 3D 雷达和 RGB-D 相机；`--lidar`、`--camera`、`--all-sensors` 可按需开启。
三个场景的 world、出生位姿、目标和围捕半径统一保存在
`go2_ws_v2/src/multi_go2_nav2/config/scenes/*.yaml`。脚本直接读取 YAML，不再依赖 launch 内重复
填写的默认出生点。

按需另开终端启动键盘或 RViz：

```bash
ros2 launch go2_config teleop_three_go2.launch.py
ros2 launch go2_config view_three_go2_velodyne.launch.py
```

## 多狗模式：三只 Go2 纯 Nav2 静态围捕

airport 当前使用一个共享 `/map` 和三套命名空间隔离的 Nav2。每套 Nav2 使用 SmacPlanner2D、
Regulated Pure Pursuit Controller 和 velocity smoother，实现整条路径连续跟踪。自定义协调器只生成
围捕候选点、调用 Nav2 路径 action 做三狗目标分配，再并行发送 `NavigateToPose`；它不实现 A*，
也不直接发布速度。

终端 1 启动 airport 和三只 Go2：

```bash
cd $DELIVERY_ROOT/Scripts
./start_three_go2_velodyne.sh airport
```

终端 2 启动共享地图、三套 Nav2、围捕协调器和 RViz：

```bash
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch multi_go2_nav2 multi_go2_nav2.launch.py
```

修改机场参数只需编辑 `multi_go2_nav2/config/scenes/airport.yaml`。出生位姿、目标位姿和围捕半径
改变后不需要重新生成地图；只有 world 的静态 collision、地图边界、分辨率或高度切片改变时才需
重新运行 `world_to_grid`。

RViz 固定 frame 为 `map`，会在同一地图上显示最终 `/go2_i/selected_plan`、
`/go2_i/actual_path`、三只狗模型、目标和分配终点。候选查询会覆盖的 `/go2_i/plan` 默认关闭。
若只想检查 Nav2 而暂不执行围捕：

```bash
ros2 launch multi_go2_nav2 multi_go2_nav2.launch.py start_coordinator:=false
```

旧 `multi_go2_waypoint` 自研 A*/waypoint 控制代码作为历史对照保留，但 airport 纯 Nav2 运行时不要
同时启动它，否则会与 Nav2 竞争 `/go2_i/cmd_vel`。完整设计、实施记录和验收命令见 `Docs/1.txt`。

### 从 airport world 离线生成原始栅格地图

`world_to_grid` 会先通过 `gz sdf -p` 展开 world 中的模型引用，再读取所有 `collision`（不读取
`visual`），把与 Go2 高度切片相交的三维世界 AABB 保守投影到二维栅格。当前 airport 地图范围为
`[-180,-75] → [180,75] m`，分辨率为 `0.20 m/cell`，高度切片为 `0.03–0.80 m`。
该地图表示原始碰撞体，不包含机器人半径或安全距离膨胀。

airport 引用了 Gazebo 官方模型，正式生成前请确保模型完整位于 `~/.gazebo/models`。编译并重新生成：

```bash
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select world_to_grid multi_go2_waypoint
source install/setup.bash

ros2 run world_to_grid world_to_grid \
  --world "$KD_MODEL_ROOT/world/airport" \
  --output-prefix "$DELIVERY_ROOT/go2_ws_v2/src/multi_go2_waypoint/maps/airport" \
  --bounds -180 -75 180 75 \
  --resolution 0.20 \
  --z-min 0.03 \
  --z-max 0.80 \
  --border-cells 1 \
  --ignore-model uav1_iris_depth_camera
```

命令生成 `airport.pgm`、`airport.yaml`、逐碰撞体诊断表 `airport_collisions.csv` 和
`airport_preview.svg`。其中 PGM/YAML 会随 `multi_go2_waypoint` 安装到
`share/multi_go2_waypoint/maps`，YAML 内使用相对图像路径，生成后需重新构建该包。
额外模型搜索目录可重复使用 `--model-path PATH` 指定，也会与 `GAZEBO_MODEL_PATH` 合并。

可用 Nav2 map_server 验证安装后的地图：

```bash
MAP_YAML="$(ros2 pkg prefix multi_go2_waypoint)/share/multi_go2_waypoint/maps/airport.yaml"
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:="$MAP_YAML"

# 在另一个已 source 的终端执行
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
ros2 topic echo /map nav_msgs/msg/OccupancyGrid --once \
  --qos-reliability reliable --qos-durability transient_local --field info
```


森林场景也可以支持一键启动静态围捕:

```bash
# 森林：围捕飞机残骸
cd $DELIVERY_ROOT/Scripts
./start_three_go2_forest.sh
```

## 多狗模式：三只 Go2 联合“拦截式围捕”动态行人追踪

> **现状说明**：三狗联合“拦截式围捕”已**初步调通**——go2_1 用 RGB-D 相机在线视觉感知
> （YOLO 检测 + 深度反投影）估计行人位置作为目标源，go2_1 稳定跟在行人身后做感知，
> go2_2/go2_3 绕到行人前方后**同步冲刺合围**。


一键启动（务必先 `conda deactivate`，确认 `which python3` 为 `/usr/bin/python3`）：

```bash
cd $DELIVERY_ROOT/Scripts
./start_three_go2_dynamic_tracking.sh
```

脚本会按顺序拉起：target_seek 世界 → go2_1（RGB-D 相机）→ go2_2/go2_3 → 行人真值桥接
`actor_state_publisher` → 目标感知 `target_perception` → 动态追踪控制 `dynamic_encircle`
（目标源 `/go2_1/target_estimated/odom`）→ rqt 调试图 → `perception_eval` 误差评估。


## 模型参数说明

### 机器狗

```text
型号/模型名: Unitree Go2
URDF/xacro robot name: go2
主本体 link: trunk
base_link 到 trunk: 固定关节，xyz=0 0 0, rpy=0 0 0
city 场景中，3D 多狗默认出生点:
  go2_1: x=0,  y=-4,  z=0.50, yaw=1.57
  go2_2: x=10, y=-17, z=0.50, yaw=0
  go2_3: x=60, y=10,  z=0.50, yaw=0
forest / airport 通过 start_three_go2_velodyne.sh 启动时:
  go2_1: x=0, y=-4; go2_2: x=2, y=-4; go2_3: x=0, y=-6; z=0.50, yaw=0
机身碰撞盒尺寸: 0.3762 x 0.0935 x 0.114 m
机身质量: 6.921 kg
```

### Go2 传感器

2D 激光雷达：

```text
外观模型: Hokuyo, mesh=hokuyo.dae
link: front_laser
相对 base_link 固定关节: xyz=0.225 0 0.105, rpy=0 0 0
Gazebo ray sensor 额外 pose: xyz=-0.032 0 0.171, rpy=0 0 0
发布话题: /scan
frame_id: front_laser
扫描角度: -3.14159 到 3.14159 rad
采样数: 720
量程: 0.12 到 10.0 m
更新率: 20 Hz
噪声 stddev: 0.01
```

3D Velodyne 雷达：

```text
型号: Velodyne VLP-16
mesh: $(find velodyne_description)/meshes/VLP16_base_1.dae, VLP16_base_2.dae, VLP16_scan.dae
base link: velodyne_base_link
scan link: velodyne
相对 base_link 固定关节: xyz=0.2 0 0.08, rpy=0 0 0
velodyne_base_link 到 velodyne: xyz=0 0 0.0377, rpy=0 0 0
Gazebo sensor 类型: ray
发布话题: /velodyne_points；多狗 3D 模式为 /go2_i/velodyne_points
frame_id: velodyne；多狗 3D 模式为 go2_i/velodyne
水平扫描角度: -3.141592653589793 到 3.141592653589793 rad
水平采样数: 440
垂直扫描角度: -0.2617993877991494 到 0.2617993877991494 rad
垂直线数: 16
ray range: 0.3 到 131.0 m，resolution=0.001
plugin 输出 range: min_range=0.9, max_range=130.0
更新率: 10 Hz
点云组织: organize_cloud=false
噪声: gaussian_noise=0.008
Gazebo 插件: libgazebo_ros_velodyne_laser.so
```

深度相机：

```text
类型: Gazebo depth camera
camera_link 相对 base_link: xyz=0.28 0 0.12, rpy=0 0 0
RGB frame 相对 camera_link: xyz=0.02 -0.02 0, rpy=0 0 0
depth frame 相对 camera_link: xyz=0.02 0.02 0, rpy=0 0 0
optical frame 旋转: rpy=-1.5708 0 -1.5708
分辨率: 640 x 480
FOV: 1.047 rad
深度范围: 0.05 到 10.0 m
更新率: 30 Hz
主要话题: /camera/rgb/image_raw, /camera/depth/image_raw, /camera/depth/points
```

IMU：

```text
link: imu_link
相对 trunk: xyz=0 0 0, rpy=0 0 0
更新率: 100 Hz
插件输出 remap: ~/out := data
```

### 无人机

```text
外层模型: uav1_iris_depth_camera 到 uav5_iris_depth_camera
机体模型: uavN_iris
底层机型: 3DR Iris Quadrotor
深度相机子模型: uavN_depth_camera
GPS 子模型: uavN_gps
```

### 无人机传感器

深度相机：

```text
挂载位置: uavN_depth_camera 相对 uavN_iris::base_link，xyz=0.1 0 0, rpy=0 0 0
相机模型自身 pose: xyz=0 0 0.035, rpy=0 0 0
mesh: realsense_camera/meshes/realsense.dae
仿真资源名: RealSense
型号说明: realsense_camera/model.config 描述来自 Intel RealSense R200；深度相机 SDF 注释按 Intel RealSense D455 深度范围配置
分辨率: 848 x 480
FOV: 1.5009831567 rad
深度裁剪: near=0.001, far=65.535
plugin 最小深度: 0.2
更新率: 10 Hz
命名空间示例: /uav1
frame 示例: uav1/camera_link
```

GPS：

```text
挂载名称: gps0
相对 uavN_iris::base_link: xyz=0.05 0 0.04, rpy=0 0 0
更新率: 5 Hz
```

IMU 与其它仿真插件：

```text
IMU link: /imu_link
IMU 相对机体: xyz=0 0 0.02, rpy=0 0 0
IMU topic 示例: /uav1/imu
其它插件: magnetometer, barometer, groundtruth, mavlink interface
```
