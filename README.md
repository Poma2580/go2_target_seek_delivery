# Go2 多场景使用手册

**Latest Updated 2026.08.25**

本文档说明如何在 Gazebo Classic 中启动 `target_seek`、森林、机场等场景：

- 单狗模式：一只带 2D 激光雷达与深度相机的 Unitree Go2，可在 `target_seek`、`forest`、`airport` 三个场景间切换，支持键盘控制和 SLAM。
- 2D 多狗模式：同一场景里导入 3 只带 2D 激光雷达的 Go2，各自独立键盘控制。
- 3D 多狗模式：同一场景里导入 3 只 Go2，可按需开启 3D Velodyne 和 RGB-D，相互独立控制并可用 RViz 查看数据。
- 多狗 Nav2 模式：三只 Go2 在 city、forest、airport 场景中分别进行 RTAB-Map 建图与 Nav2 自主导航。
- 动态围捕模式：三只 Go2 均使用 RGB-D + YOLO 感知目标，最先稳定发现目标的机器人持续视觉跟踪，其余两只先由 Nav2 导航到围捕位置，再安全交接给 MADDPG 编队控制。

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
  ros-humble-pointcloud-to-laserscan \
  ros-humble-rtabmap-ros \
  ros-humble-slam-toolbox \
  ros-humble-teleop-twist-keyboard \
  ros-humble-diagnostic-updater \
  ros-humble-velodyne-description \
  ros-humble-velodyne-gazebo-plugins \
  libassimp-dev \
  libignition-math6-dev \
  libtinyxml2-dev \
  python3-colcon-common-extensions \
  xterm
```

## 第 2 步：安装 Python 依赖

运行本仓库的 ROS 节点时不要使用 conda，请使用系统 Python 安装 `requirements.txt` 中的依赖：

```bash
conda deactivate
which python3  # 必须输出 /usr/bin/python3
python3 -m pip install -r requirements.txt
```

## 第 3 步：下载 OSRF/Gazebo 模型

如果目标机器还没有 Gazebo 官方模型缓存：

```bash
mkdir -p ~/.gazebo
git clone https://github.com/osrf/gazebo_models ~/.gazebo/models
```

如果 `~/.gazebo/models` 已经存在，不想覆盖原目录：

```bash
git clone https://github.com/osrf/gazebo_models ~/gazebo_models_osrf

grep -qxF \
  'export GAZEBO_MODEL_PATH="$HOME/gazebo_models_osrf:${GAZEBO_MODEL_PATH:-}"' \
  ~/.bashrc || echo \
  'export GAZEBO_MODEL_PATH="$HOME/gazebo_models_osrf:${GAZEBO_MODEL_PATH:-}"' \
  >> ~/.bashrc

source ~/.bashrc
```

## 第 4 步：写入项目环境变量

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

后续新开的终端会自动获得这些变量。

## 第 5 步：编译

编译及运行任何 ROS 命令前先退出 conda，并确认系统使用的是 `/usr/bin/python3`：

```bash
conda deactivate
which python3  # 必须输出 /usr/bin/python3
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

## 单狗 go2_1：RTAB-Map 在线建图与 Nav2 静态目标导航

该模式仅启动 target_seek 世界中的一只 go2_1：Velodyne 点云转换为
LaserScan，RTAB-Map 使用 Gazebo 真值里程计在线建图，Nav2 仅接受静态
`NavigateToPose` 目标。

启动世界、go2_1、建图与导航：

```bash
cd "$DELIVERY_ROOT"
chmod +x Scripts/start_go2_1_mapping_nav.sh Scripts/send_go2_1_static_goal.sh
./Scripts/start_go2_1_mapping_nav.sh
```

新终端发送静态指定目标导航：

```bash
cd "$DELIVERY_ROOT"
./Scripts/send_go2_1_static_goal.sh 5 0 0
./Scripts/send_go2_1_static_goal.sh 20 16 0
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
./start_three_go2_velodyne.sh city --all-sensors --mapping-nav
```

默认关闭三只 Go2 的 3D 雷达和 RGB-D 相机；`--lidar`、`--camera`、`--all-sensors` 可按需开启。
三个场景的出生位姿、传感器默认值和动态行人路线统一保存在
`go2_ws_v2/src/go2_scenario_config/config/scenes/*.yaml`。

增加 `--mapping-nav` 后，脚本会自动为三只 Go2 开启 3D Velodyne，等待各自的点云和里程计
topic 就绪，再分别启动 RTAB-Map 与 Nav2。

三狗导航目标可指定机器人，省略 `--robot` 时默认发送给 `go2_1`：

```bash
./Scripts/send_go2_1_static_goal.sh --robot go2_1 5 0 0
./Scripts/send_go2_1_static_goal.sh --robot go2_2 5 0 0
./Scripts/send_go2_1_static_goal.sh --robot go2_3 5 0 0
```

按需另开终端启动键盘或 RViz：

```bash
ros2 launch go2_config teleop_three_go2.launch.py
ros2 launch go2_config view_three_go2_velodyne.launch.py
```

## 多狗模式：三只 Go2 基于建图与 Nav2 的动态行人围捕

> **现状说明**：三只 Go2 都启用 RGB-D + YOLO 感知。首只稳定发现行人的
> Go2 会成为感知狗并持续跟踪，另外两只作为导航狗，先通过 Nav2 到达围捕位置，
> 停稳后自动切换到 MADDPG 左右编队控制。交接过程会检查 Nav2 取消、停车、
> MADDPG ready/active 状态，异常时优先安全停车。

一键启动（ROS 项目不要使用 conda，需确认 `which python3` 为 `/usr/bin/python3`）：

```bash
cd $DELIVERY_ROOT/Scripts
./start_three_go2_dynamic_tracking.sh           # city
./start_three_go2_dynamic_tracking.sh forest    # 森林
./start_three_go2_dynamic_tracking.sh airport   # 机场
```

系统就绪后脚本启动行人，并按“感知跟踪 → Nav2 靠近 → MADDPG 接管”自动运行。

## 测试模式：

目前已搭建三只 Go2 的目标感知自动化测试框架，可通过 YAML 组合
city、forest、airport 三个场景，三类行人路线和 11 组机器人位姿，
共展开 99 个正式测试 Case。框架能够自动启动并隔离运行每个 Case，
记录原始数据，计算目标识别准确率与平均定位误差，并汇总测试结果。

当前测试范围主要覆盖目标识别与定位，暂不包含建图、Nav2、动态围捕和 tracking 控制流程。配置方法、运行命令、评价指标及输出目录详见
[Go2 目标感知自动化测试框架说明](go2_ws_v2/src/go2_test_framework/README.md)。

### 工具：动态场景与机器狗位姿生成器

工具链按职责拆分为两个并列目录：

- [`tools/gazebo_map_creator`](tools/gazebo_map_creator/README.md)：独立构建
  Gazebo 地图生成器，并保存 PGM、PNG、Nav2 YAML 基准地图。
- [`tools/pedestrian_map`](tools/pedestrian_map/README.md)：消费上述地图，验证
  行人路线并生成机器人初始位姿及复核报告。

## 模型参数说明

### 行人走路控制

先启动 Gazebo 世界：

```bash
cd $DELIVERY_ROOT
ros2 launch go2_config gazebo_target_seek_world.launch.py
```

通过 ROS 2 服务控制行人运动：

```bash
# 启动或继续行走
ros2 service call /walking_target/start std_srvs/srv/Trigger "{}"

# 暂停行走
ros2 service call /walking_target/pause std_srvs/srv/Trigger "{}"

# 复位行人
ros2 service call /walking_target/reset std_srvs/srv/Trigger "{}"
```

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
  forest:
    go2_1: x=20, y=18, z=0.80, yaw=2.19
    go2_2: x=-42, y=8, z=0.80, yaw=-2.35619449
    go2_3: x=36, y=40, z=0.80, yaw=-2.92
  airport:
    go2_1: x=60, y=-4, z=0.50, yaw=0
    go2_2: x=70, y=10, z=0.50, yaw=0
    go2_3: x=95, y=-25, z=0.50, yaw=0
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
