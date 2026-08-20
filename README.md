# Go2 多场景使用手册

**Latest Updated 2026.08.04**

本文档说明如何在 Gazebo Classic 中启动 `target_seek`、森林、机场等场景：

- 单狗模式：一只带 2D 激光雷达与深度相机的 Unitree Go2，可在 `target_seek`、`forest`、`airport` 三个场景间切换，支持键盘控制和 SLAM。
- 2D 多狗模式：同一场景里导入 3 只带 2D 激光雷达的 Go2，各自独立键盘控制。
- 3D 多狗模式：同一场景里导入 3 只 Go2，可按需开启 3D Velodyne 和 RGB-D，相互独立控制并可用 RViz 查看数据。
- 多狗静态围捕模式：三只 Go2 在三个场景 city、forest、airport 下进行导航与静态追踪，支持三种方法：1.人工航点规划，2.离线栅格地图与 A* 规划，3.RTAB建图与Nav2导航自主规划。
- 动态围捕模式：go2_1 使用 RGB-D 相机在线视觉感知（YOLO + 深度）估计并持续跟随行人，go2_2/go2_3 基于三机融合地图和 Nav2 驶向固定分配的动态围捕槽位，形成三狗围捕。

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
三个场景的出生位姿、传感器默认值、目标和围捕半径统一保存在
`go2_ws_v2/src/multi_go2_waypoint/config/scenes/*.yaml`。脚本只选择 world 并传递 `scene`；
每只狗的 spawn launch 会读取相应 YAML，命令行传感器选项优先于 YAML 默认值。

增加 `--mapping-nav` 后，脚本会自动为三只 Go2 开启 3D Velodyne，等待各自的点云和里程计
topic 就绪，再分别启动独立的 RTAB-Map 与 Nav2。三套地图使用独立 frame 和数据库：
`go2_1/map`、`go2_2/map`、`go2_3/map`。可通过 `USE_RVIZ=false` 关闭三个 RViz：

```bash
USE_RVIZ=false ./start_three_go2_velodyne.sh city --mapping-nav
```

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

## 多狗模式：三只 Go2 静态围捕（三场景均实现）

先用通用脚本启动所需场景、内置 UAV 和三只 Go2。脚本不会自动启动围捕控制器：

```bash
cd $DELIVERY_ROOT/Scripts
./start_three_go2_velodyne.sh forest
```

三只狗控制器全部 active 后，另开终端启动围捕。city 和 forest 使用人工 waypoint：

```bash
cd $DELIVERY_ROOT/go2_ws_v2
conda deactivate
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run multi_go2_waypoint waypoint_encircle --ros-args \
  -p scene:=forest \
  -p planner_mode:=manual
```

airport 既可使用 `planner_mode:=manual`，也可读取离线地图执行 A*：

```bash
ros2 run multi_go2_waypoint waypoint_encircle --ros-args \
  -p scene:=airport \
  -p planner_mode:=astar
```



## 多狗模式：三只 Go2 基于建图与 Nav2 的动态行人围捕

> **现状说明**：go2_1 使用 RGB-D 相机 YOLO 检测，
> 并在行人附近持续跟随。go2_2/go2_3 使用融合地图和
> Nav2，根据行人估计位置及 go2_1 方位计算围捕点并导航至附近。

一键启动（ROS 项目不要使用 conda，需确认 `which python3` 为 `/usr/bin/python3`）：

```bash
cd $DELIVERY_ROOT/Scripts
./start_three_go2_dynamic_tracking.sh
```

脚本固定使用 `target_seek/city` 场景，依次启动：Gazebo 世界 → 三只 Go2（全部开启
Velodyne，仅 go2_1 开启 RGB-D）→ 已知位姿地图融合 → 三套 RTAB-Map/Nav2 → 统一 RViz →
行人真值桥接 `actor_state_publisher` → 目标感知 `target_perception` →
`go2_mapping_nav/dynamic_encircle.py`（目标源 `/go2_1/target_estimated/odom`）→ RQT 调试图 →
`perception_eval` 误差评估。上述组件就绪后，脚本最后调用 `/walking_target/start`，让行人开始运动。


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
    go2_2: x=-8, y=42, z=0.80, yaw=0
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
