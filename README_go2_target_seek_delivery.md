# Go2 + target_seek 使用手册

本文档说明如何在 Gazebo Classic 中启动 `target_seek` 场景、一只带 2D 激光雷达的 Unitree Go2，并使用键盘控制和 SLAM。
另提供「多狗模式」：在同一场景里同时导入 3 只 Go2、各自独立键盘控制（见下文「多狗模式：三只 Go2」）。

## 版本要求

```text
Ubuntu 22.04
ROS 2 Humble
Gazebo Classic 11
```

键盘控制使用 ROS 官方包 `teleop_twist_keyboard`，发布标准 `/cmd_vel`。

## 第 0 步：解压项目

假设收到的压缩包为 `go2_target_seek_delivery.tar.gz`：

```bash
cd /你希望存放项目的目录
tar -xzf go2_target_seek_delivery.tar.gz
cd go2_target_seek_delivery
```

后续命令都用 `DELIVERY_ROOT` 表示项目实际目录，请改成当前机器上的真实路径：

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
```

解压后主要目录应为：

```text
go2_target_seek_delivery/
  go2_ws_v2/
  QY_MODEL/
    models/
  README_go2_target_seek_delivery.md
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
  ros-humble-teleop-twist-keyboard
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

## 第 3 步：设置项目路径

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT
export QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
export GAZEBO_MODEL_PATH=$QY_MODEL_ROOT/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_DATABASE_URI=""
```

## 第 4 步：构建

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 第 5 步：启动 Gazebo + target_seek + Go2

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
export QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
export GAZEBO_MODEL_PATH=$QY_MODEL_ROOT/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_DATABASE_URI=""

ros2 launch go2_config gazebo_target_seek_2d_lidar.launch.py gui:=true rviz:=false
```

默认会加载：

```text
world: $QY_MODEL_ROOT/target_seek
robot_name: go2
spawn: x=-3.5, y=2.7, z=0.25, yaw=0.0
```

## 第 6 步：键盘控制

另开一个终端：

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

常用按键：

```text
i      前进
,      后退
j      左转
l      右转
k      停止
q / z  加速 / 减速
```

运行键盘控制时，鼠标焦点需要停留在该终端窗口内。

## 第 7 步：启动 SLAM

另开一个终端：

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch go2_config slam.launch.py sim:=true rviz:=true
```

SLAM 默认使用：

```text
scan_topic: /scan
odom_frame: odom
map_frame: map
base_frame: base_footprint
```

## 多狗模式：三只 Go2（独立键盘控制）

在同一个 `target_seek` 场景里同时导入 3 只带 2D 激光雷达（不带相机）的 Go2，
分别用 `/go2_1`、`/go2_2`、`/go2_3` 命名空间，每只用一个键盘窗口独立控制。
本阶段三狗只做键盘行走，不带 SLAM。

每只狗独立话题/坐标：

```text
go2_i/cmd_vel              键盘速度指令
go2_i/scan                 2D 激光（frame: go2_i/front_laser）
go2_i/odom                 里程计，TF: go2_i/odom -> go2_i/base_footprint
go2_i/controller_manager   各自的 ros2_control 控制器管理器
出生点: go2_1 (-3.5, 1.5)  go2_2 (-3.5, 2.7)  go2_3 (-3.5, 3.9)  z=0.30
```

启动步骤（建议每条单独开一个终端，每个终端都要先 source 环境）：

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
export QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
export GAZEBO_MODEL_PATH=$QY_MODEL_ROOT/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_DATABASE_URI=""
```

```bash
# 终端 1：只启动 target_seek 世界（不导入狗，不暂停物理）
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true

# 终端 2/3/4：逐只导入。务必等上一只的两个控制器都 active、
# 且 /go2_i/scan 出现后，再启动下一只（顺序导入，避免传感器抢命名空间）。
ros2 launch go2_config spawn_go2_1.launch.py
ros2 launch go2_config spawn_go2_2.launch.py
ros2 launch go2_config spawn_go2_3.launch.py

# 终端 5：一次性弹出 3 个 xterm 键盘窗口（go2_1 / go2_2 / go2_3）
ros2 launch go2_config teleop_three_go2.launch.py
```

键盘窗口与单狗相同（i 前进、, 后退、j 左转、l 右转、k 停）。
**注意**：键盘只对当前鼠标焦点所在的那个 xterm 窗口生效——
要控制哪只狗，先点一下对应窗口再按键。

检查（每只狗都应满足）：

```bash
# 两个控制器都 active
ros2 control list_controllers --controller-manager /go2_1/controller_manager
# joint_trajectory 同时有发布者和订阅者（控制链闭合）
ros2 topic info /go2_1/joint_group_effort_controller/joint_trajectory   # Publisher 1 / Subscription 1
# 三只各自的激光
ros2 topic list | grep scan        # /go2_1/scan /go2_2/scan /go2_3/scan
```

> 说明：多狗共用一个 Gazebo（Classic）进程，gazebo_ros 传感器插件在多实例下
> 对命名空间解析不稳，因此激光话题用绝对话题名（如 `/go2_3/scan`）强制指定；
> 控制器命名空间由 URDF 内 `gazebo_ros2_control` 插件的 `<ros><namespace>` 烧入。
> 启动时请**顺序导入**三只狗（等上一只 active 再起下一只），最稳。

## 多狗模式：三只 Go2 + 3D Velodyne（替换 2D 激光）

本模式复用同一个 `target_seek` 世界和三狗键盘控制链路，但每只 Go2 只装 3D Velodyne，
不再加载 2D 激光雷达，因此不会发布 `/go2_i/scan`。三只狗分别发布：

```text
go2_i/cmd_vel                    键盘速度指令
go2_i/velodyne_points            3D 点云（sensor_msgs/PointCloud2，frame: go2_i/velodyne）
go2_i/odom                       里程计，TF: go2_i/odom -> go2_i/base_footprint
go2_i/controller_manager         各自的 ros2_control 控制器管理器
出生点: go2_1 (-3.5, 1.5)  go2_2 (-3.5, 2.7)  go2_3 (-3.5, 3.9)  z=0.50
```

如果目标机器还没有 Velodyne 相关包，先安装：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-velodyne-description \
  ros-humble-velodyne-gazebo-plugins
```

启动步骤（建议每条单独开一个终端，每个终端都要先 source 环境）：

```bash
export DELIVERY_ROOT=/实际/项目路径/go2_target_seek_delivery
cd $DELIVERY_ROOT/go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
export QY_MODEL_ROOT=$DELIVERY_ROOT/QY_MODEL
export GAZEBO_MODEL_PATH=$QY_MODEL_ROOT/models:$GAZEBO_MODEL_PATH
export GAZEBO_MODEL_DATABASE_URI=""
```

```bash
# 终端 1：只启动 target_seek 世界（不导入狗，不暂停物理）
ros2 launch go2_config gazebo_target_seek_world.launch.py gui:=true

# 终端 2/3/4：逐只导入带 3D Velodyne 的狗。
# 务必等上一只的两个控制器都 active、且 /go2_i/velodyne_points 出现后，再启动下一只。
ros2 launch go2_config spawn_go2_velodyne_1.launch.py
ros2 launch go2_config spawn_go2_velodyne_2.launch.py
ros2 launch go2_config spawn_go2_velodyne_3.launch.py

# 终端 5：一次性弹出 3 个 xterm 键盘窗口（go2_1 / go2_2 / go2_3）
ros2 launch go2_config teleop_three_go2.launch.py

# 终端 6：RViz 同时查看三份点云（固定坐标系 world）
ros2 launch go2_config view_three_go2_velodyne.launch.py
```

> 注意：运行任何 ROS 命令前先 `conda deactivate`，确认 `which python3` 为
> `/usr/bin/python3`。conda 的 python 会让 `ground_truth_odom_relay.py`、
> `spawn_entity.py` 崩溃（找不到 `rclpy._rclpy_pybind11` / `lxml`），导致狗不生成、
> odom 不发、点云无法在 RViz 显示。

只看单只狗（推荐先用它验证点云链路，固定坐标系 `go2_1/odom`，无需 world 静态 TF）：

```bash
# 终端 1：世界；终端 2：导入 go2_1；终端 3：单狗 RViz
ros2 launch go2_config gazebo_target_seek_world.launch.py
ros2 launch go2_config spawn_go2_velodyne_1.launch.py
ros2 launch go2_config view_go2_velodyne_1.launch.py
```

检查（每只狗都应满足）：

```bash
# 三份点云都在发布
ros2 topic list | grep velodyne_points
ros2 topic hz /go2_1/velodyne_points
ros2 topic echo /go2_1/velodyne_points --field header.frame_id --once   # go2_1/velodyne

# 本 3D 模式替换掉 2D 激光，不应出现三狗 scan
ros2 topic list | grep -c /scan

# 两个控制器都 active
ros2 control list_controllers --controller-manager /go2_1/controller_manager
# joint_trajectory 同时有发布者和订阅者（控制链闭合）
ros2 topic info /go2_1/joint_group_effort_controller/joint_trajectory   # Publisher 1 / Subscription 1

# TF 中 base_link 到 Velodyne frame 正常
ros2 run tf2_ros tf2_echo go2_1/base_link go2_1/velodyne

# 完整链路（odom 经 base_footprint/base_link 到 velodyne）可解析 —— RViz 能否显示的关键判据
# 若此命令报 "frame does not exist"，多半是 EKF 未发布 base_footprint->base_link 或 conda 环境问题
ros2 run tf2_ros tf2_echo go2_1/odom go2_1/velodyne
```

> 说明：3D 版同样采用绝对点云话题名（如 `/go2_3/velodyne_points`）强制指定 Gazebo
> 传感器插件输出，避免多实例命名空间解析不稳。RViz 启动文件会额外发布
> `world -> go2_i/odom` 的静态 TF，让三棵独立 TF 树能在同一个 `world` fixed frame 下显示。

## 模型参数说明

### 机器狗

```text
型号/模型名: Unitree Go2
URDF/xacro robot name: go2
主本体 link: trunk
base_link 到 trunk: 固定关节，xyz=0 0 0, rpy=0 0 0
默认出生点: x=-3.5, y=2.7, z=0.25, yaw=0.0
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

## 常见问题

Gazebo 端口占用：

```bash
ps -ef | grep gzserver
```

确认旧仿真不需要保留后，再结束旧进程。

模型缺失或无人机不可见：

```bash
echo $QY_MODEL_ROOT
echo $GAZEBO_MODEL_PATH
ls $QY_MODEL_ROOT/models
ls ~/.gazebo/models
```

检查报错中的 `model://xxx` 是否存在于 `$QY_MODEL_ROOT/models` 或 `~/.gazebo/models`。

Go2 不动：

```bash
ros2 topic echo /cmd_vel
```

确认 `teleop_twist_keyboard` 终端正在发布速度指令，并且该终端窗口处于键盘焦点状态。

`/scan` 没有数据：

```bash
ros2 topic list | grep scan
ros2 topic echo /scan --once
```

若没有 `/scan`，优先检查 Go2 是否成功生成，以及 2D 激光雷达 xacro 是否被当前 launch 加载。
