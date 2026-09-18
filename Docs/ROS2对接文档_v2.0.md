# 三机目标追踪与协同围捕模块 ROS2 对接文档

> **代码基线**：`Poma2580/go2_target_seek_delivery`，`main@014733e9`（后续代码变更需重新核查）
> **主参考脚本**：`Scripts/start_three_go2_dynamic_tracking.sh`
> **核查方式**：源码静态核查，非运行验证；不代表依赖、模型或全部启动流程已在目标环境验证通过。

---

## 1. 模块说明

本模块用于三只 Go2 的动态目标感知、持续追踪、导航围捕及最终协同控制。

当前完整运行链为：

```text
Gazebo 场景
    ↓
三只 Go2 启动
    ↓
建图与地图融合
    ↓
RGB-D 目标感知
    ↓
感知狗角色选举
    ↓
目标持续追踪
    ↓
两只导航狗动态围捕
    ↓
Nav2 初始靠近 → MADDPG 选点
    ↓
Nav2 执行候选航点，完成三机协同围堵
```

### 1.1 主要功能模块

| 模块             | ROS 包                               | 主要功能                                 |
| ---------------- | ------------------------------------ | ---------------------------------------- |
| Go2 仿真与传感器 | `go2_config` / `go2_description` | 启动三只 Go2、RGB-D、Velodyne 及底层控制 |
| 场景配置         | `go2_scenario_config`              | 场景参数及机器人姿态检查                 |
| 建图导航         | `go2_mapping_nav`                  | RTAB-Map、地图融合、MADDPG 选点及 Nav2   |
| 目标感知         | `go2_target_perception`            | YOLO + RGB-D 目标识别与定位              |
| 角色选举         | `go2_target_perception`            | 从三只机器狗中确定感知狗                 |
| 动态围捕         | `go2_dynamic_encircle`             | 目标追踪、围捕点规划及 Nav2 Goal 管理    |
| 阶段交接         | `go2_dynamic_encircle`             | Nav2 初始靠近与 MADDPG 选点阶段交接       |
| 动态目标         | `walking_target_controller`        | Gazebo 中动态行人的运动控制              |

---

## 2. 软件环境与编译

### 2.1 软件环境

当前项目主要运行环境：

| 项目     | 环境                     |
| -------- | ------------------------ |
| 操作系统 | Ubuntu 22.04             |
| ROS      | ROS2 Humble              |
| Python   | 系统`/usr/bin/python3` |
| 仿真     | Gazebo Classic           |
| 建图     | RTAB-Map                 |
| 导航     | Nav2                     |
| 深度学习 | PyTorch                  |
| 目标检测 | Ultralytics YOLO         |

---

### 2.2 工作空间

项目主要 ROS2 工作空间：

```text
go2_target_seek_delivery/
└── go2_ws_v2/
    ├── src/
    ├── build/
    ├── install/
    └── log/
```

---

### 2.3 编译方法

进入项目后执行：

```bash
conda deactivate

which python3  # 必须为 /usr/bin/python3，否则先修正环境再继续

cd <go2_target_seek_delivery>/go2_ws_v2

source /opt/ros/humble/setup.bash

colcon build --symlink-install

source install/setup.bash
```


新的终端手工执行第 4 章命令前，统一准备环境（将仓库路径替换为实际绝对路径）：

```bash
conda deactivate
which python3  # 必须为 /usr/bin/python3，否则停止并修正环境
export DELIVERY_ROOT="/实际路径/go2_target_seek_delivery"
cd "$DELIVERY_ROOT/go2_ws_v2"
source /opt/ros/humble/setup.bash
source install/setup.bash
export QY_MODEL_ROOT="$DELIVERY_ROOT/QY_MODEL"
export KD_MODEL_ROOT="$DELIVERY_ROOT/KD_MODEL"
export GAZEBO_MODEL_PATH="$QY_MODEL_ROOT/models:$KD_MODEL_ROOT/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=""
```

以上对应主脚本的 `COMMON_ENV`。特别是森林、机场场景需要 KD 模型搜索路径；仅传入 World 文件路径不等于配置了模型搜索路径。命令中的 `<SCENE>`、`<WORLD_PATH>`、`<REPO_ROOT>` 均为待替换占位符。

---

### 2.4 运行所需模型资源

当前完整运行链还需要以下非 ROS 资源：

```text
QY_MODEL / KD_MODEL      Gazebo 场景模型
yolov8s.pt               YOLO 目标检测模型
waypoint_maddpg_v0/.../best_model.pt
                         MADDPG 离散航点选择模型
```

启动前需保证对应模型路径在当前运行环境中可访问。

---

## 3. ROS 包交付说明

### 3.1 我方 ROS 包

| ROS 包                        | 用途                           |
| ----------------------------- | ------------------------------ |
| `go2_config`                | Gazebo 世界与 Go2 仿真启动     |
| `go2_description`           | Go2 URDF / Xacro               |
| `go2_scenario_config`       | 场景参数与姿态检查             |
| `go2_mapping_nav`           | 三机建图、地图融合、MADDPG 选点与 Nav2 |
| `go2_target_perception`     | RGB-D 目标感知与感知狗角色选举 |
| `go2_dynamic_encircle`      | 动态追踪、围捕与控制切换       |
| `walking_target_controller` | Gazebo 动态行人控制            |

---

## 4. 启动命令

本章按 `Scripts/start_three_go2_dynamic_tracking.sh` 的执行顺序列出 C01～C16 终端命令。C11 直接运行 `rviz2`，其余使用 ROS2 CLI。各命令可以分别调用，但不代表运行功能无需其他接口；须先完成第 2.3 节环境准备并满足以下依赖。

| 命令 | 功能运行所需条件 |
| --- | --- |
| C01 | 可用的 World、Gazebo 插件和模型资源 |
| C02～C04 | Gazebo 已运行且 `/spawn_entity` 可用 |
| C05 | `/gazebo/model_states` 中已有三狗状态 |
| C06 | 可先启动；收到有效单狗地图后才能产生融合地图 |
| C07 | 按角色及控制源转发速度；未获得角色时处理三狗输入，输入超时输出零速度 |
| C08～C10 | 点云、Odom、TF、时钟；当前融合地图模式还依赖 `/merged_map` |
| C11、C15 | 可单独运行；有效显示依赖相应数据，C11 还需要 TF |
| C12 | 模型、RGB-D、相机内参、TF 和一致时钟 |
| C13 | 选点模型、感知角色、三狗 Odom、两只跟随狗 LaserScan、TF 及 Action Server |
| C14 | 场景配置、角色、目标估计、Odom、TF 和 Action Server |
| C16 | `/walking_target/start` Provider 可用 |

上述是源码依赖说明；各命令能否在目标机器成功完成运行，无法从当前代码确认。

脚本支持场景：

```text
city
forest
airport
```

其中 `city` 还接受 `qy`、`target_seek` 作为别名，脚本内部统一转换为 `city`。

### 4.1 完整启动链

| 编号 | ROS 命令 / 进程                                                           | 主要功能                         |
| ---- | ------------------------------------------------------------------------- | -------------------------------- |
| C01  | `ros2 launch go2_config gazebo_target_seek_world.launch.py`             | 启动动态行人 Gazebo 世界         |
| C02  | `ros2 launch go2_config spawn_go2_velodyne_1.launch.py`                 | 生成`go2_1` 及其传感器、控制器 |
| C03  | `ros2 launch go2_config spawn_go2_velodyne_2.launch.py`                 | 生成`go2_2`                    |
| C04  | `ros2 launch go2_config spawn_go2_velodyne_3.launch.py`                 | 生成`go2_3`                    |
| C05  | `ros2 run go2_scenario_config check_three_go2_attitude`                 | 检查三只 Go2 是否翻倒            |
| C06  | `ros2 launch go2_mapping_nav three_go2_map_merge.launch.py`             | 启动三机地图融合                 |
| C07  | `ros2 run go2_dynamic_encircle follower_cmd_vel_mux`                    | 将 Nav2 速度转发到跟随狗底盘        |
| C08  | `ros2 launch go2_mapping_nav go2_1_mapping_nav.launch.py`               | 启动`go2_1` RTAB-Map + Nav2    |
| C09  | `ros2 launch go2_mapping_nav go2_2_mapping_nav.launch.py`               | 启动`go2_2` RTAB-Map + Nav2    |
| C10  | `ros2 launch go2_mapping_nav go2_3_mapping_nav.launch.py`               | 启动`go2_3` RTAB-Map + Nav2    |
| C11  | `rviz2 -d .../three_go2_mapping_nav.rviz`                               | 启动三机统一 RViz                |
| C12  | `ros2 launch go2_target_perception three_go2_target_tracking.launch.py` | 启动三机目标感知及感知狗选举     |
| C13  | `ros2 run go2_mapping_nav maddpg_waypoint_selector.py`                  | 预加载 MADDPG 选点器，向 Nav2 下发 Goal |
| C14  | `ros2 run go2_dynamic_encircle dynamic_encircle`                        | 启动目标追踪、初始靠近和选点交接     |
| C15  | `ros2 topic echo /dynamic_encircle/handoff_state`                       | 监视 Nav2 靠近 → MADDPG 选点阶段    |
| C16  | `ros2 service call /walking_target/start ...`                           | 最后启动动态行人运动             |

> 脚本中已经注释的 `actor_state_publisher`、`perception_eval`、`rqt_image_view` 等调试或评估命令，不属于当前正式启动链。

---

### 4.2 C01：启动 Gazebo 动态场景

```bash
ros2 launch go2_config gazebo_target_seek_world.launch.py \
    gui:=true \
    world:=<WORLD_PATH>
```

`WORLD_PATH` 由脚本根据场景选择：

| 场景        | World 路径                                      |
| ----------- | ----------------------------------------------- |
| `city`    | `$QY_MODEL_ROOT/target_seek`                  |
| `forest`  | `$KD_MODEL_ROOT/world/forestV3_dynamic.world` |
| `airport` | `$KD_MODEL_ROOT/world/airport_dynamic.world`  |

该 launch 仅启动 Gazebo 世界，不生成机器狗。

启动后脚本依次等待以下接口出现：

```text
/spawn_entity
/gazebo/model_states
/clock
/walking_target/start
```

其中 `/walking_target/start` 默认等待超时：

```text
ACTOR_SERVICE_TIMEOUT=30 s
```

---

### 4.3 C02～C04：生成三只 Go2

#### go2_1

```bash
ros2 launch go2_config spawn_go2_velodyne_1.launch.py \
    scene:=<SCENE> \
    use_sim_time:=true \
    enable_lidar:=true \
    enable_camera:=true
```

#### go2_2

```bash
ros2 launch go2_config spawn_go2_velodyne_2.launch.py \
    scene:=<SCENE> \
    use_sim_time:=true \
    enable_lidar:=true \
    enable_camera:=true
```

#### go2_3

```bash
ros2 launch go2_config spawn_go2_velodyne_3.launch.py \
    scene:=<SCENE> \
    use_sim_time:=true \
    enable_lidar:=true \
    enable_camera:=true
```

脚本按以下顺序依次启动三只 Go2：

```text
启动 go2_1
   ↓
等待 controller + lidar + odom + RGB-D
   ↓
等待 3 s
   ↓
启动 go2_2
   ↓
等待 controller + lidar + odom + RGB-D
   ↓
等待 3 s
   ↓
启动 go2_3
```

每只机器狗启动后，脚本等待：

```text
/go2_i/controller_manager/list_controllers
/go2_i/velodyne_points
/go2_i/odom
/go2_i/camera/image_raw
/go2_i/camera/depth/image_raw
/go2_i/camera/depth/camera_info
```

两个 ROS2 Controller 必须处于 `active` 状态：

```text
joint_group_effort_controller
joint_states_controller
```

其中：

```text
i ∈ {1,2,3}
```

---

### 4.4 C05：三狗姿态检查

```bash
ros2 run go2_scenario_config check_three_go2_attitude --ros-args \
    -p model_states_topic:=/gazebo/model_states \
    -p robot_names:='[go2_1,go2_2,go2_3]' \
    -p roll_limit_deg:=90.0 \
    -p sample_frames:=5 \
    -p timeout_seconds:=10.0
```

功能：

- 读取 `/gazebo/model_states`；
- 连续采集 5 帧三只机器狗姿态；
- 检查是否存在 `abs(roll) > 90°`；
- 判断机器狗是否发生翻倒。

当前退出码处理：

|     退出码 | 含义             | 脚本行为                       |
| ---------: | ---------------- | ------------------------------ |
|      `0` | 姿态检查通过     | 继续启动后续模块               |
|     `10` | 检测到机器狗翻倒 | 清理当前运行进程并自动重新启动 |
| 其他非零值 | 姿态检查异常     | 停止当前启动流程               |

当前默认最大自动重启次数：

```text
MAX_GO2_RESTARTS=3
```

---

### 4.5 C06：启动三机地图融合

```bash
ros2 launch go2_mapping_nav three_go2_map_merge.launch.py \
    use_sim_time:=true \
    use_rviz:=false
```

地图融合关系：

```text
/go2_1/map ─┐
/go2_2/map ─┼──> known_pose_map_merger ───> /merged_map
/go2_3/map ─┘
```

当前输出：

```text
Topic: /merged_map
Type : nav_msgs/msg/OccupancyGrid
Frame: merged_map
```

同时启动当前系统所需的统一 TF 根。

---

### 4.6 C07：Nav2 速度转发

```bash
ros2 run go2_dynamic_encircle follower_cmd_vel_mux \
    --ros-args \
    -p use_sim_time:=true
```

控制关系：

```text
/go2_i/nav_cmd_vel
        ↓
follower_cmd_vel_mux
        ↓
/go2_i/cmd_vel
```

选点版中控制源始终为：

```text
Nav2
```

虽然 Mux 仍保留兼容旧版的 `/go2_i/maddpg_cmd_vel` 输入，但新 MADDPG 选点器不发布 `Twist`，`/dynamic_encircle/use_maddpg` 在正常流程中保持 `false`。

---

### 4.7 C08～C10：启动三套 RTAB-Map + Nav2

#### go2_1

```bash
ros2 launch go2_mapping_nav go2_1_mapping_nav.launch.py \
    use_sim_time:=true \
    use_merged_map:=true \
    use_rviz:=false \
    delete_db_on_start:=true \
    cmd_vel_topic:=/go2_1/nav_cmd_vel
```

#### go2_2

```bash
ros2 launch go2_mapping_nav go2_2_mapping_nav.launch.py \
    use_sim_time:=true \
    use_merged_map:=true \
    use_rviz:=false \
    delete_db_on_start:=true \
    cmd_vel_topic:=/go2_2/nav_cmd_vel
```

#### go2_3

```bash
ros2 launch go2_mapping_nav go2_3_mapping_nav.launch.py \
    use_sim_time:=true \
    use_merged_map:=true \
    use_rviz:=false \
    delete_db_on_start:=true \
    cmd_vel_topic:=/go2_3/nav_cmd_vel
```

Nav2 当前不直接向最终 `/cmd_vel` 输出，而是：

```text
Nav2
  ↓
/go2_i/nav_cmd_vel
  ↓
follower_cmd_vel_mux
  ↓
/go2_i/cmd_vel
```

启动第一套导航后，脚本等待 `/merged_map` 的首条消息：

```text
MERGED_MAP_TIMEOUT=120 s
```

每套 Nav2 还必须等待：

```text
Managed nodes are active
```

以及对应 Action Server：

```text
/go2_i/navigate_to_pose
```

---

### 4.8 C11：启动三机统一 RViz

```bash
rviz2 -d \
$(ros2 pkg prefix go2_mapping_nav)/share/go2_mapping_nav/rviz/three_go2_mapping_nav.rviz
```

功能：

- 可视化三只 Go2；
- 可视化单狗地图及融合地图；
- 可视化 Nav2 规划与导航状态。

该进程不参与算法控制闭环。

---

### 4.9 C12：启动三机目标感知与角色选举

```bash
ros2 launch go2_target_perception \
    three_go2_target_tracking.launch.py \
    use_sim_time:=true \
    model_path:=<REPO_ROOT>/yolov8s.pt
```

启动后脚本等待 Topic：

```text
/target_role/perception_robot
```

该 Topic 用于发布最终选中的感知狗名称：

```text
go2_1
```

或：

```text
go2_2
```

或：

```text
go2_3
```

---

### 4.10 C13：预加载 MADDPG 航点选择器

```bash
ros2 run go2_mapping_nav maddpg_waypoint_selector.py \
    --ros-args \
    -p use_sim_time:=true \
    -p model_path:=<REPO_ROOT>/waypoint_maddpg_v0/.../best_model.pt \
    -p global_frame:=merged_map \
    -p robot_names:="[go2_1,go2_2,go2_3]" \
    -p perception_robot_topic:=/target_role/perception_robot \
    -p wait_for_enable:=true \
    -p enabled:=false \
    -p enable_topic:=/dynamic_encircle/maddpg_enable \
    -p controller_ready_topic:=/maddpg_waypoint/controller_ready \
    -p controller_active_topic:=/maddpg_waypoint/controller_active \
    -p decision_period:=1.0 \
    -p nav_goal_update_period:=3.0 \
    -p require_initial_formation:=false \
    -p dry_run:=false
```

其中：

```text
wait_for_enable=true
```

表示节点启动后先完成：

```text
模型加载
   ↓
三狗 Odom 与两只跟随狗 LaserScan 接收
   ↓
感知狗角色锁定
   ↓
发布 ready 状态
   ↓
等待启用信号
```

该阶段不会立即下发新航点。

后续由 `dynamic_encircle` 发布：

```text
/dynamic_encircle/maddpg_enable
```

请求 MADDPG 选点器开始推理和下发航点。MADDPG 只选择两只跟随狗的候选点，通过 `/go2_i/navigate_to_pose` 交给 Nav2 执行，不发布 `cmd_vel`。

---

### 4.11 C14：启动动态目标追踪与围捕

```bash
ros2 run go2_dynamic_encircle dynamic_encircle \
    --ros-args \
    -p use_sim_time:=true \
    -p scene:=<SCENE> \
    -p perception_robot_topic:=/target_role/perception_robot \
    -p robot_names:="[go2_1,go2_2,go2_3]" \
    -p maddpg_ready_topic:=/maddpg_waypoint/controller_ready \
    -p maddpg_active_topic:=/maddpg_waypoint/controller_active \
    -p maddpg_enable_topic:=/dynamic_encircle/maddpg_enable \
    -p switch_mux_to_maddpg:=false
```

该节点负责整套动态任务层逻辑：

```text
读取目标感知结果
      ↓
确定感知狗
      ↓
感知狗持续追踪目标
      ↓
计算另外两只机器狗的围捕目标点
      ↓
通过 NavigateToPose 下发动态导航目标
      ↓
判断围捕位置是否到达
      ↓
取消 Nav2 当前 Goal
      ↓
请求 MADDPG 选点器启用
      ↓
MADDPG 持续选点，Nav2 继续独占速度控制
```

---

### 4.12 C15：监视任务阶段

```bash
ros2 topic echo \
    /dynamic_encircle/handoff_state \
    std_msgs/msg/String \
    --qos-durability transient_local
```

该命令仅用于 CLI 状态监视，不属于算法节点。

状态枚举统一见第 6.4.6 节。

---

### 4.13 C16：启动动态行人

```bash
ros2 service call \
    /walking_target/start \
    std_srvs/srv/Trigger \
    "{}"
```

该 Service 调用位于整个启动流程最后。

即：

```text
Gazebo
  ↓
三只 Go2
  ↓
建图 / Nav2
  ↓
目标感知
  ↓
MADDPG 选点器
  ↓
dynamic_encircle 交接状态机
  ↓
此前命令已发出并完成脚本规定的检查
  ↓
/walking_target/start
```

这不保证所有模块均已准备完成。脚本等待控制器 active、融合地图首条消息、Nav2 激活日志、Action 名称、角色 Topic 名称及选点器 Ready Topic 名称；名称存在不等于已收到 `ready=true`。选点器未启用前可预加载模型，实际启用交由 C14 的交接状态机处理。依据：`Scripts/start_three_go2_dynamic_tracking.sh` 的 C12～C16 调用段。

---

## 5. ROS 节点说明

本章对应第 4 章 C01～C16，说明每条启动命令实际启动的节点或持续运行进程及其功能。

以下节点清单依据源码声明，不代表本次观察到的运行图。C05、C07、C12、C13 的 executable 分别为 `check_three_go2_attitude`、`follower_cmd_vel_mux`、`target_perception`/`target_role_selector`、`maddpg_waypoint_selector.py`；C14 executable 为 `dynamic_encircle`，实际节点名为 `/nav2_dynamic_encircle`。除 C12 的三个感知实例在各机器人 namespace 外，这些节点位于根 namespace。入口依据各包 `setup.py` 及节点构造函数。

---

### 5.1 C01：Gazebo 场景进程

`gazebo_target_seek_world.launch.py` 主要启动：

| 进程         | 功能                              |
| ------------ | --------------------------------- |
| `gzserver` | Gazebo 仿真服务器，加载指定 World |
| `gzclient` | Gazebo GUI，`gui:=true` 时启动  |

`gzserver` 加载：

```text
libgazebo_ros_init.so
libgazebo_ros_factory.so
```

从而向 ROS2 提供 Gazebo 相关 Topic / Service。

> 本节将 `gzserver` / `gzclient` 作为仿真进程说明，不将其表述为普通业务 ROS Node。

---

### 5.2 C02～C04：单只 Go2 启动节点

三份 `spawn_go2_velodyne_i.launch.py` 结构基本一致，仅机器人名称和 namespace 分别为：

```text
go2_1
go2_2
go2_3
```

统一使用 `go2_i` 表示。

| 节点 / 进程                   | Package                   | Namespace | 功能                                                  |
| ----------------------------- | ------------------------- | --------- | ----------------------------------------------------- |
| `robot_state_publisher`     | `robot_state_publisher` | `go2_i` | 根据 URDF 和 JointState 发布机器人 TF                 |
| `quadruped_controller_node` | `champ_base`            | `go2_i` | Go2 四足运动控制                                      |
| `state_estimation_node`     | `champ_base`            | `go2_i` | 机器人状态估计                                        |
| `base_to_footprint_ekf`     | `robot_localization`    | `go2_i` | EKF 状态融合及机体相关 TF/状态维护                    |
| `ground_truth_odom_relay`   | `go2_config`            | `go2_i` | 将 Gazebo ground-truth odom 转换并发布为机器狗主 odom |
| `spawn_entity.py`           | `gazebo_ros`            | —        | 将 Go2 URDF 模型生成到 Gazebo，完成后退出             |
| controller`spawner` ×2     | `controller_manager`    | —        | 加载并激活 ROS2 Control Controller，完成后退出         |

上表中 `base_to_footprint_ekf` 的 executable 为 `ekf_node`，`ground_truth_odom_relay` 为 `ground_truth_odom_relay.py`；前三个节点的 executable 与表中名称一致。`spawn_entity.py` 和 `spawner` 为可执行文件名，launch 未显式设置其节点名或 namespace，不将可执行文件名当作已确认的运行时节点名。spawner 通过 `--controller-manager /go2_i/controller_manager` 指定目标。依据：`go2_ws_v2/src/unitree-go2-ros2/robots/configs/go2_config/launch/spawn_go2_velodyne_1.launch.py` 及对应的 go2_2、go2_3 文件。

Go2 URDF 中的 `gazebo_ros2_control` 插件提供对应：

```text
/go2_i/controller_manager
```

当前脚本通过：

```text
/go2_i/controller_manager/list_controllers
```

检查以下 Controller 是否为 `active`：

```text
joint_group_effort_controller
joint_states_controller
```

---

### 5.3 C05：三狗姿态检查节点

| Node                          | Package                 | 功能                                                            |
| ----------------------------- | ----------------------- | --------------------------------------------------------------- |
| `/check_three_go2_attitude` | `go2_scenario_config` | 从`/gazebo/model_states` 读取三狗姿态并检查 roll 是否超过阈值 |

该节点属于一次性检查节点。

检查完成后退出，不作为系统持续运行节点。

---

### 5.4 C06：三机地图融合节点

当前命令采用：

```text
use_rviz:=false
```

因此实际启动：

| Node                         | Package             | 功能                                    |
| ---------------------------- | ------------------- | --------------------------------------- |
| `/known_pose_map_merger`   | `go2_mapping_nav` | 融合三只 Go2 的`OccupancyGrid` 地图   |
| `/merged_map_to_world`     | `tf2_ros`         | 发布`merged_map → world` 静态 TF     |
| `/merged_map_to_go2_1_map` | `tf2_ros`         | 发布`merged_map → go2_1/map` 静态 TF |
| `/merged_map_to_go2_2_map` | `tf2_ros`         | 发布`merged_map → go2_2/map` 静态 TF |
| `/merged_map_to_go2_3_map` | `tf2_ros`         | 发布`merged_map → go2_3/map` 静态 TF |

launch 文件中还定义：

```text
three_go2_mapping_nav_rviz
```

但由于：

```text
use_rviz:=false
```

该节点在 C06 中不会启动。

统一 RViz 由后续 C11 单独启动。

---

### 5.5 C07：Nav2 速度转发节点

| Node                      | Package                  | 功能                                         |
| ------------------------- | ------------------------ | -------------------------------------------- |
| `/follower_cmd_vel_mux` | `go2_dynamic_encircle` | 将 Nav2 的平滑速度命令转发给当前两只跟随狗 |

工作关系：

```text
/go2_i/nav_cmd_vel
       ↓
/go2_i/cmd_vel
```

角色锁定前，Mux 对三狗处理输入；锁定后，仅对另外两只导航 / 跟随狗执行转发。输入超过默认 `command_timeout=0.5 s` 未更新时输出零速度。代码中保留的 /go2_i/maddpg_cmd_vel 输入属于旧版连续控制接口，当前流程不使用。
---

### 5.6 C08～C10：单狗 RTAB-Map + Nav2 节点

三套 `go2_i_mapping_nav.launch.py` 结构一致。

| Node                              | Package                     | 功能                                    |
| --------------------------------- | --------------------------- | --------------------------------------- |
| `go2_i_pointcloud_to_laserscan` | `pointcloud_to_laserscan` | 将 Velodyne 点云转换为 2D LaserScan     |
| `rtabmap`                       | `rtabmap_slam`            | 在线 RTAB-Map 建图                      |
| `controller_server`             | `nav2_controller`         | Nav2 局部运动控制                       |
| `smoother_server`               | `nav2_smoother`           | 路径平滑                                |
| `planner_server`                | `nav2_planner`            | 全局路径规划                            |
| `behavior_server`               | `nav2_behaviors`          | Nav2 行为控制                           |
| `bt_navigator`                  | `nav2_bt_navigator`       | 执行`NavigateToPose` 行为树           |
| `waypoint_follower`             | `nav2_waypoint_follower`  | 航点任务支持                            |
| `velocity_smoother`             | `nav2_velocity_smoother`  | 平滑 Nav2 输出速度                      |
| `lifecycle_manager_navigation`  | `nav2_lifecycle_manager`  | 管理 Nav2 生命周期                      |
| `go2_i_map_to_odom`             | `tf2_ros`                 | 发布`go2_i/map → go2_i/odom` 静态 TF |

Nav2 核心节点处于对应 `go2_i` namespace 下。

表中 `rtabmap` 也在 `/go2_i` 下；`go2_i_pointcloud_to_laserscan` 和 `go2_i_map_to_odom` 在根 namespace。前者 executable 为 `pointcloud_to_laserscan_node`，后者为 `static_transform_publisher`；`lifecycle_manager_navigation` 的 executable 为 `lifecycle_manager`，其余该表节点的 executable 与表中名称一致。依据：`go2_ws_v2/src/go2_mapping_nav/launch/go2_1_mapping_nav.launch.py` 及对应的 go2_2、go2_3 文件。

launch 文件还定义了：

```text
go2_i_base_footprint_to_base_link
```

但当前默认：

```text
publish_base_footprint_tf:=false
```

因此该静态 TF 节点默认不启动。

同样，单狗 launch 定义：

```text
go2_i_mapping_nav_rviz
```

但当前脚本传入：

```text
use_rviz:=false
```

所以 C08～C10 不启动三套独立 RViz。

---

### 5.7 C11：三机统一 RViz

| Node / 进程 | Package   | 功能                                                           |
| ----------- | --------- | -------------------------------------------------------------- |
| `rviz2`   | `rviz2` | 加载`three_go2_mapping_nav.rviz`，统一显示三机地图及导航状态 |

当前整体结构为：

```text
3 × RTAB-Map / Nav2
        +
1 × Unified RViz
```

---

### 5.8 C12：目标感知与角色选举节点

`three_go2_target_tracking.launch.py` 启动 4 个节点：

| Node                         | Package                   | 功能                               |
| ---------------------------- | ------------------------- | ---------------------------------- |
| `/go2_1/target_perception` | `go2_target_perception` | `go2_1` RGB-D 行人识别与定位     |
| `/go2_2/target_perception` | `go2_target_perception` | `go2_2` RGB-D 行人识别与定位     |
| `/go2_3/target_perception` | `go2_target_perception` | `go2_3` RGB-D 行人识别与定位     |
| `/target_role_selector`    | `go2_target_perception` | 根据连续有效目标定位结果选举感知狗 |

运行逻辑：

```text
三只 Go2 同时运行 target_perception
              ↓
各自发布 target_estimated/odom
              ↓
target_role_selector
              ↓
首先达到稳定识别条件的机器人
              ↓
/target_role/perception_robot
              ↓
角色锁定
              ↓
另外两个 target_perception 实例主动退出
              ↓
仅感知狗继续执行 YOLO + RGB-D 目标定位
```

当前角色确认条件：

```text
confirmation_count  = 3
confirmation_window = 1.0 s
max_message_age     = 0.5 s
```

---

### 5.9 C13：MADDPG 航点选择节点

| Node                          | Package             | 功能                                                   |
| ----------------------------- | ------------------- | -------------------------------------------------------------- |
| `/maddpg_waypoint_selector` | `go2_mapping_nav` | 加载离散 MADDPG 模型，为两只跟随狗选点并下发 Nav2 Goal |

节点启动过程：

```text
加载 MADDPG best_model.pt
          ↓
接收三只 Go2 Odom
          ↓
接收感知狗角色
          ↓
确定 leader / follower
          ↓
接收两只 follower LaserScan 并验证 TF/时间同步
          ↓
发布 ready 状态
          ↓
等待 maddpg_enable
          ↓
每 1 s 选择两个候选点
          ↓
通过 NavigateToPose 周期性下发给 Nav2
```

当前启动参数：

```text
wait_for_enable:=true
dry_run:=false
decision_period:=1.0
nav_goal_update_period:=3.0
```

因此 MADDPG 节点启动后不会立即下发航点；启用后也不抢占速度控制权，Nav2 始终是两只跟随狗的唯一 `cmd_vel` 生成者。

---

### 5.10 C14：动态目标追踪与围捕节点

| Node                       | Package                  | 功能                                                                   |
| -------------------------- | ------------------------ | ---------------------------------------------------------------------- |
| `/nav2_dynamic_encircle` | `go2_dynamic_encircle` | 动态目标追踪、初始围捕 Goal 管理以及 Nav2 靠近 → MADDPG 选点阶段交接 |

主要职责：

#### 1. 目标状态管理

```text
/go2_i/target_estimated/odom
```

#### 2. 感知狗追踪控制

```text
/go2_i/cmd_vel
```

#### 3. 两只导航狗动态目标规划

```text
/go2_i/navigate_to_pose
```

#### 4. Nav2 靠近 → MADDPG 选点 Handoff

```text
/dynamic_encircle/maddpg_enable
/dynamic_encircle/use_maddpg
/dynamic_encircle/handoff_state
```

控制阶段由内部 Handoff 状态机管理。

`/dynamic_encircle/use_maddpg` 在选点版中保持 `false`；交接改变的是 Goal 生成方，而不是底层速度生成方。

---

### 5.11 C15 / C16：CLI 调试与一次性调用

#### C15

```bash
ros2 topic echo /dynamic_encircle/handoff_state ...
```

属于 ROS2 CLI Topic 监视进程，仅用于显示控制阶段。

#### C16

```bash
ros2 service call /walking_target/start ...
```

属于一次性 Service 调用，用于启动动态行人。

因此 C15、C16 不作为核心持续运行 ROS Node 列入算法节点清单。

---

## 6. Topic 接口

当前完整运行链中的主要 Topic 按功能划分为以下四类：

1. 仿真、机器人状态与传感器 Topic；
2. 建图与导航 Topic；
3. 目标感知 Topic；
4. 动态围捕与控制 Topic。

其中：

```text
go2_i ∈ {go2_1, go2_2, go2_3}
```

本节主要整理对模块启动、运行及上下游系统对接具有直接意义的 Topic。Nav2、RTAB-Map 等第三方模块内部使用的 Costmap、Bond、Parameter Event 等内部 Topic 不逐项展开。

QoS 仅填写当前源码中能够明确确认的配置；对于 Gazebo、Nav2、RTAB-Map 等第三方节点未在我方代码中显式指定的 Publisher QoS，不做无依据推断。

---

### 6.1 仿真、机器人状态与传感器 Topic

#### 6.1.1 仿真基础 Topic

| Topic                    | 消息类型                        | Publisher | Subscriber / 使用方             | Frame               | QoS         | 含义                              |
| ------------------------ | ------------------------------- | --------- | ------------------------------- | ------------------- | ----------- | --------------------------------- |
| `/clock`               | `rosgraph_msgs/msg/Clock`     | Gazebo    | 所有`use_sim_time:=true` 节点 | 无                  | 无法从当前代码确认发布端 QoS | ROS2 仿真时间                     |
| `/gazebo/model_states` | `gazebo_msgs/msg/ModelStates` | Gazebo    | `/check_three_go2_attitude`   | 无统一 Header Frame | 无法从当前代码确认发布端 QoS | Gazebo 中全部模型的位姿及速度状态 |

当前启动链中的主要节点均配置：

```text
use_sim_time:=true
```

因此 `/clock` 是当前仿真系统的基础时间接口。

`/gazebo/model_states` 当前主要用于三只机器狗生成后的姿态检查，不参与后续目标追踪和导航控制。

---

#### 6.1.2 Go2 里程计 Topic

Gazebo 中每只 Go2 首先由 P3D Plugin 发布世界坐标系下的 Ground Truth Odometry：

| Topic                        | 消息类型                  | Publisher                     | Subscriber                         | Frame     | QoS                | 含义                            |
| ---------------------------- | ------------------------- | ----------------------------- | ---------------------------------- | --------- | ------------------ | ------------------------------- |
| `/go2_i/odom/ground_truth` | `nav_msgs/msg/Odometry` | Gazebo`p3d_base_controller` | `/go2_i/ground_truth_odom_relay` | `world` | 无法从当前代码确认发布端 QoS | Gazebo 中机器狗的真值位姿和速度 |

当前 Gazebo Plugin 配置：

```text
frame_name  = world
update_rate = 10 Hz
```

随后 `ground_truth_odom_relay` 将其转换为当前系统统一使用的机器人 Odom：

| Topic           | 消息类型                  | Publisher                          | Subscriber                                                                     | Frame                                                  | QoS                     | 含义                          |
| --------------- | ------------------------- | ---------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------ | ----------------------- | ----------------------------- |
| `/go2_i/odom` | `nav_msgs/msg/Odometry` | `/go2_i/ground_truth_odom_relay` | RTAB-Map、Nav2、`/nav2_dynamic_encircle`、`/maddpg_waypoint_selector` | Header:`go2_i/odom`; Child: `go2_i/base_footprint` | 默认 ROS2 QoS，depth=10 | 当前系统使用的 Go2 位姿和速度 |

当前消息明确设置：

```text
header.frame_id = go2_i/odom
child_frame_id  = go2_i/base_footprint
```

`/go2_i/odom` 是当前建图、导航、动态围捕和 MADDPG 航点选择共同使用的重要基础状态接口。选点器对三只机器人均以 depth=20 订阅 Odom。

---

#### 6.1.3 RGB-D 相机 Topic

当前三只 Go2 均启用：

```text
enable_camera:=true
```

RGB-D 相机由 `camera_rgb_3d.xacro` 中的 Gazebo Camera Plugin 发布。

| Topic                               | 消息类型                        | Publisher           | Subscriber                   | Frame                                | QoS                            | 含义         |
| ----------------------------------- | ------------------------------- | ------------------- | ---------------------------- | ------------------------------------ | ------------------------------ | ------------ |
| `/go2_i/camera/image_raw`         | `sensor_msgs/msg/Image`       | Gazebo RGB-D Camera | `/go2_i/target_perception` | `go2_i/camera_depth_optical_frame` | 感知订阅侧采用 Sensor Data QoS | RGB 图像     |
| `/go2_i/camera/camera_info`       | `sensor_msgs/msg/CameraInfo`  | Gazebo RGB-D Camera | 当前主感知链未直接使用       | `go2_i/camera_depth_optical_frame` | 无法从当前代码确认发布端 QoS | RGB 相机参数 |
| `/go2_i/camera/depth/image_raw`   | `sensor_msgs/msg/Image`       | Gazebo RGB-D Camera | `/go2_i/target_perception` | `go2_i/camera_depth_optical_frame` | 感知订阅侧采用 Sensor Data QoS | 深度图       |
| `/go2_i/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo`  | Gazebo RGB-D Camera | `/go2_i/target_perception` | `go2_i/camera_depth_optical_frame` | 感知订阅侧采用 Sensor Data QoS | 深度相机内参 |
| `/go2_i/camera/points`            | `sensor_msgs/msg/PointCloud2` | Gazebo RGB-D Camera | 当前主任务链未使用           | `go2_i/camera_depth_optical_frame` | 无法从当前代码确认发布端 QoS | RGB-D 点云   |

当前目标感知模块实际依赖：

```text
/go2_i/camera/image_raw
/go2_i/camera/depth/image_raw
/go2_i/camera/depth/camera_info
```

`target_perception` 对 RGB、Depth 和 CameraInfo 使用：

```text
qos_profile_sensor_data
```

即 Sensor Data QoS，其主要配置为：

```text
Reliability : BEST_EFFORT
Durability  : VOLATILE
```

其中 RGB 与 Depth 使用：

```text
ApproximateTimeSynchronizer
```

进行近似时间同步。

当前同步容差：

```text
sync_slop = 0.05 s
```

---

#### 6.1.4 3D 激光雷达 Topic

当前三只 Go2 均启用：

```text
enable_lidar:=true
```

| Topic                      | 消息类型                        | Publisher              | Subscriber                        | Frame              | QoS                  | 含义               |
| -------------------------- | ------------------------------- | ---------------------- | --------------------------------- | ------------------ | -------------------- | ------------------ |
| `/go2_i/velodyne_points` | `sensor_msgs/msg/PointCloud2` | Gazebo Velodyne Plugin | `go2_i_pointcloud_to_laserscan` | `go2_i/velodyne` | 无法从当前代码确认发布端 QoS | Go2 的 3D 激光点云 |

当前 Xacro 明确设置：

```text
points_topic = /go2_i/velodyne_points
frame_name   = go2_i/velodyne
```

该 Topic 是当前 RTAB-Map 建图链的主要激光传感器输入。

---

#### 6.1.5 TF Topic

当前目标感知、RTAB-Map、Nav2、动态围捕和 MADDPG 航点选择模块均依赖 TF2。

| Topic          | 消息类型                   | Publisher                                                    | Subscriber / 使用方        | 含义         |
| -------------- | -------------------------- | ------------------------------------------------------------ | -------------------------- | ------------ |
| `/tf`        | `tf2_msgs/msg/TFMessage` | `robot_state_publisher`、Odometry/TF 节点等                | 感知、建图、Nav2、动态围捕、MADDPG 选点 | 动态坐标变换 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `static_transform_publisher`、`robot_state_publisher` 等 | 感知、建图、Nav2、动态围捕、MADDPG 选点 | 静态坐标变换 |

主要坐标系及完整 TF 关系在第 8 章单独说明。

---

### 6.2 建图与导航 Topic

当前单狗建图链为：

```text
/go2_i/velodyne_points
        ↓
pointcloud_to_laserscan
        ↓
/go2_i/scan
        ↓
RTAB-Map
        ↓
/go2_i/map
```

三机地图融合链为：

```text
/go2_1/map ─┐
             │
/go2_2/map ─┼──> known_pose_map_merger ───> /merged_map
             │
/go2_3/map ─┘
```

---

#### 6.2.1 PointCloud 转 LaserScan

| Topic                      | 消息类型                        | Publisher                         | Subscriber                              | Frame              | QoS                | 含义                             |
| -------------------------- | ------------------------------- | --------------------------------- | --------------------------------------- | ------------------ | ------------------ | -------------------------------- |
| `/go2_i/velodyne_points` | `sensor_msgs/msg/PointCloud2` | Gazebo Velodyne                   | `pointcloud_to_laserscan`             | `go2_i/velodyne` | 见 6.1             | 原始 3D 点云                     |
| `/go2_i/scan`            | `sensor_msgs/msg/LaserScan`   | `go2_i_pointcloud_to_laserscan` | `/go2_i/rtabmap`、Nav2 Obstacle Layer、`/maddpg_waypoint_selector` | `go2_i/velodyne` | 选点器订阅侧为 Sensor Data QoS | 由 3D 点云转换得到的二维激光数据 |

当前转换节点使用：

```text
target_frame = go2_i/velodyne
```

选点器在角色锁定前为支持任意感知狗而订阅三路 `/scan`；角色锁定后，策略观测只使用两只跟随狗的扫描数据。当前扫描至少需包含模型要求的 108 条射线；实现先按训练方向重采样为 108 条，再压缩为 36 个雷达扇区特征。

---

#### 6.2.2 单狗地图

| Topic          | 消息类型                       | Publisher          | Subscriber                 | Frame         | QoS                                           | 含义                          |
| -------------- | ------------------------------ | ------------------ | -------------------------- | ------------- | --------------------------------------------- | ----------------------------- |
| `/go2_i/map` | `nav_msgs/msg/OccupancyGrid` | `/go2_i/rtabmap` | `/known_pose_map_merger` | `go2_i/map` | Map Merger 订阅侧：Reliable + Transient Local | 单只 Go2 的 RTAB-Map 栅格地图 |

RTAB-Map 当前参数：

```text
frame_id      = go2_i/base_link
odom_frame_id = go2_i/odom
map_frame_id  = go2_i/map
```

三只机器狗分别发布：

```text
/go2_1/map
/go2_2/map
/go2_3/map
```

---

#### 6.2.3 三机融合地图

| Topic           | 消息类型                       | Publisher                  | Subscriber                     | Frame          | QoS                                              | 含义                          |
| --------------- | ------------------------------ | -------------------------- | ------------------------------ | -------------- | ------------------------------------------------ | ----------------------------- |
| `/merged_map` | `nav_msgs/msg/OccupancyGrid` | `/known_pose_map_merger` | 三套 Nav2 Global Costmap、RViz | `merged_map` | Reliable / Transient Local / Keep Last / depth=1 | 三只 Go2 的融合 OccupancyGrid |

`known_pose_map_merger` 对地图输入和融合地图输出均使用：

```text
Reliability : RELIABLE
Durability  : TRANSIENT_LOCAL
History     : KEEP_LAST
Depth       : 1
```

默认输出参数：

```text
output_topic = /merged_map
output_frame = merged_map
publish_rate = 1.0 Hz
```

---

#### 6.2.4 Nav2 速度 Topic

当前 Nav2 不直接向机器狗最终 `/cmd_vel` 输出速度。

实际链路：

```text
Nav2 Controller / Behavior
        ↓
/go2_i/raw_cmd_nav_vel
        ↓
velocity_smoother
        ↓
/go2_i/nav_cmd_vel
        ↓
follower_cmd_vel_mux
        ↓
/go2_i/cmd_vel
```

主要 Topic：

| Topic                      | 消息类型                    | Publisher                    | Subscriber                   | Frame  | QoS                         | 含义                   |
| -------------------------- | --------------------------- | ---------------------------- | ---------------------------- | ------ | --------------------------- | ---------------------- |
| `/go2_i/raw_cmd_nav_vel` | `geometry_msgs/msg/Twist` | Nav2 Controller / Behavior   | `/go2_i/velocity_smoother` | 不适用 | 无法从当前代码确认发布端 QoS | Nav2 原始速度命令      |
| `/go2_i/nav_cmd_vel`     | `geometry_msgs/msg/Twist` | `/go2_i/velocity_smoother` | `/follower_cmd_vel_mux`    | 不适用 | Mux 订阅侧 default depth=10 | 平滑后的 Nav2 速度命令 |

其中：

```text
/go2_i/nav_cmd_vel
```

是当前 Nav2 与我方速度仲裁模块之间的主要控制接口，而不是机器人底盘最终速度接口。

---

### 6.3 目标感知 Topic

目标感知数据链为：

```text
RGB
 +
Depth
 +
CameraInfo
 +
TF
 ↓
target_perception
 ↓
目标 Pose / Odom / Status
 ↓
target_role_selector
 ↓
感知狗角色
```

---

#### 6.3.1 目标位置输出

| Topic                            | 消息类型                          | Publisher                    | Subscriber                                            | Frame                                                    | QoS                            | 含义                   |
| -------------------------------- | --------------------------------- | ---------------------------- | ----------------------------------------------------- | -------------------------------------------------------- | ------------------------------ | ---------------------- |
| `/go2_i/target_pose_estimated` | `geometry_msgs/msg/PoseStamped` | `/go2_i/target_perception` | 当前主闭环无直接订阅，可供外部使用                    | `go2_i/odom`                                           | Reliable / Volatile / depth=10 | 目标估计位置           |
| `/go2_i/target_estimated/odom` | `nav_msgs/msg/Odometry`         | `/go2_i/target_perception` | `/target_role_selector`、`/nav2_dynamic_encircle` | Header:`go2_i/odom`; Child: `go2_i/target_estimated` | Reliable / Volatile / depth=10 | 目标位置及平面速度估计 |

其中：

```text
/go2_i/target_estimated/odom
```

是当前目标追踪和动态围捕的核心目标状态接口。

`child_frame_id=go2_i/target_estimated` 仅是消息字段，`target_perception` 未广播该目标 TF，不能据此查询实际 TF 边。依据：`go2_ws_v2/src/go2_target_perception/go2_target_perception/target_perception.py` 的 `_publish()`。

消息中的：

```text
pose.pose.position
```

表示目标估计位置；

```text
twist.twist.linear.x
twist.twist.linear.y
```

表示基于相邻有效目标位置差分并经过低通滤波得到的平面速度估计。

---

#### 6.3.2 感知结果状态

| Topic                                      | 消息类型                | Publisher                    | Subscriber               | Frame | QoS                            | 含义                    |
| ------------------------------------------ | ----------------------- | ---------------------------- | ------------------------ | ----- | ------------------------------ | ----------------------- |
| `/go2_i/target_perception/result_status` | `std_msgs/msg/String` | `/go2_i/target_perception` | 测试、记录及外部诊断模块 | 无    | Reliable / Volatile / depth=10 | 单次 RGB-D 感知结果状态 |

该 Topic 的 String 内容为严格 JSON，主要字段包括：

```text
schema_version
stamp
sample_id
recognition_success
confidence
bbox
localization_success
```

该接口主要用于感知测试、数据记录和状态诊断，不参与当前运动控制闭环。

---

#### 6.3.3 感知调试图像

| Topic                                    | 消息类型                  | Publisher                    | Subscriber                    | Frame                    | QoS             | 含义                               |
| ---------------------------------------- | ------------------------- | ---------------------------- | ----------------------------- | ------------------------ | --------------- | ---------------------------------- |
| `/go2_i/target_perception/debug_image` | `sensor_msgs/msg/Image` | `/go2_i/target_perception` | `rqt_image_view` 等调试工具 | 继承输入 RGB 图像 Header | Sensor Data QoS | 带 YOLO 框及目标定位信息的调试图像 |

该 Topic 仅在：

```text
publish_debug = true
```

时发布。

---

#### 6.3.4 感知狗角色 Topic

| Topic                             | 消息类型                | Publisher                 | Subscriber                                                                                                           | Frame | QoS                                              | 含义                 |
| --------------------------------- | ----------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------ | -------------------- |
| `/target_role/perception_robot` | `std_msgs/msg/String` | `/target_role_selector` | 三个`target_perception`、`/follower_cmd_vel_mux`、`/nav2_dynamic_encircle`、`/maddpg_waypoint_selector` | 无    | Reliable / Transient Local / Keep Last / depth=1 | 当前锁定的感知狗名称 |

Topic 数据为：

```text
go2_1
```

或：

```text
go2_2
```

或：

```text
go2_3
```

当前 QoS：

```text
Reliability : RELIABLE
Durability  : TRANSIENT_LOCAL
History     : KEEP_LAST
Depth       : 1
```

因此感知狗角色被锁定后，后续启动的订阅节点仍可以获得当前角色。

---

### 6.4 动态围捕与控制 Topic

当前动态围捕与控制链为：

```text
target_estimated/odom
        ↓
dynamic_encircle
        ↓
NavigateToPose（初始靠近 Goal）
        ↓
Nav2
        ↓
nav_cmd_vel
        ↓
cmd_vel mux
        ↓
接近围捕位置
        ↓
MADDPG 选点阶段交接
        ↓
Odom + LaserScan + TF
        ↓
MADDPG 选择两个候选航点
        ↓
NavigateToPose（选点 Goal）
        ↓
Nav2 → nav_cmd_vel → cmd_vel mux → cmd_vel
```

MADDPG 在新链路中只负责选点，不生成 `Twist`；Nav2 在初始靠近和 MADDPG 选点两个阶段均保持两只跟随狗的速度控制权。

---

#### 6.4.1 最终机器人速度接口

| Topic              | 消息类型                    | Publisher                                               | Subscriber                           | Frame  | QoS                                            | 含义                 |
| ------------------ | --------------------------- | ------------------------------------------------------- | ------------------------------------ | ------ | ---------------------------------------------- | -------------------- |
| `/go2_i/cmd_vel` | `geometry_msgs/msg/Twist` | `/nav2_dynamic_encircle` 或 `/follower_cmd_vel_mux` | `/go2_i/quadruped_controller_node` | 不适用 | 我方 Publisher：Reliable / Volatile / depth=10 | Go2 最终运动控制速度 |

当前控制关系：

```text
感知狗：

nav2_dynamic_encircle
        ↓
/go2_i/cmd_vel
```

另外两只导航 / 跟随狗：

```text
MADDPG 产生候选 Goal
        ↓
Nav2 产生速度
        ↓
follower_cmd_vel_mux
        ↓
/go2_i/cmd_vel
```

因此：

```text
/go2_i/cmd_vel
```

是当前系统与 Go2 底层运动控制直接连接的最终速度接口。

正常角色锁定后，`nav2_dynamic_encircle` 发布感知狗速度，Mux 转发两只跟随狗的 Nav2 速度。Mux 未锁定角色时处理三狗输入，输入超过默认 0.5 s 未更新则输出零速度。代码中仍保留的 `/go2_i/maddpg_cmd_vel` 是旧版连续控制兼容分支，不是当前选点流程的对接接口。依据：`node.py`、`follower_cmd_vel_mux.py` 和 `maddpg_waypoint_selector.py`。

---

#### 6.4.2 MADDPG 选点阶段控制 Topic

| Topic                               | 消息类型              | Publisher                  | Subscriber                         | Frame | QoS                                  | 含义                       |
| ----------------------------------- | --------------------- | -------------------------- | ---------------------------------- | ----- | ------------------------------------ | -------------------------- |
| `/dynamic_encircle/maddpg_enable` | `std_msgs/msg/Bool` | `/nav2_dynamic_encircle` | `/maddpg_waypoint_selector` | 无 | Reliable / Transient Local / depth=1 | 请求启用或停用 MADDPG 选点 |
| `/dynamic_encircle/use_maddpg` | `std_msgs/msg/Bool` | `/nav2_dynamic_encircle` | `/follower_cmd_vel_mux` | 无 | Reliable / Transient Local / depth=1 | 旧速度 Mux 选择信号；选点版始终为 `false` |

当 `/dynamic_encircle/maddpg_enable=true` 时，选点器重置策略内部状态、恢复 Nav2 Goal 下发，并在下一次决策周期开始推理。为 `false` 时暂停 Goal 下发并取消选点器已接受的活动 Goal。

当前 C14 显式设置 `switch_mux_to_maddpg:=false`，所以进入 `MADDPG_ACTIVE` 后 `/dynamic_encircle/use_maddpg` 仍为 `false`。这表示 Nav2 继续控制速度，不表示 MADDPG 选点未运行。

---

#### 6.4.3 MADDPG Ready / Active 状态

| Topic                                     | 消息类型              | Publisher                     | Subscriber                 | Frame | QoS                                  | 含义                            |
| ----------------------------------------- | --------------------- | ----------------------------- | -------------------------- | ----- | ------------------------------------ | ------------------------------- |
| `/maddpg_waypoint/controller_ready`  | `std_msgs/msg/Bool` | `/maddpg_waypoint_selector` | `/nav2_dynamic_encircle` | 无 | Reliable / Transient Local / depth=1 | 选点器输入是否齐全且未超时 |
| `/maddpg_waypoint/controller_active` | `std_msgs/msg/Bool` | `/maddpg_waypoint_selector` | `/nav2_dynamic_encircle` | 无 | Reliable / Transient Local / depth=1 | 选点器是否已启用并进入决策回调 |
| `/maddpg_waypoint/ready` | `std_msgs/msg/Bool` | `/maddpg_waypoint_selector` | 调试/监视模块 | 无 | Reliable / Volatile / depth=10 | 本次决策周期是否可用，并包含初始阵型就绪语义 |

当前 `ready=true` 要求：

```text
MADDPG 模型加载成功
+
感知狗角色已经确定
+
三只 Go2 Odom 均已收到且未超时（默认 1.0 s）
+
两只跟随狗 LaserScan 均已收到且未超时（默认 1.0 s）
```

`controller_active=true` 表示选点器已激活，并不表示它发布速度。当前主脚本设置 `require_initial_formation:=false`，因此不再等待跟随狗进入默认候选点后才开始策略推理。

---

#### 6.4.4 MADDPG 选点结果与调试 Topic

| Topic | 消息类型 | Publisher | Frame / 数据布局 | QoS | 含义 |
| --- | --- | --- | --- | --- | --- |
| `/maddpg_waypoint/actions` | `std_msgs/msg/Int32MultiArray` | `/maddpg_waypoint_selector` | 无 Header；长度 2 | Reliable / Volatile / depth=10 | 两只跟随狗选中的离散动作索引 0～4 |
| `/maddpg_waypoint/goals` | `geometry_msgs/msg/PoseArray` | `/maddpg_waypoint_selector` | Header Frame=`merged_map`；2 个 Pose | Reliable / Volatile / depth=10 | 当前选中的两个航点 |
| `/maddpg_waypoint/observations` | `std_msgs/msg/Float32MultiArray` | `/maddpg_waypoint_selector` | 无 Header；2×83 展平为长度 166 | Reliable / Volatile / depth=10 | 本次策略推理的观测输入 |
| `/maddpg_waypoint/errors` | `std_msgs/msg/Float32MultiArray` | `/maddpg_waypoint_selector` | 无 Header；长度 2，单位 m | Reliable / Volatile / depth=10 | 两只跟随狗到当前选中航点的距离 |
| `/maddpg_waypoint/decision_diagnostics` | `std_msgs/msg/String` | `/maddpg_waypoint_selector` | 无 ROS Header；严格 JSON | Reliable / Volatile / depth=10 | 雷达新鲜度、候选点净空、blocked、掩码与选择结果 |
| `/maddpg_waypoint/markers` | `visualization_msgs/msg/MarkerArray` | `/maddpg_waypoint_selector` | 各 Marker Frame=`merged_map` | Reliable / Volatile / depth=10 | RViz 中的当前位置、选中航点、误差线与文字 |

`actions`、`goals`、`errors` 和 `markers` 的第 0/1 个元素按当前 `follower_names` 顺序对应两只跟随狗。这些 Topic 用于联调、记录和可视化；实际导航命令由选点器内部 Action Client 发送，不是由 `/maddpg_waypoint/goals` Topic 驱动 Nav2。

每只跟随狗的 83 维观测依次为：36 维雷达扇区、25 维候选点特征、2 维默认点相对位置、2 维自身速度、2 维 leader 相对位置、2 维 leader 速度、2 维 teammate 相对位置、2 维 teammate 相对速度、2 维角色 one-hot、5 维上一动作 one-hot、2 维当前 Goal 相对位置和 1 维上次进度。

---

#### 6.4.5 Handoff 状态 Topic

| Topic                               | 消息类型                | Publisher                  | Subscriber                             | Frame | QoS                                  | 含义                             |
| ----------------------------------- | ----------------------- | -------------------------- | -------------------------------------- | ----- | ------------------------------------ | -------------------------------- |
| `/dynamic_encircle/handoff_state` | `std_msgs/msg/String` | `/nav2_dynamic_encircle` | C15 CLI 监视器，也可由上游任务模块订阅 | 无    | Reliable / Transient Local / depth=1 | 当前 Nav2 初始靠近 → MADDPG 选点阶段 |

当前可能状态：

```text
ROLE_WAIT
NAV2_ACTIVE
ARRIVAL_HOLD
NAV2_CANCELLING
WAITING_FOR_STOP
WAITING_FOR_MADDPG_READY
ENABLING_MADDPG
MADDPG_ACTIVE
HANDOFF_FAILED
```

该 Topic 可作为上层系统监视当前任务执行阶段的状态接口。`MADDPG_ACTIVE` 表示选点器正在生成 Nav2 Goal，不表示 MADDPG 已取代 Nav2 输出速度。

---

#### 6.4.6 Nav Goal 状态的当前限制

当前 `NavGoalManager` 会记录 Goal generation、Action Goal Handle 及取消状态，但源码未创建 `/dynamic_encircle/nav_goal_status` Publisher。因此旧文档中该 Topic 及其 JSON schema 不是当前可用的 ROS2 对接接口，上游不应依赖它。

当前可通过 Nav2 Action 状态、节点日志以及第 6.4.4 节的选点调试 Topic 观察 Goal 下发与执行情况。

---

### 6.5 主要 Topic 数据链

当前主要跨模块数据关系可以概括为：

```text
                         Gazebo / Go2
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼
 /go2_i/camera/*        /go2_i/odom      /go2_i/velodyne_points
          │                   │                    │
          ▼                   │                    ▼
 target_perception            │         pointcloud_to_laserscan
          │                   │                    │
          ▼                   │                    ▼
target_estimated/odom         │              /go2_i/scan
          │                   │                    │
          ├──── role ─────────┤                    ▼
          │                   │                 RTAB-Map
          ▼                   │                    │
 dynamic_encircle ◄───────────┘                    ▼
          │                                     /go2_i/map
          │                                         │
          │                                known_pose_map_merger
          │                                         │
          │                                         ▼
          │                                    /merged_map
          │                                         │
          ▼                                         ▼
/go2_i/navigate_to_pose                          Nav2
          │                                         │
          │                                         ▼
          │                                /go2_i/nav_cmd_vel
          │                                         │
          │                               follower_cmd_vel_mux
          │                                         │
          │                                         ▼
          │                                  /go2_i/cmd_vel
          │
          │
          └──── 到达初始围捕位置 ──> MADDPG 选点交接

/go2_i/odom + /go2_i/scan + TF
          │
          ▼
maddpg_waypoint_selector
          │
          ├──> /maddpg_waypoint/actions, goals, observations, errors
          │
          └──> /go2_i/navigate_to_pose ──> Nav2
                                                │
                                                ▼
                                      /go2_i/nav_cmd_vel
                                                │
                                                ▼
                                     follower_cmd_vel_mux
                                                │
                                                ▼
                                         /go2_i/cmd_vel
```

---

## 7. Service / Action 接口

本章整理当前正式启动链中实际使用，或由我方功能模块明确向外提供的主要 Service 和 Action。

ROS2 节点默认生成的参数服务，以及 Nav2、RTAB-Map、Controller Manager 内部大量生命周期或管理 Service 不逐项展开；仅保留当前启动、状态检查及上下游集成直接相关的接口。

---

### 7.1 Service 接口

#### 7.1.1 Service 总表

| Service                                        | 类型                                            | Provider                                      | Caller / 使用方                         | 用途                                            |
| ---------------------------------------------- | ----------------------------------------------- | --------------------------------------------- | --------------------------------------- | ----------------------------------------------- |
| `/spawn_entity`                              | `gazebo_msgs/srv/SpawnEntity`                 | Gazebo ROS Factory                            | `gazebo_ros/spawn_entity.py`          | 将三只 Go2 模型生成到 Gazebo                    |
| `/go2_i/controller_manager/list_controllers` | `controller_manager_msgs/srv/ListControllers` | `/go2_i/controller_manager`                 | `start_three_go2_dynamic_tracking.sh` | 检查 Go2 Controller 是否已经进入`active` 状态 |
| `/walking_target/start`                      | `std_srvs/srv/Trigger`                        | Gazebo 内`walking_target_controller` Plugin | C16 启动脚本                            | 启动或继续动态目标运动                          |
| `/walking_target/pause`                      | `std_srvs/srv/Trigger`                        | `walking_target_controller`                 | 外部调试/控制                           | 暂停动态目标运动                                |
| `/walking_target/reset`                      | `std_srvs/srv/Trigger`                        | `walking_target_controller`                 | 外部调试/控制                           | 将动态目标重置到轨迹起点并暂停                  |
| `/maddpg_waypoint/set_enabled`              | `std_srvs/srv/SetBool`                        | `/maddpg_waypoint_selector`                  | 外部控制/调试                           | 手动启用或停用 MADDPG 选点                      |

---

#### 7.1.2 `/spawn_entity`

C01 加载的 `libgazebo_ros_factory.so` 提供该服务；C02～C04 的 `gazebo_ros/spawn_entity.py` 调用它生成 Go2。主脚本先等待服务出现，再启动机器人生成命令。类型与 Provider 见第 7.1.1 节。

---

#### 7.1.3 Controller 状态检查 Service

主脚本调用各狗的 `list_controllers`，等待 `joint_group_effort_controller` 和 `joint_states_controller` 均为 `active`；该检查不参与后续追踪控制。类型与 Provider 见第 7.1.1 节。

> Controller `spawner` 还使用加载、配置和切换等管理服务，属于 ROS2 Control 内部启动接口，此处不展开。

---

#### 7.1.4 动态目标控制 Service

三个 Trigger 服务的类型与 Provider 见第 7.1.1 节；其行为由 `go2_ws_v2/src/walking_target_controller/src/walking_target_controller.cpp` 实现。

##### `/walking_target/start`

功能：

```text
running = true
```

用于启动或继续动态目标沿预设轨迹运动。

当前主启动链最后执行：

```bash
ros2 service call \
    /walking_target/start \
    std_srvs/srv/Trigger \
    "{}"
```

##### `/walking_target/pause`

功能：

```text
running = false
```

用于暂停目标运动，但不重置当前轨迹位置。

调用示例：

```bash
ros2 service call \
    /walking_target/pause \
    std_srvs/srv/Trigger \
    "{}"
```

##### `/walking_target/reset`

功能：

```text
停止运动
+
重置到轨迹起点
```

调用后目标处于暂停状态。

示例：

```bash
ros2 service call \
    /walking_target/reset \
    std_srvs/srv/Trigger \
    "{}"
```

当前正式 `.sh` 启动流程仅调用：

```text
/walking_target/start
```

`pause` 和 `reset` 属于当前系统已经提供但未在主启动链中自动调用的控制接口。

---

#### 7.1.5 MADDPG 选点器手动启停 Service

服务名称、SetBool 类型及 Provider 见第 7.1.1 节。

请求：

```text
data = true
```

表示请求启用 MADDPG 选点；

```text
data = false
```

表示停用 MADDPG 选点。

关闭时节点会：

```text
停止 active 状态
+
暂停并取消选点器已接受的 Nav2 Goal
+
重置策略内部状态
```

当前 Service 调用会设置 `enabled` 状态并返回：

```text
success = true
```

启用不等于立即可用。若角色、Odom 或 LaserScan 尚未齐全，节点保持 `controller_active=false`，直到后续决策周期检测到输入就绪才进入：

```text
active = true
```

当前正式自动运行流程主要通过 Topic：

```text
/dynamic_encircle/maddpg_enable
```

完成 MADDPG 选点启停，因此该 Service 主要作为人工调试或外部系统手动控制接口。

该 Service 不操作 Mux，也不发布速度。`active=true` 表示选点器已开始推理并可向 Nav2 下发 Goal。完整自动交接使用 `/dynamic_encircle/maddpg_enable`；`/dynamic_encircle/use_maddpg` 在选点版中保持 `false`。依据：`maddpg_waypoint_selector.py`、`handoff_manager.py` 及主启动脚本的 `switch_mux_to_maddpg:=false`。

---

### 7.2 Action 接口

当前我方动态围捕节点和 MADDPG 选点器共同使用的主要 ROS2 Action 为 Nav2：

```text
NavigateToPose
```

三只机器狗分别提供：

```text
/go2_1/navigate_to_pose
/go2_2/navigate_to_pose
/go2_3/navigate_to_pose
```

---

#### 7.2.1 NavigateToPose Action 总表

| Action                      | 类型                                | Server                      | Client                                               | Goal Frame   | 用途 |
| --------------------------- | ----------------------------------- | --------------------------- | ---------------------------------------------------- | ------------ | ---- |
| `/go2_1/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | `go2_1` Nav2 BT Navigator | `/nav2_dynamic_encircle`、`/maddpg_waypoint_selector` | `merged_map` | 向 go2_1 下发初始靠近或 MADDPG 选点目标 |
| `/go2_2/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | `go2_2` Nav2 BT Navigator | `/nav2_dynamic_encircle`、`/maddpg_waypoint_selector` | `merged_map` | 向 go2_2 下发初始靠近或 MADDPG 选点目标 |
| `/go2_3/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | `go2_3` Nav2 BT Navigator | `/nav2_dynamic_encircle`、`/maddpg_waypoint_selector` | `merged_map` | 向 go2_3 下发初始靠近或 MADDPG 选点目标 |

两个节点各自创建三套 Action Client，但角色锁定后均仅向：

```text
除感知狗之外的两只导航狗
```

发送导航目标。阶段交接时，`dynamic_encircle` 先取消并完结初始靠近 Goal，再启用选点器；正常流程不会让两个 Client 同时持续刷新 Goal。

---

#### 7.2.2 Goal 内容

Goal 有两种来源：`dynamic_encircle` 根据目标状态和围捕几何关系生成初始靠近点；`maddpg_waypoint_selector` 根据每只跟随狗的 83 维观测从 5 个候选点中选择一个。两者均生成：

```text
x
y
yaw
```

并转换为：

```text
nav2_msgs/action/NavigateToPose.Goal
```

核心 Goal：

```text
goal.pose.header.frame_id = merged_map
goal.pose.pose.position.x = goal_x
goal.pose.pose.position.y = goal_y
goal.pose.pose.orientation = yaw 对应四元数
```

当前默认：

```text
global_frame = merged_map
```

因此所有动态导航 Goal 默认在：

```text
merged_map
```

坐标系下表达。

---

#### 7.2.3 动态 Goal 更新

当前 Nav Goal 并非一次性静态航点，两个阶段分别更新 Goal。

数据链：

```text
目标最新位置
     ↓
FormationPlanner
     ↓
计算两只导航狗新的目标位置
     ↓
NavGoalManager
     ↓
/go2_i/navigate_to_pose
```

初始靠近阶段，围捕计划默认以 `encircle_update_rate=1.0 Hz` 更新，Goal 以 `nav_goal_update_rate=0.2 Hz`（约每 5 秒一次）下发。进入交接后，该 `NavGoalManager` 完结并不再下发。

MADDPG 选点阶段，策略按 `decision_period=1.0 s` 生成最新候选点，但只按主脚本设置的 `nav_goal_update_period=3.0 s` 最多向 Nav2 刷新一次 Goal。两个 Action Server 均就绪后才会成组发送。当前没有 `/dynamic_encircle/nav_goal_status` Topic；Goal 生命周期可通过 Nav2 Action 和节点日志观察。

---

#### 7.2.4 Action 取消

在以下情况中，当前程序会主动取消已经接受的 Nav2 Goal：

```text
目标状态失效
```

或：

```text
准备从初始 Nav2 Goal 交接至 MADDPG 选点
```

或：

```text
dynamic_encircle 节点退出
```

取消通过当前 Action Goal Handle：

```text
cancel_goal_async()
```

执行。

Nav2 初始靠近 → MADDPG 选点 Handoff 阶段的核心流程为：

```text
两只导航狗到达围捕区域
        ↓
NAV2_CANCELLING
        ↓
取消当前 NavigateToPose Goal
        ↓
WAITING_FOR_STOP
        ↓
确认机器人停止
        ↓
WAITING_FOR_MADDPG_READY
        ↓
启用 MADDPG 选点器
        ↓
选点器向同一 NavigateToPose Action 发送新 Goal
```

因此 Nav2 Action 的 Goal 取消能力是避免初始靠近 Goal 与选点 Goal 重叠的重要条件。选点器停用或节点退出时，也会取消由自身接受的活动 Goal。

---

#### 7.2.5 Action 就绪检查

当前主启动脚本在每套 Nav2 启动后均检查：

```text
/go2_i/navigate_to_pose
```

是否已经出现在：

```bash
ros2 action list
```

中。

脚本在对应 Action 名称出现后继续；名称存在检查不等于成功发送 Goal。围捕节点和选点器实际下发前都调用 `server_is_ready()` 检查两只跟随狗的 Server。

可人工检查：

```bash
ros2 action info /go2_1/navigate_to_pose
ros2 action info /go2_2/navigate_to_pose
ros2 action info /go2_3/navigate_to_pose
```

因此对上游系统而言，如果使用自身导航模块替换当前 Nav2 启动链，则只要继续向我方提供兼容的：

```text
/go2_i/navigate_to_pose
```

Action Server，可同时保持 `dynamic_encircle` 和 `maddpg_waypoint_selector` 的目标下发接口名称与类型不变；完整运动控制还须满足第 10.3 节的重复 Goal、取消、结果反馈和速度输出要求。

---

## 8. TF / 坐标系

当前系统主要使用以下坐标系：

| Frame                                | 含义                     |
| ------------------------------------ | ------------------------ |
| `merged_map`                       | 三机统一全局地图坐标系   |
| `go2_i/map`                        | 单只 Go2 建图坐标系      |
| `go2_i/odom`                       | 单只 Go2 里程计坐标系    |
| `go2_i/base_footprint`             | 机器人地面投影坐标系     |
| `go2_i/base_link`                  | 机器人本体坐标系         |
| `go2_i/velodyne`                   | 3D 激光雷达坐标系        |
| `go2_i/camera_depth_optical_frame` | RGB-D 深度相机光学坐标系 |

其中：

```text
go2_i ∈ {go2_1, go2_2, go2_3}
```

### 8.1 主要 TF 链

当前核心 TF 关系可概括为：

```text
merged_map
    ↓
go2_i/map
    ↓
go2_i/odom
    ↓
go2_i/base_footprint
    ↓
go2_i/base_link
    ├── go2_i/velodyne
    └── go2_i/camera_depth_optical_frame
```

其中：

- `merged_map → go2_i/map`：由 `three_go2_map_merge.launch.py` 中的静态 TF 发布；
- `go2_i/map → go2_i/odom`：由 `go2_i_mapping_nav.launch.py` 中的静态 TF 发布；
- `go2_i/odom → go2_i/base_footprint`：由 `ground_truth_odom_relay` 动态发布；
- `go2_i/base_footprint → go2_i/base_link`：由 `/go2_i/base_to_footprint_ekf` 动态发布；
- `base_link → 传感器 Frame`：由 `robot_state_publisher` 根据 URDF/Xacro 发布。

传感器链由 Xacro 中的固定关节构成（图中省略中间 Link），其几何关系固定；不要据此推断第三方版本的具体发布 Topic 行为。当前 mapping 配置为 RTAB-Map `publish_tf: false`，`map → odom` 由单独静态发布者提供，不是 SLAM 动态修正；融合 launch 还发布零变换 `merged_map → world`。替换定位模块时，同一 TF 边应由明确的唯一来源提供，避免新定位 TF 与旧静态发布者并存。

依据：`go2_ws_v2/src/unitree-go2-ros2/robots/configs/go2_config/launch/spawn_go2_velodyne_1.launch.py`、`go2_ws_v2/src/go2_mapping_nav/config/rtabmap/go2_1_mapping.yaml` 及对应三狗 launch/config。

### 8.2 感知模块 TF 要求

目标感知首先在：

```text
go2_i/camera_depth_optical_frame
```

下得到目标三维位置，再通过 TF 转换到：

```text
go2_i/odom
```

因此感知模块需要保证：

```text
go2_i/camera_depth_optical_frame
        ↔
go2_i/odom
```

TF 连通。

### 8.3 动态围捕与 MADDPG 选点 TF 要求

`dynamic_encircle` 默认使用：

```text
global_frame = merged_map
```

机器人 Odom 和目标估计均需要转换到统一全局坐标系，因此需要保证：

```text
go2_i/odom
    ↔
merged_map
```

TF 连通。

当前仿真中 `merged_map`、`world`、`go2_i/map` 和 `go2_i/odom` 之间采用恒等变换，即零平移、零旋转；不是机器狗出生位置为零。`ground_truth_odom_relay` 保留世界坐标的 x/y，并投影为平面 Odom。

MADDPG 选点器会将三狗 Odom 中的 Pose 通过 TF2 转换到 `global_frame`（默认 `merged_map`），再生成候选点和 Nav2 Goal；如果 Odom Header Frame 已经等于 `merged_map`，则不进行转换。因此必须保证每个 `go2_i/odom ↔ merged_map` 可查询，默认 TF 查询超时为 0.2 s。

选点器不通过 TF 变换 LaserScan 点，而是根据跟随狗 Odom yaw、leader yaw 和训练配置中的雷达前向偏移重建障碍点。因此 `/go2_i/scan` 必须符合当前物理雷达坐标约定，不能只改 Topic 或 `frame_id` 而不同步调整预处理。Odom 的 `twist.twist.linear.x/y` 按机体坐标解释，并使用转换后的 yaw 旋转到全局坐标。

TF 连通仍不是整套系统兼容的充分条件：感知狗控制直接比较自身 Odom 与目标坐标，追踪及围捕朝向还使用预配置路线。各数据源仍须满足一致的轴向、速度坐标和时间约定。

依据：`go2_ws_v2/src/unitree-go2-ros2/robots/configs/go2_config/scripts/ground_truth_odom_relay.py`；`go2_ws_v2/src/go2_mapping_nav/go2_mapping_nav/maddpg_waypoint_selector.py`；`go2_ws_v2/src/go2_dynamic_encircle/go2_dynamic_encircle/` 下的 `perception_controller.py`、`formation_planner.py`、`node.py`。

---

## 9. URDF / Xacro

当前三只 Go2 均通过：

```text
go2_description/xacro/robot_3d_lidar_nocam.xacro
```

生成机器人模型。

虽然文件名包含 `nocam`，但当前正式启动时：

```text
enable_lidar:=true
enable_camera:=true
```

因此实际同时加载 3D 激光雷达和 RGB-D 相机。

### 9.1 当前主要 Xacro

| Xacro                          | 作用                             |
| ------------------------------ | -------------------------------- |
| `robot_3d_lidar_nocam.xacro` | Go2 主模型入口                   |
| `gazebo.xacro`               | Gazebo、ROS2 Control、IMU 等插件 |
| `velodyne_3d.xacro`          | 3D Velodyne 激光雷达             |
| `camera_rgb_3d.xacro`        | RGB-D 相机                       |

### 9.2 关键参数

每只 Go2 启动时主要传入：

```text
robot_namespace:=/go2_i
frame_prefix:=go2_i/
points_topic:=/go2_i/velodyne_points
name_suffix:=go2_i
enable_velodyne:=true
enable_camera:=true
```

其中：

- `robot_namespace`：区分三只机器狗的 ROS Namespace；
- `frame_prefix`：区分三只机器狗的 TF Frame；
- `points_topic`：指定独立 Velodyne 点云 Topic；
- `name_suffix`：避免 Gazebo Plugin 名称冲突；
- `enable_velodyne`：启用 3D 激光雷达；
- `enable_camera`：启用 RGB-D 相机。

当前三个机器人通过：

```text
go2_1
go2_2
go2_3
```

三个 Namespace / Frame Prefix 实现 Topic、TF 和传感器接口隔离。

---

## 10. 与上游系统对接要求

本章说明当前实现的对接条件。上游已有仿真、建图、导航或机器人控制模块时，可在满足下述接口和行为约定后替换对应模块；是否兼容不能仅由名称和消息类型判断。

### 10.1 机器人状态与传感器接口

若上游不使用当前 Gazebo / Go2 仿真模块，需提供三狗 `/go2_i/odom`（`nav_msgs/msg/Odometry`）及配套 TF。保留 MADDPG 选点器时，还需提供三狗 `/go2_i/scan`（`sensor_msgs/msg/LaserScan`）；角色锁定后策略实际使用两只跟随狗的扫描。保留我方感知时还需提供第 10.6 节的 RGB-D 输入。

三狗 Odom Pose 须能通过 TF 转换到 `global_frame=merged_map`。选点器按机体系解释 Odom 的线速度 `x/y`，再根据机器人 yaw 转到全局坐标；上游若发布世界系 twist，必须在对接层转换。LaserScan 的角度方向、雷达安装偏移和量程须与选点预处理及训练配置一致。依据：`maddpg_waypoint_selector.py`、`waypoint_maddpg_v0/config.py` 和 `target_perception.py`。

如继续使用当前建图模块，还需提供：

```text
/go2_i/velodyne_points
```

并保证相关 TF 可用。

保留 `use_sim_time=true` 时必须提供 `/clock`，且图像、目标、Odom 和 LaserScan 的时间来源与节点时钟一致。当前角色选举默认仅接受消息年龄不超过 0.5 s 的目标估计，围捕机器人 Odom 超时阈值默认 0.5 s，选点器的 Odom 和 LaserScan 接收超时均默认 1.0 s，全部 Odom/Scan 接收时刻的最大偏差默认不得超过 0.2 s。

替换 Gazebo 后，原脚本的 `/spawn_entity`、`/gazebo/model_states`、Controller Manager 检查以及 `/walking_target/start` 不能默认继续适用。它们是原仿真启动链要求，不是追踪和选点算法本身全部必需的上游服务；需由集成启动流程对应处理。依据：主脚本及 `target_role_selector.py`、围捕 `config.py`、`maddpg_waypoint_selector.py`。

---

### 10.2 建图接口

若上游使用自身三机建图模块替换：

```text
go2_mapping_nav
```

则需要提供统一的全局地图和坐标系。

当前默认接口为：

```text
/merged_map
Type: nav_msgs/msg/OccupancyGrid
Frame: merged_map
```

同时需要保证：

```text
go2_i/odom
    ↔
merged_map
```

TF 连通。

`dynamic_encircle` 和 `maddpg_waypoint_selector` 均使用 `global_frame`（当前主脚本为 `merged_map`）表达 Nav2 Goal。原 mapping launch 的 `use_merged_map` 只在 `merged_map` 与对应 `go2_i/map` 间选择，原融合 launch 的输出 Frame 和静态 TF 则写为当前名称。不能把单个节点参数可配置理解为整条启动链提供了统一的任意 Frame 参数。适配时还必须满足第 8.3 节的数值坐标、雷达安装和路线约束。

若保留当前 `known_pose_map_merger`，输入地图须已处于共同坐标约定；它不通过 TF 对任意独立地图做配准，并明确拒绝非零 `initial_se2_transforms`。若连导航也由上游替换，感知、围捕和选点节点本身均不直接订阅 `/merged_map`；地图是否需要由保留的导航实现决定，但 `merged_map` Frame 仍须通过 TF 可用。

依据：`go2_ws_v2/src/go2_mapping_nav/launch/three_go2_map_merge.launch.py`、`go2_1_mapping_nav.launch.py`，以及 `go2_ws_v2/src/go2_mapping_nav/go2_mapping_nav/known_pose_map_merger.py`。

---

### 10.3 导航接口

若上游使用自身 Nav2 或其他导航系统，需要继续提供三只机器狗对应的导航 Action：

```text
/go2_1/navigate_to_pose
/go2_2/navigate_to_pose
/go2_3/navigate_to_pose
```

类型：

```text
nav2_msgs/action/NavigateToPose
```

`dynamic_encircle` 在初始靠近阶段通过该 Action 下发围捕目标；交接完成后，`maddpg_waypoint_selector` 通过同一 Action 下发策略选中的候选航点。

保留当前 Mux 时，上游导航还需把速度输出到 `/go2_i/nav_cmd_vel`（`geometry_msgs/msg/Twist`），而不是直接向最终 `/cmd_vel` 竞争发布。Mux 默认输入超过 0.5 s 未更新即输出零速度。

Action 必须兼容周期性替换 Goal、取消请求及结果返回。交接不仅等待初始 Goal 取消完成，还通过 Odom 确认两只导航狗停止：默认线速度不超过 0.08 m/s、角速度不超过 0.12 rad/s，保持 0.5 s；之后选点器默认每 1 s 决策、每 3 s 最多刷新一次 Goal。Goal Frame 默认 `merged_map`，修改时须同时适配围捕节点、选点器、TF 和导航系统。依据：`go2_dynamic_encircle` 中的 `nav_goal_manager.py`、`node.py`、`handoff_manager.py`，以及 `go2_mapping_nav/maddpg_waypoint_selector.py`。

---

### 10.4 运动控制接口

当前机器人最终速度接口为：

```text
/go2_i/cmd_vel
```

类型：

```text
geometry_msgs/msg/Twist
```

当前控制关系见第 6.4 节：感知狗由 `dynamic_encircle` 直接发布速度，两只跟随狗在全流程中都由 Nav2 生成速度，再由 Mux 转发。MADDPG 只为跟随狗选择 Nav2 Goal，不发布 `/go2_i/maddpg_cmd_vel` 或其他 `Twist`。

如果上游已有自己的速度控制或速度仲裁模块，应统一最终 `/cmd_vel` 的控制权，避免多个节点同时向同一机器狗发布速度指令。替换 Mux 时只需保留 Nav2 速度链，不应把旧 `/maddpg_cmd_vel` 分支视为新版必需接口。

---

### 10.5 机器人命名约定

当前默认机器人名称：

```text
go2_1
go2_2
go2_3
```

对应 Topic 和 Frame 均以该名称作为前缀。

例如：

```text
/go2_1/odom
/go2_1/camera/image_raw
go2_1/odom
go2_1/base_link
```

若上游采用其他机器人名称，应同步适配各节点 `robot_names`、选点器的 `leader_name/follower_1/follower_2`或动态角色参数、感知 `robot_namespace`、Topic、Frame 及角色字符串。选点器会由机器人名生成 `/<name>/odom`、`/<name>/scan` 和 `/<name>/navigate_to_pose`。多处 Topic 使用绝对路径，仅在外层增加 namespace 不会自动修改这些名称；现有三狗 launch 也固定使用 `go2_1/2/3`。

ROS2 Topic remap 不修改消息内 `header.frame_id`、`child_frame_id` 或角色字符串。角色选举与目标追踪严格要求目标估计 `header.frame_id == "<robot>/odom"`；不能仅将感知输出 Frame 改为任意全局 Frame。依据：`go2_ws_v2/src/go2_target_perception/go2_target_perception/target_role_selector.py`、`go2_ws_v2/src/go2_dynamic_encircle/go2_dynamic_encircle/target_tracker.py`。

---

### 10.6 感知模块输入要求

若上游希望直接使用我方目标感知模块，每只机器狗至少需要提供：

```text
RGB Image
Depth Image
CameraInfo
TF
```

当前默认 Topic：

```text
/go2_i/camera/image_raw
/go2_i/camera/depth/image_raw
/go2_i/camera/depth/camera_info
```

并保证：

```text
go2_i/camera_depth_optical_frame
        ↔
go2_i/odom
```

TF 可用。

RGB 与深度图须已像素对齐，内参须对应深度图。实现直接在 RGB 检测框对应像素采样深度，不执行 RGB-D 配准；RGB 需能经 `cv_bridge` 转为 `bgr8`。深度 `16UC1`/`mono16` 按毫米转为米，浮点深度应以米提供，不能仅靠 Topic 名称判断单位。

RGB 与 Depth 采用近似时间同步，默认容差 0.05 s；CameraInfo 单独接收。深度消息的 `header.frame_id` 是实际 TF 源 Frame，须保证其与配置的目标输出 Frame 连通。当前输出默认 `go2_i/odom`，保留角色选举与追踪时还须满足第 10.5 节校验。

此外，当前 `dynamic_encircle` 读取 `scene/scene_config` 中的 `dynamic_target.route`，用于追踪路径和围捕朝向；该配置不只是出生位置配置。替换场景或坐标系时，须保证路线数值与参与运算的机器人、目标坐标一致。依据：`go2_ws_v2/src/go2_target_perception/go2_target_perception/target_perception.py`；`go2_ws_v2/src/go2_dynamic_encircle/go2_dynamic_encircle/node.py`、`perception_controller.py`、`formation_planner.py`。

---

### 10.7 推荐对接边界

上游系统与我方模块之间可按照以下接口进行对接：

```text
上游系统
│
├── Go2 状态
│   └── /go2_i/odom
│
├── 2D 雷达
│   └── /go2_i/scan
│
├── RGB-D
│   ├── /go2_i/camera/image_raw
│   ├── /go2_i/camera/depth/image_raw
│   └── /go2_i/camera/depth/camera_info
│
├── 全局地图 / TF
│   └── merged_map 或等价全局 Frame
│
└── 导航 Action
    └── /go2_i/navigate_to_pose

                ↓

我方目标感知 + dynamic_encircle + MADDPG 选点器

                ↓

MADDPG 选中 Goal → /go2_i/navigate_to_pose → Nav2

                ↓

/go2_i/nav_cmd_vel → Mux → /go2_i/cmd_vel
```

该图仅概括接口方向。完整对接还包括角色 Topic、`/dynamic_encircle/maddpg_enable`、选点器 ready/active 状态、导航输出 `/go2_i/nav_cmd_vel` 到 Mux，以及上游底盘订阅 `/go2_i/cmd_vel`。新版 MADDPG 不与 Mux 交换速度；其与导航系统的主接口是 `NavigateToPose` Action。

上游可以替换 Gazebo、RTAB-Map 或 Nav2 的实现，但必须满足第 10.1～10.6 节的数据、时间、坐标、雷达、路线及 Action 行为约定。保留 MADDPG 选点器时，LaserScan 与 Odom/TF 是策略的必需输入；不能因为上游 Nav2 自身能够避障，就省略选点器的 `/scan` 接口。

---
