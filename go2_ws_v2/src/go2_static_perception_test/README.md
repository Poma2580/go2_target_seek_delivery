# 静态五目标感知测试（第一阶段）

独立 City world、单狗单 prompt YOLOE、轻量 smoke 验证与静态 100 Case 测试框架。只使用 `go2_1`，不接入 T1/T2/T3，不启动 walking target、角色选择器、多狗、Nav2 或 RTAB-Map。

## 静态 100 Case Runner

五类目标各使用 20 个确定性机器人位姿，按目标优先顺序展开为 100 Cases。正式运行、单类、单 Case 与只读展开命令如下：

```bash
ros2 run go2_static_perception_test static_test_runner --suite static_100cases.yaml --model-path yoloe-26s-seg.pt --device cuda:0
ros2 run go2_static_perception_test static_test_runner --target person --model-path yoloe-26s-seg.pt --device cuda:0
ros2 run go2_static_perception_test static_test_runner --case SP-PERSON-P01 --model-path yoloe-26s-seg.pt --device cuda:0
ros2 run go2_static_perception_test static_test_runner --case SP-PERSON-P01 --gui --rqt --model-path yoloe-26s-seg.pt --device cuda:0
ros2 run go2_static_perception_test static_test_runner --no-gui --no-rqt --model-path yoloe-26s-seg.pt --device cuda:0
ros2 run go2_static_perception_test static_test_runner --dry-run
```

`static_100cases.yaml` 默认与 T1 一致启用 Gazebo GUI 和 RQT；命令行 `--gui/--no-gui`、`--rqt/--no-rqt` 可覆盖。RQT 固定查看 `/go2_1/static_perception/debug_image`。Runner 在启动 Recorder 前等待 YOLOE 至少发布一条真实 `result_status`，仅用于确认感知流已完成首轮推理，不改变 2 Hz、10 s、0.4 s 时间戳匹配或任何指标定义。

结果目录按 T1 层级保存：默认根目录固定为仓库根目录下的 `TestResults/`，不受启动命令当前工作目录影响；可用 `--results-root PATH` 显式覆盖。`TestResults/batch_YYYYmmdd_HHMMSS/resolved_cases.yaml` 位于批次根目录，静态专项位于 `static_target_test/`，Case 使用 `case_001` 到 `case_100`。每个 Case 根目录保存 `case_config.yaml`、`case_summary.yaml` 和 `attempts/`；原始 CSV、指标和完整日志只保存在最终/各次 `attempts/attempt_XX/` 内，不再复制到 Case 根目录。批次逐类与总体加权统计位于 `static_target_test/summary/`。基础设施或倒地失败最多重启八次，识别/定位算法失败不会重试。Runner 不调用静态 world 验证器。

## 场景和正式目标

`worlds/city_static_objects.world` 从 `QY_MODEL/target_seek` 派生，保留城市布局、地面、光照、原有环境物体和 `/gazebo/model_states`。原 world 的保存状态已落实到环境模型/局部 link 定义，再移除历史 `<state>`，以免覆盖正式目标 pose。环境原有 `dumpster_94` 和其他行人仍保留；它们不是本次配置中的正式目标。

| target key   | model_name      | 唯一 prompt  |       x |       y |                z |      roll |    pitch |      yaw |
| ------------ | --------------- | ------------ | ------: | ------: | ---------------: | --------: | -------: | -------: |
| airplane     | cessna_c172     | airplane     | 45.1531 |      28 |         0.427145 | -0.000106 | -0.06111 | -1.97173 |
| person       | static_person   | person       |      62 |       5 |                0 |         0 |        0 |        0 |
| pickup_truck | pickup_truck    | pickup truck |      46 |      10 |    0.00734785678 |         0 |        0 |        0 |
| ground_robot | pr2             | ground robot |       2 |       2 |                0 |         0 |        0 |        0 |
| dumpster     | static_dumpster | dumpster     |     -20 |      17 | 0.00137753173193 |         0 |        0 |     3.14 |

位置单位米、角度单位弧度；YAML 中 pose 是 **Gazebo world 坐标系的模型原点**。五个正式目标均显式设为 `<static>true</static>`。飞机保留任务提供的 x/z 和姿态，仅将 y 调整为 28.0，使其 10–12 m 测试圆环位于现有 City occupancy map 内；不使用旧 `<state>` 中另一个飞机 pose。

其他位姿的来源：

- 行人使用 `~/.gazebo/models/person_walking` 的完整模型结构，保留其底部碰撞盒和 visual/collision 的局部 z=`-0.02`，在派生 world 中重命名为 `static_person` 并显式设为静态；模型原点 pose 保持为 `(62, 5, 0, 0, 0, 0)`。
- Pickup 实际来源是 `~/.gazebo/models/pickup/model.sdf`，不是 `model://pickup_truck`。保留其局部 mesh yaw `-1.57079632679` 和模型默认朝向。DAE 单位为英寸，最低顶点 z 为 `-0.2517014`，场景节点 z 平移 `-0.0375843`；落地高度为 `(0.2517014 + 0.0375843) * 0.0254`。
- PR2 保留原 SDF 的基座默认朝向及 link pose；`base_footprint` 底部 box 中心 z=0.071、高 0.142，底面为 0。复制到新 world 后删除全部 18 个 sensor 和 2 个 plugin，仅保留外观、碰撞和静态机构。
- Dumpster DAE 单位为英寸，最低顶点 z=`-8.806948`，节点平移 z=`8.72673`，节点 Z scale=`0.4507179`，SDF mesh scale=`1.5`。落地高度为 `(8.806948-8.72673)*0.4507179*0.0254*1.5`；yaw=3.14 沿用原 City `dumpster_94` 的摆放朝向，roll/pitch 沿用模型直立定义。
- 原 world 的 `ground_plane` 高度为 0。以上高度结合模型变换计算，并通过 Gazebo 静止检查和相机画面检查。

原 `walking_target` actor、轨迹与 controller 已移除。背景 UAV 保留外观并设为静态，展开其嵌套模型后删除相机、传感器和控制插件；新 world 唯一运行插件是 `gazebo_ros_state`。共享 `QY_MODEL` 和 `~/.gazebo/models` 均未修改。

## 环境与构建

使用 ROS 2 Humble / Gazebo Classic、系统 Python、已构建的 `go2_config` 及其依赖。沿用现有 City 场景资源：`QY_MODEL/models` 与 `~/.gazebo/models`；后者需包含 person_walking、pickup、pr2、dumpster 等场景使用的模型。

YOLOE 使用系统 Python 环境中的 Ultralytics、PyTorch、OpenCV 和 NumPy。没有需要单独安装的 `yoloe` pip 包，YOLOE API 包含在 `ultralytics==8.4.82` 中。YOLOE-26 的文字 tokenizer 由 Ultralytics 的 CLIP fork 提供，默认模型使用 TorchScript 文本编码器 `mobileclip2_b.ts`，不需要另装 `mobileclip` Python 包。

新机器先退出 conda，并从仓库根目录安装已固定的 Python 依赖：

```bash
cd "$DELIVERY_ROOT"
conda deactivate                # 无活动 conda 环境时可跳过
which python3                   # 必须输出 /usr/bin/python3
/usr/bin/python3 -m pip install --user -r requirements.txt
```

`requirements.txt` 已包含 `ultralytics==8.4.82` 及验证过 commit 的 Ultralytics CLIP fork。CPU 可直接使用 `device:=cpu`；GPU 机器应先按照本机 NVIDIA 驱动和 CUDA 环境安装匹配的 `torch`/`torchvision`，不要盲目复制开发机的 CUDA wheel。

YOLOE 主模型和文本编码器是两个独立资产：默认分别为 `yoloe-26s-seg.pt` 和 `mobileclip2_b.ts`。首次联网启动时，节点会将缺失资产自动下载到 `model_cache_dir`，默认是 `~/.cache/go2_static_yolo_smoke`。离线迁移时，应从已准备好的机器复制这两个文件：

```bash
mkdir -p "$HOME/.cache/go2_static_yolo_smoke"
cp /path/to/yoloe-26s-seg.pt "$HOME/.cache/go2_static_yolo_smoke/"
cp /path/to/mobileclip2_b.ts "$HOME/.cache/go2_static_yolo_smoke/"
```

节点本身不安装或改写全局 Python 依赖，模型缓存也不提交到仓库。

首次使用或修改本包源码后，才需要重新构建：

```bash
cd "$DELIVERY_ROOT/go2_ws_v2"
conda deactivate                # 无活动 conda 环境时可跳过，但必须确认下面结果
which python3                   # 必须输出 /usr/bin/python3
source /opt/ros/humble/setup.bash
colcon build --packages-select go2_static_perception_test --symlink-install
source install/setup.bash
```

构建完成后，每个日常 ROS 终端只需加载环境：

```bash
cd "$DELIVERY_ROOT/go2_ws_v2"
conda deactivate
which python3                   # 必须输出 /usr/bin/python3
source /opt/ros/humble/setup.bash
source install/setup.bash
```

这里沿用仓库根 README 已写入 `~/.bashrc` 的模型目录，不设置测试专用的 `ROS_DOMAIN_ID` 或 `GAZEBO_MASTER_URI`。同一个 Gazebo master 上只启动一个 world。

## 启动顺序

**1. 只启动场景。**

```bash
ros2 launch go2_config gazebo_target_seek_world.launch.py \
  world:="$(ros2 pkg prefix go2_static_perception_test)/share/go2_static_perception_test/worlds/city_static_objects.world" \
  gui:=true
```

**2. 在另一终端验证 world，等待成功后再生成狗。**

```bash
ros2 run go2_static_perception_test validate_static_world
```

验证器等待五个模型出现，检查 YAML 初值，持续检查 10 秒仿真时间内的全部采样；位置容差 0.01 m、四元数角距离容差 0.01 rad。默认启动总超时 120 秒墙钟时间，开始采样后消息或仿真时间停滞 10 秒即失败。成功返回 0，否则返回非 0，并输出 JSON 诊断。可使用 `--targets`、`--duration`、`--timeout`、`--stall-timeout` 覆盖。

**3. 生成唯一的 go2_1。**

使用 `spawn_go2_velodyne_1.launch.py`，该入口支持相机开关。不要使用无相机的 `spawn_go2_1.launch.py`。

Person 的观察位置：

```bash
ros2 launch go2_config spawn_go2_velodyne_1.launch.py \
  scene:=city spawn_x:=58.0 spawn_y:=5.0 spawn_z:=0.4 spawn_yaw:=0.0 \
  enable_camera:=true enable_lidar:=false
```

等待关节控制器激活，并检查狗已站稳、相机朝向正确。曾在 world 与狗几乎同时启动时观察到倒地及不完整的相机场景，故必须按上述顺序启动，不能仅凭 model_states 中有模型就认定 RGB 可用。若狗倒地，可在控制器已激活后恢复当前测试狗的位姿，再检查画面：

```bash
ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState \
  '{state: {name: go2_1, pose: {position: {x: 58.0, y: 5.0, z: 0.4}, orientation: {w: 1.0}}, reference_frame: world}}'
```

**4. 启动 person 感知。**

```bash
ros2 launch go2_static_perception_test single_go2_static_yoloe.launch.py \
  target_prompt:=person model_path:=yoloe-26s-seg.pt device:=cuda:0 use_sim_time:=true
```

无 CUDA 时使用 `device:=cpu`。节点优先使用明确指定的本地权重；裸权重名在 `model_cache_dir` 中查找，缓存不存在时由原 YOLOE API 下载；不自动更换模型。启动位置不影响缓存查找，也不需要进入缓存目录。

**5. 执行轻量 RGB-D smoke。**

```bash
cd "$DELIVERY_ROOT"
/usr/bin/python3 tools/static_perception_map/check_rgbd_smoke.py \
  --prompt person --output /tmp/static_smoke_person
```

该程序只观察当前一个人工布置的目标，检查 RGB/Depth/CameraInfo/debug、严格 JSON、递增 sample_id，以及成功定位对应的同时间戳 pose。至少观察 10 秒仿真时间且获得 3 个成功样本才通过；默认最多等待 180 秒墙钟时间。只保存一份结果 JSON 和末帧 RGB/debug 图，不展开 Case 或记录数据集。

**6. 切换 PR2。**

停止 person 感知节点。新一次场景启动时，Go2 观察位置改为 `(6,2,0.4)`、yaw=`3.141592653589793`，其余步骤相同；已有站稳的单狗也可通过 `/gazebo/set_entity_state` 人工移到该位置，四元数 `{z: 1.0, w: 0.0}`，无需生成第二只狗。

```bash
ros2 launch go2_static_perception_test single_go2_static_yoloe.launch.py \
  target_prompt:="ground robot" model_path:=yoloe-26s-seg.pt device:=cuda:0 use_sim_time:=true
```

随后将 smoke 命令改为 `--prompt "ground robot" --output /tmp/static_smoke_ground_robot`。

## 节点接口

默认输入：

- `/go2_1/camera/image_raw`
- `/go2_1/camera/depth/image_raw`
- `/go2_1/camera/depth/camera_info`

默认输出：

- `/go2_1/static_perception/target_pose_estimated`：`geometry_msgs/PoseStamped`
- `/go2_1/static_perception/result_status`：`std_msgs/String`，严格 JSON
- `/go2_1/static_perception/debug_image`：`sensor_msgs/Image`，best-effort

默认目标坐标系为 `go2_1/odom`。本次 spawn 使用现有 ground-truth odom relay，其位置来源于 Gazebo 世界位姿，适合本次 smoke；节点本身只使用 TF，不读取目标真值。切换其他 target_frame 前须提供相应 TF，不能把任意 odom 与 YAML world 坐标直接比较。

姿态字段固定单位四元数，不估计物体朝向。定位值是所选 bbox 中心射线与 ROI 深度给出的可见表面点，不是模型原点或几何中心。本阶段没有位置误差指标。

参数详见 `go2_static_perception_test/config/static_yoloe_perception.yaml`。至少包含任务要求的所有输入输出 topic、模型、缓存目录、prompt、设备、深度、同步、频率和 TF 参数，另保留四个 ROI 比例参数。参数均在启动时固定；`target_prompt` 空白拒绝，运行中切换需重启节点。只有一个 prompt 被传给 `get_text_pe` 和 `set_classes`，不加载五类别联合检测。

每次实际 YOLOE 调用恰好输出一次状态，sample_id 从 0 开始递增；RGB 转换失败、重复或过期图像在推理前丢弃，不生成状态。状态及 pose 使用深度图时间戳；debug 使用 RGB 时间戳。

```json
{"schema_version":1,"stamp":{"sec":12,"nanosec":0},"sample_id":3,"prompt":"ground robot","recognition_success":true,"confidence":0.83,"bbox":[100,80,220,300],"localization_success":true}
```

没有检测时 confidence/bbox 为 null；检测成功而深度、内参、对齐条件或 TF 失败时保留检测结果、定位标为 false，不发布 pose。TF 先查图像时刻，再回退最新变换。调试图显示 prompt、confidence、Z、x/y 和失败原因，调试失败不改变状态。

## 测试与资源重建

```bash
# 已退出 conda，确认 which python3 为 /usr/bin/python3，且 source 过 ROS 和工作区。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest \
  go2_ws_v2/src/go2_target_perception/test \
  go2_ws_v2/src/go2_static_perception_test/test -q
```

如确需重新从源模型生成资产，运行 `tools/static_perception_map/build_static_world.py`（系统 Python，需要 NumPy/SciPy/PyYAML）。它只改写本包的派生 world 和 YAML，不修改源 world、源模型或模型缓存；不会在 launch 时自动运行。模型源文件发生变化后，必须重新验证几何高度、world 静止性和两项感知 smoke。

本次验收记录见仓库 `tools/static_perception_map/artifacts/phase1/`。

## 静态测试位姿生成

五类目标的 100 个确定性 `go2_1` 初始位姿由独立工具
`tools/static_perception_pose_generator/` 生成，正式配置位于
`config/poses/static_robot_pose_cases.yaml`。工具只使用 City occupancy map 的
free/unknown/occupied 状态、0.8 m 障碍 clearance 和 Go2 相机水平视场，不接入
Runner、Recorder 或指标计算。生成与只读 `--check` 命令详见该工具 README。
