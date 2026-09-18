# Go2 T1/T2/T3 自动化测试框架

`go2_test_framework` 用同一个 `target_test_runner` 执行三类任务：

| task_type         | 套件 | 任务                           | 正式 Batch 阈值                         |
| ----------------- | ---- | ------------------------------ | --------------------------------------- |
| `perception`    | T1   | 目标识别与相对定位             | 沿用 recognition/localization Case 判据 |
| `tracking`      | T2   | 感知狗连续跟踪                 | 70%                                     |
| `path_planning` | T3   | 两只导航狗同时到达且无静态碰撞 | 90%                                     |

正式套件都按 `city → forest → airport`、`straight → rectangle → v_shape`、`group_01 → group_11` 展开，均为 99 Cases。T1 的目录、Case ID、
顺序和既有 CSV 保持兼容；T2/T3 使用独立结果目录。

## 目录与文件职责

下面是当前 `go2_test_framework` 包的文件架构。相对旧版 T1 文档新增的
T2/T3、场景解析、生命周期与探针文件已标注为“新增”。

```text
go2_test_framework/
├── README.md                              # 当前 T1/T2/T3 使用说明
├── Readme_backup.md                       # 旧版 T1 文档备份，不作为当前行为依据
├── package.xml                            # ROS 2 包依赖
├── setup.py                               # Python 包、资源及 console scripts 安装定义
├── setup.cfg                              # ament_python 脚本安装位置
├── resource/go2_test_framework            # ament 资源索引标记
├── config/
│   ├── suites/
│   │   ├── T1_smoke_city.yaml             # T1 city/rectangle 单 Case 联调
│   │   ├── T1_target_test.yaml            # T1 正式 99 Cases
│   │   ├── T2_smoke_city.yaml             # 新增：T2 单 Case 联调
│   │   ├── T2_tracking_test.yaml          # 新增：T2 正式 99 Cases
│   │   ├── T3_smoke_city.yaml             # 新增：T3 路径规划单 Case 联调
│   │   ├── T3_collision_smoke_city.yaml   # 新增：T3 碰撞注入验收
│   │   └── T3_path_planning_test.yaml     # 新增：T3 正式 99 Cases
│   ├── parameters/
│   │   ├── target_routes.yaml             # 三个场景、三种路线及 World 来源
│   │   └── robot_pose_groups.yaml         # group_01～group_11 三狗绝对位姿
│   └── metrics/
│       ├── recognition.yaml               # T1 识别通过阈值
│       ├── localization.yaml              # T1 定位误差通过阈值
│       ├── tracking.yaml                  # 新增：T2 半径、连续时长及捕获超时
│       └── path_planning.yaml             # 新增：T3 端点误差及路径超时
├── worlds/
│   ├── city_{straight,rectangle,v_shape}.world
│   ├── forest_{straight,rectangle,v_shape}.world
│   └── airport_{straight,rectangle,v_shape}.world
├── launch/
│   └── target_test.launch.py              # 单独启动 T1 Recorder 的 launch
├── go2_test_framework/
│   ├── common/
│   │   ├── config.py                      # Suite/YAML 读取、字段及 Case 配置校验
│   │   ├── execution.py                   # execution 配置模型、校验和 CLI 覆盖
│   │   └── scene_resolution.py            # 新增：生成并校验 Case 专属场景配置
│   ├── evaluators/
│   │   ├── recognition.py                 # T1 识别准确率
│   │   ├── localization.py                # T1 平均二维相对定位误差
│   │   ├── tracking.py                    # 新增：T2 连续跟踪状态与判定
│   │   └── path_planning.py               # 新增：T3 最新 generation 到达判定
│   ├── ground_truth/
│   │   └── visibility.py                  # 相机投影与姿态坐标计算
│   ├── recorders/
│   │   ├── cache.py                       # 最近时间样本缓存及一次性消费
│   │   ├── target_recorder.py             # T1 可见性、识别与定位采样
│   │   ├── tracking_recorder.py           # 新增：T2 跟踪采样与结果输出
│   │   ├── path_planning_recorder.py      # 新增：T3 目标点、真值与碰撞采样
│   │   ├── nav_goal_status.py             # 新增：NavGoal strict JSON 共享解析器
│   │   ├── nav_chain_probe.py             # 新增：T3 角色、dispatch 与 mux 链路探针
│   │   ├── collisions.py                  # 新增：碰撞消息解析、过滤与 active-set 去重
│   │   └── collision_probe_spawner.py     # 新增：碰撞 smoke 使用的静态方块注入器
│   ├── reporting/
│   │   └── results.py                     # CSV 离线评价与 Case/Batch YAML 汇总
│   ├── runner/
│   │   ├── cases.py                       # 三类 Case 展开、编号和 Case ID 生成
│   │   ├── main.py                        # CLI、批次循环和最终退出码
│   │   ├── orchestration.py               # T1/T2/T3 启动编排、重试与结果聚合
│   │   ├── processes.py                   # 进程启动、日志和所属进程管理
│   │   ├── ros_wait.py                    # ROS topic/service/action 就绪等待
│   │   ├── runtime.py                     # 批次锁、运行标记与退出信号
│   │   ├── health.py                      # 新增：有墙钟上限的 ROS worker/启动探针
│   │   └── lifecycle.py                   # 新增：进程身份、分阶段退出与 DDS 诊断
│   └── world_generator.py                 # 由路线 YAML 生成/检查 9 个 World
└── test/
    ├── test_cache_visibility.py            # 缓存匹配与相机可见性
    ├── test_cases_and_config.py            # 三类配置、Case 展开与校验
    ├── test_execution_and_orchestration.py # execution、任务启动与重试编排
    ├── test_metrics.py                     # T1 指标与结果汇总
    ├── test_processes.py                   # 子进程启动、日志及回收
    ├── test_runtime.py                     # 批次锁、信号及残留恢复
    ├── test_walking_target_start.py        # walking target 启动与运动确认
    ├── test_world_generator.py             # World 生成及轨迹一致性
    ├── test_tracking.py                    # 新增：T2 连续跟踪判定
    ├── test_path_planning.py               # 新增：T3 generation、到达及超时判定
    ├── test_collisions.py                  # 新增：碰撞过滤与去重
    ├── test_nav_chain_probe.py             # 新增：T3 导航链路及消息 schema
    ├── test_scene_resolution.py            # 新增：路线展开与场景覆写边界
    └── test_lifecycle.py                   # 新增：退出顺序、进程身份及健康探针
```

T3 还依赖同一工作区中的以下跨包实现；它们不属于 `go2_test_framework`，
因此单独列出：

```text
go2_ws_v2/src/
├── go2_dynamic_encircle/
│   ├── go2_dynamic_encircle/nav_goal_manager.py  # 发布 NavGoal generation/status
│   └── test/test_nav_goal_status.py               # 新增：状态事件及旧回调测试
├── go2_mapping_nav/launch/
│   └── three_go2_map_merge.launch.py              # 三狗地图融合与 Nav2 启动入口
└── unitree-go2-ros2/champ/champ_gazebo/src/
    └── body_contact_monitor.cpp                    # 新增：Gazebo trunk 接触集合桥接
```

安装后提供以下命令：

| 命令                        | 用途                                       |
| --------------------------- | ------------------------------------------ |
| `target_test_runner`      | 展开、选择并执行 T1/T2/T3 Case             |
| `target_test_recorder`    | 记录 T1 感知 Case                          |
| `tracking_test_recorder`  | 新增：记录并评价 T2 Case                   |
| `path_planning_recorder`  | 新增：记录并评价 T3 Case                   |
| `nav_chain_probe`         | 新增：验收 T3 导航消息链路                 |
| `collision_probe_spawner` | 新增：为 T3 collision smoke 注入静态障碍物 |
| `generate_test_worlds`    | 生成 World 或执行`--check` 漂移检查      |

## 构建与环境

ROS 项目不得使用 conda：

```bash
cd /home/bit/go2_target_seek_delivery
conda deactivate
which python3                 # 必须是 /usr/bin/python3
source /opt/ros/humble/setup.bash
colcon --log-base go2_ws_v2/log build \
  --base-paths go2_ws_v2/src \
  --build-base go2_ws_v2/build \
  --install-base go2_ws_v2/install \
  --packages-up-to champ_gazebo go2_test_framework \
  --symlink-install
source go2_ws_v2/install/setup.bash
```

回归测试也必须禁用外部 pytest 插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest \
  go2_ws_v2/src/go2_scenario_config/test \
  go2_ws_v2/src/go2_dynamic_encircle/test \
  go2_ws_v2/src/go2_mapping_nav/test \
  go2_ws_v2/src/go2_target_perception/test \
  go2_ws_v2/src/go2_test_framework/test \
  go2_ws_v2/src/walking_target_controller/test -q
```

## Suite 与指标

Suite 的 `task_type` 是必填项，只允许 `perception`、`tracking`、
`path_planning`。正式/联调配置如下：

```text
config/suites/T1_target_test.yaml
config/suites/T1_smoke_city.yaml
config/suites/T2_tracking_test.yaml
config/suites/T2_smoke_city.yaml
config/suites/T3_path_planning_test.yaml
config/suites/T3_smoke_city.yaml
config/suites/T3_collision_smoke_city.yaml   # 只用于注入碰撞验收
```

指标分别位于 `config/metrics/{recognition,localization,tracking, path_planning}.yaml`。可用 CLI 参数覆盖：

- `--recognition-metrics`、`--localization-metrics`
- `--tracking-metrics`
- `--path-planning-metrics`

其他主要参数是 `--suite`、可重复的 `--case-id`、`--all`、
`--model-path`、`--results-root`、`--dry-run`，以及 `--[no-]gui`、
`--[no-]rqt`、`--[no-]rviz`、`--[no-]lidar`、`--[no-]check-attitude`、
`--max-restarts`。不带 `--case-id/--all` 时只执行 suite 的第一个 Case。

## RViz 可视化

在 Suite 的 `execution` 下设置 `rviz: true/false`，与 `gazebo_gui`、
`rqt` 并列。T1/T2 正式套件默认关闭，T3 正式套件默认开启；smoke
套件和未填写该字段的旧配置默认关闭。`--rviz` / `--no-rviz` 优先覆盖 YAML，
最终值写入执行配置，`--dry-run` 可检查且不会打开窗口。

Runner 复用动态跟踪脚本使用的 `three_go2_mapping_nav.rviz`，显示融合地图、
三狗模型和全局规划路径；雷达、原始地图和代价地图可在窗口内按需开启。
T3 在融合地图和三套导航 action 就绪后、感知启动前打开窗口，使用仿真时间。
T1/T2 显式开启时也在感知启动前打开，但不会因此启动雷达或建图导航，
未发布的地图和路径不会显示。运行需要可用的图形会话。

每个 Attempt 最多一个 RViz，日志位于 `attempts/attempt_<NN>/logs/rviz.log`。
Attempt 结束或重试会清理窗口；手动关闭窗口不影响测试评分。

## 路线解析

每个 Case 根目录同时写入：

- `case_config.yaml`：完整 Case、task type、指标和执行参数；
- `resolved_scene_config.yaml`：从场景 YAML 深拷贝，仅覆盖动态目标的
  `speed`、`turn_duration` 和 `route`。

`closed` 路线保持原点列；三点及以上的 `ping_pong` 生成
`[A,…,末点,…,B]`；两点 `ping_pong` 生成 `[A, midpoint, B]`。生成结果
会通过 `go2_scenario_config` loader 校验，world、robots、services 等其余
场景字段不变。dynamic encircle 和 walking target 使用同一 resolved 配置。

## T1：感知

T1 沿用既有启动链与输出。完整 GT、CameraInfo 和 TF 就绪且目标首次可见后
开始固定窗口，按 2 Hz 记录：

- recognition：识别准确率默认至少 80%；
- localization：平均二维相对定位误差默认不高于 15%。

原始文件仍为 `raw/target_samples.csv`，指标为
`metrics/recognition_summary.yaml` 与 `metrics/localization_summary.yaml`。

```bash
# city smoke
ros2 run go2_test_framework target_test_runner \
  --suite go2_ws_v2/src/go2_test_framework/config/suites/T1_smoke_city.yaml \
  --model-path /home/bit/go2_target_seek_delivery/yolov8s.pt

# 正式 suite 的一个 Case
ros2 run go2_test_framework target_test_runner \
  --suite go2_ws_v2/src/go2_test_framework/config/suites/T1_target_test.yaml \
  --case-id T1-CITY-RECTANGLE-G01 \
  --model-path /home/bit/go2_target_seek_delivery/yolov8s.pt
```

## T2：连续跟踪

T2 不启动 RTAB-Map、merged map、Nav2 或 MADDPG。GT、CameraInfo、TF 完整后
的首个正式评价时刻是 acquisition 起点。2 Hz 评价规则：

- `distance <= 4 m && visible` 首次成立后开始连续计时；
- 连续 5 秒即成功；
- 建立后任一正式时刻不可见为 `visibility_lost`，超距为
  `tracking_radius_exceeded`；
- 50 秒未建立为 `acquisition_timeout`；
- 缺输入、TF 无效或进程异常属于 infrastructure failure。

原始文件是 `raw/tracking_samples.csv`。

```bash
# city smoke
ros2 run go2_test_framework target_test_runner \
  --suite go2_ws_v2/src/go2_test_framework/config/suites/T2_smoke_city.yaml \
  --model-path /home/bit/go2_target_seek_delivery/yolov8s.pt

# 单 Case / 小批量 / 全部
ros2 run go2_test_framework target_test_runner \
  --suite go2_ws_v2/src/go2_test_framework/config/suites/T2_tracking_test.yaml \
  --case-id T2-CITY-RECTANGLE-G01 \
  --case-id T2-FOREST-RECTANGLE-G01 \
  --model-path /home/bit/go2_target_seek_delivery/yolov8s.pt
```

把上述选择项换成 `--all` 可运行全部 99 Cases；正式运行前建议先加
`--dry-run --results-root /tmp/t2_dry_run`。

## T3：路径规划与静态碰撞

T3 强制三狗 lidar，启动顺序为 World、三狗、姿态门禁、actor、map merge、
mux、三套 mapping/Nav2、perception、角色锁定、Recorder、dynamic encircle、
walking target。Runner 逐项等待第一条有效 `/merged_map`、三套
`NavigateToPose` action server、角色和 NavGoal 状态。不会启动 MADDPG；
`arrival_hold_duration` 大于路径总超时，使测试窗口内 mux owner 保持 Nav2。

`/dynamic_encircle/nav_goal_status` 是 reliable、transient-local 的 strict JSON，
包含 `schema_version`、`stamp`、`event`、`generation`、`frame_id`、
`navigation_dogs`、`goals`、`robot`、`action_status`。两次
`send_goal_async()` 后才发布共享 generation 的 `DISPATCHED`；旧回调保留旧
generation。

Recorder 只用 `DISPATCHED` 更新 latest generation，新 generation 立即淘汰
旧目标。600 秒总超时从首个有效 dispatch 开始且不会重置。两只当前导航狗
必须在同一 2 Hz 样本、同一 latest generation 下二维端点误差都不超过 1 m；
忽略 yaw，不要求 hold。`merged_map → world` 使用显式 identity TF。
当前 generation 的 `ABORTED/REJECTED` 是有效失败，旧 generation 回调忽略。

Gazebo contact bridge 发布当前 trunk 接触集合。Recorder 忽略 ground、同一
机器人、其他 Go2 和 walking target，对外部静态模型按规范化 pair 做 active-set
去重：首次接触一条、持续接触不重复、分离后再接触产生新事件。T3 成功必须
同时满足 `all_reached && collision_count == 0`。

原始文件：

- `raw/nav_goal_events.csv`
- `raw/path_samples.csv`
- `raw/collision_events.csv`

```bash
# city smoke
ros2 run go2_test_framework target_test_runner \
  --suite go2_ws_v2/src/go2_test_framework/config/suites/T3_smoke_city.yaml \
  --model-path /home/bit/go2_target_seek_delivery/yolov8s.pt

# 三场景 rectangle 各一例
ros2 run go2_test_framework target_test_runner \
  --suite go2_ws_v2/src/go2_test_framework/config/suites/T3_path_planning_test.yaml \
  --case-id T3-CITY-RECTANGLE-G01 \
  --case-id T3-FOREST-RECTANGLE-G01 \
  --case-id T3-AIRPORT-RECTANGLE-G01 \
  --model-path /home/bit/go2_target_seek_delivery/yolov8s.pt
```

正式 99 Case 解析检查使用 `--all --dry-run`；只有确实要执行完整仿真时才
去掉 `--dry-run`。

## 结果与 Batch 判定

默认结构：

```text
TestResults/batch_<timestamp>/
├── resolved_cases.yaml
└── T{1,2,3}_..._test/
    ├── batch_summary.yaml
    └── case_<NNN>/
        ├── case_config.yaml
        ├── resolved_scene_config.yaml
        ├── case_summary.yaml
        └── attempts/attempt_<NN>/
            ├── case_summary.yaml
            ├── raw/
            └── logs/
```

`batch_summary.yaml` 记录 scheduled、eligible、success、valid failure、
infrastructure failure、success rate、阈值和 `batch_pass`。只有
infrastructure-valid Case 进入成功率分母；只要存在 infrastructure failure，
`completion_status` 为 `incomplete`、`incomplete_reason` 为
`infrastructure_failed`，且不得正式通过。T2 阈值 70%，T3 阈值 90%。

姿态门禁确认摔倒或出现 infrastructure failure 时只重启当前 Case，两类失败
共用 `max_restarts` 预算并计入 `restarts_used`；即使关闭姿态门禁，基础设施
失败仍会重试。每次 Attempt 无论结果如何都会保存日志，并只终止自身进程组；
未知启动/数据/TF 错误不会混入算法有效失败。`--max-restarts 0` 表示不重试。

## World 漂移检查

```bash
ros2 run go2_test_framework generate_test_worlds \
  --config go2_ws_v2/src/go2_test_framework/config/parameters/target_routes.yaml \
  --repo-root /home/bit/go2_target_seek_delivery \
  --output-dir go2_ws_v2/src/go2_test_framework/worlds \
  --check
```

成功输出 `checked 9 test worlds`。`--check` 只比较、不写文件。

## 循环测试退出与启动门禁

每个 Attempt（正常结束、失败重试或用户中断）先退出业务、可视化和机器人
进程，最后退出 World/Gazebo。每阶段先发送 SIGINT 并等待最多 15 秒，仍未退出
才发送 SIGTERM（最多 5 秒）和 SIGKILL（再确认最多 3 秒）。已全部退出时立即
结束等待；重试间隔不会代替退出确认。

Runner 检查所属进程组、运行标记及 PID 启动时间，跟踪父进程退出后仍存活或
另开会话的子进程；只在确认没有所属存活进程后报告 `cleanup complete`。
清理期间再次收到退出信号会记录并延后处理。Batch 启动前的残留清理使用相同
流程；没有归属证据的历史进程只诊断，不终止。

首次启动和每个 Attempt 启动前都执行独立 ROS 探针，节点创建、销毁及退出
总共受 10 秒墙钟超时约束。控制器、角色选择和目标启动等 ROS 等待也在独立
辅助进程中执行，避免节点初始化卡住导致 runner 无限等待。机器人启动日志
会报告生成节点初始化、生成请求、模型生成及控制器激活阶段。

诊断文件：

- Batch 目录的 `cleanup_summary.yaml`、`startup_health.yaml`：启动前检查。
- Attempt 目录的同名文件：该轮退出确认及启动探针结果。
- `health_logs/`：探针输出与超时诊断；`logs/`：ROS 等待辅助进程日志及诊断。

清理记录包含发送信号、耗时、残留进程、僵尸进程和重复退出信号；探针失败
记录已到达阶段、退出错误以及可观察到的共享内存关联。不会保存完整环境变量，
也不会删除或隔离共享内存文件。仅检查进程引用的 `fastrtps_port<数字>` 数据
文件；零字节的 `_el`、`_sl` 锁文件不会按损坏数据处理。

`cleanup_failed` 或 `startup_health_failed` 会阻断整个 Batch，停止重试和后续
Case；摘要标记 `incomplete`、`batch_pass: false`，保留已有指标和原始数据。
`scheduled_case_ids`、`started_case_ids`、`not_run_case_ids` 及对应计数区分计划、
已启动与未运行 Case；未运行 Case 不作为算法失败。处理诊断原因后可用
`--case-id` 选择需要补跑的 Case。
