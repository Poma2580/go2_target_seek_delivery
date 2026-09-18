# 三个 MADDPG 启动脚本简要对比

本文对比 `Scripts` 目录下三个与三 Go2、MADDPG 和 Nav2 有关的启动脚本。

## 1. start_maddpg_waypoint_nav2.sh

- 功能：只启动 MADDPG 航点选择相关节点，不启动完整仿真。
- 前提：Gazebo、三只 Go2、`merged_map` 和三套 Nav2 已经运行。
- 领航犬：固定为 `go2_1`，负责跟踪动态行人。
- 跟随犬：固定为 `go2_2` 和 `go2_3`。
- 目标来源：Gazebo 行人真值，由 `actor_state_publisher` 提供。
- 决策周期：1 秒。
- Nav2 航点刷新周期：3 秒。
- 初始队形容差：0.5 米，达到队形后再进入正常决策。
- 默认模式：`--dry-run`，只计算动作，不向 Nav2 发送真实目标。
- 执行模式：添加 `--execute`，实际向 Nav2 发送航点。
- 执行时将 go2_2/go2_3 的 Nav2 线速度上限设为 0.30 m/s。
- 使用模型：`waypoint_maddpg_v0/runs/two_obstacles_108rays_final_gpu_20260826/best_model.pt`。
- 适用场景：已有完整仿真，只想单独检查或执行 MADDPG 航点策略。

## 2. start_three_go2_dynamic_tracking.sh

- 功能：从零启动完整的三 Go2 动态行人追踪实验。
- 启动内容：Gazebo、三只 Go2、雷达、RGB-D、地图融合、RTAB-Map、Nav2 和 RViz。
- 感知方式：使用 RGB-D、YOLO 和目标角色选举，不依赖固定领航犬。
- 领航犬：从 go2_1/go2_2/go2_3 中动态选举当前感知犬。
- 跟随犬：除当前感知犬以外的另外两只 Go2。
- 控制流程：先由 Nav2 接近行人，再启用 MADDPG 选择跟随航点。
- MADDPG 只发布 NavigateToPose 目标，不直接发布 `cmd_vel`。
- 决策周期：1 秒。
- Nav2 航点刷新周期：8 秒。
- 支持场景：`city`、`forest`、`airport`，默认 `city`。
- Gazebo GUI：固定开启。
- 使用模型：`waypoint_maddpg_v0/runs/three_seed_obstacle_curriculum_20260909_091240/seed_27/stage2_two_obstacles/best_model.pt`。
- 适用场景：正式运行完整的三狗动态目标感知、接近和航点跟随实验。

## 3. start_three_go2_dynamic_waypoint_maddpg.sh

- 功能：包装第二个脚本，复用完整仿真前半段并替换 MADDPG 控制尾部。
- 基础系统：同样启动 Gazebo、三只 Go2、传感器、地图融合和 Nav2。
- 感知方式：正常模式下仍使用 YOLO 感知和动态角色选举。
- 控制流程：Nav2 接近行人，交接完成后启用 MADDPG 航点选择。
- 附加功能：启动 `maddpg_waypoint_monitor.py` 监视雷达、候选航点和动作。
- 支持参数：`--gui`、`--headless` 和 `--check`。
- 支持通过环境变量更换选择器程序、追加参数或强制指定感知犬。
- 决策周期：1 秒；Nav2 航点刷新周期：8 秒。
- 使用模型：`waypoint_maddpg_v0/runs/three_seed_obstacle_curriculum_20260909_091240/seed_27/stage2_two_obstacles/best_model.pt`。
- 适用场景：需要完整实验，同时需要额外决策监控或自定义 MADDPG 参数。
- 当前注意：文本截断和 GUI/Mux 替换规则与基础脚本不一致，可能重复启动节点。
- 当前建议：修复上述规则前，完整实验优先使用 `start_three_go2_dynamic_tracking.sh`。

## 快速选择

- 只调试 MADDPG，底层仿真已经启动：使用 `start_maddpg_waypoint_nav2.sh`。
- 从零运行完整动态行人追踪：使用 `start_three_go2_dynamic_tracking.sh`。
- 需要可配置 MADDPG 和动作监控：修复后使用 `start_three_go2_dynamic_waypoint_maddpg.sh`。
