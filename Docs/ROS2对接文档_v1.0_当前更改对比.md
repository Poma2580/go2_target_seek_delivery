# ROS2 对接文档 v1.0 当前更改对比

## 对比基准

- 初始文档：`/home/zhj/wat/ROS2对接文档_v1.0.md`
- 当前文档：`/home/zhj/wat/go2_target_seek_delivery-main/Docs/ROS2对接文档_v1.0.md`
- 对比方式：逐行 `diff -u`

> 注：项目内的当前文档是 Git 未跟踪文件，因此不能直接用普通 `git diff` 取得初始版。本对比使用用户最初提供且未经修改的文档作为基准。

## 当前改动汇总

| 位置 | 初始文档 | 当前文档 |
| --- | --- | --- |
| 1. 完整运行链 | `Nav2 → MADDPG 控制切换` | `Nav2 初始靠近 → MADDPG 选点`，再由 Nav2 执行候选航点 |
| 1.1 功能模块 | `go2_mapping_nav` 只写 RTAB-Map、地图融合及 Nav2 | 增加 MADDPG 选点职责 |
| 1.1 功能模块 | `Nav2 与 MADDPG 控制切换` | `Nav2 初始靠近与 MADDPG 选点阶段交接` |
| 2.4 模型资源 | `-MADDPG/.../best_model.pt`，MADDPG 控制模型 | `waypoint_maddpg_v0/.../best_model.pt`，MADDPG 离散航点选择模型 |
| 3.1 ROS 包 | `go2_mapping_nav` 负责建图、融合与 Nav2 | 增加 MADDPG 选点 |
| 4. 运行依赖 | C13/C14 合并描述旧控制器 | C13 明确需要选点模型、三狗 Odom、两只跟随狗 LaserScan、TF 和 Action；C14 单独描述交接节点依赖 |
| 4.1 C07 | Nav2/MADDPG 速度命令仲裁 | 选点版中只转发 Nav2 速度，Mux 不切换至旧 MADDPG 速度分支 |
| 4.1 C13 | `gazebo_leader_slot_controller` 直接输出速度 | `maddpg_waypoint_selector.py` 选择航点并通过 NavigateToPose 交给 Nav2 |
| 4.1 C14 | 围捕后切换两只跟随狗的速度源 | 初始靠近后启用选点器，Nav2 继续独占速度控制 |
| 4.1 C15 | `Nav2 → MADDPG` 控制阶段 | `Nav2 靠近 → MADDPG 选点` 任务阶段 |
| 5.5 C07 节点 | 在 Nav2/MADDPG 速度输入之间仲裁 | 选点版保持 Nav2 分支，旧速度分支仅作代码兼容保留 |
| 5.9 C13 节点 | `/gazebo_leader_slot_controller` 输出 `maddpg_cmd_vel` | `/maddpg_waypoint_selector` 读取 Odom/LaserScan，每 1 s 选点，每 3 s 最多向 Nav2 刷新一次 Goal |
| 5.10 C14 节点 | Nav2 → MADDPG 速度控制切换 | Nav2 初始靠近 → MADDPG 选点阶段交接，`use_maddpg=false` |

## 当前未改部分

当前第 6～10 章尚未完成新版选点接口替换，仍包含以下旧版描述：

- `/go2_i/maddpg_cmd_vel`作为 MADDPG 速度输出。
- `/gazebo_leader_slot_controller/ready` 和 `/active`。
- `/gazebo_leader_slot_controller/set_enabled`。
- MADDPG 接管 Mux 并直接控制跟随狗速度。
- Action Client 只记载为 `/nav2_dynamic_encircle`，尚未加入 `/maddpg_waypoint_selector`。
- MADDPG 仅使用 Odom 的旧输入说明，尚未改为 Odom + LaserScan + TF + 83 维观测。

因此，当前文档是修改中的中间版本，不应直接作为最终对接文档交付。

## 查看完整逐行差异

```bash
diff -u \
  --label '初始文档/ROS2对接文档_v1.0.md' \
  --label '当前文档/Docs/ROS2对接文档_v1.0.md' \
  /home/zhj/wat/ROS2对接文档_v1.0.md \
  /home/zhj/wat/go2_target_seek_delivery-main/Docs/ROS2对接文档_v1.0.md
```
