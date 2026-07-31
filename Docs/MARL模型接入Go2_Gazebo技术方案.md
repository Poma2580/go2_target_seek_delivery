# MARL模型接入Go2 Gazebo动态围堵技术方案

## 1. 方案目标

本方案用于将已有四智能体Leader-Follower MADDPG编队模型接入当前三台Unitree Go2的Gazebo城市动态目标围堵系统。由于项目当前只使用三只真实机器狗，而已有模型包含四个策略智能体，因此采用“三只真实Go2 + 一个虚拟智能体”的方式完成模型适配。

核心思想为：规则控制器先负责三只Go2在城市环境中的安全接近、绕行建筑和动态目标追踪；当三只真实机器狗均接近目标后，切换到MARL队形围堵阶段。MARL推理时仍构造四个智能体观测，其中缺少的智能体由虚拟状态补全，并始终保持在其对应的理想编队槽位上。模型输出四组动作，但系统只向三只真实Go2发布控制指令，虚拟智能体动作不执行。

整体流程为：

```text
目标感知
→ 规则控制安全接近
→ 三只真实Go2接近目标
→ 触发MARL切换条件
→ 构造三真实+一虚拟的四智能体观测
→ MADDPG策略推理
→ 三只真实Go2执行队形围堵
→ 目标丢失/任务结束/人工停止
```

## 2. 当前系统基础

当前动态目标围堵脚本为：

```text
Scripts/start_three_go2_dynamic_tracking.sh
```

该脚本启动以下模块：

| 模块 | 作用 |
| --- | --- |
| `go2_world` | 启动Gazebo城市目标场景 |
| `spawn_go2_1` | 生成带RGB-D相机的感知狗 |
| `spawn_go2_2` | 生成第一只运动控制狗 |
| `spawn_go2_3` | 生成第二只运动控制狗 |
| `actor_state` | 发布行人真值状态 |
| `target_perception` | 基于go2_1相机估计目标状态 |
| `dynamic_encircle` | 当前规则式动态追踪与拦截围捕 |
| `rqt_debug_image` | 显示目标检测调试图 |
| `perception_eval` | 评估目标估计误差 |

当前主要ROS2接口为：

| Topic | 类型 | 作用 |
| --- | --- | --- |
| `/go2_1/odom` | `nav_msgs/msg/Odometry` | go2_1状态 |
| `/go2_2/odom` | `nav_msgs/msg/Odometry` | go2_2状态 |
| `/go2_3/odom` | `nav_msgs/msg/Odometry` | go2_3状态 |
| `/go2_1/cmd_vel` | `geometry_msgs/msg/Twist` | go2_1速度控制 |
| `/go2_2/cmd_vel` | `geometry_msgs/msg/Twist` | go2_2速度控制 |
| `/go2_3/cmd_vel` | `geometry_msgs/msg/Twist` | go2_3速度控制 |
| `/go2_1/target_estimated/odom` | `nav_msgs/msg/Odometry` | 视觉估计目标状态 |
| `/walking_target/odom` | `nav_msgs/msg/Odometry` | Gazebo行人真值状态 |

当前`dynamic_encircle`节点中的规则状态机为：

```text
go2_1：catch_up → formation
go2_2/go2_3：approach → to_stage → staged → charge → done
```

新版方案中，原来“进入5m后完成”的含义调整为“进入MARL队形围堵阶段的触发条件”。

## 3. 已有MADDPG模型信息

当前使用的模型为四智能体编队模型：

```text
三角形MADDPG/runs/stage4_b512_usteps20_g0.99_t0.005_alr5e-05_clr0.0005_n0.14_minn0.02_h128,128_20260430_132728/best_model.pt
```

模型结构核对结果：

| 项目 | 内容 |
| --- | --- |
| 智能体数量 | 4 |
| 角色结构 | 1个Leader + 3个Follower |
| Actor数量 | 4 |
| Critic数量 | 4 |
| 每个Actor输入 | 23维观测 |
| 每个Actor输出 | 2维动作 |
| 网络结构 | `23 → 128 → 128 → 2` |
| 训练动作含义 | 二维平面速度/位移方向 |
| 训练阵型 | Leader在前，3个Follower在后方横向排列 |

训练环境中的角色关系为：

```text
agent_0：Leader
agent_1：左后Follower
agent_2：中后Follower
agent_3：右后Follower
```

Follower相对Leader的理想槽位为：

```text
agent_1: [-0.60, -0.65]
agent_2: [ 0.00, -0.65]
agent_3: [ 0.60, -0.65]
```

## 4. 三真实Go2加虚拟智能体方案

由于项目中只使用三只真实机器狗，因此将四智能体模型映射为：

```text
agent_0 -> go2_1，真实Leader/感知狗
agent_1 -> go2_2，真实左后Follower
agent_2 -> virtual_agent，虚拟中后Follower
agent_3 -> go2_3，真实右后Follower
```

虚拟智能体不在Gazebo中生成，不订阅真实里程计，也不发布`cmd_vel`。它只在MARL观测构造阶段存在，用于补全四智能体模型所需的状态输入。

虚拟智能体位置由Leader状态实时计算：

```text
p_virtual = p_leader + [0.00, -0.65]
```

虚拟智能体速度建议取Leader速度：

```text
v_virtual = v_leader
```

这样模型看到的是完整四智能体编队，且虚拟Follower始终保持在中后理想槽位。模型输出动作：

```text
a0, a1, a2, a3
```

实际执行时只发布：

```text
a0 -> /go2_1/cmd_vel
a1 -> /go2_2/cmd_vel
a3 -> /go2_3/cmd_vel
```

虚拟智能体动作`a2`直接丢弃。

## 5. MARL切换时机

### 5.1 切换条件

MARL队形围堵不应在仿真一开始就启动，而应在三只真实机器狗均接近动态目标后启动。建议采用以下条件：

```text
目标状态有效
&& go2_1/go2_2/go2_3里程计有效
&& go2_1到目标距离 < marl_switch_radius
&& go2_2到目标距离 < marl_switch_radius
&& go2_3到目标距离 < marl_switch_radius
&& MADDPG模型已成功加载
&& 策略推理输出有效
```

初始建议参数：

```text
marl_switch_radius = 5.0 m
```

代码表达为：

```python
def _all_real_dogs_near_target(self):
    if not self.target_ok:
        return False

    for name in ("go2_1", "go2_2", "go2_3"):
        dog = self.dogs[name]
        if not dog.received:
            return False
        if self._age(dog.last_stamp) > self.odom_timeout:
            return False

        dist = math.hypot(self.target_x - dog.x, self.target_y - dog.y)
        if dist > self.marl_switch_radius:
            return False

    return True
```

### 5.2 状态机表达

建议在控制节点中增加控制模式：

```text
rule_approach：规则控制接近目标
marl_encircle：MARL队形围堵
safe_stop：异常停车
```

控制循环伪代码为：

```python
def control_loop(self):

    self._resolve_target()

    if self.mode == "rule_approach":
        if self._all_real_dogs_near_target():
            self.mode = "marl_encircle"
            self.get_logger().info("三只Go2均接近目标，切换到MARL队形围堵")
        else:
            self._run_rule_controller()
            return

    if self.mode == "marl_encircle":
        observations = self._build_marl_observations()
        actions = self.maddpg.act(observations, add_noise=False)
        self._publish_real_go2_actions(actions)
```

### 5.3 防止频繁切换

切换应采用一次性触发逻辑。进入`marl_encircle`后，不应因为某一只狗短暂超过5m又切回规则控制，否则会导致控制源频繁变化和速度抖动。

推荐逻辑：

```text
rule_approach -> marl_encircle：满足接近条件后单向切换
marl_encircle -> safe_stop：目标丢失、里程计超时、模型异常或人工停止
```

## 6. MARL观测构造

训练环境中的23维观测结构为：

```text
own_pos(2)
own_vel(2)
own_slot_rel(2)
leader_rel(2)
target_rel(2)
other_agents_rel(6)
role_flag(1)
obstacles_rel(6)
```

四个智能体均按相同顺序构造观测。ROS2侧应保持以下顺序：

```text
agent_0 = go2_1
agent_1 = go2_2
agent_2 = virtual_agent
agent_3 = go2_3
```

各部分含义为：

| 字段 | 维度 | 构造方式 |
| --- | --- | --- |
| `own_pos` | 2 | 当前智能体世界坐标位置 |
| `own_vel` | 2 | 当前智能体世界坐标速度 |
| `own_slot_rel` | 2 | 自身期望槽位相对自身的位置 |
| `leader_rel` | 2 | Leader相对自身的位置 |
| `target_rel` | 2 | 目标相对自身的位置 |
| `other_agents_rel` | 6 | 其他三个智能体相对自身的位置 |
| `role_flag` | 1 | Leader为1，Follower为0 |
| `obstacles_rel` | 6 | 三个障碍物相对自身的位置 |

第一阶段接入时，如不从Gazebo解析障碍物，可暂时将`obstacles_rel`置零，但这会造成训练环境和部署环境存在差异。后续若要提升城市环境稳定性，应将建筑或关键障碍物抽象为训练环境中的三类障碍输入。

目标位置在原训练环境中是固定点，当前项目中应替换为实时行人位置：

```text
target_pos = /go2_1/target_estimated/odom
```

调试阶段可先使用：

```text
target_pos = /walking_target/odom
```

## 7. 动作转换与发布

模型每个Actor输出二维动作：

```text
[u_x, u_y]
```

Go2底层控制接口为：

```text
Twist.linear.x
Twist.angular.z
```

因此需要将二维平面动作转换为机器人前向速度和角速度：

```text
desired_yaw = atan2(u_y, u_x)
desired_speed = k_v * sqrt(u_x^2 + u_y^2)
yaw_error = normalize_angle(desired_yaw - robot_yaw)
linear.x = clamp(desired_speed * heading_gate, 0, v_max)
angular.z = clamp(k_omega * yaw_error, -omega_max, omega_max)
```

其中：

```text
heading_gate = max(cos(yaw_error), 0.0)
```

初始测试建议参数：

```text
v_max = 0.10 m/s
omega_max = 0.30 rad/s
control_rate = 10 Hz
```

确认方向、坐标和队形趋势正确后，再逐步提高速度。当前规则控制中常用上限为：

```text
max_linear = 0.65 m/s
max_angular = 0.9 rad/s
```

MARL初期不建议直接使用该上限。

## 8. 控制权管理

任一时刻，每只Go2的`cmd_vel`只能有一个有效发布源。切换到MARL后，规则控制器必须停止向以下话题发布非零速度：

```text
/go2_1/cmd_vel
/go2_2/cmd_vel
/go2_3/cmd_vel
```

推荐实现方式是在同一个控制节点内集成规则接近和MARL围堵两种模式：

```text
DynamicEncircleWithMARL
```

该方式可以复用当前`dynamic_encircle`已有的目标保活、里程计超时、速度限幅和停车逻辑，同时避免两个节点同时控制同一只机器狗。

如果采用独立`marl_formation_controller`节点，则必须增加控制仲裁机制，例如：

```text
rule_controller发布候选速度
marl_controller发布候选速度
cmd_vel_mux根据当前mode选择一路输出
```

在未实现仲裁器前，不建议让`dynamic_encircle`和`marl_formation_controller`同时直接发布`/go2_i/cmd_vel`。

## 9. ROS2节点设计

建议新增或改造节点：

```text
marl_virtual_agent_encircle
```

节点职责：

1. 加载四个Actor权重；
2. 订阅三只真实Go2里程计；
3. 订阅动态目标状态；
4. 判断MARL切换条件；
5. 生成虚拟智能体状态；
6. 构造四个23维观测；
7. 执行MADDPG推理；
8. 丢弃虚拟智能体动作；
9. 将三只真实Go2动作转换为`cmd_vel`；
10. 执行速度、角速度和加速度限幅；
11. 目标丢失、里程计超时或模型异常时发布零速度；
12. 记录切换时刻、队形误差和推理耗时。

输入接口：

| Topic | 类型 | 作用 |
| --- | --- | --- |
| `/go2_1/odom` | `nav_msgs/msg/Odometry` | Leader状态 |
| `/go2_2/odom` | `nav_msgs/msg/Odometry` | 真实Follower左侧状态 |
| `/go2_3/odom` | `nav_msgs/msg/Odometry` | 真实Follower右侧状态 |
| `/go2_1/target_estimated/odom` | `nav_msgs/msg/Odometry` | 在线目标估计 |
| `/walking_target/odom` | `nav_msgs/msg/Odometry` | 调试阶段目标真值 |

输出接口：

| Topic | 类型 | 作用 |
| --- | --- | --- |
| `/go2_1/cmd_vel` | `geometry_msgs/msg/Twist` | Leader速度控制 |
| `/go2_2/cmd_vel` | `geometry_msgs/msg/Twist` | 左侧Follower速度控制 |
| `/go2_3/cmd_vel` | `geometry_msgs/msg/Twist` | 右侧Follower速度控制 |
| `/marl/mode` | `std_msgs/msg/String` | 当前控制模式，可选 |
| `/marl/formation_error` | `std_msgs/msg/Float32` | 队形误差，可选 |
| `/marl/virtual_agent/odom` | `nav_msgs/msg/Odometry` | 虚拟智能体状态，可选 |

## 10. 安全与异常处理

MARL队形围堵阶段必须保留安全保护：

| 异常 | 处理 |
| --- | --- |
| 目标丢失超过`target_hold` | 三只Go2发布零速度 |
| 任一Go2里程计超时 | 三只Go2发布零速度 |
| 模型输出NaN或Inf | 丢弃动作并停车 |
| 动作持续越界 | 限幅并记录告警 |
| 机器人距离过近 | 降低速度或停车 |
| 人工Ctrl+C | 发布零速度后退出 |

目标短时丢失时可沿用当前`dynamic_encircle`的coast外推逻辑，即在`target_timeout < age <= target_hold`内使用最后一次目标速度外推位置。

## 11. 环境与模型验证

当前GPU环境已经配置为：

```text
conda环境：maddpg_gpu
PyTorch：2.5.1+cu121
CUDA：12.1
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
```

模型接入前应先完成离线回放：

```bash
conda activate maddpg_gpu
cd /home/wangantong/KD_all/go2_target_seek_delivery/三角形MADDPG

python run.py \
  --model-path "runs/stage4_b512_usteps20_g0.99_t0.005_alr5e-05_clr0.0005_n0.14_minn0.02_h128,128_20260430_132728/best_model.pt" \
  --env-name formation_navigation_v0 \
  --episodes 1 \
  --max-steps 120 \
  --training-stage 4 \
  --seed 42
```

若离线回放正常，再进入ROS2只读验证，即只订阅Gazebo数据、构造观测并打印模型输出，不发布`cmd_vel`。

## 12. 分阶段实施步骤

建议按以下顺序推进：

1. 离线验证`best_model.pt`能正常加载和回放；
2. 编写只读推理脚本，输入固定观测，检查四组动作输出；
3. 在ROS2中订阅三狗里程计和目标状态，不发布控制；
4. 生成虚拟`agent_2`状态并发布到调试话题；
5. 构造四个23维观测并打印数值范围；
6. 执行MADDPG推理但不下发动作；
7. 实现三狗都小于`5m`的MARL切换条件；
8. 低速发布三只真实Go2的`cmd_vel`；
9. 使用`/walking_target/odom`真值目标进行低速测试；
10. 切换到`/go2_1/target_estimated/odom`视觉目标；
11. 调整速度上限、切换半径和队形误差指标；
12. 整合进动态启动脚本。

## 13. 验收标准

满足以下条件后，可认为该方案完成初步接入：

1. 四Actor模型能够在`maddpg_gpu`环境稳定加载；
2. ROS2节点能够构造四个23维观测；
3. 虚拟智能体始终位于Leader对应理想槽位；
4. 三只真实Go2均小于`marl_switch_radius`后能切换到MARL模式；
5. MARL模式下只有一个控制源发布`/go2_i/cmd_vel`；
6. 模型输出动作能被正确转换为Go2速度指令；
7. 三只真实Go2能够形成与虚拟智能体补全后的队形关系；
8. 目标丢失或里程计异常时能够停车；
9. Gazebo中不存在持续反向运动、原地高速旋转或明显振荡；
10. 使用视觉目标输入时能够完成一轮动态队形围堵演示。

## 14. 方案边界

本方案复用的是四智能体Leader-Follower编队模型，目标是让三只真实Go2在虚拟智能体补全条件下形成队形围堵。它不是重新训练得到的三智能体围堵模型，也不是四只真实机器狗的物理协同。

因此报告中建议表述为：

```text
基于虚拟智能体补全的四智能体MADDPG编队围堵控制方法
```

不建议表述为：

```text
四只真实机器狗协同围堵
```

若后续要求目标位于多机器人闭合包围圈中心，或要求三只真实机器狗完全独立完成三角围堵，则仍需基于当前任务重新训练或微调三智能体/三真实平台模型。
