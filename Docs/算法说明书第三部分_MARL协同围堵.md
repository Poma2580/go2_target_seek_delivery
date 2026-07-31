# 3 基于多智能体强化学习的协同围堵策略

## 3.1 方法概述

针对复杂动态环境下目标持续运动、多机器人协同决策困难的问题，设计一种基于多智能体强化学习的 Leader-Follower 协同围堵方法。该方法以搭载 RGB-D 相机的 go2_1 作为感知 Leader，实时获取目标位置与运动状态并向其他机器人共享；go2_2 和 go2_3 作为 Follower，根据目标状态、自身位姿和队友信息，利用 MADDPG 算法生成连续运动控制策略，从不同方向向目标靠拢并形成围堵队形，实现三台机器狗对动态目标的持续跟踪与协同围控。

## 3.2 算法流程

**图3 基于多智能体强化学习的协同围堵算法流程**

如图3所示，协同围堵算法首先接收感知 Leader 输出的动态目标位置与运动状态，并按照项目设定确定 go2_1 为感知 Leader、go2_2 和 go2_3 为 Follower。随后根据目标状态和各机器狗角色生成动态围堵参考位置，结合三台机器狗的自身位姿、目标相对状态、队友相对位置及参考位置误差构建多智能体状态观测。各机器狗利用训练完成的 MADDPG Actor 网络生成线速度和角速度控制量并执行运动，同时更新目标状态、机器人状态和围堵队形误差。当满足围堵完成判据时结束任务；否则重新获取目标状态并进入下一轮协同决策，实现对动态目标的持续逼近与协同围控。

## 3.3 关键技术

### （1）协同围堵问题建模与机制设计

面向目标定位信息已知条件下的动态目标处置任务，建立由 1 台感知 Leader 和 2 台 Follower 组成的协同围堵模型。感知 Leader 负责获取目标位置和速度，并在目标后方保持跟踪；两个 Follower 根据 Leader 共享的目标状态和自身位姿分别向目标前方及侧前方机动。通过动态参考位置、队形误差和任务完成判据，将协同围堵任务描述为多机器人对动态目标的持续锁定、分角色逼近和联合围控过程。

设动态目标在时刻 \(t\) 的位置和速度分别为

\[
p_T(t)=[x_T(t),y_T(t)]^T,\qquad
v_T(t)=[v_{Tx}(t),v_{Ty}(t)]^T.
\]

目标运动航向定义为

\[
\psi_T(t)=\operatorname{atan2}\left(v_{Ty}(t),v_{Tx}(t)\right).
\]

感知 Leader 需要在目标后方保持距离 \(r_L\)，其期望位置为

\[
p_L^{*}(t)=p_T(t)-r_L
\begin{bmatrix}
\cos\psi_T(t)\\
\sin\psi_T(t)
\end{bmatrix}.
\]

为形成随目标运动方向变化的围堵结构，定义旋转矩阵

\[
R(\psi_T)=
\begin{bmatrix}
\cos\psi_T&-\sin\psi_T\\
\sin\psi_T&\cos\psi_T
\end{bmatrix}.
\]

第 \(i\) 个 Follower 相对 Leader 的期望偏移为 \(\Delta_i\)，则其动态期望位置为

\[
p_i^{*}(t)=p_L(t)+R(\psi_T(t))\Delta_i,\qquad i=2,3.
\]

其中，\(p_L(t)\) 为 Leader 的实时位置，\(\Delta_2\) 和 \(\Delta_3\) 分别对应前方拦截位置和侧前方封堵位置。项目中可结合目标矩形运动回路、Follower 外扩车道及前向 staging 距离对上述期望位置进行约束，避免机器人直接穿越建筑区域。

第 \(i\) 台机器狗的槽位误差定义为

\[
e_i(t)=\left\|p_i(t)-p_i^{*}(t)\right\|_2.
\]

三台机器狗的平均队形误差为

\[
E_{form}(t)=\frac{1}{3}\sum_{i=1}^{3}e_i(t).
\]

机器人之间的距离定义为

\[
d_{ij}(t)=\left\|p_i(t)-p_j(t)\right\|_2,\qquad i\ne j.
\]

当 Leader 位于目标后方有效感知区域、两个 Follower 进入对应围堵区域、平均队形误差小于阈值 \(\varepsilon_f\)，且任意两台机器狗之间满足 \(d_{ij}>d_{safe}\) 时，判定协同围堵完成。该判据避免仅以单台机器人接近目标作为任务完成条件，保证最终形成具有空间结构的团队围控状态。

### （2）基于多智能体强化学习的协同围堵策略

面向动态目标运动变化和多机器人实时协同决策问题，将三机器狗围堵任务建模为多智能体马尔可夫决策过程

\[
\langle N,S,A,P,R,\gamma\rangle,
\]

其中，\(N=3\) 表示智能体数量，\(S\) 表示联合状态空间，\(A=A_1\times A_2\times A_3\) 表示联合动作空间，\(P\) 表示状态转移过程，\(R=\{R_1,R_2,R_3\}\) 表示各智能体奖励函数，\(\gamma\) 表示折扣因子。

第 \(i\) 个智能体的局部观测由自身运动状态、目标相对状态、期望位置误差、队友相对位置、角色标志和目标有效状态构成：

\[
o_i(t)=\left[v_i,\omega_i,p_T-p_i,v_T,p_i^{*}-p_i,q_i^{rel},\rho_i,c_T\right].
\]

其中，\(v_i\) 和 \(\omega_i\) 分别为机器狗线速度和角速度，\(q_i^{rel}\) 表示其他机器狗相对位置，\(\rho_i\) 表示 Leader/Follower 角色，\(c_T\) 表示目标状态是否有效。

每个智能体输出二维连续动作

\[
a_i(t)=[u_{v,i}(t),u_{\omega,i}(t)]^T,
\qquad u_{v,i},u_{\omega,i}\in[-1,1].
\]

动作映射为 Go2 的线速度和角速度控制量：

\[
v_i^{cmd}=v_{max}u_{v,i},\qquad
\omega_i^{cmd}=\omega_{max}u_{\omega,i}.
\]

奖励函数根据不同角色的任务进行设计。第 \(i\) 个智能体的总奖励表示为

\[
R_i=R_i^{target}+R_i^{form}+R_i^{safe}+R_i^{step}+R^{success}.
\]

其中，\(R_i^{target}\) 用于奖励机器人向动态参考位置靠近；\(R_i^{form}\) 用于减小槽位误差和整体队形误差；\(R_i^{safe}\) 用于惩罚机器人间距离过近、速度突变和碰撞风险；\(R_i^{step}\) 为每步时间惩罚；\(R^{success}\) 为团队围堵成功奖励。Leader 额外考虑目标距离和相机朝向，以保证持续感知；Follower 重点根据期望位置误差和团队合围状态获得奖励。

参考位置趋近奖励可表示为

\[
R_i^{target}=w_p\left[e_i(t-1)-e_i(t)\right]-w_e e_i(t).
\]

队形奖励可表示为

\[
R_i^{form}=-w_f E_{form}(t).
\]

安全奖励可表示为

\[
R_i^{safe}=-w_s\sum_{j\ne i}
\max\left(d_{safe}-d_{ij}(t),0\right).
\]

本项目采用 MADDPG 的集中式训练、分布式执行框架。第 \(i\) 个 Actor 根据局部观测输出动作：

\[
a_i=\mu_i(o_i\mid\theta_i^{\mu}).
\]

第 \(i\) 个 Critic 在训练阶段接收全部智能体的联合观测和联合动作：

\[
Q_i(x,a_1,a_2,a_3\mid\theta_i^Q),
\qquad x=[o_1,o_2,o_3].
\]

Critic 的目标值为

\[
y_i=r_i+\gamma Q_i'(x',a_1',a_2',a_3')(1-d),
\]

其中，\(a_j'=\mu_j'(o_j')\)，\(d\) 为回合终止标志。Critic 通过最小化均方误差进行更新：

\[
L_i=\frac{1}{M}\sum_{m=1}^{M}
\left(Q_i(x,a_1,a_2,a_3)-y_i\right)^2.
\]

目标网络采用软更新方式：

\[
\theta_i'\leftarrow\tau\theta_i+(1-\tau)\theta_i'.
\]

训练完成后，各机器狗仅保留本机 Actor 网络进行在线推理，根据实时观测独立输出运动控制量。通过目标趋近、队形保持、安全约束和团队成功奖励的共同引导，使 Leader 与 Follower 在动态目标运动过程中形成持续跟踪、协同机动和联合围控策略。

## 3.4 ROS2 输入输出与接口设计

### 3.4.1 输入接口

| Topic | 类型 | 发布节点 | 作用 |
| --- | --- | --- | --- |
| `/go2_1/target_estimated/odom` | `nav_msgs/msg/Odometry` | `target_perception` | 提供动态目标的位置和速度，是在线围堵任务的目标状态输入 |
| `/go2_1/odom` | `nav_msgs/msg/Odometry` | go2_1 里程计节点 | 提供感知 Leader 的位置、姿态和速度，用于构建 Leader 观测及计算后方跟踪误差 |
| `/go2_2/odom` | `nav_msgs/msg/Odometry` | go2_2 里程计节点 | 提供前方 Follower 的位置、姿态和速度，用于构建 Follower 观测及计算前方槽位误差 |
| `/go2_3/odom` | `nav_msgs/msg/Odometry` | go2_3 里程计节点 | 提供侧前方 Follower 的位置、姿态和速度，用于构建 Follower 观测及计算侧前方槽位误差 |

`/go2_1/target_estimated/odom` 中，`pose.pose.position` 表示目标位置，`twist.twist.linear` 表示目标速度。三个 `/go2_i/odom` 中，`pose.pose.position` 表示机器狗位置，`pose.pose.orientation` 用于计算航向角，`twist.twist` 表示机器狗运动状态。MARL 协同围堵节点订阅上述话题后，计算目标相对位置、队友相对位置和角色参考位置误差，并组合形成各 Actor 网络的状态观测。

在训练和算法对比阶段，可使用 `/walking_target/odom` 替代 `/go2_1/target_estimated/odom`。该话题类型同为 `nav_msgs/msg/Odometry`，由 `actor_state_publisher` 根据 Gazebo 行人模型状态生成，用于提供不含视觉估计误差的目标真值。两个目标话题为可切换关系，正式在线运行时使用视觉估计话题，不同时作为控制输入。

### 3.4.2 输出接口

| Topic | 类型 | 订阅节点 | 作用 |
| --- | --- | --- | --- |
| `/go2_1/cmd_vel` | `geometry_msgs/msg/Twist` | go2_1 底层运动控制器 | 控制 Leader 接近目标并在目标后方保持感知距离和朝向 |
| `/go2_2/cmd_vel` | `geometry_msgs/msg/Twist` | go2_2 底层运动控制器 | 控制前方 Follower 向目标前方参考位置机动并执行合围 |
| `/go2_3/cmd_vel` | `geometry_msgs/msg/Twist` | go2_3 底层运动控制器 | 控制侧前方 Follower 向侧前方参考位置机动并执行合围 |

三个输出话题均使用 `geometry_msgs/msg/Twist`。其中，`linear.x` 为机器狗前向线速度，`angular.z` 为绕竖直轴的角速度，其余分量保持为零。Actor 网络输出的归一化动作经过速度映射、限幅和加速度约束后写入对应字段，从而转换为 Go2 可执行的运动指令。围堵完成、目标长时间丢失或机器人里程计超时时，协同围堵节点通过上述话题发布零速度，使机器狗停止运动。

### 3.4.3 接口关系

各接口之间的数据关系为：

```text
go2_1 RGB-D相机
        ↓
target_perception目标定位节点
        ↓ /go2_1/target_estimated/odom
MARL协同围堵节点 ← /go2_1/odom
        ↑            /go2_2/odom
        └────────────/go2_3/odom
        ↓
状态观测构建 → 动态参考位置生成 → Actor策略推理
        ↓
/go2_1/cmd_vel → go2_1底层运动控制器
/go2_2/cmd_vel → go2_2底层运动控制器
/go2_3/cmd_vel → go2_3底层运动控制器
```

目标定位模块只负责输出统一坐标系下的目标状态，不直接控制机器狗。协同围堵模块是目标信息、机器人状态与运动控制之间的连接节点：其订阅一个目标状态话题和三个机器人里程计话题，在内部完成 Leader-Follower 参考生成、状态观测构建和 Actor 策略推理，再分别发布三路速度指令。Follower 不需要直接订阅相机图像，也不需要单独发布目标点话题，其所需目标信息由协同围堵模块从 Leader 的目标估计中统一获得。
