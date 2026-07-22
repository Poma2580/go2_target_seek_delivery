# multi_go2_nav2

该包在一个共享 `/map` 上启动三套命名空间隔离的 Nav2。Gazebo 仿真使用
ground-truth odometry，因此由静态 TF 建立 `map -> go2_i/odom`，不启动 AMCL。

机场运行顺序：

```bash
cd /home/zhj/xhk/go2_target_seek_delivery
./Scripts/start_three_go2_velodyne.sh airport

cd go2_ws_v2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch multi_go2_nav2 multi_go2_nav2.launch.py
```

默认启动不再让四个 lifecycle manager 同时 autostart。包内的
`nav2_bringup_sequencer` 会严格按照 `map -> go2_1 -> go2_2 -> go2_3`
逐套启动并确认 active，随后协调器才开始规划。这避免三套大型 costmap 同时配置时
出现 lifecycle `change_state` 响应超时。正常使用时启动命令不变。

需要调试生命周期时，可关闭自动顺序启动：

```bash
ros2 launch multi_go2_nav2 multi_go2_nav2.launch.py \
  autostart:=false start_coordinator:=false
```

场景参数统一位于 `config/scenes/*.yaml`。修改出生位姿、目标位姿和围捕半径
不需要重新生成地图；只有 world 内静态 collision 或地图边界/分辨率改变时才需
重新运行 `world_to_grid`。

Go2 保守控制配置
-----------------

Nav2 的默认轮式底盘恢复树会在跟踪失败后反复执行 Spin/BackUp，不适合 CHAMP
四足步态。本包默认安装并使用 `navigate_to_pose_no_recovery.xml`：保持 1 Hz
重规划和连续 FollowPath，但控制失败时直接安全终止。`controller_server` 和
`behavior_server` 的速度统一进入 `cmd_vel_nav -> velocity_smoother -> cmd_vel`，
不会再有恢复动作绕过平滑器与零速命令竞争。

当前首轮保守值为 0.22 m/s 平移、0.22 rad/s 原地转向，Nav2 控制频率 10 Hz，
速度平滑器采用 OPEN_LOOP。应先通过单狗直线、转向和单目标测试，再逐步提高速度，
不要直接恢复到 0.45 m/s、0.60 rad/s。
