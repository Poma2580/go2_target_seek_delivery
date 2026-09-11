# 行人地图生成与场景验证工具链

本目录负责基于二维占用地图验证动态行人路线，以及生成三只 Go2 的初始位姿组。
源码、命令入口、测试和验证产物彼此分离。

## 目录结构

```text
pedestrian_map/
├── scripts/                    # 面向使用者的命令入口
│   ├── routes/                 # 行人路线验证
│   └── poses/                  # 机器人初始位姿生成
├── src/pedestrian_map/         # 可测试、可复用的 Python 实现
├── tests/                      # 单元测试
└── artifacts/                  # 已验证的基准产物
    ├── route_validation/       # 各场景的路线验证图
    └── robot_pose_generation/  # 位姿验证图和 JSON 报告
```

`artifacts` 中的 city、forest、airport 验证产物纳入版本管理。正式运行配置
`go2_ws_v2/src/go2_test_framework/config/parameters/robot_pose_groups.yaml`
仍保存在 ROS 测试框架中。

## 地图依赖

默认地图由兄弟工具 [`gazebo_map_creator`](../gazebo_map_creator/README.md) 生成，
位置为 `tools/gazebo_map_creator/artifacts/maps/<scene>/`。路线和位姿命令分别可用
`--map-root`、`--maps-root` 覆盖该默认位置。

## 验证行人路线

```bash
/usr/bin/python3 tools/pedestrian_map/scripts/routes/validate_route.py \
  --scene city --shape rectangle \
  --point -13 4 --point 41 4 --point 41 36 --point -13 36 \
  --safety-distance 0.4 --plot
```

其他路线示例：

```bash
/usr/bin/python3 tools/pedestrian_map/scripts/routes/validate_route.py \
  --scene airport --shape straight \
  --point 54 -17 --point 100 -17 --plot

/usr/bin/python3 tools/pedestrian_map/scripts/routes/validate_route.py \
  --scene forest --shape v \
  --point -50 0 --point -10 0 --point -6 -43 \
  --safety-distance 0.5 --plot
```

不带参数时进入交互模式。三种路线连接方式为：

- `straight`：P1-P2
- `rectangle`：P1-P2-P3-P4-P1
- `v`：P1-P2-P3，返回时沿相同线段反向运动

地图中的未知栅格按占用处理，默认安全距离为 `0.4 m`。退出码 `0` 表示
PASS，`1` 表示路线有效但碰撞或安全距离检查失败，`2` 表示输入、地图或工具错误。

`--plot` 默认写入
`artifacts/route_validation/<scene>/<shape>_validation.png`；可用
`--plot-root` 更换根目录，或用 `--plot-output` 指定单张图片。

## 生成机器人初始位姿

每个场景独立维护 11 组三狗位姿，正式 YAML 使用 `schema_version: 2` 和
`coordinate_mode: scene_absolute`，数据路径为 `scenes.<scene>.pose_groups.<group>`。
city 固定 go2_1，forest 固定 go2_2，airport 固定 go2_3。迁移前的 33 个感知狗
完整位姿保存在 `artifacts/robot_pose_generation/reference_robot_poses.yaml`，
生成时只读取该基准，不从输出 YAML 提取基准，也不重新计算固定狗的 yaw。

```bash
conda deactivate 2>/dev/null || true
which python3  # 必须输出 /usr/bin/python3
/usr/bin/python3 tools/pedestrian_map/scripts/poses/generate_robot_pose_groups.py \
  --neighbor-radius-min 3.0 \
  --neighbor-radius-max 8.0 \
  --spawn-clearance 0.8 \
  --min-robot-separation 2.0 \
  --seed 20260901
```

每组另外两只导航狗围绕固定感知狗进行圆环面积均匀采样，只检查当前场景的
地图范围、free space、障碍距离、参考狗距离以及两只导航狗之间的间距。
未知栅格按占用处理。不同组之间不施加间距条件，导航狗不要求看到行人。
新狗默认继承该组固定狗的 z（当前 `0.60 m`），yaw 朝向该场景三条路线 P1
的平均 anchor；`--spawn-z` 仅覆盖新狗高度。坐标舍入至两位小数后再检查约束。
每个场景、每个组使用独立且可复现的随机序列。

常用参数：

- `--routes`：目标路线 YAML，默认使用测试框架正式配置，只读。
- `--maps-root`：地图根目录，默认 `tools/gazebo_map_creator/artifacts/maps`。
- `--reference-poses`：冻结感知狗基准，默认上述 `reference_robot_poses.yaml`。
- `--output`：正式位姿 YAML，默认写回测试框架配置目录。
- `--report-dir`：报告与图片目录，默认 `artifacts/robot_pose_generation`。
- `--max-attempts-per-robot`：每组每只新狗的尝试上限，默认 1000000。
- `--check`：按相同参数重新生成、比较并复验现有 YAML，不写入文件。

旧的 `--radius-min/max`、FOV、目标可见距离和跨组间距参数已移除，使用上面的
邻域参数。默认参数的只读复验命令：

```bash
/usr/bin/python3 tools/pedestrian_map/scripts/poses/generate_robot_pose_groups.py --check
```

采样或验证失败时不放宽约束、不替换正式输出。成功后输出 YAML、三张场景图和
`validation_report.json`。报告包括 99 条机器人记录、固定基准文件校验和、采样统计、
距离与 clearance（米）、朝向误差（弧度）及各项检查结果。固定狗记录的半径和
导航狗间距检查标记为不适用。每张图仅显示本场景位姿，包含全图概览及局部放大，
标出机器人颜色、组号、朝向、组内连线、P1 和 anchor。

## 运行测试

```bash
conda deactivate 2>/dev/null || true
which python3  # 必须输出 /usr/bin/python3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q \
  tools/pedestrian_map/tests
```
