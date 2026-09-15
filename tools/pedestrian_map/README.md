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

使用固定 seed 生成并验证 11 组位姿，同时更新正式 YAML、JSON 报告和三张复核图：

```bash
/usr/bin/python3 tools/pedestrian_map/scripts/poses/generate_robot_pose_groups.py \
  --radius-min 6.0 \
  --radius-max 12.0 \
  --spawn-z 0.4 \
  --spawn-clearance 0.2 \
  --max-target-distance 25.0 \
  --camera-hfov-deg 60.0 \
  --min-pose-separation 0.5 \
  --go2-2-camera-hfov-deg 60.0 \
  --go2-2-min-pose-separation 0.5 \
  --seed 20260901 \
  --max-attempts-per-robot 1000000
```

常用路径参数：

- `--routes`：目标路线 YAML，默认使用测试框架正式配置。
- `--maps-root`：地图根目录，默认使用 `tools/gazebo_map_creator/artifacts/maps`。
- `--output`：正式位姿 YAML，默认写回测试框架配置目录。
- `--report-dir`：报告目录，默认使用 `artifacts/robot_pose_generation`。
- `--check`：只复验现有 YAML，不写入任何结果。

其余采样参数及默认值以 `--help` 为准。当前正式 YAML 的出生高度已调整为
`0.8 m`，其余二维位姿仍来自已保存报告中的生成参数。对应的只读复验命令为：

```bash
/usr/bin/python3 tools/pedestrian_map/scripts/poses/generate_robot_pose_groups.py \
  --check --radius-min 8.0 --radius-max 14.0 --spawn-z 0.8 \
  --spawn-clearance 0.2 --max-target-distance 25.0 \
  --camera-hfov-deg 50.0 --min-pose-separation 0.5 \
  --go2-2-camera-hfov-deg 60.0 --go2-2-min-pose-separation 0.5 \
  --seed 20260901 --max-attempts-per-robot 1000000
```

## 运行测试

```bash
conda deactivate 2>/dev/null || true
which python3  # 必须输出 /usr/bin/python3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q \
  tools/pedestrian_map/tests
```
