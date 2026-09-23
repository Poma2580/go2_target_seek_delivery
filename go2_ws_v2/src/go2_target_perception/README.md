# go2_target_perception

三只 Go2 共用的 RGB-D 目标感知、感知角色选择和动态行人状态发布包。检测器可在
`config/target_perception.yaml` 中切换，定位、速度估计和 ROS 输出链保持一致。

## 检测后端

默认配置使用 YOLOE：

```yaml
detector_backend: yoloe
yoloe_model_path: yoloe-26s-seg.pt
model_cache_dir: ~/.cache/go2_static_yolo_smoke
```

回退到 YOLOv8 只需修改一项：

```yaml
detector_backend: yolov8
```

配置同时保存 `yolov8_model_path` 和 `yoloe_model_path`。YOLOE 的文字 prompt
会重新编号，因此不能把 YOLOv8 的 COCO ID 传给 YOLOE；节点会分别为两个后端
建立正确的类别映射。`person` 是唯一进入 RGB-D 定位和围捕控制链的类别，其他
配置类别只出现在调试图中。

YOLOE 首次启动需要主模型和文字编码器。部署到离线机器前应准备：

```text
~/.cache/go2_static_yolo_smoke/yoloe-26s-seg.pt
~/.cache/go2_static_yolo_smoke/mobileclip2_b.ts
```

正常启动由 YAML 决定后端：

```bash
ros2 launch go2_target_perception three_go2_target_tracking.launch.py \
  use_sim_time:=true
```

测试或诊断时可以临时覆盖，但不修改 YAML：

```bash
ros2 launch go2_target_perception three_go2_target_tracking.launch.py \
  use_sim_time:=true \
  detector_backend_override:=yolov8 \
  model_path_override:=/home/bit/go2_target_seek_delivery/yolov8s.pt
```

ROS 项目必须使用系统 Python，不要在 conda 环境中运行。
