#!/usr/bin/env python3
"""单 Go2 RGB-D 在线感知节点：YOLO 检测行人 + 深度反投影 + TF 转全局。

链路（最小闭环的第 4~7 步）：
  RGB + Depth(对齐) --YOLO--> person bbox
  bbox 中心像素 (u,v) + bbox 内 ROI 稳健深度 Z --反投影--> camera_depth_optical_frame 下 3D 点
  --tf2--> 目标系(默认 go2_1/odom ≈ 世界系) 下的全局位置
  --发布--> /go2_1/target_pose_estimated (PoseStamped)
            /go2_1/target_estimated/odom (Odometry，twist 用相邻帧差分+低通)

设计要点：
- target_estimated/odom 的字段与 actor_state_publisher 的 /walking_target/odom 完全同构，
  后续把 dynamic_encircle 的订阅从真值切到这里即可完成“真值->估计”替换。
- 深度优先取 bbox 内人体 ROI 的低分位数，减少背景深度污染；ROI 无效时回退中心窗口。
- 找不到 person / 查不到 TF：节流告警并跳过，不发布伪造位姿。

坐标/单位约定：
- YOLO 输出的是图像像素坐标 bbox=(x1,y1,x2,y2)，原点在图像左上角，u 向右、v 向下。
- depth 图每个像素存的是相机到该像素射线方向上的深度 Z，本节点统一转成“米”。
- 相机内参 K 中 fx/fy/cx/cy 用于把像素 (u,v,Z) 还原成相机光学坐标系下的 3D 点：
    X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=Z
- TF 再负责把这个相机坐标点变换到 target_frame，默认是 go2_1/odom。

运行（先 conda deactivate，确认 which python3 为 /usr/bin/python3）：
  ros2 run multi_go2_waypoint target_perception --ros-args -p use_sim_time:=true
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (注册 PointStamped 的 do_transform)


class TargetPerception(Node):
    def __init__(self):
        super().__init__('target_perception')

        # ---- 参数 ----
        # 这些参数都可以在 ros2 run/launch 时覆盖。话题类参数留空时，会自动根据
        # robot_namespace 拼出默认话题，方便同一套代码复用到 go2_1/go2_2/...。
        self.declare_parameter('robot_namespace', 'go2_1')
        self.declare_parameter('rgb_topic', '')          # 留空则按 namespace 自动拼
        self.declare_parameter('depth_topic', '')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('pose_topic', '')
        self.declare_parameter('odom_topic', '')
        self.declare_parameter('debug_image_topic', '')
        self.declare_parameter('target_frame', '')        # 留空则用 <ns>/odom

        self.declare_parameter('model_path', 'yolov8m.pt')  # m 比 nano 对远/小目标召回强很多
        self.declare_parameter('conf', 0.7)               # 阈值放低，捞回远处低分检测
        self.declare_parameter('imgsz', 960)               # 推理分辨率，上采样提升小目标召回
        self.declare_parameter('device', 'cuda:0')        # 无 GPU 改 'cpu'
        self.declare_parameter('depth_window', 5)         # 中值采样半窗口(像素)
        self.declare_parameter('depth_percentile', 30.0)  # ROI 内取偏近分位，减少背景污染
        self.declare_parameter('min_depth_samples', 20)
        self.declare_parameter('roi_x_min', 0.25)
        self.declare_parameter('roi_x_max', 0.75)
        self.declare_parameter('roi_y_min', 0.20)
        self.declare_parameter('roi_y_max', 0.70)
        self.declare_parameter('max_depth', 25.0)         # 超过视为无效(与相机 far 一致)
        self.declare_parameter('min_depth', 0.3)
        self.declare_parameter('sync_slop', 0.05)         # rgb/depth 时间同步容差(s)
        self.declare_parameter('vel_lpf_alpha', 0.3)      # 速度低通系数
        self.declare_parameter('tf_timeout', 0.1)         # 单次 TF 查询超时(s)
        self.declare_parameter('publish_debug', True)

        ns = self.get_parameter('robot_namespace').value.strip('/')

        def _topic(param, default):
            # 小工具：如果用户显式传了某个话题名就用用户的；否则用默认值。
            v = self.get_parameter(param).value
            return v if v else default

        rgb_topic = _topic('rgb_topic', f'/{ns}/camera/image_raw')
        depth_topic = _topic('depth_topic', f'/{ns}/camera/depth/image_raw')
        info_topic = _topic('camera_info_topic', f'/{ns}/camera/depth/camera_info')
        pose_topic = _topic('pose_topic', f'/{ns}/target_pose_estimated')
        odom_topic = _topic('odom_topic', f'/{ns}/target_estimated/odom')
        dbg_topic = _topic('debug_image_topic', f'/{ns}/target_perception/debug_image')
        self.target_frame = _topic('target_frame', f'{ns}/odom')

        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.device = self.get_parameter('device').value
        self.win = int(self.get_parameter('depth_window').value)
        self.depth_percentile = float(self.get_parameter('depth_percentile').value)
        self.min_depth_samples = int(self.get_parameter('min_depth_samples').value)
        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        slop = float(self.get_parameter('sync_slop').value)
        self.alpha = float(self.get_parameter('vel_lpf_alpha').value)
        self.tf_timeout = float(self.get_parameter('tf_timeout').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)

        # ---- YOLO ----
        model_path = self.get_parameter('model_path').value
        from ultralytics import YOLO   # 延迟导入，避免无依赖时整包失败
        self.get_logger().info(f'加载 YOLO 模型: {model_path} (device={self.device}) ...')
        self.model = YOLO(model_path)
        # COCO 数据集中 person 的类别 id 是 0，所以 classes=[0] 表示只让 YOLO 返回人。
        # 这样后面不需要处理车、椅子等其它类别，也能减少误检传播到导航侧。
        self._yolo_kwargs = dict(classes=[0], conf=self.conf, imgsz=self.imgsz,
                                 verbose=False)
        try:
            self._yolo_kwargs['device'] = self.device
        except Exception:
            pass

        # ---- 内部状态 ----
        self.bridge = CvBridge()
        self.K = None             # 相机内参，收到 CameraInfo 后填成 (fx, fy, cx, cy)
        # 以下 last_* 和 v* 用于由相邻两帧目标位置差分出速度，并做低通滤波。
        # 注意这里的 last_stamp 使用节点当前时间，而发布消息仍使用传感器时间戳。
        self.last_x = None
        self.last_y = None
        self.last_stamp = None
        self.vx = 0.0
        self.vy = 0.0
        self._warn_throttle = self.get_clock()

        # ---- TF ----
        # Buffer 保存 TF 树里的历史变换；TransformListener 负责订阅 /tf 和 /tf_static
        # 并不断把变换写进 Buffer。后面 transform() 就从这个缓存里查。
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- 发布 ----
        # PoseStamped 只表达“目标在哪里”；Odometry 额外带线速度，方便下游控制器直接订阅。
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        # debug 图用与相机 image_raw 相同的 sensor_data QoS（BEST_EFFORT），
        # 这样 rqt_image_view 能像显示 image_raw 一样显示带框图。
        self.dbg_pub = (self.create_publisher(Image, dbg_topic, qos_profile_sensor_data)
                        if self.publish_debug else None)

        # ---- 订阅（rgb + depth 近似时间同步）----
        # CameraInfo 不需要严格和图像同步，只要拿到内参即可；RGB 和 depth 必须配对，
        # 否则 YOLO 框中心对应的深度可能来自另一时刻，目标移动时会产生明显误差。
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data)
        rgb_sub = message_filters.Subscriber(
            self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=slop)
        self.sync.registerCallback(self._rgbd_cb)

        self.get_logger().info(
            f'target_perception 已启动：\n'
            f'  rgb   = {rgb_topic}\n'
            f'  depth = {depth_topic}\n'
            f'  info  = {info_topic}\n'
            f'  -> pose = {pose_topic}\n'
            f'  -> odom = {odom_topic}  (frame={self.target_frame})')

    # -----------------------------------------------------------------
    def _info_cb(self, msg):
        """保存相机针孔模型内参。

        CameraInfo.k 是 3x3 矩阵按行展开：
          [fx, 0, cx,
           0, fy, cy,
           0,  0,  1]
        本节点只需要 fx/fy/cx/cy 做深度反投影。
        """
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _warn(self, text):
        # ROS 日志节流：同一类问题每 2 秒最多打印一次，避免图像回调高频刷屏。
        self.get_logger().warn(text, throttle_duration_sec=2.0)

    def _rgbd_cb(self, rgb_msg, depth_msg):
        """RGB-D 同步回调：整条感知链路都在这里串起来。

        一帧进入后的处理顺序是：
        1. ROS Image -> OpenCV/numpy；
        2. YOLO 在 RGB 图上找 person；
        3. 取 bbox 内人体 ROI 的稳健深度，必要时回退中心窗口；
        4. 用相机内参把像素点反投影成相机系 3D 点；
        5. 用 TF 转成 target_frame 下的全局点；
        6. 发布 PoseStamped/Odometry，并可选发布带框 debug 图。
        """
        if self.K is None:
            self._warn('尚未收到 camera_info，无法反投影。')
            return

        try:
            # rgb 用 bgr8 是因为 OpenCV 默认颜色顺序是 BGR；ultralytics 也能接受 numpy 图像。
            # depth 用 passthrough 保留原始深度编码，后面再根据 encoding 判断单位。
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:  # noqa: BLE001
            self._warn(f'cv_bridge 转换失败：{e}')
            return

        depth = np.asarray(depth, dtype=np.float32)  # 米；16UC1 时单位是 mm，见下
        # 若是 16UC1（毫米），转成米
        if depth_msg.encoding in ('16UC1', 'mono16'):
            depth = depth / 1000.0

        # ---- YOLO 检测 ----
        # results 里可能包含多个检测框；_pick_person 会挑一个“面积最大的人”作为跟踪目标。
        results = self.model(rgb, **self._yolo_kwargs)
        best = self._pick_person(results)
        if best is None:
            self._warn('未检测到 person。')
            if self.dbg_pub is not None:
                self._publish_debug(rgb, None, rgb_msg.header)
            return

        x1, y1, x2, y2, conf = best
        # 用检测框中心作为反投影像素，深度则优先从 bbox 内 ROI 鲁棒采样。
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))

        Z, depth_source = self._sample_depth_from_bbox(depth, (x1, y1, x2, y2), u, v)
        if Z is None:
            self._warn(f'bbox 深度无效 (u={u}, v={v})。')
            if self.dbg_pub is not None:
                self._publish_debug(rgb, (x1, y1, x2, y2, conf, None), rgb_msg.header)
            return

        # ---- 反投影到 camera_depth_optical_frame ----
        fx, fy, cx, cy = self.K
        # 针孔相机反投影公式：像素点离主点越远，在相同深度下 X/Y 偏移越大。
        # 这里得到的是深度相机光学坐标系下的点，不是 odom/world 下的点。
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        pt = PointStamped()
        # 用深度图的 frame_id 作为源系（已是 <ns>/camera_depth_optical_frame）
        pt.header.frame_id = depth_msg.header.frame_id or 'camera_depth_optical_frame'
        pt.header.stamp = depth_msg.header.stamp
        pt.point.x = float(X)
        pt.point.y = float(Y)
        pt.point.z = float(Z)

        # ---- TF 转到全局系 ----
        try:
            # 优先按深度图时间戳查询 TF，保证“图像那一刻”的相机位姿和深度匹配。
            pt_world = self.tf_buffer.transform(
                pt, self.target_frame,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
        except Exception:  # noqa: BLE001
            # 回退：用最新可用变换
            try:
                # 有些仿真/启动时刻 TF 历史不完整，按图像时间查不到；
                # 这里用最新 TF 兜底，宁可给一个近似估计，也不直接让整帧失败。
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, pt.header.frame_id,
                    rclpy.time.Time())
                pt_world = tf2_geometry_msgs.do_transform_point(pt, tf)
            except Exception as e:  # noqa: BLE001
                self._warn(f'TF {pt.header.frame_id} -> {self.target_frame} 失败：{e}')
                return

        gx = pt_world.point.x
        gy = pt_world.point.y
        gz = pt_world.point.z

        self._publish(gx, gy, gz, depth_msg.header.stamp)

        # 成功时节流打印估计 odom，让启动终端持续看到结果（也兼作轻量诊断）
        self.get_logger().info(
            f'估计行人全局位置 x={gx:+.2f} y={gy:+.2f}  '
            f'距离 Z={Z:.2f}m/{depth_source}  conf={conf:.2f}  '
            f'速度 vx={self.vx:+.2f} vy={self.vy:+.2f}',
            throttle_duration_sec=0.5)

        if self.dbg_pub is not None:
            self._publish_debug(rgb, (x1, y1, x2, y2, conf, Z), rgb_msg.header,
                                gx=gx, gy=gy, depth_source=depth_source)

    # -----------------------------------------------------------------
    def _pick_person(self, results):
        """从 YOLO 结果里挑面积最大的一个 person bbox。

        YOLOv8 的每个 box 都带类别、置信度和 xyxy 坐标。虽然推理时已经设置
        classes=[0]，这里仍检查 cls==0，相当于再做一层保险。

        当前策略是“面积最大的人 = 目标人”。这适合场景里通常只有一个主要行人，
        或者最近的人在图像上最大；如果未来要多目标跟踪，可以在这里改成按 ID、
        距离、上一帧位置连续性等规则选择。

        Returns:
            (x1, y1, x2, y2, conf)，如果没有 person 则返回 None。
        """
        best = None
        best_area = 0.0
        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for b in boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                if cls != 0:
                    continue
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                # 用 bbox 面积选“主体目标”。max 防止异常框出现负宽/负高。
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area > best_area:
                    best_area = area
                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    best = (x1, y1, x2, y2, conf)
        return best

    def _sample_depth(self, depth, u, v):
        """在 bbox 中心附近取一个稳健深度值。

        真实/仿真深度图常见问题是 0、NaN、inf、边缘飞点，直接读取中心一个像素
        很容易失败。因此这里取 (2*win+1)^2 的小窗口，过滤掉无效深度和超出范围
        的深度，再用中位数抑制噪声。
        """
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        win = self.win
        u0, u1 = max(0, u - win), min(w, u + win + 1)
        v0, v1 = max(0, v - win), min(h, v + win + 1)
        patch = depth[v0:v1, u0:u1].reshape(-1)
        valid = patch[np.isfinite(patch) &
                      (patch > self.min_depth) & (patch < self.max_depth)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _sample_depth_from_bbox(self, depth, bbox, u, v):
        """优先在 bbox 内人体 ROI 取深度，样本不足时回退中心窗口。"""
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = bbox

        bx0 = max(0, min(w, int(round(x1))))
        bx1 = max(0, min(w, int(round(x2))))
        by0 = max(0, min(h, int(round(y1))))
        by1 = max(0, min(h, int(round(y2))))
        if bx1 <= bx0 or by1 <= by0:
            return self._sample_depth(depth, u, v), 'center'

        bw = bx1 - bx0
        bh = by1 - by0
        rx0 = bx0 + int(round(bw * self.roi_x_min))
        rx1 = bx0 + int(round(bw * self.roi_x_max))
        ry0 = by0 + int(round(bh * self.roi_y_min))
        ry1 = by0 + int(round(bh * self.roi_y_max))
        rx0, rx1 = max(0, min(w, rx0)), max(0, min(w, rx1))
        ry0, ry1 = max(0, min(h, ry0)), max(0, min(h, ry1))

        if rx1 > rx0 and ry1 > ry0:
            patch = depth[ry0:ry1, rx0:rx1].reshape(-1)
            valid = patch[np.isfinite(patch) &
                          (patch > self.min_depth) & (patch < self.max_depth)]
            if valid.size >= self.min_depth_samples:
                pct = max(0.0, min(100.0, self.depth_percentile))
                return float(np.percentile(valid, pct)), 'roi'

        return self._sample_depth(depth, u, v), 'center'

    def _publish(self, gx, gy, gz, stamp):
        """发布 ROI 深度反投影后的目标估计结果。"""
        now = self.get_clock().now()

        # 用相邻帧位置差分自算世界系速度，再低通（复用 actor_state_publisher 写法）
        if self.last_stamp is not None:
            dt = (now - self.last_stamp).nanoseconds * 1e-9
            if dt > 1e-4:
                raw_vx = (gx - self.last_x) / dt
                raw_vy = (gy - self.last_y) / dt
                self.vx = self.alpha * raw_vx + (1.0 - self.alpha) * self.vx
                self.vy = self.alpha * raw_vy + (1.0 - self.alpha) * self.vy
        self.last_x, self.last_y, self.last_stamp = gx, gy, now

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.target_frame
        pose.pose.position.x = float(gx)
        pose.pose.position.y = float(gy)
        pose.pose.position.z = float(gz)
        pose.pose.orientation.w = 1.0

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.target_frame
        odom.child_frame_id = 'target_estimated'
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.linear.z = 0.0

        self.pose_pub.publish(pose)
        self.odom_pub.publish(odom)

    def _publish_debug(self, rgb, det, header, gx=None, gy=None, depth_source=None):
        """发布带检测框的调试图。

        det 为 None 时表示这一帧没有检测到人；否则 det 里包含 bbox、置信度和
        可选深度 Z。调试图只用于人工观察，不参与控制闭环。
        """
        import cv2
        img = rgb.copy()
        if det is not None:
            x1, y1, x2, y2, conf, Z = det
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            label = f'person {conf:.2f}'
            if Z is not None:
                label += f' Z={Z:.2f}m'
            if depth_source is not None:
                label += f' {depth_source}'
            if gx is not None:
                label += f' ({gx:.1f},{gy:.1f})'
            cv2.putText(img, label, (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(img, 'no person', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            out.header = header
            self.dbg_pub.publish(out)
        except Exception:  # noqa: BLE001
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TargetPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
