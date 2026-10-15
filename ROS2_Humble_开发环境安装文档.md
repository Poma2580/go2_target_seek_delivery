# Ubuntu 24.04 工作站开发环境安装文档

**目标环境**：ROS 2 Humble + Python 开发 + YOLOE（开放词汇目标检测）+ Gazebo Sim
**实现方式**：宿主机 Ubuntu 24.04 不动，通过 Docker 容器运行 Ubuntu 22.04 环境
**适用读者**：计算机视觉 / 目标跟踪方向，无 Docker 与底层系统经验

---

## 0. 方案总览

ROS 2 Humble 官方只支持 Ubuntu 22.04（Jammy），不支持 24.04。因此采用容器方案：

```
┌─────────────────────────────────────┐
│  容器：Ubuntu 22.04 用户态           │
│  ROS 2 Humble + Gazebo Fortress     │
│  + PyTorch + Ultralytics(YOLOE)     │
├─────────────────────────────────────┤
│  Docker Engine + NVIDIA Toolkit     │
├─────────────────────────────────────┤
│  Ubuntu 24.04 宿主机（NVIDIA 驱动） │
├─────────────────────────────────────┤
│  硬件：CPU / GPU / 磁盘             │
└─────────────────────────────────────┘
```

- Gazebo 版本说明：ROS 2 Humble 官方配对的 Gazebo Sim 版本是 **Fortress（LTS）**，本文档安装此版本（命令为 `gz sim`）。
- YOLOE 已并入 ultralytics 官方包，直接 `pip install ultralytics` 即可使用 `YOLOE` 类，无需单独编译。

---

## 1. 宿主机配置（Ubuntu 24.04）

以下命令全部在**宿主机的终端**中执行。

### 1.1 安装 NVIDIA 驱动

```bash
# 查看推荐驱动版本
ubuntu-drivers devices

# 自动安装推荐驱动（或手动指定版本，如 nvidia-driver-570）
sudo ubuntu-drivers autoinstall

# 装完重启
sudo reboot
```

重启后验证：

```bash
nvidia-smi   # 能显示显卡信息即成功
```

### 1.2 安装 Docker Engine

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

> 国内网络可把所有 `download.docker.com` 替换为 `mirrors.aliyun.com/docker-ce`。

### 1.3 安装 NVIDIA Container Toolkit（容器内使用 GPU 的桥梁）

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1.4 免 sudo 运行 docker

```bash
sudo usermod -aG docker $USER
# 注销并重新登录后生效
```

### 1.5 允许容器显示 GUI 窗口（Gazebo / RViz 需要）

```bash
# 每次重启宿主机后执行一次（容器内默认是 root 用户）
xhost +SI:localuser:root
```

> 建议把上面这行追加到 `~/.profile` 末尾，开机自动生效。

### 1.6 验证宿主机侧

```bash
docker run hello-world                                        # Docker 正常
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # GPU 透传正常
```

---

## 2. 编写 Dockerfile

在宿主机创建工作目录（代码和数据今后都放这里，挂载进容器）：

```bash
mkdir -p ~/work/ros2_ws/src
cd ~/work
```

新建文件 `~/work/Dockerfile`，内容如下：

```dockerfile
# 基于官方 ROS 2 Humble 桌面完整版（已内置 Ubuntu 22.04 + ROS 2 Humble）
FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# 基础工具 + Python 开发环境 + OpenGL 图形库（Gazebo/RViz 需要）
RUN apt-get update && apt-get install -y \
    python3-pip python3-colcon-common-extensions python3-rosdep \
    git wget curl vim nano \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Gazebo Sim Fortress + ROS-Gazebo 桥（Humble 官方配对版本）
RUN apt-get update && apt-get install -y \
    ros-humble-ros-gz \
    && rm -rf /var/lib/apt/lists/*

# PyTorch（pip 版自带 CUDA 运行时，无需单独装 CUDA Toolkit）
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision

# Ultralytics（已包含 YOLOE）+ 常用视觉库
RUN pip3 install -U ultralytics opencv-python

# 进入容器自动加载 ROS 环境
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /workspace
CMD ["bash"]
```

> 说明：`osrf/ros:humble-desktop-full` 的 apt 源中已包含 Gazebo Fortress 软件包，
> 因此 `ros-humble-ros-gz` 可直接安装，无需额外添加 OSRF 源。

---

## 3. 构建镜像与启动容器

### 3.1 构建镜像（约 20~40 分钟，取决于网络）

```bash
cd ~/work
docker build -t ros2-humble-dev .
```

### 3.2 启动容器（GPU + GUI + 目录挂载）

```bash
xhost +SI:localuser:root   # 若已写入 ~/.profile 可跳过

docker run -it --name ros2dev \
  --gpus all \
  --net=host --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/work:/workspace \
  ros2-humble-dev bash
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--gpus all` | 透传 GPU 给容器 |
| `--net=host` | 共享宿主机网络（ROS 2 节点通信、GUI 显示都更方便） |
| `-e DISPLAY` + `-v /tmp/.X11-unix` | 让容器内 Gazebo/RViz 窗口显示在宿主机屏幕上 |
| `-e NVIDIA_DRIVER_CAPABILITIES=all` | 容器内可用 GPU 做 OpenGL 渲染（Gazebo 必需） |
| `-v ~/work:/workspace` | 宿主机目录挂载进容器，**代码数据不会因容器删除而丢失** |

### 3.3 日常使用命令（宿主机终端执行）

```bash
docker start ros2dev            # 启动已存在的容器
docker exec -it ros2dev bash    # 进入容器开一个新终端（可开多个）
docker stop ros2dev             # 停止容器
docker rm ros2dev               # 删除容器（镜像和 ~/work 数据都在）
```

---

## 4. 验证安装（容器内执行）

### 4.1 ROS 2 Humble

```bash
ros2 --version          # 应显示 ros2 cli 信息
printenv ROS_DISTRO     # 应输出 humble（若为空，先 source /opt/ros/humble/setup.bash）

# 双终端收发测试：
# 终端A: ros2 run demo_nodes_cpp talker
# 终端B: ros2 run demo_nodes_cpp listener
```

### 4.2 Gazebo Sim

```bash
gz sim --version        # 应显示 7.x（Fortress）
gz sim shapes.sdf       # 应弹出 GUI 窗口，显示三个几何体
```

ROS 2 与 Gazebo 联合启动测试：

```bash
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="shapes.sdf"
```

> 无显示器的场景可用无头模式：`gz sim -s -r shapes.sdf`（只跑服务端不弹窗）。

### 4.3 PyTorch / GPU

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出类似: 2.x.x+cu12x True
```

### 4.4 YOLOE

```bash
cd /workspace
wget https://ultralytics.com/images/bus.jpg -O bus.jpg

python3 - <<'EOF'
from ultralytics import YOLOE

# 首次运行自动下载权重（yoloe-11s-seg.pt，约几十 MB）
model = YOLOE("yoloe-11s-seg.pt")
results = model.predict("bus.jpg", save=True)
print("检测完成，结果保存在 runs/ 目录")

# 开放词汇（自定义文本类别）用法示例：
# model.set_classes(["person", "helmet"], model.get_text_pe(["person", "helmet"]))
# results = model.predict("bus.jpg", save=True)
EOF
```

---

## 5.（可选）国内镜像加速

如果构建镜像时下载很慢，在 Dockerfile 的**最前面**（FROM 之后）加入：

```dockerfile
# Ubuntu apt 换清华源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g; s@//ports.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# ROS 2 源换清华镜像
RUN sed -i 's@packages.ros.org@mirrors.tuna.tsinghua.edu.cn/ros@g' /etc/apt/sources.list.d/ros2.list
```

pip 换源（容器内执行一次即可）：

```bash
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 6. 常见问题排查

| 现象 | 原因与解决 |
|---|---|
| `docker: command not found` | 第 1.2 步未完成，或安装后未重开终端 |
| `permission denied` 访问 docker.sock | 第 1.4 步未做或未重新登录 |
| 容器内 `nvidia-smi` 报错 | 启动时漏了 `--gpus all`，或第 1.3 步 toolkit 未装好 |
| Gazebo/RViz 弹不出窗口，`cannot connect to X server` | 宿主机没执行 `xhost +SI:localuser:root`，或启动参数漏了 DISPLAY / X11 挂载 |
| Gazebo 窗口黑屏 / libGL 报错 | 确认 `-e NVIDIA_DRIVER_CAPABILITIES=all`；仍不行则在容器内执行 `export LIBGL_ALWAYS_SOFTWARE=1` 改用软件渲染兜底（速度慢，仅调试用） |
| `ros2` 命令找不到 | 当前终端没加载环境：`source /opt/ros/humble/setup.bash` |
| `apt install ros-humble-ros-gz` 找不到包 | 确认基于 `osrf/ros:humble-desktop-full` 构建；如手动安装 ROS，需先配置好 ROS apt 源（见附录） |
| YOLOE 首次运行卡住 | 正在下载模型权重，等待即可；或手动从 ultralytics 发布页下载后放入工作目录 |

---

## 7. 附录：不使用 Dockerfile 的手动安装流程

如果已经在某个 `ubuntu:22.04` 容器里手动操作（等同于 Dockerfile 每一步），顺序如下：

```bash
# 1. ROS 2 apt 源
apt update && apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list

# 2. ROS 2 Humble 桌面完整版 + 开发工具
apt update
apt install -y ros-humble-desktop-full python3-colcon-common-extensions \
  python3-pip python3-rosdep git wget libgl1-mesa-glx libglib2.0-0

# 3. Gazebo Fortress + ROS 桥
apt install -y ros-humble-ros-gz

# 4. Python 视觉栈
pip3 install -U pip
pip3 install torch torchvision
pip3 install -U ultralytics opencv-python

# 5. 环境变量
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

手动装完后建议把容器固化为镜像，避免重复劳动：

```bash
# 在宿主机执行，把容器当前状态保存为新镜像
docker commit ros2dev ros2-humble-dev:snapshot
```

---

## 8. 推荐工作流小结

1. 宿主机开机 → 确保 `xhost +SI:localuser:root` 已生效
2. `docker start ros2dev && docker exec -it ros2dev bash`
3. 代码写在宿主机 `~/work`（可用任意编辑器），容器内 `/workspace` 实时同步
4. colcon 工作空间放在 `/workspace/ros2_ws`，编译：`cd /workspace/ros2_ws && colcon build && source install/setup.bash`
5. 环境有重要变更时 `docker commit` 打快照，或更新 Dockerfile 重新构建
