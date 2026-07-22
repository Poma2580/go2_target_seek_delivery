#!/usr/bin/env python3
"""轻量级二值栅格地图和 A* 路径规划。

本模块不依赖 Nav2 规划器。它直接读取 Nav2 map_server 使用的 YAML/PGM，
以 Gazebo 世界坐标规划路径，再把路径转换成 waypoint_encircle 可执行的航点。
"""

from dataclasses import dataclass
import heapq
import math
from pathlib import Path

import yaml


SQRT2 = math.sqrt(2.0)


class MapFormatError(ValueError):
    """地图文件格式或参数无效。"""


class PlanningError(RuntimeError):
    """A* 无法生成一条有效路径。"""


def _read_pgm_token(stream):
    """读取一个 PGM token，同时跳过空白和 # 注释。"""
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            raise MapFormatError('PGM 文件意外结束。')
        if char == b'#':
            stream.readline()
            continue
        if not char.isspace():
            token.extend(char)
            break

    while True:
        char = stream.read(1)
        if not char or char.isspace():
            break
        if char == b'#':
            stream.readline()
            break
        token.extend(char)
    return bytes(token)


def _load_pgm(path):
    """返回 (width, height, 0..255 灰度字节)，支持 P5/P2 PGM。"""
    try:
        with path.open('rb') as stream:
            magic = _read_pgm_token(stream)
            if magic not in (b'P5', b'P2'):
                raise MapFormatError(
                    f'仅支持 P5/P2 PGM，{path} 的格式为 {magic!r}。')
            width = int(_read_pgm_token(stream))
            height = int(_read_pgm_token(stream))
            max_value = int(_read_pgm_token(stream))
            if width <= 0 or height <= 0:
                raise MapFormatError('PGM 宽度和高度必须大于零。')
            if not 0 < max_value <= 65535:
                raise MapFormatError('PGM max_value 必须位于 1..65535。')

            count = width * height
            if magic == b'P5':
                bytes_per_pixel = 1 if max_value < 256 else 2
                raw = stream.read(count * bytes_per_pixel)
                if len(raw) != count * bytes_per_pixel:
                    raise MapFormatError('PGM 像素数据长度不足。')
                if bytes_per_pixel == 1:
                    values = raw
                else:
                    values = [
                        (raw[index] << 8) | raw[index + 1]
                        for index in range(0, len(raw), 2)
                    ]
            else:
                values = [int(_read_pgm_token(stream)) for _ in range(count)]
    except OSError as error:
        raise MapFormatError(f'无法读取 PGM 地图 {path}: {error}') from error
    except ValueError as error:
        raise MapFormatError(f'PGM 头部包含非法数字：{path}') from error

    if any(value < 0 or value > max_value for value in values):
        raise MapFormatError('PGM 像素值超出 max_value。')
    if max_value == 255:
        return width, height, bytes(values)
    return width, height, bytes(
        round(value * 255.0 / max_value) for value in values)


class GridMap:
    """以世界 Y 正方向向上的顺序保存二值占用栅格。"""

    def __init__(self, width, height, resolution, origin_x, origin_y, blocked):
        if width <= 0 or height <= 0 or resolution <= 0.0:
            raise MapFormatError('地图尺寸和分辨率必须大于零。')
        if len(blocked) != width * height:
            raise MapFormatError('占用数组尺寸与地图宽高不一致。')
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.blocked = bytearray(blocked)

    @classmethod
    def from_yaml(cls, yaml_path):
        yaml_path = Path(yaml_path).expanduser().resolve()
        try:
            config = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError) as error:
            raise MapFormatError(f'无法读取地图 YAML {yaml_path}: {error}') from error
        if not isinstance(config, dict):
            raise MapFormatError('地图 YAML 根节点必须是字典。')
        required = ('image', 'resolution', 'origin', 'negate',
                    'occupied_thresh', 'free_thresh')
        missing = [name for name in required if name not in config]
        if missing:
            raise MapFormatError(f'地图 YAML 缺少字段：{", ".join(missing)}。')
        if config.get('mode', 'trinary') != 'trinary':
            raise MapFormatError('第一版 A* 仅支持 mode: trinary。')

        origin = config['origin']
        if not isinstance(origin, list) or len(origin) < 2:
            raise MapFormatError('地图 YAML origin 至少需要 x、y 两项。')
        resolution = float(config['resolution'])
        occupied_thresh = float(config['occupied_thresh'])
        free_thresh = float(config['free_thresh'])
        negate = int(config['negate'])
        if resolution <= 0.0:
            raise MapFormatError('地图分辨率必须大于零。')
        if not 0.0 <= free_thresh < occupied_thresh <= 1.0:
            raise MapFormatError('地图 free/occupied 阈值无效。')
        if negate not in (0, 1):
            raise MapFormatError('地图 negate 必须为 0 或 1。')

        image_path = Path(str(config['image'])).expanduser()
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path
        width, height, pixels = _load_pgm(image_path.resolve())

        # PGM 从顶部一行开始；内部数组改为 gy=0 对应世界最下方一行。
        blocked = bytearray(width * height)
        for row in range(height):
            gy = height - 1 - row
            source_offset = row * width
            target_offset = gy * width
            for gx in range(width):
                pixel = pixels[source_offset + gx] / 255.0
                occupancy = pixel if negate else 1.0 - pixel
                # 本项目地图无 unknown；若将来出现阈值中间值，保守地按障碍处理。
                blocked[target_offset + gx] = occupancy > free_thresh
        return cls(width, height, resolution, origin[0], origin[1], blocked)

    def copy(self):
        return GridMap(
            self.width, self.height, self.resolution,
            self.origin_x, self.origin_y, self.blocked)

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def index(self, gx, gy):
        return gy * self.width + gx

    def cell(self, index):
        return index % self.width, index // self.width

    def is_free(self, gx, gy):
        return self.in_bounds(gx, gy) and not self.blocked[self.index(gx, gy)]

    def world_to_grid(self, x, y):
        gx = math.floor((float(x) - self.origin_x) / self.resolution)
        gy = math.floor((float(y) - self.origin_y) / self.resolution)
        if not self.in_bounds(gx, gy):
            raise PlanningError(
                f'世界坐标 ({x:.3f}, {y:.3f}) 位于地图范围外。')
        return gx, gy

    def grid_to_world(self, gx, gy):
        if not self.in_bounds(gx, gy):
            raise PlanningError(f'栅格 ({gx}, {gy}) 位于地图范围外。')
        return (
            self.origin_x + (gx + 0.5) * self.resolution,
            self.origin_y + (gy + 0.5) * self.resolution,
        )

    def is_free_world(self, x, y):
        try:
            gx, gy = self.world_to_grid(x, y)
        except PlanningError:
            return False
        return self.is_free(gx, gy)

    def inflated(self, radius):
        """用圆形整数核膨胀障碍，返回一份新地图。"""
        radius = float(radius)
        if radius < 0.0:
            raise ValueError('inflation_radius 不能小于零。')
        if radius == 0.0:
            return self.copy()
        cell_radius = int(math.ceil(radius / self.resolution))
        offsets = [
            (dx, dy)
            for dy in range(-cell_radius, cell_radius + 1)
            for dx in range(-cell_radius, cell_radius + 1)
            if dx * dx + dy * dy <= cell_radius * cell_radius
        ]
        result = bytearray(self.blocked)
        occupied = [index for index, value in enumerate(self.blocked) if value]
        for index in occupied:
            gx, gy = self.cell(index)
            for dx, dy in offsets:
                nx, ny = gx + dx, gy + dy
                if self.in_bounds(nx, ny):
                    result[self.index(nx, ny)] = 1
        return GridMap(
            self.width, self.height, self.resolution,
            self.origin_x, self.origin_y, result)


@dataclass(frozen=True)
class AStarResult:
    cells: tuple
    cost: float
    expanded: int


def _octile_distance(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


def astar(grid_map, start, goal, max_expansions=None):
    """在二值地图上执行确定性的八邻域 A*。"""
    sx, sy = start
    gx, gy = goal
    if not grid_map.is_free(sx, sy):
        raise PlanningError(f'A* 起点栅格 {start} 被占用或越界。')
    if not grid_map.is_free(gx, gy):
        raise PlanningError(f'A* 终点栅格 {goal} 被占用或越界。')
    if max_expansions is None:
        max_expansions = grid_map.width * grid_map.height
    if max_expansions <= 0:
        raise ValueError('max_expansions 必须大于零。')

    start_index = grid_map.index(sx, sy)
    goal_index = grid_map.index(gx, gy)
    if start_index == goal_index:
        return AStarResult((start,), 0.0, 0)

    # 固定邻居顺序和序号，使同样输入每次产生相同路径。
    neighbors = (
        (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
        (1, 1, SQRT2), (-1, 1, SQRT2),
        (-1, -1, SQRT2), (1, -1, SQRT2),
    )
    open_heap = []
    serial = 0
    start_h = _octile_distance(sx, sy, gx, gy)
    heapq.heappush(open_heap, (start_h, start_h, serial, start_index))
    g_score = {start_index: 0.0}
    parent = {}
    closed = bytearray(grid_map.width * grid_map.height)
    expanded = 0

    while open_heap:
        _, _, _, current_index = heapq.heappop(open_heap)
        if closed[current_index]:
            continue
        if current_index == goal_index:
            path = []
            cursor = current_index
            while True:
                path.append(grid_map.cell(cursor))
                if cursor == start_index:
                    break
                cursor = parent[cursor]
            path.reverse()
            return AStarResult(tuple(path), g_score[current_index], expanded)

        closed[current_index] = 1
        expanded += 1
        if expanded > max_expansions:
            raise PlanningError(
                f'A* 超过最大展开节点数 {max_expansions}，规划已停止。')

        x, y = grid_map.cell(current_index)
        current_g = g_score[current_index]
        for dx, dy, move_cost in neighbors:
            nx, ny = x + dx, y + dy
            if not grid_map.is_free(nx, ny):
                continue
            if dx and dy:
                # 禁止从两个障碍的接触角斜穿过去。
                if not grid_map.is_free(x + dx, y) or not grid_map.is_free(x, y + dy):
                    continue
            neighbor_index = grid_map.index(nx, ny)
            if closed[neighbor_index]:
                continue
            tentative_g = current_g + move_cost
            if tentative_g + 1e-12 >= g_score.get(neighbor_index, math.inf):
                continue
            g_score[neighbor_index] = tentative_g
            parent[neighbor_index] = current_index
            heuristic = _octile_distance(nx, ny, gx, gy)
            serial += 1
            heapq.heappush(
                open_heap,
                (tentative_g + heuristic, heuristic, serial, neighbor_index))

    raise PlanningError(f'A* 无法从 {start} 到达 {goal}。')


def line_is_free(grid_map, start, goal):
    """检查两个栅格中心的 supercover 线段是否完全位于自由空间。"""
    x, y = start
    goal_x, goal_y = goal
    if not grid_map.is_free(x, y) or not grid_map.is_free(goal_x, goal_y):
        return False
    dx = goal_x - x
    dy = goal_y - y
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    t_delta_x = math.inf if dx == 0 else 1.0 / abs(dx)
    t_delta_y = math.inf if dy == 0 else 1.0 / abs(dy)
    t_max_x = math.inf if dx == 0 else 0.5 * t_delta_x
    t_max_y = math.inf if dy == 0 else 0.5 * t_delta_y

    while (x, y) != (goal_x, goal_y):
        if t_max_x < t_max_y:
            x += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            y += step_y
            t_max_y += t_delta_y
        else:
            # 穿过栅格角点时，两侧格子也必须自由。
            if (not grid_map.is_free(x + step_x, y) or
                    not grid_map.is_free(x, y + step_y)):
                return False
            x += step_x
            y += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        if not grid_map.is_free(x, y):
            return False
    return True


def simplify_grid_path(grid_map, cells):
    """用可视直连贪心删除 A* 路径中的多余栅格点。"""
    cells = tuple(cells)
    if len(cells) <= 2:
        return cells
    result = [cells[0]]
    anchor = 0
    while anchor < len(cells) - 1:
        candidate = len(cells) - 1
        while candidate > anchor + 1:
            if line_is_free(grid_map, cells[anchor], cells[candidate]):
                break
            candidate -= 1
        result.append(cells[candidate])
        anchor = candidate
    return tuple(result)


def grid_path_to_waypoints(
        grid_map, cells, start_world, goal_world, goal_yaw,
        max_spacing=1.5):
    """把已简化栅格路径转换成现有控制器使用的 (x,y,yaw) 航点。"""
    if max_spacing <= 0.0:
        raise ValueError('max_waypoint_spacing 必须大于零。')
    if not cells:
        raise PlanningError('不能把空路径转换为航点。')

    # 先从真实位置走到起点格中心，最后从终点格中心走到精确目标。
    # 这样所有较长线段都与 simplify_grid_path 检查过的“格中心连线”一致；
    # 两端的短线只位于各自已经确认自由的起点/终点格内部。
    controls = [tuple(start_world)]
    for cell in cells:
        center = grid_map.grid_to_world(*cell)
        if math.hypot(
                center[0] - controls[-1][0],
                center[1] - controls[-1][1]) > 1e-9:
            controls.append(center)
    exact_goal = tuple(goal_world)
    if math.hypot(
            exact_goal[0] - controls[-1][0],
            exact_goal[1] - controls[-1][1]) > 1e-9:
        controls.append(exact_goal)
    else:
        controls[-1] = exact_goal
    if len(controls) == 1:
        return [(exact_goal[0], exact_goal[1], goal_yaw)]
    waypoints = []
    for segment_index in range(1, len(controls)):
        x0, y0 = controls[segment_index - 1]
        x1, y1 = controls[segment_index]
        distance = math.hypot(x1 - x0, y1 - y0)
        pieces = max(1, int(math.ceil(distance / max_spacing)))
        for piece in range(1, pieces + 1):
            fraction = piece / pieces
            x = x0 + fraction * (x1 - x0)
            y = y0 + fraction * (y1 - y0)
            is_final = segment_index == len(controls) - 1 and piece == pieces
            waypoints.append((x, y, goal_yaw if is_final else None))
    return waypoints


def path_length_world(waypoints, start_world):
    """计算控制航点折线的世界坐标长度。"""
    length = 0.0
    previous = tuple(start_world)
    for x, y, _ in waypoints:
        length += math.hypot(x - previous[0], y - previous[1])
        previous = (x, y)
    return length
