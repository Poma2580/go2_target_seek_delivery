import math

import pytest

from multi_go2_waypoint.grid_astar import (
    GridMap,
    PlanningError,
    astar,
    grid_path_to_waypoints,
    line_is_free,
    simplify_grid_path,
)


def test_load_map_world_grid_conversion_and_pgm_y_flip(tmp_path):
    # PGM 第一行是世界最高的 gy=2，障碍分别位于 (0,2) 和 (2,1)。
    pgm = tmp_path / 'tiny.pgm'
    pgm.write_bytes(
        b'P5\n# test map\n4 3\n255\n' +
        bytes([0, 254, 254, 254,
               254, 254, 0, 254,
               254, 254, 254, 254]))
    yaml = tmp_path / 'tiny.yaml'
    yaml.write_text(
        'image: tiny.pgm\n'
        'mode: trinary\n'
        'resolution: 0.5\n'
        'origin: [-1.0, -2.0, 0.0]\n'
        'negate: 0\n'
        'occupied_thresh: 0.65\n'
        'free_thresh: 0.25\n', encoding='utf-8')

    grid = GridMap.from_yaml(yaml)
    assert (grid.width, grid.height) == (4, 3)
    assert not grid.is_free(0, 2)
    assert not grid.is_free(2, 1)
    assert grid.is_free(0, 0)
    assert grid.world_to_grid(-0.75, -1.75) == (0, 0)
    assert grid.grid_to_world(3, 2) == pytest.approx((0.75, -0.75))
    with pytest.raises(PlanningError):
        grid.world_to_grid(-1.01, -1.0)


def test_circular_obstacle_inflation():
    blocked = bytearray(7 * 7)
    blocked[3 * 7 + 3] = 1
    grid = GridMap(7, 7, 1.0, 0.0, 0.0, blocked)
    inflated = grid.inflated(1.1)  # ceil(1.1 / 1.0) = 2 cells
    assert not inflated.is_free(3, 3)
    assert not inflated.is_free(5, 3)
    assert not inflated.is_free(4, 4)
    assert inflated.is_free(5, 5)
    assert grid.is_free(5, 3)  # 原地图没有被修改


def test_astar_detours_around_wall_and_is_deterministic():
    width = height = 10
    blocked = bytearray(width * height)
    for y in range(9):
        blocked[y * width + 4] = 1
    grid = GridMap(width, height, 1.0, 0.0, 0.0, blocked)
    first = astar(grid, (1, 1), (8, 1))
    second = astar(grid, (1, 1), (8, 1))
    assert first.cells == second.cells
    assert any(y == 9 for _, y in first.cells)
    assert all(grid.is_free(x, y) for x, y in first.cells)


def test_astar_forbids_diagonal_corner_cutting():
    blocked = bytearray(4)
    blocked[1] = 1       # (1, 0)
    blocked[2] = 1       # (0, 1)
    grid = GridMap(2, 2, 1.0, 0.0, 0.0, blocked)
    with pytest.raises(PlanningError):
        astar(grid, (0, 0), (1, 1))
    assert not line_is_free(grid, (0, 0), (1, 1))


def test_path_simplification_and_waypoint_spacing():
    grid = GridMap(20, 20, 1.0, 0.0, 0.0, bytearray(400))
    cells = tuple((index, index) for index in range(10))
    simplified = simplify_grid_path(grid, cells)
    assert simplified == ((0, 0), (9, 9))
    waypoints = grid_path_to_waypoints(
        grid, simplified, (0.5, 0.5), (9.5, 9.5), 1.2, max_spacing=2.0)
    previous = (0.5, 0.5)
    for x, y, yaw in waypoints:
        assert math.hypot(x - previous[0], y - previous[1]) <= 2.0 + 1e-9
        previous = (x, y)
    assert all(yaw is None for _, _, yaw in waypoints[:-1])
    assert waypoints[-1] == pytest.approx((9.5, 9.5, 1.2))
