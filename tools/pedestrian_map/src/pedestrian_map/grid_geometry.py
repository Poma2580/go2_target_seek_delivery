"""Shared world/grid coordinate and line traversal helpers."""

import math
from typing import Protocol


GRID_SNAP_EPSILON = 1e-9


class GridMap(Protocol):
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int


def snap_grid_coordinate(value: float) -> float:
    """Snap floating-point noise near an exact grid boundary."""
    nearest = round(value)
    if abs(value - nearest) <= GRID_SNAP_EPSILON:
        return float(nearest)
    return value


def supercover_cells(
    map_data: GridMap, start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[int, int]]:
    """Traverse every grid cell touched by a segment using 2-D DDA."""
    x0 = snap_grid_coordinate((start[0] - map_data.origin_x) / map_data.resolution)
    y0 = snap_grid_coordinate((start[1] - map_data.origin_y) / map_data.resolution)
    x1 = snap_grid_coordinate((end[0] - map_data.origin_x) / map_data.resolution)
    y1 = snap_grid_coordinate((end[1] - map_data.origin_y) / map_data.resolution)
    gx, gy = math.floor(x0), math.floor(y0)
    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    t_delta_x = abs(1.0 / dx) if dx else math.inf
    t_delta_y = abs(1.0 / dy) if dy else math.inf
    next_x = gx + 1 if step_x > 0 else gx
    next_y = gy + 1 if step_y > 0 else gy
    t_max_x = (next_x - x0) / dx if dx else math.inf
    t_max_y = (next_y - y0) / dy if dy else math.inf

    grid_cells: list[tuple[int, int]] = []

    def add(cell_x: int, cell_y: int) -> None:
        if 0 <= cell_x < map_data.width and 0 <= cell_y < map_data.height:
            item = (map_data.height - 1 - cell_y, cell_x)
            if not grid_cells or grid_cells[-1] != item:
                grid_cells.append(item)

    def add_endpoint_cells(grid_x: float, grid_y: float) -> None:
        columns = {math.floor(grid_x)}
        rows = {math.floor(grid_y)}
        if grid_x.is_integer():
            columns.add(int(grid_x) - 1)
        if grid_y.is_integer():
            rows.add(int(grid_y) - 1)
        for cell_y in sorted(rows):
            for cell_x in sorted(columns):
                add(cell_x, cell_y)

    add_endpoint_cells(x0, y0)
    max_steps = int(math.ceil(abs(dx)) + math.ceil(abs(dy)) + 4)
    for _ in range(max_steps):
        next_t = min(t_max_x, t_max_y)
        if next_t > 1.0 + GRID_SNAP_EPSILON:
            add_endpoint_cells(x1, y1)
            return grid_cells
        if math.isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-12):
            add(gx + step_x, gy)
            add(gx, gy + step_y)
            gx += step_x
            gy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            gx += step_x
            t_max_x += t_delta_x
        else:
            gy += step_y
            t_max_y += t_delta_y
        add(gx, gy)
    raise RuntimeError("grid traversal exceeded its parameter-bound safety limit")
