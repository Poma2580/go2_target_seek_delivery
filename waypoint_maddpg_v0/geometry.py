"""Small dependency-free 2-D geometry helpers."""

import math

import numpy as np


def rotate(vector, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = float(vector[0]), float(vector[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def world_to_body(point, origin, yaw):
    return rotate(np.asarray(point, dtype=np.float32) - origin, -yaw)


def point_aabb_clearance(point, lower, upper):
    """Unsigned point-to-axis-aligned-box distance (zero inside the box)."""
    point = np.asarray(point, dtype=np.float32)
    delta = np.maximum(np.maximum(lower - point, 0.0), point - upper)
    return float(np.linalg.norm(delta))


def circle_aabb_clearance(center, radius, lower, upper):
    return point_aabb_clearance(center, lower, upper) - float(radius)


def ray_aabb_distance(origin, direction, lower, upper, max_range):
    """Return first ray/AABB intersection distance or max_range."""
    t_min, t_max = 0.0, float(max_range)
    for axis in range(2):
        if abs(float(direction[axis])) < 1e-9:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return float(max_range)
            continue
        inv = 1.0 / float(direction[axis])
        t1 = (float(lower[axis]) - float(origin[axis])) * inv
        t2 = (float(upper[axis]) - float(origin[axis])) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return float(max_range)
    return float(t_min) if 0.0 <= t_min <= max_range else float(max_range)


def ray_circle_distance(origin, direction, center, radius, max_range):
    offset = origin - center
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset) - radius * radius)
    disc = b * b - c
    if disc < 0.0:
        return float(max_range)
    root = math.sqrt(disc)
    first, second = -b - root, -b + root
    if first >= 0.0:
        distance = first
    elif second >= 0.0:
        distance = second
    else:
        return float(max_range)
    return float(distance) if distance <= max_range else float(max_range)


def point_segment_distance(point, start, end):
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _orientation(a, b, c):
    return float(np.cross(b - a, c - a))


def segments_intersect(a, b, c, d):
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return o1 * o2 <= 0.0 and o3 * o4 <= 0.0


def segment_segment_distance(a, b, c, d):
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )
