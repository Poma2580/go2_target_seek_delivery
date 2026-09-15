"""Pure geometry helpers for three-robot dynamic encirclement."""

import itertools
import math


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lower, upper):
    """Clamp a numeric value to an inclusive interval."""
    return max(lower, min(upper, value))


def quaternion_to_yaw(quaternion):
    """Extract planar yaw from a quaternion-like object."""
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion_components(yaw):
    """Convert finite yaw to planar quaternion (z, w) components."""
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    return math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def navigation_dog_names(robot_names, perception_dog):
    """Return the two navigation robots in configured order."""
    names = tuple(robot_names)
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("robot_names must contain exactly three unique names")
    if perception_dog not in names:
        raise ValueError("perception dog must be present in robot_names")
    return tuple(name for name in names if name != perception_dog)


def solve_encircle_points(target_x, target_y, radius, num_dogs, start_angle):
    """Return uniformly spaced (x, y, yaw) slots around a target."""
    if num_dogs < 1:
        raise ValueError("num_dogs must be greater than zero")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("formation_radius must be finite and greater than zero")
    if not all(math.isfinite(value) for value in (target_x, target_y, start_angle)):
        raise ValueError("encircle geometry inputs must be finite")

    points = []
    for index in range(num_dogs):
        angle = normalize_angle(start_angle + 2.0 * math.pi * index / num_dogs)
        points.append(
            (
                target_x + radius * math.cos(angle),
                target_y + radius * math.sin(angle),
                normalize_angle(angle + math.pi),
            )
        )
    return points


def assign_remaining_slots(dog_positions, points):
    """Assign slots 1..N to navigation robots with minimum total distance."""
    names = tuple(dog_positions)
    candidate_indices = tuple(range(1, len(points)))
    if len(names) != len(candidate_indices):
        raise ValueError("navigation dog count must match remaining slot count")

    best_indices = None
    best_cost = float("inf")
    for permutation in itertools.permutations(candidate_indices):
        cost = sum(
            math.hypot(
                points[index][0] - dog_positions[name][0],
                points[index][1] - dog_positions[name][1],
            )
            for name, index in zip(names, permutation)
        )
        if cost < best_cost:
            best_cost = cost
            best_indices = permutation
    return dict(zip(names, best_indices))


def update_assigned_slots(slot_indices, points):
    """Update slot coordinates while retaining the initial assignment."""
    if any(index <= 0 or index >= len(points) for index in slot_indices.values()):
        raise ValueError("assigned slot index is outside the remaining points")
    return {name: points[index] for name, index in slot_indices.items()}


def navigation_slots_with_heading(slot_indices, points, heading):
    """Return assigned slot positions with one shared global heading."""
    if not math.isfinite(heading):
        raise ValueError("navigation slot heading must be finite")
    assigned = update_assigned_slots(slot_indices, points)
    return {
        name: (point[0], point[1], normalize_angle(heading))
        for name, point in assigned.items()
    }


def encircle_reached(dog_poses, assigned_points, position_tolerance, yaw_tolerance):
    """Return whether all navigation robots simultaneously meet both tolerances."""
    if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise ValueError("success_tolerance must be finite and greater than zero")
    if (
        not math.isfinite(yaw_tolerance)
        or yaw_tolerance <= 0.0
        or yaw_tolerance > math.pi
    ):
        raise ValueError(
            "success_yaw_tolerance must be finite, greater than zero, "
            "and no greater than pi"
        )
    if set(dog_poses) != set(assigned_points):
        raise ValueError("dog poses and assigned points must have matching names")
    return all(
        math.hypot(
            dog_poses[name][0] - assigned_points[name][0],
            dog_poses[name][1] - assigned_points[name][1],
        )
        <= position_tolerance
        and abs(normalize_angle(dog_poses[name][2] - assigned_points[name][2]))
        <= yaw_tolerance
        for name in dog_poses
    )


class Loop:
    """Closed polygon represented by cumulative arc length."""

    def __init__(self, corners):
        """预计算闭合多边形的边和累计弧长。"""
        self.corners = tuple(corners)
        self.edges = []
        self.cumulative = [0.0]
        for index, (x0, y0) in enumerate(self.corners):
            x1, y1 = self.corners[(index + 1) % len(self.corners)]
            length = math.hypot(x1 - x0, y1 - y0)
            self.edges.append((x0, y0, x1, y1, length))
            self.cumulative.append(self.cumulative[-1] + length)
        self.length = self.cumulative[-1]

    def project_with_heading(self, px, py):
        """Return nearest arc length, edge index, and directed edge yaw."""
        best_s = 0.0
        best_index = 0
        best_heading = 0.0
        best_distance_squared = float("inf")
        for index, (x0, y0, x1, y1, length) in enumerate(self.edges):
            if length < 1e-9:
                continue
            fraction = (
                (px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)
            ) / (length * length)
            fraction = clamp(fraction, 0.0, 1.0)
            closest_x = x0 + fraction * (x1 - x0)
            closest_y = y0 + fraction * (y1 - y0)
            distance_squared = (px - closest_x) ** 2 + (py - closest_y) ** 2
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_s = self.cumulative[index] + fraction * length
                best_index = index
                best_heading = math.atan2(y1 - y0, x1 - x0)
        return best_s, best_index, best_heading

    def project(self, px, py):
        """Return only the nearest arc length for a point."""
        return self.project_with_heading(px, py)[0]

    def point_at(self, arc_length):
        """Return the planar point at a wrapped loop arc length."""
        arc_length %= self.length
        for index, (x0, y0, x1, y1, length) in enumerate(self.edges):
            if arc_length <= self.cumulative[index + 1] or index == len(self.edges) - 1:
                fraction = (
                    (arc_length - self.cumulative[index]) / length
                    if length > 1e-9
                    else 0.0
                )
                return (
                    x0 + fraction * (x1 - x0),
                    y0 + fraction * (y1 - y0),
                )
        return self.corners[0]

    def signed_arc(self, start, end):
        """Return the shortest signed loop distance from start to end."""
        distance = (end - start) % self.length
        if distance > self.length / 2.0:
            distance -= self.length
        return distance
