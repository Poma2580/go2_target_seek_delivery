"""Camera-projection visibility using robot GT and camera TF."""

import math


def quaternion_conjugate_rotate(vector, quaternion):
    """Rotate a world vector by the inverse of quaternion (x, y, z, w)."""
    x, y, z, w = quaternion
    norm = x * x + y * y + z * z + w * w
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("robot quaternion is invalid")
    x, y, z, w = -x / math.sqrt(norm), -y / math.sqrt(norm), -z / math.sqrt(norm), w / math.sqrt(norm)
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def project_camera_point(point, camera_info, min_depth, max_depth):
    """Return visibility and projected pixel for a point in optical frame."""
    x, y, z = point
    if not all(math.isfinite(value) for value in point):
        raise ValueError("camera point must be finite")
    if z < min_depth or z > max_depth:
        return False, None
    fx, fy, cx, cy = camera_info
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    u = fx * x / z + cx
    v = fy * y / z + cy
    width, height = camera_info.width, camera_info.height
    return 0.0 <= u < width and 0.0 <= v < height, (u, v)


class CameraIntrinsics(tuple):
    def __new__(cls, fx, fy, cx, cy, width, height):
        obj = super().__new__(cls, (float(fx), float(fy), float(cx), float(cy)))
        obj.width = int(width)
        obj.height = int(height)
        return obj
