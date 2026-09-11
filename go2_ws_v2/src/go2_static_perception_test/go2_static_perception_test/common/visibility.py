"""Camera projection helpers matching the established T1 implementation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def quaternion_conjugate_rotate(vector, quaternion):
    x, y, z, w = quaternion
    norm = x*x + y*y + z*z + w*w
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("robot quaternion is invalid")
    scale = math.sqrt(norm)
    x, y, z, w = -x/scale, -y/scale, -z/scale, w/scale
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+(y*tz-z*ty), vy+w*ty+(z*tx-x*tz), vz+w*tz+(x*ty-y*tx))


def project_camera_point(point, intrinsics, min_depth, max_depth):
    x, y, z = point
    if not all(math.isfinite(value) for value in point):
        raise ValueError("camera point must be finite")
    if z < min_depth or z > max_depth:
        return False, None
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    return 0 <= u < intrinsics.width and 0 <= v < intrinsics.height, (u, v)
