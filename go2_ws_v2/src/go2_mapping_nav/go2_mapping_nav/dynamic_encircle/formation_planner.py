"""Stateful fixed-slot formation planning built on pure geometry helpers."""

import math

from .geometry import (
    assign_remaining_slots,
    encircle_reached,
    navigation_slots_with_heading,
    solve_encircle_points,
)
from .models import FormationPlan


class FormationPlanner:
    """Keep navigation robot slot indices fixed while target geometry moves."""

    def __init__(self, config, loop):
        """使用共享配置和闭合路线初始化固定槽位规划器。"""
        self.config = config
        self.loop = loop
        self._slot_indices = None

    @property
    def slot_indices(self):
        """Return a copy of the fixed assignment, if one has been created."""
        return None if self._slot_indices is None else dict(self._slot_indices)

    def update(self, target_xy, dog_poses, perception_dog, navigation_dogs):
        """Build current slots, retaining the first minimum-distance assignment."""
        target_x, target_y = target_xy
        start_angle = math.atan2(
            dog_poses[perception_dog][1] - target_y,
            dog_poses[perception_dog][0] - target_x,
        )
        _, _, route_heading = self.loop.project_with_heading(target_x, target_y)
        points = solve_encircle_points(
            target_x,
            target_y,
            self.config.formation_radius,
            len(self.config.robot_names),
            start_angle,
        )

        assignment_created = self._slot_indices is None
        if assignment_created:
            self._slot_indices = assign_remaining_slots(
                {name: dog_poses[name] for name in navigation_dogs},
                points,
            )

        slots = navigation_slots_with_heading(
            self._slot_indices,
            points,
            route_heading,
        )
        completed = encircle_reached(
            {name: dog_poses[name] for name in navigation_dogs},
            slots,
            self.config.success_tolerance,
            self.config.success_yaw_tolerance,
        )
        return FormationPlan(
            slots=slots,
            route_heading=route_heading,
            slot_indices=dict(self._slot_indices),
            completed=completed,
            assignment_created=assignment_created,
        )
