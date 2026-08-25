"""Unit tests for the ROS-independent known-pose map merge core."""

import itertools

import pytest

from go2_mapping_nav.known_pose_map_merger import (
    GridData,
    classify_cell,
    merge_grids,
    validate_grid,
)


def grid(data, *, width=1, height=1, origin=(0.0, 0.0), resolution=1.0):
    return GridData(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=origin[0],
        origin_y=origin[1],
        origin_yaw=0.0,
        data=tuple(data),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, -1),
        (0, 0),
        (25, 0),
        (26, -1),
        (64, -1),
        (65, 100),
        (100, 100),
    ],
)
def test_classify_thresholds(value, expected):
    assert classify_cell(value, 25, 65) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((100, 100, -1), 100),
        ((100, 0, 0), 0),
        ((100, 0, -1), 100),
        ((0, 0, -1), 0),
        ((-1, -1, -1), -1),
        ((100, -1, -1), 100),
        ((0, -1, -1), 0),
    ],
)
def test_conflict_policy(values, expected):
    assert merge_grids(grid([value]) for value in values).data == (expected,)


def test_input_order_does_not_change_result():
    inputs = (grid([100]), grid([0]), grid([-1]))
    results = {
        merge_grids(permutation).data
        for permutation in itertools.permutations(inputs)
    }
    assert results == {(100,)}


def test_union_of_different_origins_and_sizes():
    left = grid([0, 100], width=2, origin=(-1.0, 0.0))
    right = grid([100, 0], width=2, origin=(1.0, 0.0))
    merged = merge_grids((left, right))
    assert (merged.origin_x, merged.origin_y) == (-1.0, 0.0)
    assert (merged.width, merged.height) == (4, 1)
    assert merged.data == (0, 100, 100, 0)


def test_expanded_input_expands_merged_map_and_keeps_old_cells():
    initial = merge_grids((grid([0, 100], width=2),))
    expanded = merge_grids((grid([0, 100, 0], width=3),))
    assert expanded.width == 3
    assert expanded.data[:2] == initial.data


@pytest.mark.parametrize(
    "bad_grid",
    [
        grid([0], resolution=0.1),
        GridData(1.0, 1, 1, 0.0, 0.0, 0.01, (0,)),
        GridData(1.0, 2, 1, 0.0, 0.0, 0.0, (0,)),
    ],
)
def test_invalid_grid_is_rejected(bad_grid):
    if bad_grid.resolution == 0.1:
        with pytest.raises(ValueError, match="differs"):
            validate_grid(bad_grid, expected_resolution=0.05)
    elif bad_grid.origin_yaw:
        with pytest.raises(ValueError, match="origin yaw"):
            validate_grid(bad_grid)
    else:
        with pytest.raises(ValueError, match="data length"):
            validate_grid(bad_grid)


def test_one_or_two_maps_can_be_merged():
    assert merge_grids((grid([0]),)).data == (0,)
    assert merge_grids((grid([0]), grid([100]))).data == (100,)
