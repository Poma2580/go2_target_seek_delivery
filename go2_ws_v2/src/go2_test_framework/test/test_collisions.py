import json

import pytest

from go2_test_framework.recorders.collisions import (
    CollisionDebouncer, parse_body_contacts,
)


TRUNK = "go2_1::trunk::collision"
GAZEBO_TRUNK = (
    "go2_3::base_link::base_link_fixed_joint_lump__trunk_collision"
)
BOX = "phase6_test_box::link::collision"


def test_body_contact_parser_is_strict():
    stamp, contacts = parse_body_contacts(json.dumps({
        "schema_version": 1,
        "stamp": {"sec": 12, "nanosec": 34},
        "contacts": [{"collision1": TRUNK, "collision2": BOX}],
    }))
    assert stamp == {"sec": 12, "nanosec": 34}
    assert contacts == [(TRUNK, BOX)]
    with pytest.raises(ValueError, match="unexpected fields"):
        parse_body_contacts(json.dumps({
            "schema_version": 1, "stamp": {"sec": 1, "nanosec": 0},
            "contacts": [], "extra": True,
        }))


def test_contact_active_set_debounces_and_rearms_after_separation():
    debouncer = CollisionDebouncer()
    assert len(debouncer.update([(TRUNK, BOX)])) == 1
    assert debouncer.update([(BOX, TRUNK)]) == []
    assert debouncer.update([]) == []
    assert len(debouncer.update([(TRUNK, BOX)])) == 1


def test_contact_filter_ignores_non_static_test_targets():
    debouncer = CollisionDebouncer()
    assert debouncer.update([(TRUNK, "ground_plane::link::collision")]) == []
    assert debouncer.update([(TRUNK, "go2_2::trunk::collision")]) == []
    assert debouncer.update([(TRUNK, "go2_1::FL_calf::collision")]) == []
    assert debouncer.update([(TRUNK, "walking_target::link::collision")]) == []


def test_contact_event_retains_actual_scoped_collision_names():
    event = CollisionDebouncer().update([(BOX, TRUNK)])[0]
    assert event == {
        "robot": "go2_1",
        "robot_collision": TRUNK,
        "object_model": "phase6_test_box",
        "object_collision": BOX,
    }


def test_actual_gazebo_fixed_joint_lump_name_is_recognized():
    event = CollisionDebouncer().update([(BOX, GAZEBO_TRUNK)])[0]
    assert event["robot"] == "go2_3"
    assert event["robot_collision"] == GAZEBO_TRUNK
