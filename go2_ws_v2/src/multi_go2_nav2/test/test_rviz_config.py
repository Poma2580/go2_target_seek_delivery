from pathlib import Path

import yaml


RVIZ_FILE = Path(__file__).parents[1] / 'rviz' / 'multi_go2_nav2.rviz'


def test_selected_plans_are_enabled_and_internal_plans_are_disabled():
    config = yaml.safe_load(RVIZ_FILE.read_text())
    displays = config['Visualization Manager']['Displays']
    groups = {
        display.get('Name'): display
        for display in displays
        if display.get('Class') == 'rviz_common/Group'
    }

    selected = groups['Selected Assignment Paths']
    internal = groups['Nav2 Internal Last Plans']
    assert selected['Enabled'] is True
    assert internal['Enabled'] is False
    assert {
        display['Topic']['Value'] for display in selected['Displays']
    } == {
        '/go2_1/selected_plan',
        '/go2_2/selected_plan',
        '/go2_3/selected_plan',
    }
