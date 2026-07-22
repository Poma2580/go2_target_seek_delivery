"""Small shell-friendly reader for the shared scene configuration."""

import argparse
from pathlib import Path
import sys

from multi_go2_nav2.scene_config import ROBOT_NAMES, load_scene_config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--repository-root', required=True)
    parser.add_argument('--format', choices=('spawn-tsv',), default='spawn-tsv')
    args = parser.parse_args(argv)

    try:
        config = load_scene_config(args.config)
        world = config.resolve_world(Path(args.repository_root)).resolve()
        if not world.is_file():
            raise ValueError(f'world file does not exist: {world}')
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2

    # A stable, eval-free format consumed by start_three_go2_velodyne.sh.
    print(f'world\t{world}')
    for name in ROBOT_NAMES:
        pose = config.robots[name].spawn
        print(f'{name}\t{pose.x:g}\t{pose.y:g}\t{pose.z:g}\t{pose.yaw:g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
