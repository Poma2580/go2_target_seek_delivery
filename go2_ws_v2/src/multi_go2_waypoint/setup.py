from glob import glob

from setuptools import setup

package_name = 'multi_go2_waypoint'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/maps', [
            'maps/airport.pgm',
            'maps/airport.yaml',
        ]),
        ('share/' + package_name + '/config/scenes',
         glob('config/scenes/*.yaml')),
        ('share/' + package_name + '/config/planner',
         glob('config/planner/*.yaml')),
        ('share/' + package_name + '/config/controller',
         glob('config/controller/*.yaml')),
        ('share/' + package_name + '/config/visualization',
         glob('config/visualization/*.yaml')),
        ('share/' + package_name + '/config/perception',
         glob('config/perception/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bit',
    maintainer_email='tinsleybalmer@gmail.com',
    description='多 Go2 联合 waypoint 围捕控制节点（三狗等边三角形围捕）',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_encircle = multi_go2_waypoint.waypoint_encircle:main',
            'actor_state_publisher = multi_go2_waypoint.actor_state_publisher:main',
            'dynamic_encircle = multi_go2_waypoint.dynamic_encircle:main',
            'target_perception = multi_go2_waypoint.target_perception:main',
            'target_role_selector = multi_go2_waypoint.target_role_selector:main',
            'perception_eval = multi_go2_waypoint.perception_eval:main',
            'astar_visualizer = multi_go2_waypoint.astar_visualizer:main',
            'gazebo_leader_slot_controller = multi_go2_waypoint.gazebo_leader_slot_controller:main',
        ],
    },
)
