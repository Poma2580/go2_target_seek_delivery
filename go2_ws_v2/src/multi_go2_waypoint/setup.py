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
            'waypoint_forest = multi_go2_waypoint.waypoint_forest:main',
            'actor_state_publisher = multi_go2_waypoint.actor_state_publisher:main',
            'dynamic_encircle = multi_go2_waypoint.dynamic_encircle:main',
            'target_perception = multi_go2_waypoint.target_perception:main',
            'perception_eval = multi_go2_waypoint.perception_eval:main',
        ],
    },
)
