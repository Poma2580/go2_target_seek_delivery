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
            'walking_target_actor_controller = multi_go2_waypoint.walking_target_actor_controller:main',
            'gazebo_alignment_check = multi_go2_waypoint.gazebo_alignment_check:main',
            'dynamic_encircle = multi_go2_waypoint.dynamic_encircle:main',
            'marl_readonly_observer = multi_go2_waypoint.marl_readonly_observer:main',
            'marl_three_real_controller = multi_go2_waypoint.marl_three_real_controller:main',
            'maddpg_follower_slot_controller = multi_go2_waypoint.maddpg_follower_slot_controller:main',
            'gazebo_maddpg_reset = multi_go2_waypoint.gazebo_maddpg_reset:main',
            'gazebo_maddpg_train_stage1 = multi_go2_waypoint.gazebo_maddpg_train_stage1:main',
            'gazebo_leader_slot_train_stage1 = multi_go2_waypoint.gazebo_leader_slot_train_stage1:main',
            'gazebo_leader_slot_controller = multi_go2_waypoint.gazebo_leader_slot_controller:main',
            'cmd_odom_monitor = multi_go2_waypoint.cmd_odom_monitor:main',
            'target_perception = multi_go2_waypoint.target_perception:main',
            'perception_eval = multi_go2_waypoint.perception_eval:main',
        ],
    },
)
