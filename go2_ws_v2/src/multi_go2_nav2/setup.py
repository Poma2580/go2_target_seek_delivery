from glob import glob
from setuptools import find_packages, setup


package_name = 'multi_go2_nav2'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/behavior_trees',
         glob('behavior_trees/*.xml')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/scenes',
         glob('config/scenes/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bit',
    maintainer_email='tinsleybalmer@gmail.com',
    description=(
        'Three namespaced Nav2 stacks sharing one map, with encirclement task '
        'coordination and RViz trajectory visualization.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_config_dump = multi_go2_nav2.scene_config_dump:main',
            'encircle_coordinator = multi_go2_nav2.encircle_coordinator:main',
            'nav2_bringup_sequencer = '
            'multi_go2_nav2.nav2_bringup_sequencer:main',
            'trajectory_recorder = multi_go2_nav2.trajectory_recorder:main',
        ],
    },
)
