import os
from setuptools import setup

package_name = 'target_search'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            ['config/camera_params.yaml']),
        (os.path.join('share', package_name, 'launch'),
            ['launch/target_search_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Target search package for Go2 robot using YOLO and RGB-D camera',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_detector = target_search.target_detector:main',
        ],
    },
)