from pathlib import Path
from setuptools import find_packages, setup

package_name = 'go2_static_perception_test'
resources = [(f'share/{package_name}', ['package.xml', 'README.md']),
             ('share/ament_index/resource_index/packages', ['resource/' + package_name])]
for directory in ('config', 'launch', 'worlds'):
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file():
            resources.append((f'share/{package_name}/{path.parent}', [str(path)]))

setup(
    name=package_name, version='0.2.0', packages=find_packages(),
    data_files=resources, install_requires=['setuptools'], tests_require=['pytest'],
    zip_safe=True, maintainer='bit', maintainer_email='tinsleybalmer@gmail.com',
    description='Isolated City world, static YOLOE perception, and smoke validation.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'static_yoloe_perception = go2_static_perception_test.static_yoloe_perception:main',
        'validate_static_world = go2_static_perception_test.validate_world:main',
        'static_test_runner = go2_static_perception_test.runner.main:main',
        'static_summary_preview = go2_static_perception_test.reporting.results:main',
        'static_perception_recorder = go2_static_perception_test.recorders.static_perception_recorder:main',
        'check_static_go2_attitude = go2_static_perception_test.runner.attitude:main',
    ]},
)
