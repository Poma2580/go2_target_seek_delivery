from glob import glob

from setuptools import find_packages, setup


package_name = "go2_test_framework"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config/parameters", glob("config/parameters/*.yaml")),
        ("share/" + package_name + "/config/suites", glob("config/suites/*.yaml")),
        ("share/" + package_name + "/config/metrics", glob("config/metrics/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="bit",
    maintainer_email="tinsleybalmer@gmail.com",
    description="Configuration-driven Go2 simulation test framework.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "generate_test_worlds = go2_test_framework.world_generator:main",
            "target_test_recorder = go2_test_framework.recorders.target_recorder:main",
            "tracking_test_recorder = go2_test_framework.recorders.tracking_recorder:main",
            "nav_chain_probe = go2_test_framework.recorders.nav_chain_probe:main",
            "path_planning_recorder = go2_test_framework.recorders.path_planning_recorder:main",
            "collision_probe_spawner = go2_test_framework.recorders.collision_probe_spawner:main",
            "target_test_runner = go2_test_framework.runner.main:main",
        ],
    },
)
