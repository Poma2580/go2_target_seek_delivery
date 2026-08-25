from glob import glob

from setuptools import setup


package_name = "go2_scenario_config"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/scenes", glob("config/scenes/*.yaml")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="bit",
    maintainer_email="tinsleybalmer@gmail.com",
    description="Shared scene configuration for multi-Go2 simulation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "check_three_go2_attitude = go2_scenario_config.check_three_go2_attitude:main",
        ],
    },
)
