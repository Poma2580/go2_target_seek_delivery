from glob import glob

from setuptools import setup


package_name = "go2_target_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="bit",
    maintainer_email="tinsleybalmer@gmail.com",
    description="Multi-Go2 RGB-D pedestrian perception and role selection.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "target_perception = go2_target_perception.target_perception:main",
            "target_role_selector = go2_target_perception.target_role_selector:main",
            "perception_eval = go2_target_perception.perception_eval:main",
        ],
    },
)
