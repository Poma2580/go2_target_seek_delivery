from setuptools import setup


package_name = "go2_dynamic_encircle"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="bit",
    maintainer_email="tinsleybalmer@gmail.com",
    description="Nav2 dynamic encirclement and MADDPG handoff for multiple Go2 robots.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dynamic_encircle = go2_dynamic_encircle.main:main",
            "follower_cmd_vel_mux = go2_dynamic_encircle.follower_cmd_vel_mux:main",
            "gazebo_leader_slot_controller = go2_dynamic_encircle.gazebo_leader_slot_controller:main",
        ],
    },
)
