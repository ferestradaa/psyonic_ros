from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_launch_package = FindPackageShare("urdf_launch")

    urdf_display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([urdf_launch_package, "launch", "display.launch.py"])
        ),
        launch_arguments={
            "urdf_package": "ah_urdf",
            "urdf_package_path": "urdf/ability_hand_right_large.urdf",
        }.items(),
    )

    return LaunchDescription(   
        [
            DeclareLaunchArgument(
                "write_thread",
                default_value="False",
                description="Enable write thread",
                choices=["True", "False"],
            ),
            DeclareLaunchArgument(
                "port",
                default_value="",
                description="Serial Port",
            ),
            DeclareLaunchArgument(
                "baud_rate",
                default_value="0",
                description="Baud Rate",
            ),
            DeclareLaunchArgument(
                "hand_side",
                default_value="Right",
                description="Ability Hand Side",
                choices=["Right", "Left"],
            ),
            DeclareLaunchArgument(
                "js_publisher",
                default_value="True",
                description="Publish Joint States for Joint State Publisher",
                choices=["True", "False"],
            ),
            DeclareLaunchArgument(
                "simulated_hand",
                default_value="False",
                description="Simulated Ability Hand",
                choices=["True", "False"],
            ),
            Node(
                package="ah_ros_py",
                executable="ah_node",
                name="ah_node",
                output="screen",
                parameters=[
                    {"write_thread": LaunchConfiguration("write_thread")},
                    {"port": LaunchConfiguration("port")},
                    {"baud_rate": LaunchConfiguration("baud_rate")},
                    {"js_publisher": LaunchConfiguration("js_publisher")},
                    {"simulated_hand": LaunchConfiguration("simulated_hand")},
                ],
            ),
            Node(
                package="ah_ros_py",
                executable="fsr_marker_node",
                name="fsr_marker_node",
                output="screen",
            ),
            urdf_display_launch,
        ]
    )