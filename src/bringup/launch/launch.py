from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    rosbridge = ExecuteProcess(
        cmd=[
            'ros2', 'launch',
            'rosbridge_server',
            'rosbridge_websocket_launch.xml'
        ],
        output='screen'
    )

    delayed_rosbridge = TimerAction(
        period=3.0,
        actions=[rosbridge]
    )

    vision_node = Node(
        package='vision',
        executable='vision_node',
        name='vision_node',
        output='screen',
    )

    ultrasonic_fuser_node = Node(
        package='ultrasonic',
        executable='ultrasonic_fuser_node',
        name='ultrasonic_fuser_node',
        output='screen',
    )

    use_dummy_ultrasonic = LaunchConfiguration('use_dummy_ultrasonic')

    declare_use_dummy_arg = DeclareLaunchArgument(
        'use_dummy_ultrasonic',
        default_value='false',
        description='Use dummy ultrasonic node instead of GPIO ultrasonic node'
    )

    gpio_ultrasonic_node = Node(
        package='ultrasonic',
        executable='gpio_ultrasonic_node',
        name='gpio_ultrasonic_node',
        output='screen',
        condition=UnlessCondition(use_dummy_ultrasonic)
    )

    dummy_ultrasonic_node = Node(
        package='ultrasonic',
        executable='dummy_ultrasonic_node',
        name='dummy_ultrasonic_node',
        output='screen',
        condition=IfCondition(use_dummy_ultrasonic)
    )

    waypoint_node = Node(
        package='navigation',
        executable='waypoint_node',
        name='waypoint_node',
        output='screen',
    )

    arbiter_node = Node(
        package='navigation',
        executable='arbiter_node',
        name='arbiter_node',
        output='screen',
    )

    gpio_bridge_node = Node(
        package='vision',
        executable='gpio_bridge_node',
        name='gpio_bridge_node',
        output='screen',
    )

    music_player_node = Node(
        package='vision',
        executable='music_player_node',
        name='music_player_node',
        output='screen',
    )

    vel_to_dist_node = Node(
        package='navigation',
        executable='vel_to_dist_node',
        name='vel_to_dist_node',
        output='screen',
    )

    return LaunchDescription([

        delayed_rosbridge,

        vision_node,

        ultrasonic_fuser_node,

        declare_use_dummy_arg,
        gpio_ultrasonic_node,
        dummy_ultrasonic_node,
        
        waypoint_node,

        arbiter_node,

        gpio_bridge_node,

        music_player_node,

        vel_to_dist_node,
    ])