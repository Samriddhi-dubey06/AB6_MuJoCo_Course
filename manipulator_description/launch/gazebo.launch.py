import os
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_name = 'manipulator_description'
    pkg_dir = get_package_share_directory(pkg_name)

    # 1. FIX NETWORK ERROR: Force Gazebo to use localhost
    set_gz_ip = SetEnvironmentVariable(name='GZ_IP', value='127.0.0.1')

    # 2. RESOURCE PATH: Tell Gazebo where to find your meshes
    install_dir = os.path.join(pkg_dir, '..')
    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        gz_resource_path = install_dir + ':' + os.environ['GZ_SIM_RESOURCE_PATH']
    else:
        gz_resource_path = install_dir
    set_resource_path = SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=gz_resource_path)

    # Paths
    urdf_file = os.path.join(pkg_dir, 'urdf', 'final_assembly.urdf')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'default.rviz')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    # Configs
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock')
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Start RViz')

    # Parse URDF
    robot_description = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    # Nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_description}]
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # Spawn Robot
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'manipulator',
            '-x', '0.0', '-y', '0.0', '-z', '0.0' # Z is 0 now because we anchored it
        ],
        output='screen'
    )

    # 3. BRIDGE: Added /tf mapping to fix RViz errors
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen'
    )

    # RViz
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    # ... (keep all your existing code)

    # 1. Spawn the Joint State Broadcaster
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    # 2. Spawn the Trajectory Controller (Wait for broadcaster to finish starting)
    joint_trajectory_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
        output="screen",
    )

    # Ensure the controller starts AFTER the robot is spawned
    delay_controller_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[joint_state_broadcaster, joint_trajectory_controller],
        )
    )

    # Add 'delay_controller_spawn' to your return list instead of adding the nodes directly
    return LaunchDescription([
        set_gz_ip,
        set_resource_path,
        declare_use_sim_time,
        declare_use_rviz,
        gazebo,
        robot_state_publisher,
        spawn,
        bridge,
        rviz,
        delay_controller_spawn  # <--- ADD THIS
    ])