from launch import LaunchDescription
from launch_ros.actions import Node

#Alterar os valores dos parâmetros para o local de uso


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cti_sensors",
            executable="er_ukf_imu",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"gravity": 1.0},
                {"magneticIntensity": 1.0},
                {"inclination": 1.0},
            ]
        )
    ])