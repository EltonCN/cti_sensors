import rclpy
from rclpy.node import Node

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


from sensor_msgs.msg import Imu, MagneticField

class DatasetTest(Node):
    def __init__(self):
        super().__init__("dataset_test")

        self.estimationSub = self.create_subscription(Imu, 'ukf_estimation', self.estimationCallback, 10)

        self.imuPublish = self.create_publisher(Imu, "imu", 10)
        self.magPublish = self.create_publisher(MagneticField, "mag", 10)

        self.timer_period = (1495463203.30-1495463203.29)*(38274.0/37602)

        self.publishTimer = self.create_timer(self.timer_period, self.publishCallback)

        dataDirectory = "E:\\Datasets\\Oxford Inertial Odometry\\Oxford Inertial Odometry Dataset_2.0\\Oxford Inertial Odometry Dataset\\handheld\\data1\\syn"

        self.sensorFrame = pd.read_csv(dataDirectory+"\\imu1.csv")
        self.truthFrame = pd.read_csv(dataDirectory+"\\vi1.csv")

        self.sensorFrame.columns = ["Time", "attitude_roll", "attitude_pitch", "attitude_yaw",
                                    "rotation_rate_x","rotation_rate_y","rotation_rate_z",
                                    "gravity_x","gravity_y","gravity_z",
                                    "user_acc_x","user_acc_y","user_acc_z",
                                    "magnetic_field_x","magnetic_field_y","magnetic_field_z"]

        self.truthFrame.columns = ["Time",  "Header",  
                                    "translation.x", "translation.y", "translation.z", 
                                    "rotation.x", "rotation.y", "rotation.z", "rotation.w"]

        self.index = 0

    def estimationCallback(self, msg):
        truthOrientation = np.zeros(4, dtype=np.float32)
        truthOrientation[0] = self.truthFrame["rotation.x"][self.index]
        truthOrientation[1] = self.truthFrame["rotation.y"][self.index]
        truthOrientation[2] = self.truthFrame["rotation.z"][self.index]
        truthOrientation[3] = self.truthFrame["rotation.w"][self.index]

        estimateOrientation = np.zeros(4, dtype=np.float32)
        estimateOrientation[0] = msg.orientation.x
        estimateOrientation[1] = msg.orientation.y
        estimateOrientation[2] = msg.orientation.z
        estimateOrientation[3] = msg.orientation.w

        truthR = Rotation.from_quat(truthOrientation)
        estimateR = Rotation.from_quat(estimateOrientation)

        truthDegree = truthR.as_euler("xyz",degrees=True)
        estimateDegree = estimateR.as_euler("xyz", degrees=True)

        truthDegree[0] *= -1

        #print(str(truthDegree)+" "+str(estimateDegree)+" "+str(truthDegree-estimateDegree))

        print(str(self.index)+" "+str(truthDegree[0])+" "+str(estimateDegree[0])+" "+str(truthDegree[0]-estimateDegree[0]))

    def publishCallback(self):
        msg = Imu()

        covariance = np.zeros((3,3), dtype=np.float64)

        accel = -self.sensorFrame["gravity_x"][self.index]
        accel *= 9.80665
        msg.linear_acceleration.x = accel
        
        accel = -self.sensorFrame["gravity_y"][self.index]
        accel *= 9.80665
        msg.linear_acceleration.y = accel

        accel = -self.sensorFrame["gravity_z"][self.index]
        accel *= 9.80665
        msg.linear_acceleration.z = accel

        msg.linear_acceleration_covariance = covariance

        msg.angular_velocity.x = self.sensorFrame["rotation_rate_x"][self.index]
        msg.angular_velocity.y = self.sensorFrame["rotation_rate_y"][self.index]
        msg.angular_velocity.z = self.sensorFrame["rotation_rate_z"][self.index]
        msg.angular_velocity_covariance = covariance

        msg.orientation.x = self.truthFrame["rotation.x"][self.index]
        msg.orientation.y = self.truthFrame["rotation.y"][self.index]
        msg.orientation.z = self.truthFrame["rotation.z"][self.index]
        msg.orientation.w = self.truthFrame["rotation.w"][self.index]

        msg.orientation_covariance = covariance

        timeSec = self.timer_period*float(self.index)
        msg.header.stamp.sec = int(timeSec)
        msg.header.stamp.nanosec = int((timeSec*1000000000)%1000000000)

        self.imuPublish.publish(msg)

        msg = MagneticField()

        msg.magnetic_field.x = self.sensorFrame["magnetic_field_x"][self.index]*0.000001
        msg.magnetic_field.y = self.sensorFrame["magnetic_field_y"][self.index]*0.000001
        msg.magnetic_field.z = self.sensorFrame["magnetic_field_z"][self.index]*0.000001 

        msg.magnetic_field_covariance = covariance

        msg.header.stamp.sec = int(timeSec)
        msg.header.stamp.nanosec = int((timeSec*1000000000)%1000000000)

        self.magPublish.publish(msg)

        self.index += 1

def main(args=None):
    rclpy.init(args=args)

    node = DatasetTest()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    executor.spin()

    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()