import rclpy
from rclpy.node import Node

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import PoseStamped, Vector3

class DatasetTest(Node):
    def __init__(self):
        super().__init__("dataset_test")

        dataDirectory = "E:\\Datasets\\Oxford Inertial Odometry\\Oxford Inertial Odometry Dataset_2.0\\Oxford Inertial Odometry Dataset\\handheld\\data1\\raw"

        self.sensorFrame = pd.read_csv(dataDirectory+"\\imu1.csv")
        self.truthFrame = pd.read_csv(dataDirectory+"\\vi1.csv")

        self.sensorFrame.columns = ["time", "attitude_roll", "attitude_pitch", "attitude_yaw",
                                    "rotation_rate_x","rotation_rate_y","rotation_rate_z",
                                    "gravity_x","gravity_y","gravity_z",
                                    "user_acc_x","user_acc_y","user_acc_z",
                                    "magnetic_field_x","magnetic_field_y","magnetic_field_z"]

        self.truthFrame.columns = ["time",  "Header",  
                                    "translation.x", "translation.y", "translation.z", 
                                    "rotation.x", "rotation.y", "rotation.z", "rotation.w"]

        self.resultFrame = self.sensorFrame[["time"]]

        self.resultFrame = pd.DataFrame(columns=["time", "truth.x", "truth.y", "truth.z", 
                                                        "reference.x", "reference.y", "reference.z",
                                                        "estimation.x", "estimation.y", "estimation.z"])

        self.sensorTime = self.sensorFrame["time"][0]
        self.sensorIndex = 0

        self.dataPercent = 0.05

        self.estimationSub = self.create_subscription(PoseStamped, '/ukf_imu/estimate_pose', self.estimationCallback, 10)
        self.referenceSub = self.create_subscription(PoseStamped, '/ukf_imu/reference_pose', self.referenceCallback, 10)
        
        self.imuPublish = self.create_publisher(Imu, "imu", 10)
        self.magPublish = self.create_publisher(MagneticField, "mag", 10)

        self.timer_period = 1e-3
        self.info_period = 1

        self.publishTimer = self.create_timer(self.timer_period, self.publishCallback)
        self.infoTimer = self.create_timer(self.info_period, self.infoCallback)

    def infoCallback(self):
        if self.sensorIndex <= len(self.sensorFrame)*self.dataPercent:
            print(str(self.sensorIndex/len(self.sensorFrame)*100)+"%")

    def publishCallback(self):
        if self.sensorIndex >= len(self.sensorFrame)*self.dataPercent:
            print("Adicionando truth")
            self.addTruth()
            self.save()
            self.destroy_node()
            print("Terminou")
            return

        msg = Imu()

        covariance = np.zeros((3,3), dtype=np.float64)

        accel = -self.sensorFrame["gravity_x"][self.sensorIndex]+self.sensorFrame["user_acc_x"][self.sensorIndex]
        accel *= 9.80665
        msg.linear_acceleration.x = accel
        
        accel = -self.sensorFrame["gravity_y"][self.sensorIndex]+self.sensorFrame["user_acc_y"][self.sensorIndex]
        accel *= 9.80665
        msg.linear_acceleration.y = accel

        accel = -self.sensorFrame["gravity_z"][self.sensorIndex]+self.sensorFrame["user_acc_z"][self.sensorIndex]
        accel *= 9.80665
        msg.linear_acceleration.z = accel

        msg.linear_acceleration_covariance = covariance

        msg.angular_velocity.x = self.sensorFrame["rotation_rate_x"][self.sensorIndex]
        msg.angular_velocity.y = self.sensorFrame["rotation_rate_y"][self.sensorIndex]
        msg.angular_velocity.z = self.sensorFrame["rotation_rate_z"][self.sensorIndex]
        msg.angular_velocity_covariance = covariance

        msg.orientation.x = self.truthFrame["rotation.x"][self.sensorIndex]
        msg.orientation.y = self.truthFrame["rotation.y"][self.sensorIndex]
        msg.orientation.z = self.truthFrame["rotation.z"][self.sensorIndex]
        msg.orientation.w = self.truthFrame["rotation.w"][self.sensorIndex]

        msg.orientation_covariance = covariance

        self.sensorTime = self.sensorFrame["time"][self.sensorIndex]

        timeSec = float(self.sensorTime)
        msg.header.stamp.sec = int(timeSec)
        msg.header.stamp.nanosec = int((timeSec*1000000000)%1000000000)

        self.imuPublish.publish(msg)

        msg = MagneticField()

        msg.magnetic_field.x = self.sensorFrame["magnetic_field_x"][self.sensorIndex]*0.000001
        msg.magnetic_field.y = self.sensorFrame["magnetic_field_y"][self.sensorIndex]*0.000001
        msg.magnetic_field.z = self.sensorFrame["magnetic_field_z"][self.sensorIndex]*0.000001 

        msg.magnetic_field_covariance = covariance

        msg.header.stamp.sec = int(timeSec)
        msg.header.stamp.nanosec = int((timeSec*1000000000)%1000000000)

        self.magPublish.publish(msg)

        self.sensorIndex += 1

    def eulerFromOrientation(self, orientation):
        quat = np.zeros(4, dtype=np.float64)
        quat[0] = orientation.x
        quat[1] = orientation.y
        quat[2] = orientation.z
        quat[3] = orientation.w
        
        r = Rotation.from_quat(quat)
        return r.as_euler("xyz",degrees=True)

    def addTimeResult(self, time):
        a = self.resultFrame.isin([time])['time']
        columns = a.index[a].tolist()

        if len(columns) == 0:
            nan = np.nan
            self.resultFrame.loc[len(self.resultFrame)] = [time, nan,nan,nan, nan,nan,nan, nan,nan,nan]

    def estimationCallback(self, msg):
        if self.sensorIndex >= len(self.sensorFrame)*self.dataPercent:
            return

 
            
        time = float(msg.header.stamp.sec)+(float(msg.header.stamp.nanosec)*(10**-9))

        self.addTimeResult(time)

        euler = self.eulerFromOrientation(msg.pose.orientation)

        self.resultFrame.loc[self.resultFrame.time == time, 'estimation.x'] = euler[0]
        self.resultFrame.loc[self.resultFrame.time == time, 'estimation.y'] = euler[1]
        self.resultFrame.loc[self.resultFrame.time == time, 'estimation.z'] = euler[2]


    def referenceCallback(self, msg):
        if self.sensorIndex >= len(self.sensorFrame)*self.dataPercent:
            return


        time = float(msg.header.stamp.sec)+(float(msg.header.stamp.nanosec)*(10**-9))

        self.addTimeResult(time)

        euler = self.eulerFromOrientation(msg.pose.orientation)

        self.resultFrame.loc[self.resultFrame.time == time, 'reference.x'] = euler[0]
        self.resultFrame.loc[self.resultFrame.time == time, 'reference.y'] = euler[1]
        self.resultFrame.loc[self.resultFrame.time == time, 'reference.z'] = euler[2]



    def addTruth(self):
        truthIndex = 0

        self.resultFrame = self.resultFrame.sort_values("time")
        self.resultFrame = self.resultFrame.reset_index(drop=True)

        for i in range(len(self.resultFrame)):
            time = self.resultFrame['time'][i] * (10**9)

            while(self.truthFrame['time'][truthIndex] < time):
                truthIndex += 1

            if truthIndex == 0:
                truthIndex += 1

            quat = np.zeros(4, dtype=np.float64)

            if abs(self.truthFrame['time'][truthIndex]-time)<abs(self.truthFrame['time'][truthIndex-1]-time):
                quat[0] = self.truthFrame['rotation.x'][truthIndex]
                quat[1] = self.truthFrame['rotation.y'][truthIndex]
                quat[2] = self.truthFrame['rotation.z'][truthIndex]
                quat[3] = self.truthFrame['rotation.w'][truthIndex]
            else:
                quat[0] = self.truthFrame['rotation.x'][truthIndex-1]
                quat[1] = self.truthFrame['rotation.y'][truthIndex-1]
                quat[2] = self.truthFrame['rotation.z'][truthIndex-1]
                quat[3] = self.truthFrame['rotation.w'][truthIndex-1]

            r = Rotation.from_quat(quat)
            euler = r.as_euler("xyz",degrees=True)

            self.resultFrame['truth.x'][i] = -euler[0]
            self.resultFrame['truth.y'][i] = -euler[1]

            euler[2] += 90

            if euler[2] > 180:
                euler[2] -= 180

            self.resultFrame['truth.z'][i] = euler[2]

    def save(self):
        self.resultFrame.to_csv(r'result.csv')


def main(args=None):
    rclpy.init(args=args)

    node = DatasetTest()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    #executor.spin()
    rclpy.spin(node)

    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()