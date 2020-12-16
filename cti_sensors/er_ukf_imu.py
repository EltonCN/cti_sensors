import rclpy
from rclpy.node import Node

import numpy as np
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, PoseStamped, Quaternion

from .er_ukf_imu_modules.attitude_computation import AttitudeComputation
from .er_ukf_imu_modules.error_compensation import GyroErrorCompensation
from .er_ukf_imu_modules.measurement_handler import MeasurementHandler
from .er_ukf_imu_modules.er_ukf_imu import ErUkfImu

class UkfImuNode(Node):
    def __init__(self, gravity=9.78613):
        super().__init__("ukf_imu")

        self.imuSub = self.create_subscription(Imu, 'imu', self.imuCallback, 10)
        self.magSub = self.create_subscription(MagneticField, 'mag', self.magCallback, 10)
        self.publisher = self.create_publisher(Imu, "ukf_estimation", 10)

        self.accel = np.array([0.0,0.0,0.0], dtype=np.float64)
        self.gravity = gravity
        self.header = Header()

        self.attitudeComputation = AttitudeComputation()
        self.gyroErrorCompensation = GyroErrorCompensation()
        self.measurementHandler = MeasurementHandler()
        self.ukf = ErUkfImu()

        self.kalmanTime = np.ones(2,dtype=np.float64)*-1
        self.orientationTime = np.ones(2,dtype=np.float64)*-1

        timer_period = 1e-3#0.016
        self.kalmanTimer = self.create_timer(timer_period, self.kalmanCallback)

        timer_period = 1e-3#0.016
        self.orientationTimer = self.create_timer(timer_period, self.orientationCallback)

        self.debugTimer = self.create_timer(timer_period, self.debugCallback)

        self.debugPublisher = []
        self.debugPublisher.append(self.create_publisher(PoseStamped, '~/estimate_pose', 10))
        self.debugPublisher.append(self.create_publisher(PoseStamped, '~/reference_pose', 10))
        self.debugPublisher.append(self.create_publisher(PoseStamped, '~/without_correction', 10))
        self.debugPublisher.append(self.create_publisher(Vector3, '~/estimate', 10))
        self.debugPublisher.append(self.create_publisher(Vector3, '~/reference', 10))

    def publishDebug(self, vec, channel=1):
        msg = Vector3()
        msg.x = vec[0]*1
        msg.y = vec[1]*1
        msg.z = vec[2]*1

        if channel ==1:
            self.debugPublisher.publish(msg)
        elif channel == 2:
            self.debugPublisher2.publish(msg)

    def imuCallback(self, msg):
        self.accel[0] = msg.linear_acceleration.x
        self.accel[1] = msg.linear_acceleration.y
        self.accel[2] = msg.linear_acceleration.z

        self.header = msg.header

        gyro = np.zeros(3,dtype=np.float64)
        gyro[0] = msg.angular_velocity.x
        gyro[1] = msg.angular_velocity.y
        gyro[2] = msg.angular_velocity.z

        self.gyroErrorCompensation.setMeasuredOmega(gyro)
        self.measurementHandler.setAccelRead(self.accel)


        time = float(msg.header.stamp.sec)+(float(msg.header.stamp.nanosec)*(10**-9))

        self.orientationTime[0] = time

        self.kalmanTime[0] = time

        if self.orientationTime[1] == -1:
            self.orientationTime[1] = time
            self.kalmanTime[1] = time
    
    def magCallback(self, msg):
        mag = np.zeros(3,dtype=np.float64)
        mag[0] = msg.magnetic_field.x
        mag[1] = msg.magnetic_field.y
        mag[2] = msg.magnetic_field.z

        self.measurementHandler.setMagRead(mag)

    def kalmanCallback(self):
        
        if self.kalmanTime[0] == -1:
            return

        self.ukf.setMeasurement(self.measurementHandler.getErrorMeasurement())
        self.ukf.setEstimateOmega(self.gyroErrorCompensation.getCorrectedOmega())
        self.ukf.setEstimateTheta(self.attitudeComputation.getTheta())

        deltaT = self.kalmanTime[0] - self.kalmanTime[1]
        
        try:
            self.ukf.compute(deltaT)

            self.attitudeComputation.setThetaError(self.ukf.getThetaError())
            self.gyroErrorCompensation.setPredictedOmegaError(self.ukf.getOmegaError())

            self.kalmanTime[1] = self.kalmanTime[0]
            self.kalmanTime[0] = -1
        
        except Exception as e:
            self._logger.error("Filtro gerou excecao: "+str(e)+". Reiniciando filtro")
            self.ukf = ErUkfImu()

    def orientationCallback(self):

        if self.orientationTime[0] == -1:
            return

        deltaT = self.orientationTime[0] - self.orientationTime[1]

        omega = self.gyroErrorCompensation.getCorrectedOmega()

        self.attitudeComputation.setOmega(omega)
        self.attitudeComputation.computeAll(deltaT)

        theta = self.attitudeComputation.getTheta()

        self.measurementHandler.setTheta(theta)


        self.publish(theta, omega)
       
        self.orientationTime[1] = self.orientationTime[0]
        self.orientationTime[0] = -1
    
    def publish(self, theta, omega):
        
        msg = Imu()

        msg.header = self.header

        msg.angular_velocity.x = omega[0]*1
        msg.angular_velocity.y = omega[1]*1
        msg.angular_velocity.z = omega[2]*1

        gravityRotated = np.array([-self.gravity*np.sin(theta[1]), 
                                    self.gravity*np.cos(theta[1])*np.sin(theta[0]),
                                    self.gravity*np.cos(theta[1])*np.sin(theta[0]) ],
                                    dtype=np.float64)
        pureAcceleration = self.accel - gravityRotated
        msg.linear_acceleration.x = pureAcceleration[0]*1
        msg.linear_acceleration.y = pureAcceleration[1]*1
        msg.linear_acceleration.z = pureAcceleration[2]*1

        cov = np.zeros(9, dtype=np.float64)
        msg.angular_velocity_covariance = cov
        msg.orientation_covariance = cov
        msg.linear_acceleration_covariance = cov

        msg.orientation = self.thetaToOrientation(theta)

        self.publisher.publish(msg)

    def thetaToOrientation(self, theta):
        r = Rotation.from_euler('xyz',theta)
        quat = r.as_quat()

        orientation = Quaternion()

        orientation.x = quat[0]*1.0
        orientation.y = quat[1]*1.0
        orientation.z = quat[2]*1.0
        orientation.w = quat[3]*1.0

        return orientation

    def debugCallback(self):
        msg = PoseStamped()
        msg.header = self.header
        msg.pose.orientation = self.thetaToOrientation(self.attitudeComputation.getTheta())
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0

        self.debugPublisher[0].publish(msg)

        msg = PoseStamped()
        msg.header = self.header
        msg.pose.orientation = self.thetaToOrientation(self.measurementHandler.referenceOrientation)
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0

        self.debugPublisher[1].publish(msg)

        msg = PoseStamped()
        msg.header = self.header
        msg.pose.orientation = self.thetaToOrientation(self.attitudeComputation.computedTheta)
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.0

        self.debugPublisher[2].publish(msg)

        msg = Vector3()
        degree = np.degrees(self.attitudeComputation.getTheta())
        msg.x = degree[0]
        msg.x = degree[1]
        msg.x = degree[2]

        self.debugPublisher[3].publish(msg)

        msg = Vector3()
        degree = np.degrees(self.measurementHandler.referenceOrientation)
        msg.x = degree[0]
        msg.x = degree[1]
        msg.x = degree[2]

        self.debugPublisher[3].publish(msg)

def main(args=None):
    rclpy.init(args=args)

    node = UkfImuNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    #executor.spin()

    rclpy.spin(node)

    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()