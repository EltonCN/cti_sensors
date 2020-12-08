import rclpy
from rclpy.node import Node

import numpy as np
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, PoseStamped

from .er_ukf_imu_modules.attitude_computation import AttitudeComputation
from .er_ukf_imu_modules.error_compensation import GyroErrorCompensation
from .er_ukf_imu_modules.measurement_handler import MeasurementHandler
from .er_ukf_imu_modules.ukf import UKF

class UkfImuNode(Node):
    def __init__(self, gravity=9.78613):
        super().__init__("ukf_imu")

        self.imuSub = self.create_subscription(Imu, 'imu', self.imuCallback, 10)
        self.magSub = self.create_subscription(MagneticField, 'mag', self.magCallback, 10)
        self.publisher = self.create_publisher(Imu, "ukf_estimation", 10)
        self.tfPublisher = self.create_publisher(PoseStamped, 'ukf_tf_transform', 10)
        self.debugPublisher = self.create_publisher(Vector3,"ukf_debug",10)
        self.debugPublisher2 = self.create_publisher(Vector3,"ukf_debug2",10)

        self.accel = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.gravity = gravity
        self.header = Header()

        self.attitudeComputation = AttitudeComputation()
        self.gyroErrorCompensation = GyroErrorCompensation()
        self.measurementHandler = MeasurementHandler()

        self.ukf = UKF()

        timer_period = 0.016
        self.kalmanTimer = self.create_timer(timer_period, self.kalmanCallback)

        timer_period = 0.016
        self.orientationTimer = self.create_timer(timer_period, self.orientationCallback)

        self.kalmanTime = self.get_clock().now()
        self.orientationTime =self.get_clock().now()

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

        gyro = np.zeros(3,dtype=np.float32)
        gyro[0] = msg.angular_velocity.x
        gyro[1] = msg.angular_velocity.y
        gyro[2] = msg.angular_velocity.z

        self.gyroErrorCompensation.setMeasuredOmega(gyro)
        self.measurementHandler.setAccelRead(self.accel)

    
    def magCallback(self, msg):
        mag = np.zeros(3,dtype=np.float32)
        mag[0] = msg.magnetic_field.x
        mag[1] = msg.magnetic_field.y
        mag[2] = msg.magnetic_field.z

        self.measurementHandler.setMagRead(mag)

    def kalmanCallback(self):

        self.ukf.setMeasurement(self.measurementHandler.getMeasurement())
        self.ukf.setEstimateOmega(self.gyroErrorCompensation.getCorrectedOmega())
        self.ukf.setEstimateTheta(self.attitudeComputation.getTheta())

        deltaT = self.get_clock().now() - self.kalmanTime
        deltaT = float(deltaT.nanoseconds) * (10**-9)
        
        try:
            self.ukf.compute(deltaT)

            self.attitudeComputation.setThetaError(self.ukf.getThetaError())
            self.gyroErrorCompensation.setPredictedOmegaError(self.ukf.getOmegaError())

            self.kalmanTime = self.get_clock().now()
        except Exception as e:
            self._logger.error("Filtro gerou excecao: "+str(e)+". Reiniciando filtro")
            self.ukf = UKF()

    def orientationCallback(self):
        deltaT = self.get_clock().now() - self.orientationTime
        deltaT = float(deltaT.nanoseconds) * (10**-9)

        omega = self.gyroErrorCompensation.getCorrectedOmega()

        self.attitudeComputation.setOmega(omega)
        self.attitudeComputation.computeAll(deltaT)

        theta = self.attitudeComputation.getTheta()

        self.measurementHandler.setTheta(theta)

        degree = np.degrees(theta)

        self.publishDebug(np.degrees(self.measurementHandler.referenceOrientation))
        self.publishDebug(degree,2)

        self.publish(theta, omega)
        #self.publish(self.measurementHandler.referenceOrientation,omega)

        self.orientationTime =self.get_clock().now()
    
    def publish(self, theta, omega):
        
        msg = Imu()

        header = self.header
        timeStamp = self.get_clock().now().to_msg()
        header.stamp = timeStamp
        msg.header = header

        msg.angular_velocity.x = omega[0]*1
        msg.angular_velocity.y = omega[1]*1
        msg.angular_velocity.z = omega[2]*1

        gravityRotated = np.array([-self.gravity*np.sin(theta[1]), 
                                    self.gravity*np.cos(theta[1])*np.sin(theta[0]),
                                    self.gravity*np.cos(theta[1])*np.sin(theta[0]) ],
                                    dtype=np.float32)
        pureAcceleration = self.accel - gravityRotated
        msg.linear_acceleration.x = pureAcceleration[0]*1
        msg.linear_acceleration.y = pureAcceleration[1]*1
        msg.linear_acceleration.z = pureAcceleration[2]*1

        cov = np.zeros(9, dtype=np.float64)
        msg.angular_velocity_covariance = cov
        msg.orientation_covariance = cov
        msg.linear_acceleration_covariance = cov

        r = Rotation.from_euler('xyz',theta)
        quat = r.as_quat()
        msg.orientation.x = quat[0]*1
        msg.orientation.y = quat[1]*1
        msg.orientation.z = quat[2]*1
        msg.orientation.w = quat[3]*1

        self.publisher.publish(msg)

        msg2 = PoseStamped()

        msg2.header = header
        msg2.pose.orientation = msg.orientation
        msg2.pose.position.x = 0.0
        msg2.pose.position.y = 0.0
        msg2.pose.position.z = 0.0

        self.tfPublisher.publish(msg2)


def main(args=None):
    rclpy.init(args=args)

    node = UkfImuNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    executor.spin()

    #rclpy.spin(node)

    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()