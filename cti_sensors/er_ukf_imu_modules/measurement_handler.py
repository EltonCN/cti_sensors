import numpy as np
from numpy import arctan2, arcsin, cos, sin

class MeasurementHandler():
    def __init__(self, magneticIntensity=22902.5e-9, inclination=-39.2538, gravity=9.78613):
        self.referenceOrientation = np.array([0,0,0], dtype=np.float64)
        self.measurement = np.array([0,0,0], dtype=np.float64)
        self.accel = np.array([0,0,0], dtype=np.float64)
        self.mag = np.array([0,0,0], dtype=np.float64)

        self.correctedTheta = np.array([0,0,0], dtype=np.float64)

        self.calculated = True

        self.magneticIntensity = magneticIntensity
        self.inclination = np.radians(inclination)

        ''' Oxford
        self.magneticIntensity = 48956.6e-9
        self.inclination = np.radians(66.6442)'''

        self.gravity = gravity

    def setMagneticIntensity(self, magneticIntensity):
        self.magneticIntensity = magneticIntensity
    
    def setInclination(self, inclination):
        self.inclination = np.radians(inclination)

    def setGravity(self, gravity):
        self.gravity = gravity

    def setTheta(self, theta):
        self.correctedTheta = theta

        self.calculated = False

    def setAccelRead(self, accel):
        self.accel = accel

        self.calculated = False
    
    def setMagRead(self, mag):
        self.mag = mag

        self.calculated = False

    def computeReference(self):

        #phi = 0, theta = 1, psi = 2
        phi = arctan2(self.accel[1], self.accel[2])

        if self.accel[0] >= self.gravity:
            self.accel[0] = self.gravity
        elif self.accel[0] <= -self.gravity:
            self.accel[0] = -self.gravity

        thetaArc = -self.accel[0]/self.gravity

        if(thetaArc <= -1):
            thetaArc = -1
        elif(thetaArc >= 1):
            thetaArc = 1

        theta = arcsin(thetaArc)

        a = cos(theta)*((self.mag[2]*sin(phi))-(self.mag[1]*cos(phi)))
        b = self.mag[0] + (self.magneticIntensity*sin(self.inclination)*sin(theta))

        psi = arctan2(a, b)

        self.referenceOrientation[0] = phi
        self.referenceOrientation[1] = theta
        self.referenceOrientation[2] = psi

        for i in range(3):
            if self.referenceOrientation[i] > np.pi:
                self.referenceOrientation[i] = np.pi
            elif self.referenceOrientation[i] < -np.pi:
                self.referenceOrientation[i] = -np.pi



    def compute(self):
        self.computeReference()

        self.measurement = self.referenceOrientation - self.correctedTheta

        for i in range(3):
            if self.measurement[i] > np.pi:
                self.measurement[i] = np.pi
            elif self.measurement[i] < -np.pi:
                self.measurement[i] = -np.pi

        self.calculated = True

    def getMeasurement(self):
        if self.calculated == False:
            self.compute()

        return self.measurement
    


