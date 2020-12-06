import numpy as np
from numpy import sin, cos, tan
from scipy.linalg import ldl

class UKF():
    def __init__(self):
        self.measurement = np.array([0,0,0], dtype=np.float32)


        I = np.eye(3, dtype=np.float32)
        O = np.zeros((3,3), dtype=np.float32)
        self.H = np.concatenate((I,O), axis=1)

        self.estimateTheta = np.zeros(3,dtype=np.float32)
        self.estimateOmega = np.zeros(3,dtype=np.float32)

        self.measurement = np.zeros(3,dtype=np.float32)

        self.estimateState = np.zeros(6, dtype=np.float32)
        self.stateCovariance = np.eye(6,dtype=np.float32) *1

        self.processNoise = np.eye(6,dtype=np.float32)*(72e-3) #Q
        self.measurementNoise = np.eye(3,dtype=np.float32)*(72e-3) #R

        self.N = 6.0
        self.k = 3.0-self.N


    def h(self, x):
        return self.H@x

    def computeWb(self, x):
        '''!
            Calcula a matriz Wb descrita na equação (6) de [3], que possibilita computar dtheta_dt
        '''

        wb = np.eye(3, dtype=np.float32)

        wb[0][1] = sin(x[0])*tan(x[1])
        wb[0][2] = cos(x[0])*tan(x[1])
        
        wb[1][1] = cos(x[0])
        wb[1][2] = -sin(x[0])

        wb[2][1] = sin(x[0])/cos(x[1])
        wb[2][2] = cos(x[0])/cos(x[1])

        return wb

    def computeVb(self, x):
        #equacao 11, Intertial Head-Tracker Sensor Fusion by a Complementary Separate-Bias Kalman Filter 
        psi = x[0]
        theta = x[1]
        phi = x[2]
        wX = x[3]
        wY = x[4]
        wZ = x[5]

        vB = np.zeros((3,3), dtype=np.float32)

        vB[0][0] = ((cos(psi)*sin(theta)*wY)/cos(theta)) - ((sin(psi)*sin(theta)*wZ)/cos(theta))
        vB[0][1] = ((sin(psi)*wY)/np.power(cos(theta), 2)) + ((cos(psi)*wZ)/np.power(cos(theta), 2))

        vB[1][0] = -(sin(psi)*wY) - (cos(psi)*wZ)

        vB[2][0] = ((cos(psi)*wY)/cos(theta)) - ((sin(psi)*wZ)/cos(theta))
        vB[2][1] = ((sin(psi)*sin(theta)*wY)/np.power(cos(theta),2))+((cos(psi)*sin(theta)*wZ)/np.power(cos(theta),2))

        return vB

    def f(self, x, w2, deltaT):
        wB = self.computeWb(x)
        vB = self.computeVb(x)


        theta = x[0:3]
        w = x[3:6]

        theta2 = theta
        theta2 += ((w+w2)/2)*deltaT

        theta2 += ((vB@wB)@w)*np.power(deltaT,2)

        x2 = np.hstack((theta2, w2))
        
        return x2

    def setEstimateTheta(self, theta):
        self.estimateTheta = theta

    def setEstimateOmega(self, omega):
        self.estimateOmega = omega

    def setMeasurement(self, measurement):
        self.measurement = measurement

    def getState(self):
        return self.estimateState

    def getOmegaError(self):
        return self.estimateState[3:6]

    def getThetaError(self):
        return self.estimateState[:3]

    def computeSigmaPoint(self, state, covariance):

        L = np.linalg.cholesky(covariance)

        sigmaPoint = np.zeros((13,6), dtype=np.float32)

        sigmaPoint[0] = state

        for i in range(6):
            sigmaPoint[i+1] = state + (np.sqrt(self.N+self.k)*L[:,i])

        for i in range(6):
            sigmaPoint[i+7] = state - (np.sqrt(self.N+self.k)*L[:,i])

        return sigmaPoint

    def compute(self, deltaT):
        previousState = self.estimateState
        previousCovariance = self.stateCovariance

        ##Predicao##

        sigmaPoint = self.computeSigmaPoint(previousState, previousCovariance)
        
        #Propaga os sigma points
        predictedStateSigma = np.zeros((13,6), dtype=np.float32)

        for i in range(13):
            predictedStateSigma[i] = self.f(sigmaPoint[i],sigmaPoint[i][3:6],deltaT)

        #Calcula a predicao da media e covariancia
        
        predictedState = np.zeros(6,dtype=np.float32)
        predictedCovariance = np.zeros((6,6),dtype=np.float32)
        
        for i in range(13):
            alpha = self.computeAlpha(i)

            predictedState += alpha*predictedStateSigma[i]
        
        for i in range(13):
            alpha = self.computeAlpha(i)

            a = (predictedStateSigma[i]-predictedState)
            b = a

            a.shape = (6,1)
            b.shape = (6,1)

            predictedCovariance += alpha*(a@b.T)

        predictedCovariance += self.processNoise

        ##Correcao##

        #Propagar sigmaPoint para a medicao
        predictedMeasureSigma = np.zeros((13,3), dtype=np.float32)

        for i in range(13):
            predictedMeasureSigma[i] = self.h(predictedStateSigma[i])

        #Estimar media e covariancia das prediçoes das medicoes

        predictedMeasure = np.zeros(3,dtype = np.float32)
        predictedMeasureCov = np.zeros((3,3), dtype=np.float32)

        for i in range(13):
            alpha = self.computeAlpha(i)

            predictedMeasure += alpha*predictedMeasureSigma[i]

        for i in range(13):
            alpha = self.computeAlpha(i)

            a = (predictedMeasureSigma[i]-predictedMeasure)
            b = a

            a.shape = (3,1)
            b.shape = (3,1)

            predictedMeasureCov += alpha*(a@b.T)

        predictedMeasureCov += self.measurementNoise

        #Calcular cross-covariance e Kalman gain
        crossCovariance = np.zeros((6,3), dtype=np.float32)

        for i in range(13):
            alpha = self.computeAlpha(i)

            a = (predictedStateSigma[i]-predictedState)
            b = (predictedMeasureSigma[i]-predictedMeasure)

            a.shape = (6,1)
            b.shape = (3,1)

            crossCovariance += alpha*(a@b.T)
        
        kalmanGain = crossCovariance @ np.linalg.inv(predictedMeasureCov)

        #Corrigir media e covariancia
        self.estimateState = predictedState + (kalmanGain@(self.measurement-predictedMeasure))

        self.stateCovariance = predictedCovariance - ((kalmanGain@predictedMeasureCov)@kalmanGain.T)

    def computeAlpha(self, i):
        if(i == 0):
            return self.k/(self.N+self.k)
        else:
            return 0.5*(1/(self.N+self.k))



