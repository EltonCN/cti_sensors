import numpy as np
from numpy import sin, cos, tan

class AttitudeComputation():
    def __init__(self):
        self.computedTheta = np.array([0,0,0], dtype=np.float32)
        self.correctedTheta = np.array([0,0,0], dtype=np.float32)
        self.correctedOmega = np.array([0,0,0], dtype=np.float32)
        self.estimateThetaError = np.array([0,0,0], dtype=np.float32)
        

        self.calculated = True

    def computeWb(self):
        '''!
            Calcula a matriz Wb descrita na equação (6) de [3], que possibilita computar dtheta_dt
        '''

        wb = np.eye(3, dtype=np.float32)

        wb[0][1] = sin(self.computedTheta[0])*tan(self.computedTheta[1])
        wb[0][2] = cos(self.computedTheta[0])*tan(self.computedTheta[1])
        
        wb[1][1] = cos(self.computedTheta[0])
        wb[1][2] = -sin(self.computedTheta[0])

        wb[2][1] = sin(self.computedTheta[0])/cos(self.computedTheta[1])
        wb[2][2] = cos(self.computedTheta[0])/cos(self.computedTheta[1])

        return wb

    def computeDThetaDt(self):
        return self.computeWb() @ self.correctedOmega

    def computeTheta(self, deltaT):
        dThetaDt = self.computeDThetaDt()

        self.computedTheta = self.correctedTheta + (dThetaDt*deltaT)

    def correctTheta(self):
        '''!
            Calcula o vetor Theta corrigido

            @todo Verificar se se deve somar ou subtrair o estimateThetaError
        '''

        self.correctedTheta = self.computedTheta + self.estimateThetaError

    def computeAll(self, deltaT):
        self.computeTheta(deltaT)
        self.correctTheta()

        self.calculated = True

    def setThetaError(self, thetaError):
        self.estimateThetaError = thetaError

        self.calculated = False

    def setOmega(self, omega):
        self.correctedOmega = omega

        self.calculated = False
    
    def getTheta(self):

        return self.correctedTheta


