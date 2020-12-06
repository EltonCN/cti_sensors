import numpy as np

class GyroErrorCompensation():
    def __init__(self):
        self.correctedOmega = np.array([0,0,0], dtype=np.float32)
        self.measuredOmega = np.array([0,0,0], dtype=np.float32)

        self.predictedOmegaError = np.array([0,0,0], dtype=np.float32)
        self.calculate = True

    def setMeasuredOmega(self, measuredOmega):
        self.measuredOmega = measuredOmega
        self.calculate = False

    def setPredictedOmegaError(self, predictedOmegaError):
        self.predictedOmegaError = predictedOmegaError
        self.calculate = False

    def getCorrectedOmega(self):
        if(self.calculate == False):
            self.correctedOmega = self.measuredOmega + self.predictedOmegaError

        

        return self.correctedOmega