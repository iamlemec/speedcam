import numpy as np

# simple kalman filter
# no control (B = u = 0)
# no model noise (Q = w = 0)
class KalmanTracker:
    def __init__(self, ndim, std0):
        self.ndim = ndim
        self.std0 = std0

        Z, I = np.zeros((ndim, ndim)), np.eye(ndim)
        self.F = np.block([[I, I], [Z, I]])
        self.H = np.block([I, Z])

        self.R = np.diag(std0**2)
        self.P0 = np.block([[self.R, Z], [Z, self.R]])

    def start(self, z):
        vel = np.zeros(self.ndim)
        self.x = np.hstack([z, vel])
        self.P = self.P0

    def update(self, z, restart=False):
        if restart:
            return self.start(z)

        x1, P1 = self.predict()

        A = self.H @ P1 @ self.H.T + self.R
        B = P1 @ self.H.T
        K = np.linalg.solve(A, B.T).T

        I = np.eye(2*self.ndim)
        G = I - K @ self.H

        self.x = G @ x1 + K @ z
        self.P = G @ P1

    def predict(self):
        x1 = self.F @ self.x
        P1 = self.F @ self.P @ self.F.T
        return x1, P1
