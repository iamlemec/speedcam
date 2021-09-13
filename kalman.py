import numpy as np

# simple kalman filter
# no control (B = u = 0)
# no model noise (Q = w = 0)
class KalmanTracker:
    def __init__(self, ndim, σz, σv):
        self.ndim = ndim

        self.I = np.eye(ndim)
        self.Z = np.zeros((ndim, ndim))
        self.H = np.block([self.I, self.Z])

        self.R = np.diag(np.square(σz))
        Pv = np.diag(np.square(σv))
        self.P0 = np.block([
            [self.R, self.Z],
            [self.Z, Pv]
        ])

    def start(self, z):
        vel = np.zeros(self.ndim)

        x = np.hstack([z, vel])
        P = self.P0

        return x, P

    def update(self, x, P, z, dt=1):
        x1, P1 = self.predict(x, P, dt=dt)

        A = self.H @ P1 @ self.H.T + self.R
        B = P1 @ self.H.T
        K = np.linalg.solve(A, B.T).T

        I = np.eye(2*self.ndim)
        G = I - K @ self.H

        x2 = G @ x1 + K @ z
        P2 = G @ P1

        return x2, P2

    def predict(self, x, P, dt=1):
        F = np.block([
            [self.I, dt*self.I],
            [self.Z, self.I]
        ])

        x1 = F @ x
        P1 = F @ P @ F.T

        return x1, P1
