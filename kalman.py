import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.stats import chi2

from operator import itemgetter
from collections import deque

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
        K = la.solve(A, B.T).T

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

    def position(self, x, P, dt=1):
        x, P = self.predict(x, P, dt=dt)
        x1 = x[:self.ndim]
        P1 = P[:self.ndim,:self.ndim] + self.R
        return x1, P1

##
## object tracking
##

# ensure covmat is positive-semidefinite
def robust_inverse(V):
    try:
        Vi = la.inv(V)
    except la.LinAlgError:
        λ, U = la.eig(V)
        λ1 = np.maximum(0, λ.real)
        Λ1 = np.diagflat(λ1)
        V1 = U @ Λ1 @ U.T
        Vi = la.inv(V1)
    return Vi

def box_area(l, t, r, b):
    w = np.maximum(0, r-l)
    h = np.maximum(0, b-t)
    return w*h

def box_overlap(box1, box2):
    l1, t1, w1, h1 = box1
    l2, t2, w2, h2 = box2

    r1, b1 = l1 + w1, t1 + h1
    r2, b2 = l2 + w2, t2 + h2

    lx = np.maximum(l1, l2)
    tx = np.maximum(t1, t2)
    rx = np.minimum(r1, r2)
    bx = np.minimum(b1, b2)

    a1 = box_area(l1, t1, r1, b1)
    a2 = box_area(l2, t2, r2, b2)
    ax = box_area(lx, tx, rx, bx)

    sim = ax/np.maximum(a1, a2)
    return 1 - sim

def mahalanobis_distance(x, P, z):
    z1 = z - x
    Pi = robust_inverse(P)
    d = z1 @ Pi @ z1
    return chi2.cdf(d, 7)

# x, y, w, h, r, g, b
kalman_args = {
    'ndim': 7,
    'σz': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'σv': [0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
}

# single object state
# i: frame index
# t: timestamp
# z: measurement
# m: thumbnail
class Track:
    def __init__(self, kalman, length, i, l, t, z):
        self.kalman = kalman
        self.l = l
        self.t = t
        self.x, self.P = kalman.start(z)
        self.hist = deque([(i, t, z, self.x, self.P)], length)

    def predict(self, t):
        dt = t - self.t
        x1, P1 = self.kalman.predict(self.x, self.P, dt=dt)
        return x1, P1

    def position(self, t):
        dt = t - self.t
        x1, P1 = self.kalman.position(self.x, self.P, dt=dt)
        return x1, P1

    def update(self, i, t, z):
        dt = t - self.t
        self.t = t
        self.x, self.P = self.kalman.update(self.x, self.P, z, dt=dt)
        self.hist.append((i, t, z, self.x, self.P))

    def dataframe(self):
        data = pd.DataFrame(
            np.vstack([np.hstack(h[:4]) for h in self.hist]),
            columns=[
                'i', 't',
                'x', 'y', 'w', 'h', 'cr', 'cg', 'cb',
                'kx', 'ky', 'kw', 'kh', 'kr', 'kg', 'kb',
                'vx', 'vy', 'vw', 'vh', 'vr', 'vg', 'vb',
            ]
        )
        data['i'] = data['i'].astype(np.int)
        return data

# entry: index, label, qual, coords
class BoxTracker:
    def __init__(self, match_timeout=2.0, match_cutoff=0.5, time_decay=2.0, track_length=250):
        self.match_timeout = match_timeout
        self.match_cutoff = match_cutoff
        self.track_length = track_length
        self.time_decay = time_decay
        self.kalman = KalmanTracker(**kalman_args)
        self.i = 0 # unique frame id
        self.reset()

    def reset(self):
        self.nextid = 0
        self.tracks = {}

    def add(self, i, l, t, z):
        ni = self.nextid
        self.nextid += 1
        self.tracks[ni] = Track(self.kalman, self.track_length, self.i, l, t, z)
        return ni

    def pop(self, i):
        return self.tracks.pop(i)

    def update(self, t, boxes):
        # precompute predicted positions for tracks
        locs = {i: trk.position(t) for i, trk in self.tracks.items()}

        # compute all pairs with difference below cutoff
        errs = []
        for k1, (l1, z1) in enumerate(boxes):
            for i2, trk in self.tracks.items():
                l2 = trk.l
                dt = t - trk.t
                x2, P2 = locs[i2]
                if l1 == l2:
                    qt = np.exp(-self.time_decay*dt)
                    e0 = mahalanobis_distance(x2, P2, z1)
                    e = e0**qt
                    if e < self.match_cutoff:
                        errs.append((k1, i2, e))

        # unravel match in increasing order of error
        final = []
        for _ in range(len(errs)):
            k, j, e = min(errs, key=itemgetter(2))
            final.append((k, j, e))
            errs = [(k1, j1, e1) for k1, j1, e1 in errs if k1 != k and j1 != j]
            if len(errs) == 0:
                break

        # update positive matches
        mapper = {}
        for k, j, e in final:
            l, z = boxes[k]
            self.tracks[j].update(self.i, t, z)
            mapper[k] = j

        # create new tracks for non-matches
        match = []
        for k, (l, z) in enumerate(boxes):
            if k not in mapper:
                mapper[k] = self.add(self.i, l, t, z)
            match.append(mapper[k])

        # clear out old tracks
        idone = [
            i for i, trk in self.tracks.items() if t > trk.t + self.match_timeout
        ]
        done = {i: self.tracks.pop(i) for i in idone}

        # increment frame id
        self.i += 1

        # return matches and final tracks
        return match, done
