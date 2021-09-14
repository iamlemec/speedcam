import time
import toml
import cv2
import torch
import traceback
import numpy as np
import pandas as pd

from threading import Thread
from operator import itemgetter
from collections import deque
from pathlib import Path

from kalman import KalmanTracker

##
## object detection
##

class Tracker:
    def __init__(self,
        qual_cutoff=0.3, hist_length=250, match_cutoff=0.4, match_timeout=2.0,
        config_path=None
    ):
        # load yolov5 model from torch hub
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.classes = self.model.names

        # open local camera
        self.stream = cv2.VideoCapture()
        self.w = self.h = None

        # set up box tracker
        self.boxes = Boxes(timeout=match_timeout, cutoff=match_cutoff, length=hist_length)

        # other options
        self.bcut = qual_cutoff

        # scene/camera config
        config = toml.load(config_path)
        if 'scene' in config:
            scene = config['scene']
            self.fov_width = scene['width']
        else:
            self.fov_width = None
        if 'camera' in config:
            camera = config['camera']
            self.params = np.array(camera['K']), np.array(camera['D'])
        else:
            self.params = None

    def __del__(self):
        self.close_stream()

    def open_stream(self, src=0, udp=None, buffer=None, size=None):
        if self.stream.isOpened():
            return

        if udp is not None:
            self.stream.open(f'udpsrc port={port} ! application/x-rtp,encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        else:
            self.stream.open(src)

        if buffer is not None:
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer)

        if size is not None:
            self.w, self.h = size
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        else:
            self.w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f'frame size: {self.w} x {self.h}')

    def close_stream(self):
        if self.stream.isOpened():
            self.stream.release()
            self.w = self.h = None

    # score a single frame
    def calc_boxes(self, frame):
        results = self.model(frame)
        data = results.xyxyn[0].to('cpu').numpy()

        coords = data[:, :4]
        quals = data[:, 4]
        labels = data[:, 5].astype('int32')

        qsel = quals >= self.bcut
        coords = coords[qsel, :]
        quals = quals[qsel]
        labels = labels[qsel]

        return coords, quals, labels

    # plot output of model
    def plot_boxes(
        self, frame, coords, quals, labels,
        box_color=(0, 255, 0), label_font=cv2.FONT_HERSHEY_SIMPLEX
    ):
        n = len(labels)
        w, h = frame.shape[1], frame.shape[0]

        for i in range(n):
            x1, y1, x2, y2 = coords[i]
            q, l = quals[i], labels[i]

            # map into real
            x1, y1 = int(x1*w), int(y1*h)
            x2, y2 = int(x2*w), int(y2*h)

            # plot boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, l, (x1, y1), label_font, 0.9, box_color, 2
            )

        return frame

    def read_frame(self, flip=True, undistort=True):
        ret, frame = self.stream.read()
        if ret:
            if flip:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))
            if undistort and self.params is not None:
                newcam, roi = cv2.getOptimalNewCameraMatrix(
                    *self.params, (self.w, self.h), 1
                )
                frame = cv2.undistort(frame, *self.params, newcam)
            return frame

    def loop_stream(self, out=None, flip=True, tick=1):
        i = 0
        s = time.time()

        while True:
            if (frame := self.read_frame(flip=flip)) is None:
                # print('no frame')
                continue
            timestamp = time.time()

            i += 1
            if (dt := timestamp - s) >= tick:
                # print(f'fps: {i/dt}')
                i = 0
                s = timestamp

            #
            coords, quals, labels = self.calc_boxes(frame)
            detect = [(l, c) for l, c in zip(labels, coords)]
            match, done = self.boxes.update(timestamp, detect)
            labels1 = [f'{self.classes[l]} {i}' for i, l in zip(match, labels)]

            for i, trk in done.items():
                lab = self.classes[trk.l]
                t0, t1 = trk.hist[0][0], trk.hist[-1][0]
                dt = t1 - t0
                num = len(trk.hist)
                pos = np.vstack([h[1][:2] for h in trk.hist])
                move = (pos.max(axis=0)-pos.min(axis=0)).max()
                if move >= 0.2:
                    print(f'{lab} #{i}: N={num}, Δt={dt:.2f}, Δx={move:.3f}')
                    trk.dataframe().to_csv(f'tracks/{lab}_{i}_{int(t0)}.csv', index=False)

            final = self.plot_boxes(frame, coords, quals, labels1)
            if out is not None:
                out.write(final)
            else:
                cv2.imshow('waroncars', final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def mark_stream(self, out=None, fps=30, flip=False, **kwargs):
        self.boxes.reset()
        self.open_stream(**kwargs)

        if out is not None:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, four_cc, fps, (self.w, self.h))
        else:
            out = None

        try:
            self.loop_stream(out=out, flip=flip)
        except KeyboardInterrupt:
            pass

        self.close_stream()

        if out is None:
            cv2.destroyAllWindows()
        else:
            out.release()

    def snapshots(self, out_dir, delay=2.0, display=True, **kwargs):
        out_path = Path(out_dir)
        self.open_stream(**kwargs)

        try:
            i = 0
            s = time.time()

            while True:
                if (frame := self.read_frame(flip=False)) is None:
                    print('no frame')
                    continue

                if (t := time.time()) >= s + delay:
                    fpath = out_path / f'snapshot_{i}.jpg'
                    print(fpath)

                    cv2.imwrite(str(fpath), frame)
                    if display:
                        cv2.imshow('snapshot', frame)

                    i += 1
                    s = t

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass

        self.close_stream()
        if display:
            cv2.destroyAllWindows()

##
## object tracking
##

def box_area(l, t, r, b):
    w = np.maximum(0, r-l)
    h = np.maximum(0, b-t)
    return w*h

def box_overlap(box1, box2):
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2

    lx = np.maximum(l1, l2)
    tx = np.maximum(t1, t2)
    rx = np.minimum(r1, r2)
    bx = np.minimum(b1, b2)

    a1 = box_area(l1, t1, r1, b1)
    a2 = box_area(l2, t2, r2, b2)
    ax = box_area(lx, tx, rx, bx)

    sim = ax/np.maximum(a1, a2)
    return 1 - sim

kalman_args = {
    'ndim': 4,
    'σz': [0.05, 0.05, 0.05, 0.05],
    'σv': [0.5, 0.5, 0.5, 0.5],
}

# single object state
class Track:
    def __init__(self, kalman, length, l, t, z):
        self.kalman = kalman
        self.l = l
        self.t = t
        self.x, self.P = kalman.start(z)
        self.hist = deque([(t, z, self.x, self.P)], length)

    def predict(self, t):
        dt = t - self.t
        x1, P1 = self.kalman.predict(self.x, self.P, dt=dt)
        return x1, P1

    def update(self, t, z):
        dt = t - self.t
        self.t = t
        self.x, self.P = self.kalman.update(self.x, self.P, z, dt=dt)
        self.hist.append((t, z, self.x, self.P))

    def dataframe(self):
        return pd.DataFrame(
            np.vstack([np.hstack(h[:3]) for h in self.hist]),
            columns=[
                'time', 'x', 'y', 'w', 'h',
                'kx', 'ky', 'kw', 'kh',
                'vx', 'vy', 'vw', 'vh'
            ]
        )

# entry: index, label, qual, coords
class Boxes:
    def __init__(self, timeout=2.0, cutoff=0.2, length=250):
        self.timeout = timeout
        self.cutoff = cutoff
        self.length = length
        self.kalman = KalmanTracker(**kalman_args)
        self.reset()

    def reset(self):
        self.nextid = 0
        self.tracks = {}

    def add(self, l, t, z):
        i = self.nextid
        self.nextid += 1
        self.tracks[i] = Track(self.kalman, self.length, l, t, z)
        return i

    def pop(self, i):
        return self.tracks.pop(i)

    def update(self, t, boxes):
        # precompute predicted positions for tracks
        locs = {i: trk.predict(t) for i, trk in self.tracks.items()}

        # compute all pairs with difference below cutoff
        errs = []
        for k1, (l1, c1) in enumerate(boxes):
            for i2, trk in self.tracks.items():
                x1, P1 = locs[i2]
                l2, c2 = trk.l, x1[:4]
                if l1 == l2:
                    e = box_overlap(c1, c2) # this can be improved
                    if e < self.cutoff:
                        errs.append((k1, i2, e))

        # unravel match in decreasing order of similarity
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
            _, c = boxes[k]
            self.tracks[j].update(t, c)
            mapper[k] = j

        # create new tracks for non-matches
        match = []
        for k, (l, c) in enumerate(boxes):
            if k not in mapper:
                mapper[k] = self.add(l, t, c)
            match.append(mapper[k])

        # clear out old tracks
        idone = [
            i for i, trk in self.tracks.items() if t > trk.t + self.timeout
        ]
        done = {i: self.tracks.pop(i) for i in idone}

        # return matches and final tracks
        return match, done

##
## testing
##

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.fps = 1/30
        self.fps_ms = int(self.fps * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.capture.release()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.fps)

    def show_frame(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.fps_ms)

    def stream(self):
        while True:
            try:
                self.show_frame()
            except AttributeError:
                pass
            except:
                break
        cv2.destroyAllWindows()
