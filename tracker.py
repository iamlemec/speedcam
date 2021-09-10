import cv2
import torch
import numpy as np
import time
import traceback
from threading import Thread
from collections import deque
from operator import itemgetter

def timestamp():
    sub = time.time() % 1
    sec = time.strftime('%Y-%m-%dT%H:%M:%S')
    mil = int(1000*sub)
    return f'{sec}.{mil:03d}'

##
## object detection
##

class Tracker:
    def __init__(self, bcut=0.2, qlen=30, qcut=0.2):
        # load yolov5 model from torch hub
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.classes = self.model.names

        # open local camera
        self.stream = cv2.VideoCapture()
        self.w = self.h = None

        # set up box tracker
        self.deque = BoxQueue(length=qlen, cutoff=qcut)

        # other options
        self.bcut = bcut

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

            # if score is less than 0.2 we avoid making a prediction.
            if q < self.bcut:
                continue

            # map into real
            x1, y1 = int(x1*w), int(y1*h)
            x2, y2 = int(x2*w), int(y2*h)

            # plot boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, l, (x1, y1), label_font, 0.9, box_color, 2
            )

        return frame

    def read_frame(self, flip=True):
        ret, frame = self.stream.read()
        if ret:
            if flip:
                return np.ascontiguousarray(np.flip(frame, axis=1))
            else:
                return frame

    def loop_stream(self, out=None, flip=True):
        while True:
            if (frame := self.read_frame(flip=flip)) is None:
                print('no frame')
                continue

            coords, quals, labels = self.calc_boxes(frame)
            match = self.deque.append(zip(labels, coords))
            labels1 = [f'{self.classes[l]} {i}' for i, l in zip(match, labels)]

            final = self.plot_boxes(frame, coords, quals, labels1)
            if out is not None:
                out.write(final)
            else:
                cv2.imshow('waroncars', final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def mark_stream(self, out=None, fps=30, flip=False, **kwargs):
        self.deque.reset()
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

# entry: index, label, qual, coords
class BoxQueue:
    def __init__(self, length=30, cutoff=0.2):
        self.length = length
        self.cutoff = cutoff
        self.deque = deque([], length)
        self.inext = 0

    def reset(self):
        self.deque.clear()
        self.inext = 0

    def append(self, boxes):
        match = []

        for l1, c1 in boxes:
            bidx = None
            merr = None

            for bs in reversed(self.deque):
                errs = [
                    (i, box_overlap(c1, c2)) for i, l2, c2 in bs if l1 == l2
                ]
                midx, merr = min(
                    errs, key=itemgetter(1), default=(None, None)
                )

                if midx is not None and merr < self.cutoff:
                    bidx = midx
                    break

            if bidx is None:
                bidx = self.inext
                self.inext += 1
                print(l1, bidx, merr)

            match.append((bidx, l1, c1))

        self.deque.append(match)

        return [i for i, _, _ in match]

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
