import cv2
import torch
import numpy as np
import time
import traceback
from threading import Thread

class Tracker:
    def __init__(self):
        # load yolov5 model from torch hub
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.classes = self.model.names

        # open local camera
        self.stream = cv2.VideoCapture()
        self.w = self.h = None

    def __del__(self):
        self.close_stream()

    def open_stream(self, src=0, buffer=None, size=None):
        if self.stream.isOpened():
            return

        self.stream.open(src)

        if buffer:
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer)

        if size:
            self.w, self.h = size
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        else:
            self.w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.h = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
        self, frame, coords, quals, labels, thresh=0.2,
        box_color=(0, 255, 0), label_font=cv2.FONT_HERSHEY_SIMPLEX
    ):
        n = len(labels)
        w, h = frame.shape[1], frame.shape[0]

        for i in range(n):
            x1, y1, x2, y2 = coords[i]
            q, l = quals[i], labels[i]

            # if score is less than 0.2 we avoid making a prediction.
            if q < thresh:
                continue

            # map into real
            cname = self.classes[l]
            x1, y1 = int(x1*w), int(y1*h)
            x2, y2 = int(x2*w), int(y2*h)

            # plot boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, cname, (x1, y1), label_font, 0.9, box_color, 2
            )

        return frame

    def read_frame(self, flip=True):
        ret, frame = self.stream.read()
        if ret:
            if flip:
                return np.ascontiguousarray(np.flip(frame, axis=1))
            else:
                return frame

    def mark_frame(self, frame):
        coords, quals, labels = self.calc_boxes(frame)
        final = self.plot_boxes(frame, coords, quals, labels)
        return final

    def loop_stream(self, out=None, fps=10, flip=True):
        tick = int(1000/fps)

        while True:
            if (frame := self.read_frame(flip=flip)) is None:
                print('no frame')
                break
            final = self.mark_frame(frame)

            if out is None:
                cv2.imshow('boxes', final)
            else:
                out.write(final)

            cv2.waitKey(tick)

    def mark_stream(self, src=0, out_path=None, fps=10, flip=True, buffer=None, size=None):
        self.open_stream(src=src, buffer=buffer, size=size)

        if out_path is None:
            out = None
        else:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, four_cc, fps, (self.w, self.h))

        try:
            self.loop_stream(out=out, fps=fps, flip=flip)
        except KeyboardInterrupt:
            pass

        self.close_stream()

        if out is None:
            cv2.destroyAllWindows()
        else:
            out.release()

class ThreadedCamera(object):
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
