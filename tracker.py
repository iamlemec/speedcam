import cv2
import torch
import numpy as np
import time
import traceback
from threading import Thread

class Tracker:
    def __init__(self, size=(800, 600), buffer=2):
        # load yolov5 model from torch hub
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.classes = self.model.names

        # open local camera
        self.buffer = buffer
        self.w, self.h = size
        self.stream = cv2.VideoCapture()

    def __del__(self):
        self.close_stream()

    def open_stream(self, src=0):
        if not self.stream.isOpened():
            self.stream.open(src)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def close_stream(self):
        if self.stream.isOpened():
            self.stream.release()

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
        return frame if ret else None

    def mark_frame(self, frame):
        coords, quals, labels = self.calc_boxes(frame)
        final = self.plot_boxes(frame, coords, quals, labels)
        return final

    def loop_stream(self, out=None, fps=30):
        tick = 1/fps
        tick_ms = int(1000*tick)

        while True:
            if (frame := self.read_frame()) is None:
                break
            final = self.mark_frame(frame)

            if out is None:
                cv2.imshow('boxes', final)
            else:
                out.write(final)

            cv2.waitKey(tick_ms)

    def mark_stream(self, src=0, out_path=None, fps=30):
        if out_path is None:
            out = None
        else:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, four_cc, fps, (self.w, self.h))

        self.open_stream(src=src)
        try:
            self.loop_stream(out=out, fps=fps)
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
