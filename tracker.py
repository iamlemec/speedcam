import time
import toml
import cv2
import torch
import traceback
import numpy as np
import pandas as pd

from threading import Thread
from pathlib import Path

from kalman import BoxTracker
from analyzer import calc_speed

##
## object detection
##

class Tracker:
    def __init__(self,
        qual_cutoff=0.2, hist_length=250, match_cutoff=0.4, match_timeout=2.0,
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
        self.boxes = BoxTracker(timeout=match_timeout, cutoff=match_cutoff, length=hist_length)

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

    def open_stream(self, src=0, udp=None, buffer=None, size=None, undistort=True):
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

        if undistort:
            self.newcam, roi = cv2.getOptimalNewCameraMatrix(
                *self.params, (self.w, self.h), 0
            )
            self.w, self.h = (roi[2] - roi[0]), (roi[3] - roi[1])

        # update full fov
        self.aspect = self.w/self.h
        self.fov = self.fov_width, self.fov_width/self.aspect

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

    def read_frame(self, flip=False, undistort=True):
        ret, frame = self.stream.read()
        if ret:
            if flip:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))
            if undistort:
                frame = cv2.undistort(frame, *self.params, self.newcam)
            return frame

    def loop_stream(self, out=None, flip=False, undistort=True, tick=1):
        # init fps tracking
        i = 0
        s = time.time()

        while True:
            # fetch next frame (blocking)
            frame = self.read_frame(flip=flip, undistort=undistort)
            if frame is None:
                continue
            timestamp = time.time()

            # fps tracking
            i += 1
            if (dt := timestamp - s) >= tick:
                # print(f'fps: {i/dt}')
                i = 0
                s = timestamp

            # get boxes and update tracker
            coords, quals, labels = self.calc_boxes(frame)
            detect = [(l, c) for l, c in zip(labels, coords)]
            match, done = self.boxes.update(timestamp, detect)
            labels1 = [f'{self.classes[l]} {i}' for i, l in zip(match, labels)]

            # display any completed tracks
            for i, trk in done.items():
                data = trk.dataframe()
                rang = data.max() - data.min()
                move = np.sqrt(rang['x']**2+rang['y']**2)
                if move >= 0.2:
                    t0 = data['time'].iloc[0]
                    lab = self.classes[trk.l]
                    N, Δ = len(data), rang['t']
                    v, σ = calc_speed(data, self.fov)
                    print(f'{lab} #{i}: N={N}, Δt={Δ:.2f}, Δz={move:.3f}, v={v:.3f}, σ={σ:.3f}')
                    data.to_csv(f'tracks/{lab}_{i}_{int(t0)}.csv', index=False)

            # display/save frame
            final = self.plot_boxes(frame, coords, quals, labels1)
            if out is not None:
                out.write(final)
            else:
                cv2.imshow('waroncars', final)

            # get user input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def mark_stream(self, out=None, fps=30, flip=False, undistort=True, **kwargs):
        self.boxes.reset()
        self.open_stream(undistort=undistort, **kwargs)

        if out is not None:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, four_cc, fps, (self.w, self.h))
        else:
            out = None

        try:
            self.loop_stream(out=out, flip=flip, undistort=undistort)
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
