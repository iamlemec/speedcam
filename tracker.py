import time
import toml
import cv2
import torch
import traceback
import numpy as np
import pandas as pd

from threading import Thread
from pathlib import Path
from collections import deque

from kalman import BoxTracker
from analyzer import calc_speed
from tools import write_video

##
## tools
##

def datestring(t=None, tz='US/Eastern'):
    if t is None:
        t = time.time()
    d = pd.to_datetime(t, unit='s', utc=True)
    d1 = d.tz_convert(tz)
    return d1.strftime('%Y%m%dT%H%M%S')

##
## object detection
##

class Tracker:
    def __init__(self,
        qual_cutoff=0.3, edge_cutoff=0.02, track_length=250, match_cutoff=0.9,
        match_timeout=2.0, video_length=100, config_path='config.toml', tracks_dir='tracks'
    ):
        # load yolov5 model from torch hub
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.classes = self.model.names

        # open local camera
        self.stream = cv2.VideoCapture()
        self.w = self.h = None

        # set up box tracker
        self.boxes = BoxTracker(timeout=match_timeout, match_cutoff=match_cutoff, track_length=track_length)
        self.tracks_dir = tracks_dir

        # video saving
        if video_length is not None:
            self.video = deque([], video_length)
        else:
            self.video = None

        # box detection options
        self.edge_cutoff = edge_cutoff # reject boxes nearly touching edges
        self.qual_cutoff = qual_cutoff # cutoff for detection quality from YOLOv5

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

        if undistort and self.params is not None:
            self.newcam, roi = cv2.getOptimalNewCameraMatrix(
                *self.params, (self.w, self.h), 0
            )
            self.w = roi[2] - roi[0] + 1
            self.h = roi[3] - roi[1] + 1

        # update full fov
        self.aspect = self.w/self.h
        self.fov = self.fov_width, self.fov_width/self.aspect

        print(f'frame size: {self.w} x {self.h}')
        print(f'fov size: {self.fov[0]:.2f} x {self.fov[1]:.2f}')

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

        edist = 0.5 - np.abs(coords-0.5)
        qsel = quals >= self.qual_cutoff
        esel = (edist >= self.edge_cutoff).all(axis=1)
        sel = qsel & esel

        coords = coords[sel, :]
        quals = quals[sel]
        labels = labels[sel]

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
            if undistort and self.params is not None:
                frame = cv2.undistort(frame, *self.params, self.newcam)
            return frame

    def process_track(self, num, trk):
        data = trk.dataframe()

        # compute differentials
        Δt = data['t'].iloc[-1] - data['t'].iloc[0]
        Δx = data['x'].iloc[-1] - data['x'].iloc[0]

        # reject short tracks
        if Δx < 0.25:
            return

        # track stats
        N = len(data)
        t0 = data['t'].iloc[0]
        lab = self.classes[trk.l]
        μv, σv = calc_speed(data, self.fov)

        # video params
        fps = int(len(self.video/Δt))
        tstr = datestring(t0)
        fpath = f'{self.tracks_dir}/{tstr}_{lab}_{num}'

        # report detection
        print(f'{lab} #{num}: N={N}, Δt={Δt:.2f}, Δx={Δx:.3f}, μv={μv:.3f}, σv={σv:.3f}')

        # store stats and video
        if self.tracks_dir is not None:
            data.to_csv(f'{fpath}.csv', index=False)
            if self.video is not None:
                write_video(f'{fpath}.mp4', self.video, 15, (self.w, self.h))

    def loop_stream(self, out=None, flip=False, undistort=True, scale=None, tick=1):
        while True:
            # fetch next frame (blocking)
            frame = self.read_frame(flip=flip, undistort=undistort)
            if frame is None:
                continue

            timestamp = time.time()

            # get boxes
            coords, quals, labels = self.calc_boxes(frame)
            coords1 = coords.copy()
            coords1[:,2:] -= coords1[:,:2] # (x1, y1, x2, y2) → (x, y, w, h)

            # update tracker
            detect = [(l, c) for l, c in zip(labels, coords1)]
            match, done = self.boxes.update(timestamp, detect)

            # handle completed tracks
            for num, trk in done.items():
                self.process_track(num, trk)

            # draw boxes and possible scale
            labels1 = [f'{self.classes[l]} {i}' for i, l in zip(match, labels)]
            final = self.plot_boxes(frame, coords, quals, labels1)
            if scale is not None:
                size1 = int(scale*self.w), int(scale*self.h)
                final = cv2.resize(final, size1, interpolation=cv2.INTER_LANCZOS4)

            # display/save frame
            if self.video is not None:
                self.video.append(final)
            if out is not None:
                out.write(final)
            else:
                cv2.imshow('waroncars', final)

            # get user input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def mark_stream(self, out=None, fps=30, flip=False, undistort=True, scale=None, **kwargs):
        self.boxes.reset()
        self.open_stream(undistort=undistort, **kwargs)

        if out is not None:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, four_cc, fps, (self.w, self.h))
        else:
            out = None

        try:
            self.loop_stream(out=out, flip=flip, undistort=undistort, scale=scale)
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
