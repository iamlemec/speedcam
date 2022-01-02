#!/usr/bin/env

# disable numpy multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

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
from tools import datestring, write_video

##
## object detection
##

class Tracker:
    def __init__(self,
        qual_cutoff=0.3, edge_cutoff=0.02, track_length=250, match_cutoff=0.8, match_timeout=1.5,
        time_decay=2.0, video_length=100, undistort=True, model_type='ultralytics/yolov5',
        model_size='yolov5x', config_path='config.toml'
    ):
        # load yolov5 model from torch hub
        if model_type is not None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = torch.hub.load(model_type, model_size, pretrained=True, device=self.device)
            self.classes = self.model.names

        # box detection options
        self.edge_cutoff = edge_cutoff # reject boxes nearly touching edges
        self.qual_cutoff = qual_cutoff # cutoff for detection quality from YOLOv5

        # set up box tracker
        self.boxes = BoxTracker(
            match_timeout=match_timeout, match_cutoff=match_cutoff,
            time_decay=time_decay, track_length=track_length
        )

        # video saving
        if video_length is not None:
            self.video = deque([], video_length)
            self.times = deque([], video_length)
        else:
            self.video = self.times = None

        # create camera interface
        self.stream = cv2.VideoCapture()
        self.w = self.h = None

        # scene/camera config
        config = toml.load(config_path) if config_path is not None else {}
        scene, camera = config.get('scene', {}), config.get('camera', None)
        self.fov_width = scene.get('width', 1)
        if undistort and camera is not None:
            self.params = np.array(camera['K']), np.array(camera['D'])
        else:
            self.params = None

    def __del__(self):
        self.close_stream()

    def open_stream(self, src=0, udp=None, buffer=None, size=None):
        if self.stream.isOpened():
            return

        if udp is not None:
            gs_ops = [
                f'udpsrc port={udp}', 'application/x-rtp,encoding-name=JPEG,payload=26',
                'rtpjpegdepay', 'jpegdec', 'videoconvert', 'appsink'
            ]
            self.stream.open(' ! '.join(gs_ops), cv2.CAP_GSTREAMER)
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

        if self.params is not None:
            self.newcam, roi = cv2.getOptimalNewCameraMatrix(
                *self.params, (self.w, self.h), 0
            )
            self.w = roi[2] - roi[0] + 1
            self.h = roi[3] - roi[1] + 1

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

    # compute features for kalman filter
    def calc_features(self, frame, coords):
        # convert to (cx, cy, w, h)
        boxes = np.hstack([
            0.5*(coords[:,:2] + coords[:,2:]), # center
            coords[:,2:] - coords[:,:2] # dimensions
        ])

        # get thumbnails
        h, w, _ = frame.shape
        pixels = (coords*[w, h, w, h]).astype(int)
        thumbs = [frame[p[1]:p[3], p[0]:p[2]]/255 for p in pixels]

        # get average color
        if len(thumbs) > 0:
            colors = np.vstack([np.mean(t, axis=(0, 1)) for t in thumbs])
        else:
            colors = np.empty((0, 3))

        # combine into features
        return np.hstack([boxes, colors])

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

    def read_frame(self, flip=False, scale=None):
        ret, frame = self.stream.read()
        if ret:
            if self.params is not None:
                frame = cv2.undistort(frame, *self.params, self.newcam)
            if flip:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))
            if scale is not None:
                size1 = int(scale*self.w), int(scale*self.h)
                frame = cv2.resize(frame, size1, interpolation=cv2.INTER_LANCZOS4)
            return frame

    def process_track(self, num, trk, tracks=None):
        data = trk.dataframe()

        # compute differentials
        N = len(data)
        Δt = data['t'].iloc[-1] - data['t'].iloc[0]
        Δx = data['x'].iloc[-1] - data['x'].iloc[0]

        # reject short tracks
        if abs(Δx) < 0.25 or N <= 2:
            return

        # track stats
        t0 = data['t'].iloc[0]
        lab = self.classes[trk.l].replace(' ', '')
        μv, σv = calc_speed(data, self.fov)

        # video params
        fdt = self.times[-1] - self.times[0]
        fps = len(self.video)/fdt
        tstr = datestring(t0)

        # report detection
        print(f'{lab} #{num}: N={N}, Δt={Δt:.2f}, Δx={Δx:.3f}, μv={μv:.3f}, σv={σv:.3f}, fps={fps:.3f}')

        # store stats and video
        if tracks is not None:
            if not os.path.isdir(tracks):
                os.mkdir(tracks)
            fpath = os.path.join(tracks, f'{tstr}_{lab}_{num}')
            data.to_csv(f'{fpath}.csv', index=False)
            if self.video is not None:
                write_video(f'{fpath}.mp4', self.video, fps, (self.w, self.h))

    def loop_stream(self, out=True, tracks=None, flip=False, scale=None):
        while True:
            t0 = time.time()

            # fetch next frame (blocking)
            frame = self.read_frame(flip=flip, scale=scale)
            if frame is None:
                continue

            t1 = time.time()

            # record receive time
            timestamp = time.time()

            # get features
            coords, quals, labels = self.calc_boxes(frame)
            feats = self.calc_features(frame, coords)

            # update tracker - SLOW
            detect = list(zip(labels, feats))
            match, done = self.boxes.update(timestamp, detect)

            # handle completed tracks
            for num, trk in done.items():
                self.process_track(num, trk, tracks=tracks)

            # draw boxes on screen
            labels1 = [f'{self.classes[l]} {i}' for i, l in zip(match, labels)]
            final = self.plot_boxes(frame, coords, quals, labels1)

            # rolling save frame
            if self.video is not None:
                self.video.append(final)
                self.times.append(timestamp)

            # display/save frame
            if out is True:
                cv2.imshow('waroncars', final)
            elif out is not False:
                out.write(final)

            t2 = time.time()

            print(f'Fetch time: {t1-t0}')
            print(f'Model time: {t2-t1}')

            # get user input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def track(self, src=0, udp=None, out=True, **kwargs):
        self.boxes.reset()
        self.open_stream(src=src, udp=udp)

        if type(out) is str:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out, four_cc, fps, (self.w, self.h))

        def cleanup():
            self.close_stream()
            if out is True:
                cv2.destroyAllWindows()
            elif out is not False:
                out.release()

        try:
            self.loop_stream(out=out, **kwargs)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            cleanup()
            raise e

        cleanup()

    def record(self, src=0, udp=None, out=None, fps=30, size=None, display=True):
        delay = 1/fps
        if type(size) is str:
            size = [int(i) for i in size.lower().split('x')]

        # open input stream
        self.open_stream(src=src, udp=udp, size=size)

        # open output stream
        if out is not None:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out, four_cc, fps, (self.w, self.h))

        try:
            s = time.time()

            while True:
                if (frame := self.read_frame(flip=False)) is None:
                    print('no frame')
                    continue

                if (t := time.time()) >= s + delay:
                    if out is not None:
                        out.write(frame)

                    if display:
                        cv2.imshow('snapshot', frame)

                    s = t

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass

        # close input stream
        self.close_stream()

        # close output stream
        if out is not None:
            out.release()

        if display:
            cv2.destroyAllWindows()

    def snapshots(self, src=0, udp=None, out=None, delay=2.0, display=True):
        self.open_stream(src=src, udp=udp)

        try:
            i = 0
            s = time.time()

            while True:
                if (frame := self.read_frame(flip=False)) is None:
                    print('no frame')
                    continue

                if (t := time.time()) >= s + delay:
                    if out is not None:
                        fpath = os.path.join(out, f'snapshot_{i}.jpg')
                        cv2.imwrite(fpath, frame)
                        print(fpath)

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

# main entry point
if __name__ == '__main__':
    import fire
    fire.Fire(Tracker)
