#!/usr/bin/env

# disable numpy multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# disable warnings
import warnings
warnings.filterwarnings('ignore')

import time
import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque

from kalman import BoxTracker
from analyzer import calc_speed
from tools import datestring, load_config, write_video, Streamer

##
## object detection
##

class Tracker:
    def __init__(self,
        src=0, udp=None, size=None, flip=False, scale=None, qual_cutoff=0.3, edge_cutoff=0.02,
        track_length=250, match_cutoff=0.8, match_timeout=1.5, time_decay=2.0, video_length=100,
        model_type='ultralytics/yolov5', model_size='yolov5x', config_path='config.toml',
        tracks_path='tracks'
    ):
        # load yolov5 model from torch hub
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

        # track output directory
        self.tracks_path = tracks_path
        if not os.path.isdir(tracks_path):
            os.mkdir(tracks_path)

        # video saving
        if video_length is not None:
            self.video = deque([], video_length)
            self.times = deque([], video_length)
        else:
            self.video = self.times = None

        # scene/camera config
        config = load_config(config_path)
        self.fov_width = config['fov_width']
        params = config['params']

        # create streaming interface
        self.streamer = Streamer(src=src, udp=udp, size=size, params=params, flip=flip, scale=scale)

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

    def process_track(self, num, trk):
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
        if self.tracks_path is not None:
            fpath = os.path.join(self.tracks_path, f'{tstr}_{lab}_{num}')
            data.to_csv(f'{fpath}.csv', index=False)
            if self.video is not None:
                write_video(f'{fpath}.mp4', self.video, fps, self.streamer.size)

    def process_frame(self, frame):
        # record receive time
        timestamp = time.time()

        # get features
        coords, quals, labels = self.calc_boxes(frame)
        feats = self.calc_features(frame, coords)

        # update tracker
        detect = list(zip(labels, feats))
        match, done = self.boxes.update(timestamp, detect)

        # handle completed tracks
        for num, trk in done.items():
            self.process_track(num, trk)

        # draw boxes on screen
        labels1 = [f'{self.classes[l]} {i}' for i, l in zip(match, labels)]
        final = self.plot_boxes(frame, coords, quals, labels1)

        # rolling save frame
        if self.video is not None:
            self.video.append(final)
            self.times.append(timestamp)

        return final

    def stream(self, display=True):
        # start input stream
        self.streamer.start()

        # get implied fov
        w, h = self.streamer.size
        self.fov = self.fov_width, self.fov_width*(h/w)

        # cleanup for later
        def cleanup():
            self.streamer.close() # send exit signal
            if display:
                cv2.destroyAllWindows()

        try:
            for frame in self.streamer.loop():
                final = self.process_frame(frame)

                # display frame
                if display:
                    cv2.imshow('waroncars', final)

                # get user input
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        except Exception as e:
            cleanup()
            raise e

        cleanup()

# main entry point
if __name__ == '__main__':
    import fire
    fire.Fire(Tracker)
