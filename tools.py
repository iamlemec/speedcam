import cv2
import time
import toml
import numpy as np
import pandas as pd
from threading import Thread, Event

##
## dates
##

def datestring(t=None, tz='US/Eastern'):
    if t is None:
        t = time.time()
    d = pd.to_datetime(t, unit='s', utc=True)
    d1 = d.tz_convert(tz)
    return d1.strftime('%Y%m%dT%H%M%S')

##
## config
##

def load_config(path=None):
    config = toml.load(path) if path is not None else {}
    scene, camera = config.get('scene', {}), config.get('camera', None)
    params = (camera['K'], camera['D']) if camera is not None else None
    fov_width = scene.get('width', 1)

    return {
        'fov_width': fov_width,
        'params': params,
    }

##
## video
##

def write_video(path, frames, fps, dims):
    four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, four_cc, fps, dims)
    for f in frames:
        out.write(f)
    out.release()

##
## streaming
##

class Streamer(Thread):
    def __init__(self, src=0, udp=None, size=None, params=None, flip=False, scale=None):
        # source params
        self.src = src
        self.udp = udp

        # effect params
        self.flip = flip
        self.scale = scale

        # handle size string
        if type(size) is str:
            self.size = [int(i) for i in size.lower().split('x')]
        else:
            self.size = size

        # camera params
        if params is not None:
            self.params = tuple(np.asarray(x) for x in params)
        else:
            self.params = None

        # create camera interface
        self.stream = cv2.VideoCapture()

        # thread components
        self.exit = Event()
        self.started = Event()
        self.frame = None
        super().__init__(name='streamer-thread')

    def start(self):
        super().start()
        self.started.wait()

    def run(self):
        self.open_stream()

        while True:
            if self.exit.is_set():
                break
            if self.is_active():
                self.frame = self.read_frame()

        self.close_stream()

    def get(self):
        if self.frame is None:
            return None
        else:
            return self.frame.copy()

    def close(self):
        self.exit.set()
        self.join()

    def open_stream(self):
        # handle different source cases
        if self.udp is not None:
            gs_ops = [
                f'udpsrc port={self.udp}', 'application/x-rtp,encoding-name=JPEG,payload=26',
                'rtpjpegdepay', 'jpegdec', 'videoconvert', 'appsink'
            ]
            self.stream.open(' ! '.join(gs_ops), cv2.CAP_GSTREAMER)
        else:
            self.stream.open(self.src)

        # set or get dimensions
        if self.size is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        else:
            self.size = (
                int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

        # undistort source frames
        if self.params is not None:
            self.newcam, roi = cv2.getOptimalNewCameraMatrix(*self.params, self.size, 0)
            self.size = (
                roi[2] - roi[0] + 1,
                roi[3] - roi[1] + 1
            )

        # broadcast ready
        self.started.set()

    def close_stream(self):
        if self.stream.isOpened():
            self.stream.release()

    def is_active(self):
        return self.stream.isOpened()

    def read_frame(self):
        status, frame = self.stream.read()
        if status:
            if self.params is not None:
                frame = cv2.undistort(frame, *self.params, self.newcam)
            if self.flip:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))
            if self.scale is not None:
                size1 = tuple(int(scale*x) for x in self.size)
                frame = cv2.resize(frame, size1, interpolation=cv2.INTER_LANCZOS4)
            return frame

    def loop(self, fps=None):
        d = 0 if fps is None else 1/fps
        s = time.time()
        while True:
            if (t := time.time()) >= s + d:
                if (frame := self.get()) is not None:
                    s = t
                    yield frame

##
## testing
##

def stream(src=0, udp=None, size=None, fps=30, flip=False, scale=None):
    # acquire device and start thread
    stream = Streamer(src=src, udp=udp, size=size, flip=flip, scale=scale)
    stream.start()

    # cleanup for later
    def cleanup():
        stream.close()
        cv2.destroyAllWindows()

    # loop at fps
    try:
        for frame in stream.loop(fps=fps):
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        cleanup()
        raise e

    cleanup()
