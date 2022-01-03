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

class Streamer:
    def __init__(self, params=None):
        # camera params
        if params is not None:
            self.params = tuple(np.asarray(x) for x in params)
        else:
            self.params = None

        # create camera interface
        self.stream = cv2.VideoCapture()

    def __del__(self):
        self.close_stream()

    def open_stream(self, src=0, udp=None, buffer=None, size=None):
        if self.stream.isOpened():
            return

        # handle size string
        if type(size) is str:
            size = [int(i) for i in size.lower().split('x')]

        # handle different source cases
        if udp is not None:
            gs_ops = [
                f'udpsrc port={udp}', 'application/x-rtp,encoding-name=JPEG,payload=26',
                'rtpjpegdepay', 'jpegdec', 'videoconvert', 'appsink'
            ]
            self.stream.open(' ! '.join(gs_ops), cv2.CAP_GSTREAMER)
        else:
            self.stream.open(src)

        # set buffer size
        if buffer is not None:
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer)

        # set or get dimensions
        if size is not None:
            self.size = size
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
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

    def close_stream(self):
        if self.stream.isOpened():
            self.stream.release()

    def is_active(self):
        return self.stream.isOpened()

    def read_frame(self, flip=False, scale=None):
        status, frame = self.stream.read()
        if status:
            if self.params is not None:
                frame = cv2.undistort(frame, *self.params, self.newcam)
            if flip:
                frame = np.ascontiguousarray(np.flip(frame, axis=1))
            if scale is not None:
                size1 = tuple(int(scale*x) for x in self.size)
                frame = cv2.resize(frame, size1, interpolation=cv2.INTER_LANCZOS4)
            return frame

# continuously pull frames from the camera
class StreamerThread(Thread):
    def __init__(self, stream, name='camera-reader-thread', **kwargs):
        self.stream = stream
        self.args = kwargs
        self.exit = Event()
        self.frame = None
        super().__init__(name=name)

    def run(self):
        while True:
            if self.exit.is_set():
                break
            if self.stream.is_active():
                self.frame = self.stream.read_frame(**self.args)

    def get(self):
        if self.frame is None:
            return None
        else:
            return self.frame.copy()

    def close(self):
        self.exit.set()
        self.join()

##
## testing
##

def stream(src=0, size=None, fps=30, stats=False):
    delta = 1/fps

    # acquire device and start thread
    stream = Streamer()
    stream.open_stream(src=src, size=size)
    thread = StreamerThread(stream)
    thread.start()

    # cleanup for later
    def cleanup():
        thread.close()
        stream.close_stream()
        cv2.destroyAllWindows()

    # init start time
    s = time.time()

    while True:
        try:
            t = time.time()
            if (frame := thread.get()) is not None and t >= s + delta:
                if stats:
                    print(f'fps: {1/(t-s)}')
                s = t
                cv2.imshow('frame', thread.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            cleanup()
            raise e

    cleanup()
