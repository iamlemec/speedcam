import cv2
import time
import pandas as pd
from threading import Thread

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
## video
##

def write_video(path, frames, fps, dims):
    four_cc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, four_cc, fps, dims)
    for f in frames:
        out.write(f)
    out.release()

##
## testing
##

def stream(src=0, size=None, fps=30):
    delta = 1/fps

    # acquire device
    capture = cv2.VideoCapture(src)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # set video size
    if size is not None:
        w, h = size
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    data = {'status': None, 'frame': None, 'bail': False}
    def update():
        while True:
            if capture.isOpened():
                data['status'], data['frame'] = capture.read()
            if data['bail']:
                break
            time.sleep(delta)

    def cleanup():
        # close windows
        cv2.destroyAllWindows()

        # release device (causing update thread exit)
        if capture.isOpened():
            capture.release()

    # Start frame retrieval thread
    thread = Thread(target=update, daemon=True)
    thread.start()

    # init start time
    s = time.time()

    while True:
        try:
            t = time.time()
            if data['frame'] is not None and t >= s + delta:
                print(f'Frame delta: {t-s}')
                cv2.imshow('frame', data['frame'])
                s = t
            if cv2.waitKey(1) & 0xFF == ord('q'):
                data['bail'] = True
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            cleanup()
            raise e

    cleanup()
