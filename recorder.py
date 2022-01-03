#!/usr/bin/env

import time
import cv2

from tools import load_config, Streamer, StreamerThread

class Recorder:
    def __init__(self, config_path='config.toml'):
        config = load_config(config_path)
        params = config['params']
        self.streamer = Streamer(params=params)

    def video(self, src=0, udp=None, out=None, fps=30, size=None, flip=False, scale=None, display=True):
        delay = 1/fps

        # open input stream
        self.streamer.open_stream(src=src, udp=udp, size=size)
        thread = StreamerThread(self.streamer, flip=flip, scale=scale)
        thread.start()

        # open output stream
        if out is not None:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out, four_cc, fps, self.streamer.size)

        try:
            s = time.time()

            while True:
                if (frame := thread.get()) is None:
                    continue

                if (t := time.time()) >= s + delay:
                    s = t

                    if out is not None:
                        out.write(frame)

                    if display:
                        cv2.imshow('snapshot', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass

        # close input stream
        thread.close()
        self.streamer.close_stream()

        # close output stream
        if out is not None:
            out.release()

        # close display viewport
        if display:
            cv2.destroyAllWindows()

    def images(self, src=0, udp=None, out=None, delay=2.0, size=None, flip=False, scale=None, display=True):
        self.streamer.open_stream(src=src, udp=udp, size=size)
        thread = StreamerThread(self.streamer, flip=flip, scale=scale)
        thread.start()

        try:
            i = 0
            s = time.time()

            while True:
                if (frame := thread.get()) is None:
                    continue

                if (t := time.time()) >= s + delay:
                    s = t

                    if out is not None:
                        fpath = os.path.join(out, f'snapshot_{i}.jpg')
                        cv2.imwrite(fpath, frame)
                        print(fpath)

                    if display:
                        cv2.imshow('snapshot', frame)

                    i += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass

        # close input stream
        thread.close()
        self.streamer.close_stream()

        # close display viewport
        if display:
            cv2.destroyAllWindows()

# main entry point
if __name__ == '__main__':
    import fire
    fire.Fire(Recorder)
