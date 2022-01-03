#!/usr/bin/env

import time
import cv2

from tools import load_config, Streamer

class Recorder:
    def __init__(
        self, src=0, udp=None, size=None, flip=False, scale=None, config_path='config.toml'
    ):
        config = load_config(config_path)
        params = config['params']
        self.streamer = Streamer(src=src, udp=udp, size=size, params=params, flip=flip, scale=scale)

    def video(self, out=None, fps=30, display=True):
        # open input stream
        self.streamer.start()

        # open output stream
        if out is not None:
            four_cc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out, four_cc, fps, self.streamer.size)

        try:
            for frame in self.streamer.loop(fps=fps):
                if out is not None:
                    out.write(frame)

                if display:
                    cv2.imshow('snapshot', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass

        # close input stream
        self.streamer.close()

        # close output stream
        if out is not None:
            out.release()

        # close display viewport
        if display:
            cv2.destroyAllWindows()

    def images(self, out=None, delay=2.0, display=True):
        fps = 1/delay

        # open input stream
        self.streamer.start()

        try:
            for frame in self.streamer.loop(fps=fps):
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
        self.streamer.close()

        # close display viewport
        if display:
            cv2.destroyAllWindows()

# main entry point
if __name__ == '__main__':
    import fire
    fire.Fire(Recorder)
