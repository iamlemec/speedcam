# War on Cars

Speed tracking using YOLOv5 and Kalman filter matching.

![car track](promo/car_track.gif)

Estimated speed: 19.0 ± 0.5 mph

# Usage

Get the URL to your camera's RTSP stream (which we'll call `STREAM_URL` below). For me it was something like this
```
rtsp://username:password@192.168.1.232:554/live/ch1
```

This will start the tracker and output data to the `tracks` directory, while outputting match stats to the console in real time
```bash
python3 tracker.py stream --src STREAM_URL
```
There are a bunch of options you can use to tweak the algorithm if you look at the `Tracker` constructor in `tracker.py`.

You can also directly use the Python interface with something like this
```python
import tracker
track = Tracker(src=STREAM_URL)
track.stream()
```
Note that this might cause issues if you preload `numpy` in IPython. If you see very high multithreaded CPU usage, set the flag
```bash
export OPENBLAS_NUM_THREADS=1
```
before launching IPython.

# Camera Calibration

You may need to undistort the image coming from your camera. This is common for standalone security type cameras, but not for webcams. There are tutorials on this out there, but basically the steps are
1. Print out `calibrate/checkerboard.png`
2. Run `recorder.Recorder.images()` to take a bunch of still images of the checkboard in various orientations
3. Pass these to the `calibrate/calibrate.py` script to impute the camera curvature parameters
4. Copy these `K` and `D` values into the `camera` section of `config.toml`

# Scene Calibration

You need to know the width of the scene observed by your camera to determine speeds. This is of course at a particular distance from the camera itself. In this case, it would be the distance to the center of the roadway you're observing (right now you have to observe the roadway in a perpendicular manner).

One can calculate the width using the distance to the roadway and the angular field of view of your camera, but keep in mind that published FOV values my be slightly inexact and you lose some of the image in the undistortion process, so in the end setting the `scene.width` value in `config.toml` is going to be a bit of an art. See my calculation at the top of `config.toml` for some guidance. The entered width should be in meters.

# Algorithm

First we get object boxes detected by YOLOv5. Then we calculate box center position, box width and height, and mean RGB color values. Put these in a Kalman filter initialized with zero covariance terms and hard-coded variance and measurement error terms. To match new boxes to existing tracks, use the Mahalanobis distance metric, unraveling the cascade by picking off the closest matches first. There is also some time-decay factor included in the metric.
