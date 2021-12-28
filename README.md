# War on Cars

Speed tracking using YOLOv5 and Kalman filter matching

# Basic

```python
addr = '192.168.1.232' # find this
port = '554' # default
path = 'live/ch1' # ch0 for HD
username = 'username' # fill in
password = 'password' # fill in
src = f'rtsp://{username}:{password}@{addr}:{port}/{path}'
track = Tracker()
track.mark_stream(src=src)
```

# FFMPEG

Original:
ffmpeg -f v4l2 -i /dev/video0 -profile:v high -pix_fmt yuvj420p -level:v 4.1 -preset ultrafast -tune zerolatency -vcodec libx264 -r 10 -b:v 512k -s 640x360 -f h264 -flush_packets 0 "udp://192.168.1.171:6000?pkt_size=1316"

Works:
ffmpeg -f v4l2 -i /dev/video0 -profile:v high -pix_fmt yuvj420p -level:v 4.1 -preset ultrafast -tune zerolatency -vcodec libx264 -r 10 -b:v 512k -s 640x360 -f h264 "udp://127.0.0.1:5000"

# GStreamer

Server:
gst-launch-1.0 -v v4l2src device=/dev/video0 ! decodebin ! videoconvert ! jpegenc ! rtpjpegpay ! udpsink host=DEST_IP port=5000

Client:
gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! autovideosink
