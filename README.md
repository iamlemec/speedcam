# War on Cars

Speed tracking using YOLOv5 and persistent objects

# Stream Command

Original:
ffmpeg -f v4l2 -i /dev/video0 -profile:v high -pix_fmt yuvj420p -level:v 4.1 -preset ultrafast -tune zerolatency -vcodec libx264 -r 10 -b:v 512k -s 640x360 -f h264 -flush_packets 0 "udp://192.168.1.171:6000?pkt_size=1316"

Works:
ffmpeg -f v4l2 -i /dev/video0 -profile:v high -pix_fmt yuvj420p -level:v 4.1 -preset ultrafast -tune zerolatency -vcodec libx264 -r 10 -b:v 512k -s 640x360 -f h264 "udp://127.0.0.1:5000"
