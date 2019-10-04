import sys
sys.path.insert(0, '../build')
import PyDecoder
import cv2
import time

if __name__ == "__main__":
    video_path = "bigbang.mp4"
    # Test GPU decoder
    dec = PyDecoder.Decoder("CPU")
    dec.ingestVideo(video_path)
    dec.decode()
    tick = time.time()
    while True:
        frame = dec.fetchFrame()
        if frame.size == 0:
            break
    tock = time.time()
    print("time taken for cpu video reader is %.3f" % (tock - tick))

