import cv2
from lib.lanedetect import LaneDetect

import time

cap = cv2.VideoCapture('car.avi')

ld = LaneDetect()

frame_count = 0

while True:
    _, frame = cap.read()
    if not _:
        break

    if frame_count >= 2100:#1611 #254 #2100
        start = time.time()
        ld.detect(frame)
        print(time.time() - start, ' - ' ,frame_count)
        # print(frame_count)
        cv2.waitKey(0)
    frame_count += 1  