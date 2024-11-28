# we have installed cvzone and mediapipe libraries for this project
## cvzone version 1.4.1 mediapipe 0.10.0 numpy 1.21.2
import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# my camera is set on port 0 if you have issue do it 1
cap = cv2.VideoCapture(0)

# This sets the size of video capture
cap.set(3, 1280)
cap.set(4, 720)

# It sets to detect with a confidence of 0.8, which means if hand not properly visible then it will not detect
detector = HandDetector(detectionCon=1)
# until and unless confidence is more than 0.8 it will model the hand

colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # if index finger tip in rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

    if lmList:

        l, _, _ = detector.findDistance(8, 12, img, draw = False)
        print(l)
        if l < 30:
            cursor = lmList[8]  # index finger tip landmark
            for rect in rectList:
                rect.update(cursor)
    ## Draw Transperency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)
