import streamlit as st
from PIL import Image
# import torch
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading


try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# hand detection    
detector = HandDetector(maxHands=1)
classifier = Classifier("../model/keras_model.h5", "../model/labels.txt")


offset = 20
imgSize = 300
 
labels = [chr(x).upper() for x in range(97, 123)]
labels.remove("J")
labels.remove("Z")

# device = 'cpu'
# if not hasattr(st, 'classifier'):
#     model = keras.models.load_model("../model/keras_model.h5")
#     # st.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', _verbose=False)
    
while True:
    try:
        ret, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            x, y, w, h = hands[0]['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # print(prediction, index)
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
    
            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 139), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 139), 4)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)
        if cv2.waitKey(1)==ord("q"): break
    except:
        print("카메라가 경계선 밖으로 나갔습니다.")
        break
    cv2.imshow("Sign Detectoin", imgOutput)
    
cap.release()
cv2.destroyAllWindows()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        results = model(im_pil, size=112)
        bbox_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)