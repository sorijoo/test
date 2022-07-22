import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

detector = HandDetector(maxHands=1)
classifier = Classifier("./model/keras_model.h5", "./model/labels.txt")
offset = 20
imgSize = 300
labels = [chr(x).upper() for x in range(97, 123)]
labels.remove("J")
labels.remove("Z")

def process(frame):
    frame.flags.writeable = True
    hands, frame = detector.findHands(frame)
    return cv2.imshow(frame)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = process(img)
        return av.VideoFrame.from_ndarray(img)

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)