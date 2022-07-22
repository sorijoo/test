import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
# Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return cv2.flip(image, 1)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)


# import cv2
# import streamlit as st
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math

# st.title("Real Time Classification")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])

# detector = HandDetector(maxHands=1)
# classifier = Classifier("./model/keras_model.h5", "./model/labels.txt")
 
# offset = 20
# imgSize = 300
# labels = [chr(x).upper() for x in range(97, 123)]
# labels.remove("J")
# labels.remove("Z")

# try:
#     cap = cv2.VideoCapture(0)
# except:
#     cap = cv2.VideoCapture(1) 

# while run:
#     _, frame = cap.read()
#     # st.text(frame)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     imgOutput = frame
#     # imgOutput = frame.copy()
#     hands, frame = detector.findHands(frame)
#     if hands:
#         x, y, w, h = hands[0]["bbox"]
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
#         imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
#         aspectRatio =h/w
#         if aspectRatio>1:
#             k = imgSize/h
#             wCal = math.ceil(k*w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             wGap = math.ceil((imgSize-wCal)/2)
#             imgWhite[:, wGap:wCal+wGap] = imgResize
#             predition, idx = classifier.getPrediction(imgWhite, draw=False)
#         else:
#             k = imgSize/w
#             hCal = math.ceil(k*h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             hGap = math.ceil((imgSize-hCal)/2)
#             imgWhite[hGap:hCal+hGap, :] = imgResize
#             predition, idx = classifier.getPrediction(imgWhite, draw=False)
        
#         cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 139), cv2.FILLED)
#         cv2.putText(imgOutput, labels[idx], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 139), 4)       
#     FRAME_WINDOW.image(imgOutput)