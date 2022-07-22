import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

st.title("Real Time Classification")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

detector = HandDetector(maxHands=1)
classifier = Classifier("./model/keras_model.h5", "./model/labels.txt")
 
offset = 20
imgSize = 300
labels = [chr(x).upper() for x in range(97, 123)]
labels.remove("J")
labels.remove("Z")

try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1) 

while run:
    _, frame = cap.read()
    # st.text(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgOutput = frame
    imgOutput = frame.copy()
    hands, frame = detector.findHands(frame)
    if hands:
        x, y, w, h = hands[0]["bbox"]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        aspectRatio =h/w
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            predition, idx = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            predition, idx = classifier.getPrediction(imgWhite, draw=False)
        
        cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 139), cv2.FILLED)
        cv2.putText(imgOutput, labels[idx], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 139), 4)       
    FRAME_WINDOW.image(imgOutput)