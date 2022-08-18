import streamlit as st

st.header('개발 과정')
st.write("""
기본적으로 각 기능에 대해서 모듈로 테스트를 해보고, 클래스 형식으로 합치는 방식으로 진행
의도했던 모든 기능이 구현되면 py 형태로 바꿔 어플리케이션을 제작해 볼 계획이었으나 한계에 부딪힘""")


st.subheader('사용 라이브러리')
st.markdown("""
```import os
import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pytube
from ffpyplayer.player import MediaPlayer```""")

st.subheader('데이터 수집')

st.write("""
해당 프로젝트에서 사용한 데이터는 유튜브에서 다운 받은 안무 영상으로,
""")
st.write("""
mediapipe는 두 명 이상의 사람이 등장하면 객체 검출 성능이 떨어지기 때문에 기본적으로 한 명이 나오는 영상 사용
""")
st.write("""
openpose를 이용해 여러 사람에 대한 포즈 추정이나 YOLO 모델을 사용해서 처리하는 방법도 시도했으나, 나름대로의 이슈로 한 명의 사람만 등장한다는 가정 하에 프로젝트 진행
""")

st.markdown("""
```def download_video(self):
    self.__save_dance_name()
    url = input(f"{self.__dance_name}의 안무 영상 링크: ")
    if not os.path.exists(self.__video_download_path): os.mkdir(self.__video_download_path)
    yt = pytube.YouTube(url).streams.filter(res="720p").first()
    yt.download(output_path=self.__video_download_path, filename=self.__dance_name+".mp4")```""")

st.subheader('키포인트 추출')

st.markdown("""
```def extract_keypoints(self, isMirr=False, showExtract=False):
        if not os.path.exists(self.__keypoints_path): os.mkdir(self.__keypoints_path)
        
        keypoint_dict_pose = []
        
        cv2.startWindowThread()
        cap = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret: break
                if not isMirr: image = cv2.flip(image, 1)
                
                results = pose.process(image)
                # Extracting
                try: keypoint_dict_pose.append({str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.pose_landmarks.landmark)})
                except: pass
                if showExtract:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(244, 244, 244), thickness=2, circle_radius=1),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(153, 255, 153), thickness=2, circle_radius=1))
                    cv2.imshow("Extracting", image)
                    if cv2.waitKey(1)==ord("q"): break
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        # Save coord. Data for json type
        with open(self.__keypoints_path+"/"+self.__dance_name+"_keypoints.json", "w") as keypoints:
            json.dump(keypoint_dict_pose, keypoints)```""")
st.write("""
기본적인 프로세스는 영상에서 키포인트들을 추출하고 해당 키포인트를 저장해, 추후에 사용하는 방식.
디텍션 모델을 활용해 영상의 프레임 단위로 키포인트를 추출해 저장함.
""")
st.write("""
초기에는 mediapipe의 holistic 모델을 사용해서 포즈뿐만 아니라, FaceMesh와 손에 대한 키포인트도 추출했으나, 얼굴이나 손은 포즈에 비해서 키포인트 추출이 잘 되지 않아 프로토타입 제작시에는 사용하지 않음.
""")
st.write("""
일반적인 영상의 길이가 3분 대이고, 해당 영상에 대해서 평균적으로 6000~7000 프레임의 포즈가 수집되는 반면, 얼굴이나 손은 1/5 수준으로만 수집됨.
수집되지 않는 원인은 디텍션이 잘되지 않아서인데, 이 부분은 직접 모델을 생성해서 해결할 수 있지만 또 다른 문제가 되어서 포즈만 사용함""")

st.markdown("""
```keypoint_dict_pose.append(
    {str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.pose_landmarks.landmark)}
    )```""")
st.write("""
Pose 키포인트의 경우 33개가 존재하고, 위와 같은 방식으로 각 프레임에 대해 모든 pose 키포인트를 수집해 json 형식으로 저장함.

33개의 모든 키포인트가 매 프레임마다 수집되는건 아니므로 try-except로 감싸, 객체가 검출되지 않은 경우에도 다른 부분들은 수집할 수 있게 함.""")









col1, col2 = st.columns(2)

with col1:
    st.image("https://play-lh.googleusercontent.com/bWEmtRWCE9S0Gskc-VHKRjobMqm1tMx3ovmZxB0QgZAHZI7Xy1hr3R46UXC1lf07rEs=w526-h296-rw")
    st.image("https://i.gifer.com/M6J1.gif")

with col2:
    st.write("센서를 손에 장착하고, 그 센서로만 동작을 감지하여 유사도를 통해 점수를 매기는 서비스")


st.subheader("RingFit")

col3, col4 = st.columns(2)

with col3:
    st.image("https://ringfitadventure.nintendo.com/assets/img/share-tw.jpg")
    st.image("https://media1.giphy.com/media/ga9PPYHiotg8SO2yVb/giphy.gif")
with col4:
    st.write("모션 감지 및 동작 감지 기기를 이용하여 운동을 게임처럼 할 수 있는 서비스")



st.subheader("Kinect")

col5, col6 = st.columns(2)

with col5:
    st.image("https://www.gamespot.com/a/uploads/original/gamespot/images/2010/164/1518669-997628_20100614_001.jpg")
    st.image("https://images-na.ssl-images-amazon.com/images/G/01/videogames/detail-page/kinectjoyride.04.lg.jpg")
with col6:
    st.write("Kinect라는 카메라 센서를 통해 게임을 할 수 있는 서비스")




