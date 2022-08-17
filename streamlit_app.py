import numpy as np
import pandas as pd
import json

import mediapipe as mp
import cv2
import pytube

class JustDDance:
    def __init__(self):
        self.__download_path = "../dataset/video/"
        self.__keyporint_extraction_path = "../dataset/keypoint_extraction/"
        self.__dance_fps = None
        self.__dance_shape = None
        self.__user_fps = None
        self.__user_shape = None
        self.__const_k = 0.3
    
    def download_video(self):
        url = input("연습할 춤의 유튜브 링크: ")
        yt = pytube.YouTube(url)
        stream = yt.streams.filter(res="720p").first()
        stream.download(self.__download_path)
    
    def scaling_coor(self, keypoint_path): # "./keypoint_extraction/[주간아 직캠] IVE YUJIN - LOVE DIVE (아이브 유진 - 러브 다이브) l EP556_keypoints.json"
        with open(keypoint_path, "r") as file:
            data = json.load(file)
        pose_cor = pd.DataFrame(data["pose"])
        return np.array(pose_cor)
    
    def extract_keypoint(self, video_path):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_style = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        
        keypoint_dict_pose = []
        keypoint_dict_left_hand = []
        keypoint_dict_right_hand = []
        keypoint_dict = {}
        
        try: cap = cv2.VideoCapture(video_path)
        except: return -1
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                if not success: break
                self.__dance_shape = image.shape
                self.__dance_fps = cap.get(cv2.CAP_PROP_FPS)
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                try:
                    keypoint_dict_pose.append({str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.pose_landmarks.landmark)})
                    keypoint_dict_left_hand.append({str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.left_hand_landmarks.landmark)})
                    keypoint_dict_right_hand.append({str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.right_hand_landmarks.landmark)})
                except:
                    pass
                    
                keypoint_dict = {"pose": keypoint_dict_pose, "left_hand": keypoint_dict_left_hand, "right_hand": keypoint_dict_right_hand}
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        with open(self.__keyporint_extraction_path+video_path.split("/")[2].split(".")[0]+"_keypoints.json", "w") as fp:
            json.dump(keypoint_dict, fp)
            
    def show_dance_tutorial(self, video_path, keypoint_path):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_style = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic
        
        try: cap = cv2.VideoCapture(0)
        except: cap = cv2.VideoCapture(1)
        load_dance = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        dance_cors = self.scaling_coor(keypoint_path)
        dance_cors_fps = 0
        extract_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        cv2.startWindowThread()
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                ret, dance = load_dance.read()
                if not success: break
                if not ret: break
                self.__user_shape = image.shape
                self.__user_fps = cap.get(cv2.CAP_PROP_FPS)
                self.__dance_shape = dance.shape
                self.__dance_fps = load_dance.get(cv2.CAP_PROP_FPS)

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1))
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1))
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=1))
                    skeleton = {}
                    # for pose_point in range(33):
                    for pose_point in extract_points:
                        scale_x_cor_pose, scale_y_cor_pose = int(dance_cors[dance_cors_fps][pose_point][0]*self.__user_shape[1]), int(dance_cors[dance_cors_fps][pose_point][1]*self.__user_shape[0])
                        cv2.circle(image, (scale_x_cor_pose, scale_y_cor_pose), 5, (224, 224, 224), cv2.FILLED)
                        skeleton[pose_point] = (scale_x_cor_pose, scale_y_cor_pose)
                        # Acc (L2 Norm)
                        tn_x, tn_y, tn_z = dance_cors[dance_cors_fps][pose_point][0:3]
                        user_input = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                        user_x, user_y, user_z = user_input[pose_point][0:3]
                        acc = np.round(self.__const_k / (np.linalg.norm([tn_x-user_x, tn_y-user_y, tn_z-user_z]) + self.__const_k), 2)*100
                    dance_cors_fps += 1
                    cv2.putText(image, str(acc)+"%", (20, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                    # 오른쪽 스켈레톤 (붉은색)
                    cv2.line(image, skeleton[12], skeleton[14], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/팔꿈치
                    cv2.line(image, skeleton[14], skeleton[16], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오/팔꿈치 -> 오/손목
                    cv2.line(image, skeleton[12], skeleton[24], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/엉덩이
                    cv2.line(image, skeleton[24], skeleton[26], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오/엉덩이 -> 오/무릎
                    cv2.line(image, skeleton[26], skeleton[28], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오/무릎 -> 오/발목
                    cv2.line(image, skeleton[28], skeleton[30], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오/발목 -> 오/뒷꿈치
                    cv2.line(image, skeleton[30], skeleton[32], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오른발
                    cv2.line(image, skeleton[28], skeleton[32], (102, 102, 255), thickness=5, lineType=cv2.LINE_AA, shift=None) # 오른발
                    # 왼쪽 스켈레톤 (푸른색)
                    cv2.line(image, skeleton[11], skeleton[13], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/팔꿈치
                    cv2.line(image, skeleton[13], skeleton[15], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼/팔꿈치 -> 왼/손목
                    cv2.line(image, skeleton[11], skeleton[23], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/엉덩이
                    cv2.line(image, skeleton[23], skeleton[25], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼/엉덩이 -> 왼/무릎
                    cv2.line(image, skeleton[25], skeleton[27], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼/무릎 -> 왼/발목
                    cv2.line(image, skeleton[27], skeleton[29], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼/발목 -> 왼/뒷꿈치
                    cv2.line(image, skeleton[29], skeleton[31], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼발
                    cv2.line(image, skeleton[27], skeleton[31], (255, 102, 102), thickness=5, lineType=cv2.LINE_AA, shift=None) # 왼발
                    
                    cv2.line(image, skeleton[11], skeleton[12], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)
                    cv2.line(image, skeleton[23], skeleton[24], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)
                except:
                    pass
                # TODO: 싱크 문제 해결
                h_output = np.hstack((cv2.flip(dance, 1), image))
                cv2.imshow("Just DDance!", h_output)
                if cv2.waitKey(1)&0xFF==ord("q"): break
        load_dance.release()        
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)



# STREAMLIT CODE


import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        image = show_dance_tutorial(image)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)