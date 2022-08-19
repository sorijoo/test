from markdown import markdown
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

# st.markdown("""
# ```def download_video(self):
#     self.__save_dance_name()
#     url = input(f"{self.__dance_name}의 안무 영상 링크: ")
#     if not os.path.exists(self.__video_download_path): os.mkdir(self.__video_download_path)
#     yt = pytube.YouTube(url).streams.filter(res="720p").first()
#     yt.download(output_path=self.__video_download_path, filename=self.__dance_name+".mp4")```""")

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


st.subheader('스케일링 및 출력')

st.write("""
추출된 좌표들은 [x, y, z]로 3차원 좌표고, 해당 좌표들은 영상 사이즈에 맞게 0에서 1사이로 정규화되어 있음.
""")
st.write("""
예를 들어, 테스트에서 사용한 “[주간아] 아이브 이서 러브다이브“의 경우 408*720 사이즈에 맞춰 정규화되어 있음.
""")
st.write("""
사용자의 화면은 1280*720으로 설정했는데, 해당 화면에 추출된 좌표를 알맞은 위치에 출력하기 위해서는 스케일링이 필요함.
""")
st.markdown("""
```try:
    # get coors MARGIN
    cors_margin = self.__get_margin([user_input["0"], user_input["23"], user_input["24"]], [dance_cors[dance_cors_frames][0], dance_cors[dance_cors_frames][23], dance_cors[dance_cors_frames][24]])
    for pose_point in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        x_cor_pose, y_cor_pose, z_cor_pose = int((dance_cors[dance_cors_frames][pose_point][0]+cors_margin[0])*user_image.shape[1]), int((dance_cors[dance_cors_frames][pose_point][1]+cors_margin[1])*user_image.shape[0]), int((dance_cors[dance_cors_frames][pose_point][2]+cors_margin[2])*1000)
        cv2.circle(user_image, (x_cor_pose, y_cor_pose), 8, (244, 244, 244), cv2.FILLED)
        skeletons[pose_point] = (x_cor_pose, y_cor_pose)

        self.__draw_skeleton(user_image, skeletons)
        dance_cors_frames +=1
except: pass```""")
st.image("https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png")
st.write("""
불러온 데이터는 행이 프레임 번호고, 각 열이 pose에 맵핑되어 있는 키.""")
st.markdown("""
```dance_cors[0][0] # 첫번째 프레임의 코의 좌표```""")

st.write("cors_maring은 두 개의 영상에서 등장하는 사람들의 코-왼쪽 엉덩이-오른쪽 엉덩이를 연결해 얻은 삼각형의 무게 중심 좌표의 차이값. 스케일링 이전에 해당 값만큼 좌표들을 매 프레임 이동시켜주면 트랙킹 효과를 얻을 수 있음.")

st.markdown("""```SUDO:
user_ratio = Distance Two Points
dance_ratio = Distance Two Points
ratio = user_ratio/dance_ratio

x_cor_pose, y_cor_pose = x_cor_pose*ratio, y_cor_pose*ratio```""")

st.write("위와 같은 방식으로 각 영상에서의 기준 길이를 구해 각 좌표를 이동 시켜주는 등 여러가지 방법을 시도해봤지만 성공하지는 못했음. 어깨의 길이나 키의 길이를 구해 사용했는데, x축을 이동하는 방식은,")

st.markdown("""```x_cor_pose = dance_cors[dance_cors_frames][pose_point][0]*user_image.shape[1]*ratio + user_image.shape[1]*ratio```""")

st.write("""위와 같은 방식으로 대강 스케일링은 가능한데 일반적으로 스케일링이 되지 않고, 값의 변화량이 커서 그려주는 스켈레톤의 위치가 안정적이지 못해서 일단은 구햔하지 못했음.

모든 부위를 출력해주는 것이 아니라, 신체의 일부분만 출력하게 함
얼굴과 손은 사용하지 않고, 다른 부분들만 출력하게 함. 처음에는 발목까지만 그렸는데, 공간감이 많이 떨어져서 발까지 추가로 그림.

opencv에서 선을 그리는 건 점을 이용하기 때문에, 사실상 선의 각 키포인트에 대한 정보가 중요했음.""")

st.markdown("""
```python
def __draw_skeleton(self, image, skeleton):
    # 오른쪽 스켈레톤 (붉은색)
    cv2.line(image, skeleton[12], skeleton[14], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/팔꿈치
    cv2.line(image, skeleton[14], skeleton[16], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/팔꿈치 -> 오/손목
    cv2.line(image, skeleton[12], skeleton[24], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/엉덩이
    cv2.line(image, skeleton[24], skeleton[26], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/엉덩이 -> 오/무릎
    cv2.line(image, skeleton[26], skeleton[28], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/무릎 -> 오/발목
    cv2.line(image, skeleton[28], skeleton[30], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/발목 -> 오/뒷꿈치
    cv2.line(image, skeleton[30], skeleton[32], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오른발
    cv2.line(image, skeleton[28], skeleton[32], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오른발
    # 왼쪽 스켈레톤 (푸른색)
    cv2.line(image, skeleton[11], skeleton[13], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/팔꿈치
    cv2.line(image, skeleton[13], skeleton[15], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/팔꿈치 -> 왼/손목
    cv2.line(image, skeleton[11], skeleton[23], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/엉덩이
    cv2.line(image, skeleton[23], skeleton[25], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/엉덩이 -> 왼/무릎
    cv2.line(image, skeleton[25], skeleton[27], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/무릎 -> 왼/발목
    cv2.line(image, skeleton[27], skeleton[29], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/발목 -> 왼/뒷꿈치
    cv2.line(image, skeleton[29], skeleton[31], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼발
    cv2.line(image, skeleton[27], skeleton[31], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼발
    # 상체 스켈레톤 (회색)
    cv2.line(image, skeleton[11], skeleton[12], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)
    cv2.line(image, skeleton[23], skeleton[24], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)   ```""")

st.subheader('정확도 측정')

st.write("""
n-차원 공간에서 두 벡터의 유사도를 측정하는 방식에 대해 고민을 하다가, L2-Norm을 사용
""")
st.markdown("""
```acc_per_frame.append(np.round(self.__const_k / (np.linalg.norm([(x_cor_pose/user_image.shape[1]-cors_margin[0])-user_input[str(pose_point)][0], (y_cor_pose/user_image.shape[0]-cors_margin[1])-user_input[str(pose_point)][1], (z_cor_pose/1000-cors_margin[2])-user_input[str(pose_point)][2]]) + self.__const_k), 2))

acc = np.mean(acc_per_frame)*100```""")

st.latex("""
Acc = {K \over L2_Norm + K}
""")
st.write("""
L2-Norm의 경우, 두 벡터가 일치하면(기존 안무와 사용자의 안무가 일치하면) 0이 나오고, 차이가 날 수록 값이 증가한다는 점을 이용,
초기 K의 경우, 50% 일치된 동작의 L2-Norm값을 측정해, 0.6을 사용함. 해당 값을 이용해 난이도를 조절할 수 있다. K값이 감소 할 수록 좀 더 엄격하게 측정.
만약, 포즈 이외의 손이나 얼굴들의 새로운 키포인트도 추가로 이용하다면 위의 식을 변형에 사용할 수 있다. 손이나 표정 등은 포즈에 비해 상대적으로 중요도가 떨어지므로 가중치를 부여할 수 있음.

""")
st.latex("""
ACC = {K \over L2_pose + w \times L2_hand + K}
""")

st.subheader('테스트 영상')
video_file = open('data/Just DDance! 2022-08-15 23-35-45.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.markdown("[![test](https://img.youtube.com/vi/iyMM7Ysq-iA/0.jpg)](https://www.youtube.com/watch?v=iyMM7Ysq-iA)")
st.image('https://nuyhc.github.io/assets/images/sourceImg/JustDDance/justddance_1.png')
st.markdown("[![test](https://img.youtube.com/vi/6n5VLdn-j08/0.jpg)](https://www.youtube.com/watch?v=6n5VLdn-j08)")
st.image('https://nuyhc.github.io/assets/images/sourceImg/JustDDance/justddance_2.png')
st.image('https://nuyhc.github.io/assets/images/sourceImg/JustDDance/justddance_2.png')


st.subheader('한계 및 개선 방안')

st.markdown("""
#### 1. 프레임 드랍 문제
기본적으로 mediapipe 모델이 동작하면 프레임이 강제적으로 드랍된다.
드랍된 프레임을 조금이라고 보완하기 위해서 프레임수를 직접 계산해 강제적으로 프레임을 넘기는 방법을 선택했다. 이전보다는 상당 부분 개선이 되었지만 여전히 문제가 있다.
```
pTime = 0
FPS = 댄스 영상.get(cv2.CAP_PROP_FPS)

while 웹캠이 실행되는 동안:
    cTime = time.time() - pTime

    if cTime > 1./FPS:
        pTime = time.time()
        출력
```
테스트 영상에서는 소리가 함께 녹음되지 않았지만, 영상으로부터 추출된 음원은 정속도로 출력되는 영상은 프레임이 떨어져 음악이 먼저 끝난다.
음악 재생에 사용한 ffpyplayer 라이브러리도 프레임을 받아와 동작하는거 같아서, 해당 프레임을 영상이랑 맞춰주면 조금 느리게도 재생이 가능할꺼 같지만 자료가 많지 않아서 해결하지는 못했다.

#### 2. 스케일링 문제  

첫번째 영상은, 사이즈가 다른 2개의 영상을 이용했다. 먼저, 데이터를 추출한 영상은 아이브의 이서님이 춘 러브다이브고, 사용자 입력을 대신해서 넣은 오른쪽 영상은 안무 커버 영상이다.  
[![test](https://img.youtube.com/vi/iyMM7Ysq-iA/0.jpg)](https://www.youtube.com/watch?v=iyMM7Ysq-iA)

안무 커버 영상이다보니, 특정 멤버를 쭉 따라한게 아니라 노래에 맞춰 센터에 있는 멤버의 춤을 추는 모습을 확인할 수 있었다.  

대부분 안무가 비슷하나, 다른 멤버의 안무를 추는 경우에는 스켈레톤과 차이가 많이 난다.  

첫번째 영상에서 볼 수 있듯이, 스켈레톤이 인식한 사람을 따라 다니는 모습을 볼 수 있지만, 스케일링이 되지 않았다는 것을 확인할 수 있다.  

두 번째 영상은, 사용자의 화면은 왼쪽 화면과 같은 크기로 맞춘 경우인데, 이 경우에는 자동으로 스케일링이 되는 모습을 확인 할 수 있다.  
[![test](https://img.youtube.com/vi/6n5VLdn-j08/0.jpg)](https://www.youtube.com/watch?v=6n5VLdn-j08)

웹캠으로 받는 화면의 크기도 왼쪽 영상(원래 안무 영상)과 사이즈를 맞추면 스케일링이 자동으로 되지만, 원하던 프로젝트 방향이 아니라 따로 수정하지는 않았다.

해당 과정을 수행하기 위해서는 특정 스케일 기준 값을 정하고, 해당 값에 맞춰 좌표를 이동 시켜줘야한다는 사실은 알지만 쉽게 구현하지는 못했다.

#### 3. 여러 사람이 등장하는 경우

해당 프로젝트에서 사용한 mediapipe말고 openpose와 같은 디텍션 모델들은 여러명의 사람에 대해 포즈 추정을 해준다.  

YOLO 모델과 mediapipe를 사용해 해결할수도 있지만, 테스트 결과 프레임 드랍이 더 심해지고 몇명의 사람이 잡힐지에 대한 처리가 애매해서 사용하지 않았다.

#### 4. 안무 예측 및 추천 시스템Permalink  
해당 프로젝트 초기 기획했던 서비스 중 하나인데, 구현하지는 못했다.  

사용자의 춤을 입력 받아, 춤과 어울리는 노래를 추천해주는 기능도 만들고 싶었는데, CV 부분에서 너무 많은 시간을 사용해 손 대보지 못했다.  


노래를 이용해 장르와 춤에 대한 데이터 베이스를 만들어 장르로 라벨링을 해 모델을 생성해 적용하면 된다는 생각이었는데, 시도해본 팀원의 의견에 따르면 어렵다고 한다.

#### 5. 실시간 피드백 서비스Permalink
L2-Norm을 이용해 정확도를 측정하고 있는데, 정확도가 특정값 보다 적게 나오면, 영상 종류 후 비교해주는 것과 실시간으로 특정 신체 부위를 어떻게 더 수정해라와 같은 기능도 기획했지만, 역시 CV에서 너무 많은 시간을 사용해서 구현하지 못했다.
"""
)

st.subheader('전체 소스 코드')

st.markdown("""```python
class JustDDance():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_style = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    def __init__(self, const_k=0.6):
        self.__const_k = const_k
        self.__video_download_path = "video"
        self.__keypoints_path = "keypoints"
        self.__dance_name = None
        self.__accumulate_acc = []
    def set_const_k(self):
        self.__const_k = float(input("난이도 조절(0~1 사이 값): "))
    def get_const_k(self):
        print(f"현재 난이도: {self.__const_k}")
    def __get_accumlate_acc(self):
        return self.__accumulate_acc
    def __save_dance_name(self):
        self.__dance_name = input("누구의 무슨 춤?: 안유진 러브다이브")
    def set_dance_name(self, s):
        self.__dance_name = s
    def print_dance_data(self):
        acc_acc = self.__get_accumlate_acc()
        accMax, accMin, accMean = np.max(acc_acc), np.min(acc_acc), np.mean(acc_acc)
        print(f"Max Acc: {accMax}\tMin Acc: {accMin}\tAvg. Acc: {accMean}\n")
        acc_acc = pd.DataFrame(acc_acc)
        acc_acc.plot(figsize=(25, 6))
        plt.title("Accuarcy for Frames")
        plt.xlabel("Frames")
        plt.ylabel("Accuarcy")
        plt.legend("Acc")
        plt.axhline(y=70, color="r")
        plt.show()
    def download_video(self):
        self.__save_dance_name()
        url = input(f"{self.__dance_name}의 안무 영상 링크: ")
        if not os.path.exists(self.__video_download_path): os.mkdir(self.__video_download_path)
        yt = pytube.YouTube(url).streams.filter(res="720p").first()
        yt.download(output_path=self.__video_download_path, filename=self.__dance_name+".mp4")
    def __draw_skeleton(self, image, skeleton):
        # 오른쪽 스켈레톤 (붉은색)
        cv2.line(image, skeleton[12], skeleton[14], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/팔꿈치
        cv2.line(image, skeleton[14], skeleton[16], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/팔꿈치 -> 오/손목
        cv2.line(image, skeleton[12], skeleton[24], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/어깨 -> 오/엉덩이
        cv2.line(image, skeleton[24], skeleton[26], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/엉덩이 -> 오/무릎
        cv2.line(image, skeleton[26], skeleton[28], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/무릎 -> 오/발목
        cv2.line(image, skeleton[28], skeleton[30], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오/발목 -> 오/뒷꿈치
        cv2.line(image, skeleton[30], skeleton[32], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오른발
        cv2.line(image, skeleton[28], skeleton[32], (102, 102, 255), thickness=7, lineType=cv2.LINE_AA, shift=None) # 오른발
        # 왼쪽 스켈레톤 (푸른색)
        cv2.line(image, skeleton[11], skeleton[13], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/팔꿈치
        cv2.line(image, skeleton[13], skeleton[15], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/팔꿈치 -> 왼/손목
        cv2.line(image, skeleton[11], skeleton[23], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/어깨 -> 왼/엉덩이
        cv2.line(image, skeleton[23], skeleton[25], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/엉덩이 -> 왼/무릎
        cv2.line(image, skeleton[25], skeleton[27], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/무릎 -> 왼/발목
        cv2.line(image, skeleton[27], skeleton[29], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼/발목 -> 왼/뒷꿈치
        cv2.line(image, skeleton[29], skeleton[31], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼발
        cv2.line(image, skeleton[27], skeleton[31], (255, 102, 102), thickness=7, lineType=cv2.LINE_AA, shift=None) # 왼발
        # 상체 스켈레톤 (회색)
        cv2.line(image, skeleton[11], skeleton[12], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)
        cv2.line(image, skeleton[23], skeleton[24], (224, 224, 224), thickness=5, lineType=cv2.LINE_AA, shift=None)   
    def __get_margin(self, user_tri, dance_tri):
        margin = []
        ut = [(user_tri[0][0]+user_tri[1][0]+user_tri[2][0])/3, (user_tri[0][1]+user_tri[1][1]+user_tri[2][1])/3, (user_tri[0][2]+user_tri[1][2]+user_tri[2][2])/3]
        dt = [(dance_tri[0][0]+dance_tri[1][0]+dance_tri[2][0])/3, (dance_tri[0][1]+dance_tri[1][1]+dance_tri[2][1])/3, (dance_tri[0][2]+dance_tri[1][2]+dance_tri[2][2])/3]
        for u, d in zip(ut, dt): margin.append(u-d)
        return margin
    def __get_distance(self, pt1, pt2):
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    def __load_cor_data(self):
        with open(self.__keypoints_path+"/"+self.__dance_name+"_keypoints.json", "r") as keypoints:
            data = json.load(keypoints)
            return np.array(pd.DataFrame(data))
    def extract_keypoints(self, isMirr=False, showExtract=False):
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
            json.dump(keypoint_dict_pose, keypoints)
    def show_dance_tutorial(self):
        cv2.startWindowThread()
        dance = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        try: user = cv2.VideoCapture(0)
        except: user = cv2.VideoCapture(1)
        user.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        user.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        player = MediaPlayer(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        dance_cors = self.__load_cor_data()
        dance_cors_frames = 0
        skeletons = {}
        pTime = 0
        FPS = dance.get(cv2.CAP_PROP_FPS)
        
        with self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while user.isOpened():
                cTime = time.time()-pTime
                user_ret, user_image = user.read()
                dance_ret, dance_image = dance.read()
                if not user_ret: break
                if not dance_ret: break
                
                if cTime>1./FPS:
                    audio_frame, val = player.get_frame()
                    pTime = time.time()
                    acc_per_frame = []
                    user_image = cv2.cvtColor(cv2.flip(user_image, 1), cv2.COLOR_BGR2RGB)
                    user_results = pose.process(user_image)
                    user_image = cv2.cvtColor(user_image, cv2.COLOR_RGB2BGR)
                    # 사용자
                    # self.mp_drawing.draw_landmarks(user_image, user_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    #                             landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(244, 244, 244), thickness=2, circle_radius=1),
                    #                             connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(153, 255, 153), thickness=2, circle_radius=1))
                    try:
                        user_input = {str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(user_results.pose_landmarks.landmark)}
                    except: pass
                    # 추출해 온 데이터
                    try:
                        # get coors MARGIN
                        cors_margin = self.__get_margin([user_input["0"], user_input["23"], user_input["24"]], [dance_cors[dance_cors_frames][0], dance_cors[dance_cors_frames][23], dance_cors[dance_cors_frames][24]])
                        for pose_point in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
                            x_cor_pose, y_cor_pose, z_cor_pose = int((dance_cors[dance_cors_frames][pose_point][0]+cors_margin[0])*user_image.shape[1]), int((dance_cors[dance_cors_frames][pose_point][1]+cors_margin[1])*user_image.shape[0]), int((dance_cors[dance_cors_frames][pose_point][2]+cors_margin[2])*1000)
                            cv2.circle(user_image, (x_cor_pose, y_cor_pose), 8, (244, 244, 244), cv2.FILLED)
                            skeletons[pose_point] = (x_cor_pose, y_cor_pose)
                            # L2 Norm
                            acc_per_frame.append(np.round(self.__const_k / (np.linalg.norm([(x_cor_pose/user_image.shape[1]-cors_margin[0])-user_input[str(pose_point)][0], (y_cor_pose/user_image.shape[0]-cors_margin[1])-user_input[str(pose_point)][1], (z_cor_pose/1000-cors_margin[2])-user_input[str(pose_point)][2]]) + self.__const_k), 2))
                            acc = np.mean(acc_per_frame)*100
                            self.__accumulate_acc.append(acc)
                        cv2.putText(user_image, str(acc)+"%", (20, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        self.__draw_skeleton(user_image, skeletons)
                        dance_cors_frames +=1
                    except: pass
                    h_output = np.hstack((cv2.flip(dance_image, 1), user_image))
                    cv2.imshow("Just DDance!", h_output)
                if cv2.waitKey(1)&0xFF==ord("q"): break
        player.close_player()
        user.release()
        dance.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


```""")
