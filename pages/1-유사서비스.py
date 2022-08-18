import streamlit as st

st.header('유사 서비스')
st.subheader("Just Dance")



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




