import streamlit as st

st.header('유사 서비스')
st.subheader("Just Dance")
st.subheader("RingFit")
st.subheader("Kinect")


with tab1:
    # 개요 section
    section('Summary')
    image = Image.open('images/sign_language.jpg')
    st.image(image,)
    callout([
      '안녕하세요! ✌ Sign팀입니다. 🙂',
      'Kaggle의 English Sign Language 데이터를 이용하여 CNN Project를 진행하였습니다.',
      'OpenCV를 이용하여 실시간으로 알파벳 수화를 예측할 수 있도록 만들었습니다.'
    ])
    line_break()

    # 데이터 출처 section
    section('Dataset')
    link = 'https://www.kaggle.com/datasets/datamunge/sign-language-mnist'
    st.markdown(link, unsafe_allow_html=True)
    st.caption('데이터 출처 사이트로 이동하기')
    line_break()

    # Sign Team Notion section
    section('Notion')
    link = 'https://www.notion.so/likelion-aischool/English-Sign-Language-759acc98547a4e259367a63625ba2158'
    st.markdown(link, unsafe_allow_html=True)
    st.caption('팀 노션 페이지로 이동하기')
    line_break()

with tab2:
    # 개요 section
    section('Summary')
    image = Image.open('images/korean_sign_lang.png')
    st.image(image,)
    callout([
      '안녕하세요! ✌ Sign팀입니다. 🙂',
      'Kaggle의 Korean Sign Language Number 데이터를 이용하여 CNN Project를 진행하였습니다.',
      '이미지를 업로드하면 해당 수화 이미지가 어느 숫자를 의미히는지 예측합니다.'
    ])
    line_break()

    # 데이터 출처 section
    section('Dataset')
    link = 'https://www.kaggle.com/datasets/nahyunpark/korean-sign-languageksl-numbers'
    st.markdown(link, unsafe_allow_html=True)
    st.caption('데이터 출처 사이트로 이동하기')
    line_break()

    # Sign Team Notion section
    section('Notion')
    link = 'https://www.notion.so/likelion-aischool/English-Sign-Language-759acc98547a4e259367a63625ba2158'
    st.markdown(link, unsafe_allow_html=True)
    st.caption('팀 노션 페이지로 이동하기')
    line_break()