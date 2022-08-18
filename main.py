import streamlit as st
import pandas as pd
import numpy as np
from html_module import line_break, section, callout, title
from PIL import Image

# 전체 페이지 설정
st.set_page_config(
    page_title="Just DDance!",
    page_icon="🕺",  # 아이콘
    initial_sidebar_state="expanded", 
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.balloons()  # 풍선 효과
title('Motion Detection Dance Practice Service')
line_break()

# Team member section
st.header('Team Member')
st.markdown('김채현, 김동영, 서진원, 주소리')
line_break()

st.header('Background')

st.write('코로나19로 인해 재택근무, 유연근무제 등이 확산되면서 직장에 머무는 시간이 감소하고, 가정에 머무는 시간은 증가하였다(강선경, 최윤, 2021)')
st.write('국외 연구들은 코로나19로 인해 신체 활동이 감소하였고(Moore et al., 2020), 수면 시간이 증가하였으며(Xiang, Zhang, Kuwahara, 2020), 음주를 즐기는 경우도 늘어났다고 밝히고 있다(Stanton et al., 2020). 또한 미디어 콘텐츠를 이용하는 시간이 증가하였으며(Gammon & Ramshaw, 2020), 게임을 즐기는 시간도 증가하였음을 보고하고 있다(Zhu et al., 2021). 가족과 함께 여가활동을 하는 경우도 늘어났고(EasterbrookSmith, 2020)')
st.write('집은 가장 안전한 장소로 인식되면서 사람들은 집에 머무르는 시간이 많아졌고, 실내외를 막론하고 전반적으로 활동인구가 줄어들었다(문화체육관광부, 2020).')
line_break()
st.image(Image.open('data/1112333.jpg'))
st.image(Image.open('data/666.jpeg'))
line_break()
st.markdown('''
- COVID-19의 영향으로 실내 활동의 중요성이 대두된 상황에서, 실내에서 즐길 수 있는 여가 활동의 다양성 확보를 위한 일환 중 하나로 운동을 도와주는 서비스에서 아이디어를 얻음

- 장기자랑 등을 위해 춤을 연습하는 경우, 일반적으로 안무 영상을 보며 따라하는 방식으로 연습을 하게 됨. 그러나 단순히 영상을 보며 따라하는 방법은 결국 안무를 모두 스스로 암기할 수밖에 없고, 이후 본인의 춤추는 영상을 보면서 역시 스스로 틀린 부분을 찾고 고치는 방식으로 이뤄지게 됨

- 그래서 본 프로젝트는 원본 안무 영상과 본인의 안무 영상을 비교하며 틀린 부분을 직접 찾아야 하는 번거로움을 줄여주고, 추가적인 별도의 장비나 기기 없이 카메라만 있으면 특정 동작에서 어떤 부분을 어떻게 고치면 좋을지 피드백을 해주는 서비스를 기획''')



st.header('Notion')
link = '[노션페이지로 이동](https://likelion-aischool.notion.site/Just-DDance-caaee6d19c604d899285c4383779dc0f)'
st.markdown(link, unsafe_allow_html=True)
line_break()