import streamlit as st
import pandas as pd
import numpy as np
from html_module import line_break, section, callout, title

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

