import streamlit as st 

import tensorflow as tf

import numpy as np


from PIL import Image


st.set_page_config(
    page_title="Likelion AI School Sign Miniproject",
    page_icon="🐶",
    layout="wide",
)
st.sidebar.markdown("SIGN SIGN")

st.title("SIGN")

st.write("""
SIGN
""")
filename = st.file_uploader("Choose a file")

st.text(type(filename))

model = tf.keras.models.load_model('model/model_kor_num_no_augmentation.h5')

# idx = [x for x in range(0, 24)]
# alpha = [chr(x).upper() for x in range(97, 123)]

# alpha.remove("J")
# alpha.remove("Z")



def convert_letter(result):
    classLabels = {idx:c for idx, c in zip(idx, alpha)}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "err"


def img_resize_to_gray(filename):
    """파일 경로를 입력 받아 사이즈 조정과 그레이로 변환하는 함수

    Args:
        filename (str): 파일 경로
    Returns:
        arr (np.array)
    """
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((300, 300))
    return img

if filename is not None:
    st.text(filename)
    st.text(type(filename))
    img = Image.open(filename)
    img = img.convert('L')
    img = img.resize((300, 300)) 
    img = np.array(img)
    pred = np.argmax(model.predict(img.reshape(1, 300, 300, 1)))
    # text = []
    st.image(img, use_column_width=False)
    st.text(pred)